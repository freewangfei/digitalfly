"""数字果蝇的交互式 Web 控制台。

后台一条仿真线程持续跑 感觉 -> 全脑网络 -> 指令 -> 运动模式发生器 -> 物理，
前端通过两条流拿数据：
    /video   MuJoCo 离屏渲染的 MJPEG 流（3D 画面）
    /stream  SSE 数值流（全脑活动、脑区发放率、行为状态）
两条流互相独立，视频卡住不会阻塞数值面板。

可以在线切换行为（行走 / 觅食 / 拴系飞行）、注入刺激、切除神经元、
接管速度与转向、移动糖源、切换机位。
"""
from __future__ import annotations

import io
import json
import threading
import time
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

STATIC = Path(__file__).parent / "static"

# 可注入刺激的神经元群：显示名 -> NeuronIndex 上的方法名
STIM_GROUPS = {
    "糖味通路 Sugar SEL": "sugar_pathway",
    "苦味通路 Bitter SEL": "bitter_pathway",
    "唇瓣味觉 GRN": "labellar_grns",
    "食物气味通道 ORN": "food_odor_orns",
    "嗅觉（全部）": "olfactory",
    "视觉 Photoreceptors": "photoreceptors",
    "本体感觉": "proprioceptive",
    "机械感觉": "mechanosensory",
    "下行神经元 DN": "descending",
}

BEHAVIOR_LABELS = {"walk": "行走", "forage": "觅食", "flight": "拴系飞行",
                   "cube": "拧魔方"}
CTRL_EVERY = 20          # 每 20 个脑步（10 ms）更新一次运动指令


class Simulation:
    """后台仿真线程。持有大脑，以及当前行为对应的身体与模式发生器。"""

    def __init__(self, backend: str | None = None, behavior: str = "walk",
                 raster_size: int = 120):
        from ..brain import Brain
        from ..bridge import CommandBridge
        from ..calibrate import calibrated_params
        from ..connectome import load
        from ..neurons import NeuronIndex

        self.c = load()
        self.idx = NeuronIndex(self.c)
        self.brain = Brain(self.c.W, calibrated_params(), backend=backend)
        self.bridge = CommandBridge(self.c, dt_ms=self.brain.p.dt)
        self.regions = self.idx.by_region()
        self.mn9 = self.idx.proboscis_motor()

        # 全脑三维点云：神经元发放时闪烁，和任务画面拼成分屏
        from ..brainview import BrainView
        self.view = BrainView(self.c.meta, self.regions, size=(360, 420))
        self.split = True

        rng = np.random.default_rng(0)
        self.raster_idx = rng.choice(self.brain.n, size=raster_size,
                                     replace=False)

        # 两把锁：脑线程和身体线程各用各的，否则身体那几百个物理子步会把脑
        # 挡在锁外面 —— 实测共用一把锁时脑只能推进到真实时间的 1%。
        self.lock = threading.RLock()        # 大脑与共享状态
        self.blk = threading.RLock()         # 身体与渲染
        self.running = True
        self.stim: dict[str, float] = {}
        self.ablated: set[str] = set()
        self.frame: bytes | None = None
        self.latest: dict = {}
        self.error: str | None = None
        self._stop = threading.Event()
        self._ext = np.zeros(self.brain.n, dtype=np.float32)
        self._last_spikes = np.zeros(self.brain.n, dtype=np.float32)
        self.cmd = None
        self._minfo: dict = {}       # 身体线程写、脑线程读的运动状态
        self._group_cache: dict[str, np.ndarray] = {}

        # 手动接管：None 表示交给大脑
        self.manual_speed: float | None = None
        self.manual_turn: float | None = None
        self.food_xy = [2.2, 1.2]
        self.odor_on = True
        self.camera = "track1"

        self.body = None
        self.behavior = None
        self.recognizer = None
        self.recog_error = None
        self.set_behavior(behavior)

    def get_recognizer(self):
        """懒加载手写识别器（要额外建一份视觉编码器，比较重）。"""
        if self.recognizer is None and self.recog_error is None:
            try:
                from ..handwriting import Recognizer
                self.recognizer = Recognizer(self.c, verbose=False)
            except Exception as e:                        # noqa: BLE001
                self.recog_error = str(e)
        return self.recognizer

    # -- 行为切换 -----------------------------------------------------------
    def set_behavior(self, name: str) -> None:
        """切换行为。不同行为的身体配置不同，要重建身体。"""
        from ..body import FlyBody
        from ..locomotion import PreprogrammedGait, WingBeat

        if name not in BEHAVIOR_LABELS:
            raise ValueError(f"未知行为 {name}")
        with self.lock, self.blk:
            if self.body is not None:
                self.body.close()
            self.behavior = name
            self.scene = None
            if name == "cube":
                from ..cube_scene import CubeScene
                # 魔方尺寸取得和果蝇差不多大 —— 网上那些视频里就是这个比例，
                # 也只有这个尺度前腿才够得着面。
                CUBE = 0.26
                self.scene = CubeScene(size=CUBE,
                                       pos=(0.42, 0.0, CUBE / 2))
                self.body = FlyBody(mode="walk", render_size=(360, 480),
                                    cube_scene=self.scene)
                self.scene.bind(self.body.model, self.body.data)
                self._new_cube()
            else:
                self.body = FlyBody(mode=name, render_size=(360, 480))
            # Web 交互模式把物理步长从 0.1 ms 放大到 0.4 ms。flybody 默认的
            # 0.1 ms 在这台机器上纯物理只跑到真实时间的 0.25 倍，画面必然是慢动作。
            # 实测 0.4 ms 仍然稳定（不发散、不摔倒），步速 1.31 vs 1.07 体长/秒，
            # 而速度提到 1.18 倍真实时间；0.8 ms 接触就失效了（速度掉到 0.42）。
            # 离线出视频的 `behave` 命令仍用 0.1 ms，保真度不受影响。
            self.body.model.opt.timestep = 4e-4
            self.gait = self.wings = None
            if name == "flight":
                self.dt_ctrl = 2e-4
                self.wings = WingBeat(self.body, dt_ctrl=self.dt_ctrl)
                for a in self.wings.act:
                    self.body.model.actuator_gainprm[a][0] = 70.0
            else:
                self.gait = PreprogrammedGait(self.body)
                self.dt_ctrl = self.brain.p.dt / 1000.0 * CTRL_EVERY
            # "orbit" = 可鼠标拖动的自由视角，放在最前面作为默认
            self.cameras = ["orbit"] + list(self.body.camera_names)
            want = {"cube": "orbit", "flight": "flight_close"}.get(
                name, "orbit")
            if name in ("cube", "flight") or self.camera not in self.cameras:
                self.camera = (want if want in self.cameras
                               else self.cameras[0])
            self._fresh()

    def _new_cube(self, scramble: int = 8, seed: int | None = None) -> None:
        """打乱并求解一局魔方（已持锁）。"""
        import random

        from ..cube import Cube, solve
        self.scene.state = Cube()
        seed = random.randrange(10000) if seed is None else seed
        self.cube_scramble = self.scene.state.scramble(scramble, seed=seed)
        sol, src = solve(self.scene.state.copy(), fallback=self.cube_scramble)
        self.cube_solution = sol
        self.cube_source = src
        self.cube_i = 0
        self.cube_charge = 0.0
        self.cube_agree = []
        self.cube_solved_at = None
        self.scene.write()

    def _fresh(self) -> None:
        """复位大脑、身体与各层状态（已持锁）。"""
        self.brain.reset()
        self.bridge.reset()
        self.body.reset()
        if self.gait:
            self.gait.reset()
        if self.wings:
            self.wings.reset()
        self.tether = np.array(self.body.data.qpos[:7], dtype=np.float64)
        self.ctrl = self.body.rest.copy()
        self.h0 = self.body.heading().copy()
        self.step_i = 0
        self.reached = False
        self.cur_turn = 0.0
        self.cur_speed = 0.0
        self.proboscis = 0.0
        self.error = None
        if self.behavior == "cube":
            self._new_cube()

    # -- 群体索引 ----------------------------------------------------------
    def group_indices(self, name: str) -> np.ndarray:
        if name not in self._group_cache:
            method = STIM_GROUPS.get(name)
            self._group_cache[name] = (
                getattr(self.idx, method)() if method
                else np.array([], dtype=np.int64))
        return self._group_cache[name]

    # -- 控制 --------------------------------------------------------------
    def set_stim(self, name: str, rate_hz: float) -> None:
        with self.lock:
            if rate_hz <= 0:
                self.stim.pop(name, None)
            else:
                self.stim[name] = float(rate_hz)

    def ablate(self, name: str) -> int:
        idx = self.group_indices(name)
        with self.lock:
            self.brain.ablate(idx)
            self.ablated.add(name)
        return len(idx)

    def reset(self) -> None:
        with self.lock, self.blk:
            self.stim.clear()
            self.ablated.clear()
            self._fresh()

    # -- 主循环 ------------------------------------------------------------
    def loop(self) -> None:
        from PIL import Image

        dt_brain = self.brain.p.dt / 1000.0
        window = np.zeros(self.brain.n, dtype=np.float32)
        window_steps = 0
        raster_buf: list[np.ndarray] = []
        t_wall = time.time()
        info: dict = {}
        t_push = 0.0
        PUSH_DT = 0.1

        while not self._stop.is_set():
            if not self.running:
                time.sleep(0.05)
                continue
            try:
                now = time.time()
                with self.lock:
                    spikes, info = self._advance(dt_brain, dt_body=0.0)
                    self.step_i += 1
            except Exception as e:                        # noqa: BLE001
                self.error = f"{type(e).__name__}: {e}"
                self.running = False
                continue

            window += spikes
            window_steps += 1
            raster_buf.append(spikes[self.raster_idx].astype(np.uint8))

            if now - t_push >= PUSH_DT and window_steps:
                t_push = now
                secs = window_steps * dt_brain

                def rate_of(v):
                    return float(window[v].sum() / max(len(v), 1) / secs)

                now = time.time()
                d = self.body.displacement()
                self.latest = {
                    "t_ms": round(self.brain.t_ms, 1),
                    "behavior": self.behavior,
                    "mean_rate_hz": round(
                        float(window.sum() / self.brain.n / secs), 2),
                    "regions": {k: round(rate_of(v), 2)
                                for k, v in self.regions.items()},
                    "mn9_hz": round(rate_of(self.mn9), 1),
                    "dn_hz": round(rate_of(self.bridge.dn_all), 1),
                    "raster": [round(float(x), 3)
                               for x in np.mean(raster_buf, axis=0)],
                    "displacement": [round(float(x), 3) for x in d],
                    "speed_ratio": round(secs / max(now - t_wall, 1e-9), 2),
                    "stim": dict(self.stim),
                    "ablated": sorted(self.ablated),
                    "running": self.running,
                    "camera": self.camera,
                    "manual_speed": self.manual_speed,
                    "manual_turn": self.manual_turn,
                    "food": self.food_xy,
                    "odor_on": self.odor_on,
                    "body_time": round(float(self.body.data.time), 2),
                    "error": self.error,
                    **info,
                    **self._minfo,
                }
                window[:] = 0.0
                window_steps = 0
                raster_buf.clear()
                t_wall = now

            self.view.update(spikes, dt_brain * 1000)

    def body_loop(self) -> None:
        """身体 + 渲染线程：按**真实流逝的时间**推进物理，按固定帧率出图。

        和脑线程分开跑的原因：全脑一步要 3.5 ms，脑只能跑到真实时间的 14%，
        身体要是跟着脑步走，画面就是 3 FPS。拆开之后身体是实时的（20+ FPS），
        脑仍然在闭环里 —— 它通过 self.cmd 这条低带宽通道给出速度与转向。
        """
        from PIL import Image

        FRAME_DT = 1.0 / 25.0
        t_body = time.time()
        t_frame = 0.0
        while not self._stop.is_set():
            now = time.time()
            if not self.running:
                time.sleep(0.03)
                t_body = now
                continue
            # 单次最多补 60 ms，防止卡顿之后身体瞬移
            dt_body = min(now - t_body, 0.06)
            t_body = now
            try:
                cmd = getattr(self, "cmd", None)
                if cmd is not None and dt_body > 0:
                    odor_l, odor_r, taste = getattr(self, "sense",
                                                    (0.0, 0.0, 0.0))
                    with self.blk:
                        self._motion(dt_body, cmd, self._last_spikes,
                                     self._minfo, odor_l, odor_r, taste)
            except Exception as e:                        # noqa: BLE001
                self.error = f"{type(e).__name__}: {e}"

            if now - t_frame >= FRAME_DT:
                t_frame = now
                try:
                    with self.blk:
                        px = self.body.render(self.camera)
                    if self.split:
                        from ..brainview import draw_overlay, side_by_side
                        d = self.latest or {}
                        px = side_by_side(px, draw_overlay(
                            self.view.render(),
                            "全脑活动 · MaleCNS v1.0",
                            [f"{self.brain.n:,} 神经元 · 白点 = 此刻发放",
                             f"全脑 {d.get('mean_rate_hz', 0)} Hz　"
                             f"下行 {d.get('dn_hz', 0)} Hz"],
                            self.view.labels))
                    buf = io.BytesIO()
                    Image.fromarray(px).save(buf, format="JPEG", quality=75)
                    self.frame = buf.getvalue()
                except Exception:                        # noqa: BLE001
                    pass
            else:
                time.sleep(0.004)

    def _advance(self, dt_brain: float,
                 dt_body: float | None = None) -> tuple[np.ndarray, dict]:
        """推进一个脑步 + dt_body 秒的身体，返回 (脉冲向量, 行为状态)。

        **脑和身体是解耦的。** 脑必须按 0.5 ms 步长积分（LIF 的时间常数决定），
        全脑 185,348 个神经元、26,006,173 条突触，在 5090 上一步约 3.5 ms ——
        也就是说脑只能跑到真实时间的 14%。如果身体跟着脑步走，画面就是 3 FPS，
        看起来一卡一卡的。

        所以拆成两个线程：**脑线程**全速积分，把 speed / turn 指令写进 self.cmd；
        **身体线程**按真实流逝的时间推进物理并渲染。脑给的是低带宽指令，步态本身
        由真实运动学生成，所以降低指令更新频率不影响行为正确性，只影响脑内动力学
        与身体的时间对齐 —— 代价写在这里：脑的仿真时间比身体慢约 7 倍。要严格
        1:1 的话用 `behave` 命令离线出视频。
        """
        body, br = self.body, self.bridge
        info: dict = {}

        # --- 感觉输入 ---------------------------------------------------
        odor_l = odor_r = taste = 0.0
        if self.behavior == "forage" and self.odor_on:
            pos = body.root_pos[:2]
            h = body.heading()
            lat = np.array([-h[1], h[0]])
            food = np.asarray(self.food_xy, dtype=np.float64)
            sigma = 1.6

            def conc(p):
                return float(np.exp(-np.sum((food - p) ** 2) /
                                    (2 * sigma ** 2)))

            odor_l = conc(pos + 0.12 * lat + 0.1 * h)
            odor_r = conc(pos - 0.12 * lat + 0.1 * h)
            dist = float(np.linalg.norm(food - pos))
            taste = 1.0 if dist < 0.3 else 0.0
            if taste:
                self.reached = True
            info["dist"] = round(dist, 2)
            info["odor"] = [round(odor_l, 2), round(odor_r, 2)]
            info["reached"] = self.reached

        br.sensory_current(self.brain.n, body=body, odor_left=odor_l,
                           odor_right=odor_r, taste=taste, out=self._ext)
        # 行走 / 觅食需要背景驱动，否则网络静默、果蝇不会起步
        if self.behavior != "flight":
            br._drive(self._ext, br.proprio, 20.0)
        for name, rate in self.stim.items():
            br._drive(self._ext, self.group_indices(name), rate)

        spikes = self.brain.step(self._ext)
        cmd = br.read_command(spikes)
        info["cmd_speed"] = round(cmd.speed, 2)
        # 供身体线程读取的最新指令（脑 -> 身体只走这一条低带宽通道）
        self.cmd = cmd
        self.sense = (odor_l, odor_r, taste)
        self._last_spikes = spikes
        if dt_body is None:
            # 离线 1:1 模式：脑步之后紧跟身体步
            self._motion(dt_brain, cmd, spikes, info, odor_l, odor_r, taste)
        if self.behavior == "cube":
            info.update(self._advance_cube(dt_brain, spikes))
        info["turn"] = round(self.cur_turn, 3)
        info["speed"] = round(self.cur_speed, 2)
        return spikes, info

    # 运动指令的细分步长。果蝇一个步周期只有 50~100 ms，如果把整个 40 ms 的
    # 真实间隔一次性喂给步态发生器，相位一次就跳掉半个周期，腿会瞬移、打滑。
    # 2 ms 一档，相位推进才是连续的。
    MOTION_DT = 0.005

    def _motion(self, dt_b: float, cmd, spikes, info: dict,
                odor_l: float, odor_r: float, taste: float) -> None:
        """按 dt_b 秒推进身体，内部细分成 MOTION_DT 的小步。"""
        n_sub = max(1, int(np.ceil(dt_b / self.MOTION_DT)))
        dt_one = dt_b / n_sub
        for _ in range(n_sub):
            self._motion_once(dt_one, cmd, spikes, info,
                              odor_l, odor_r, taste)

    def _motion_once(self, dt_b: float, cmd, spikes, info: dict,
                     odor_l: float, odor_r: float, taste: float) -> None:
        body, br = self.body, self.bridge
        dt_brain = self.brain.p.dt / 1000.0
        sub = max(1, int(round(dt_b / body.timestep)))
        if self.behavior == "flight":
            turn = (self.manual_turn if self.manual_turn is not None
                    else float(np.clip(0.6 * cmd.turn, -0.8, 0.8)))
            self.cur_turn = turn
            self.wings.step(self.ctrl, freq_rel=0.0, asymmetry=turn,
                            amplitude=1.0)
            body.step(self.ctrl, n_substeps=sub)
            # 拴系：躯干固定，只看翅膀运动学与转向读数
            body.data.qpos[:7] = self.tether
            body.data.qvel[:6] = 0.0
            info["wing_stroke"] = round(
                float(body.data.qpos[self.wings.qadr[0]]), 2)
        else:
            # 身体线程每次都更新运动指令 —— 步态相位推进的时长必须和身体推进的
            # 时长一致，否则腿会拖在地上滑。
            if True:
                if self.manual_turn is not None:
                    turn = self.manual_turn
                elif self.behavior == "forage" and self.odor_on:
                    # 显式双侧比较趋化（不是连接组，见 sim --exp steering）
                    diff = (odor_l - odor_r) / max(odor_l + odor_r, 1e-6)
                    turn = float(np.clip(8.0 * diff, -0.5, 0.5))
                elif self.behavior == "cube":
                    # 走到魔方跟前、正对着它停下 —— 然后才用前腿去拨
                    from ..cube_hands import approach
                    sp, turn, self.cube_arrived = approach(
                        body, self.scene.origin[:2],
                        stop_dist=self.scene.size / 2 + 0.09)
                    if self.cube_arrived:
                        self.manual_speed_cube = 0.0
                    else:
                        self.manual_speed_cube = sp
                else:
                    h = body.heading()
                    err = float(self.h0[0] * h[1] - self.h0[1] * h[0])
                    turn = float(np.clip(0.3 * cmd.turn + 1.2 * err,
                                         -0.5, 0.5))
                speed = (self.manual_speed if self.manual_speed is not None
                         else float(np.clip(cmd.speed, 0.35, 1.0)))
                if self.behavior == "cube":
                    speed = getattr(self, "manual_speed_cube", speed)
                if taste:
                    speed = 0.0
                self.cur_turn, self.cur_speed = turn, speed
                self.gait.step(self.ctrl, dt_b, speed=speed, turn=turn,
                               adhesion=0.5)
                if self.behavior == "cube":
                    self._cube_leg(self.ctrl)

                if self.behavior == "forage":
                    # 伸喙由 MN9 的发放驱动 —— 这一段是连接组算出来的
                    inst = float(spikes[self.mn9].mean()) / dt_brain
                    self.proboscis += 0.25 * (
                        float(np.clip(inst / 60.0, 0, 1)) - self.proboscis)
                    for k, n in enumerate(("rostrum", "haustellum",
                                           "labrum_left", "labrum_right")):
                        a = body.act_index.get(n)
                        if a is not None:
                            lo, hi = body.ctrl_range[a]
                            self.ctrl[a] = lo + (hi - lo) * (
                                self.proboscis if k < 2
                                else 0.5 * self.proboscis)
                    info["proboscis"] = round(self.proboscis, 2)
            body.step(self.ctrl, n_substeps=sub)
            info["fallen"] = body.fallen()
            if self.behavior == "cube":
                if (self.scene.advance(dt_b)
                        and self.scene.state.is_solved()):
                    self.cube_solved_at = self.brain.t_ms / 1000.0
                self.scene.write()

    def _cube_leg(self, ctrl) -> None:
        """魔方转动期间，把对应那条前腿的"伸手 + 拨"姿态叠加到步态之上。

        腿的相位和面的转角用的是同一个 anim_t/anim_dur，所以看上去是这一拨
        把面推过去的。接触本身是运动学同步，不是物理摩擦驱动 —— 见 cube_hands。
        """
        from ..cube_hands import FACE_SIDE, FrontLegReach
        if not hasattr(self, "_legs"):
            self._legs = FrontLegReach(self.body)
        if not self._legs.available():
            return
        sc = self.scene
        if sc.anim_face is None or not getattr(self, "cube_arrived", False):
            return
        phase = min(sc.anim_t / max(sc.anim_dur, 1e-6), 1.0)
        side = FACE_SIDE.get(sc.anim_face, "right")
        # 两头淡入淡出，别在拨完的瞬间把腿弹回步态
        blend = float(np.clip(min(phase, 1.0 - phase) * 6.0, 0.0, 1.0))
        self._legs.apply(ctrl, side, phase, blend=blend)

    def _advance_cube(self, dt_brain: float, spikes: np.ndarray) -> dict:
        """魔方：连接组决定「什么时候拧」，求解器决定「拧哪一面」。"""
        FACES = ("U", "R", "F", "D", "L", "B")
        dn = self.bridge.dn_all
        if not hasattr(self, "_cube_win"):
            self._cube_win = np.zeros(len(dn), dtype=np.float32)
            self._cube_steps = 0
        self._cube_win += spikes[dn]
        self._cube_steps += 1

        if self.step_i % CTRL_EVERY == 0 and self._cube_steps:
            secs = self._cube_steps * dt_brain
            dn_hz = float(self._cube_win.sum() / len(dn) / secs)
            # 下行活动按发放率积累，满一格触发一次转动：
            # 网络越活跃，果蝇拧得越快
            self.cube_charge += dn_hz * secs / 1.6
            if (self.cube_charge >= 1.0 and not self.scene.animating
                    and self.cube_i < len(self.cube_solution)):
                self.cube_charge = 0.0
                mv = self.cube_solution[self.cube_i]
                groups = [dn[i::6] for i in range(6)]
                rates = [float(spikes[g].sum()) for g in groups]
                proposed = FACES[int(np.argmax(rates))]
                self.cube_agree.append(proposed == mv[0])
                self.scene.start_move(mv, duration=0.35)
                self.cube_i += 1
                self._cube_proposed = proposed
            self._cube_win[:] = 0.0
            self._cube_steps = 0

        # 动画的推进放在身体线程（见 _motion_once）—— 脑只跑到真实时间的
        # 三分之一，挂在脑上的话转动会拖成慢动作。
        n = len(self.cube_agree)
        return {
            "cube_move": self.cube_i,
            "cube_total": len(self.cube_solution),
            "cube_source": self.cube_source,
            "cube_solved": self.cube_solved_at is not None,
            "cube_charge": round(min(self.cube_charge, 1.0), 2),
            "cube_next": (self.cube_solution[self.cube_i]
                          if self.cube_i < len(self.cube_solution) else "—"),
            "cube_proposed": getattr(self, "_cube_proposed", "—"),
            "cube_agree": (round(100 * sum(self.cube_agree) / n)
                           if n else None),
        }

    def stop(self) -> None:
        self._stop.set()


def _brain_snapshot(sim, spikes: np.ndarray, out: dict) -> str:
    """把识别期间的全脑活动渲染成一张图，base64 回给前端。"""
    import base64

    from PIL import Image

    from ..brainview import draw_overlay

    view = sim.view
    view.heat[:] = 0.0
    s = spikes[view.idx]
    peak = float(s.max()) or 1.0
    view.heat += np.clip(s / peak, 0, 1) * 1.4
    img = draw_overlay(
        view.render(), "识别期间的全脑活动",
        [f"预测 {out['prediction']}　置信 "
         f"{out['confidence'] * 100:.0f}%　全脑 {out['brain_hz']:.1f} Hz",
         "亮度 = 这次识别里该神经元发放了多少"],
        view.labels)
    buf = __import__("io").BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=78)
    view.heat[:] = 0.0
    return base64.b64encode(buf.getvalue()).decode()


def create_app(backend: str | None = None,
               behavior: str = "walk") -> tuple[Flask, Simulation]:
    app = Flask(__name__, static_folder=None)
    sim = Simulation(backend=backend, behavior=behavior)
    threading.Thread(target=sim.loop, daemon=True).start()
    threading.Thread(target=sim.body_loop, daemon=True).start()

    @app.route("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.route("/api/info")
    def info():
        s = sim.c.stats
        return jsonify({
            "dataset": s["dataset"],
            "n_neurons": s["n_neurons"],
            "n_edges": s["n_edges"],
            "excitatory": s["excitatory_edges"],
            "inhibitory": s["inhibitory_edges"],
            "backend": sim.brain.backend,
            "behaviors": BEHAVIOR_LABELS,
            "behavior": sim.behavior,
            "cameras": sim.cameras,
            "n_actuators": sim.body.n_act,
            "bridge": sim.bridge.report.summary(),
            "groups": {k: int(len(sim.group_indices(k))) for k in STIM_GROUPS},
            "regions": {k: int(len(v)) for k, v in sim.regions.items()},
            "raster_size": len(sim.raster_idx),
        })

    @app.route("/api/control", methods=["POST"])
    def control():
        req = request.get_json(force=True)
        action = req.get("action")
        try:
            if action == "stim":
                sim.set_stim(req["group"], float(req.get("rate", 0)))
            elif action == "ablate":
                return jsonify({"ok": True,
                                "ablated": sim.ablate(req["group"])})
            elif action == "reset":
                sim.reset()
            elif action == "toggle":
                sim.running = bool(req.get("running", True))
            elif action == "behavior":
                sim.set_behavior(req["value"])
            elif action == "split":
                sim.split = bool(req.get("on", True))
            elif action == "camera":
                sim.camera = req["value"]
            elif action == "orbit":
                # 鼠标拖动：左右拖改方位角，上下拖改仰角，滚轮改距离
                with sim.lock:
                    o = sim.body.set_orbit(
                        azimuth=req.get("azimuth"),
                        elevation=req.get("elevation"),
                        distance=req.get("distance"))
                sim.camera = "orbit"
                return jsonify({"ok": True, "orbit": o})
            elif action == "manual":
                sim.manual_speed = (None if req.get("speed") is None
                                    else float(req["speed"]))
                sim.manual_turn = (None if req.get("turn") is None
                                   else float(req["turn"]))
            elif action == "food":
                sim.food_xy = [float(req["x"]), float(req["y"])]
                sim.reached = False
            elif action == "new_cube":
                with sim.lock:
                    sim._new_cube(int(req.get("scramble", 8)))
            elif action == "odor":
                sim.odor_on = bool(req.get("on", True))
            else:
                return jsonify({"ok": False,
                                "error": f"未知动作 {action}"}), 400
        except Exception as e:                            # noqa: BLE001
            return jsonify({"ok": False, "error": str(e)}), 400
        return jsonify({"ok": True})

    @app.route("/api/recognize", methods=["POST"])
    def recognize():
        """收一张手写图（28x28 灰度，0~1 的一维数组），交给果蝇的视觉系统。"""
        r = sim.get_recognizer()
        if r is None:
            return jsonify({"ok": False, "error": sim.recog_error or
                            "识别器不可用"}), 400
        try:
            req = request.get_json(force=True)
            px = np.asarray(req["pixels"], dtype=np.float32)
            n = int(round(len(px) ** 0.5))
            img = px.reshape(n, n)
            if float(img.max()) <= 0:
                return jsonify({"ok": False, "error": "画布是空的"}), 400
            with sim.lock:
                out = r.recognise(img, keep_spikes=True)
            spikes = out.pop("_spikes", None)
            # 把这次识别期间的全脑活动画出来 —— 哪些神经元亮了
            if spikes is not None:
                out["brain_png"] = _brain_snapshot(sim, spikes, out)
            out["ok"] = True
            out["model"] = r.meta
            return jsonify(out)
        except Exception as e:                            # noqa: BLE001
            return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 400

    @app.route("/api/feedback", methods=["POST"])
    def feedback():
        """用户告诉它刚才那张图真正是什么，就地更新读出层。"""
        r = sim.get_recognizer()
        if r is None:
            return jsonify({"ok": False, "error": sim.recog_error}), 400
        req = request.get_json(force=True)
        if req.get("action") == "reset":
            return jsonify(r.reset_online())
        true_char = str(req.get("true", "")).strip().upper()
        with sim.lock:
            return jsonify(r.feedback(true_char))

    @app.route("/api/recognizer")
    def recognizer_info():
        r = sim.get_recognizer()
        if r is None:
            return jsonify({"ready": False, "error": sim.recog_error})
        return jsonify({"ready": True, "charset": r.charset,
                        "grid": r.grid, **r.meta})

    @app.route("/stream")
    def stream():
        def gen():
            last = None
            while True:
                d = sim.latest
                if d and d is not last:
                    last = d
                    yield f"data: {json.dumps(d, ensure_ascii=False)}\n\n"
                time.sleep(0.04)
        return Response(gen(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache",
                                 "X-Accel-Buffering": "no"})

    @app.route("/frame")
    def frame():
        """单张 JPEG。给截图和不方便处理 MJPEG 长连接的客户端用。"""
        f = sim.frame
        if not f:
            return Response(status=503)
        return Response(f, mimetype="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.route("/video")
    def video():
        def gen():
            while True:
                f = sim.frame
                if f:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(f)).encode() +
                           b"\r\n\r\n" + f + b"\r\n")
                time.sleep(0.05)
        return Response(gen(), mimetype="multipart/x-mixed-replace; "
                                        "boundary=frame")

    return app, sim


def main(host: str = "0.0.0.0", port: int = 8080,
         backend: str | None = None, behavior: str = "walk") -> int:
    app, sim = create_app(backend, behavior)
    print(f"\n数字果蝇控制台 -> http://localhost:{port}\n")
    try:
        app.run(host=host, port=port, threaded=True, debug=False)
    finally:
        sim.stop()
    return 0
