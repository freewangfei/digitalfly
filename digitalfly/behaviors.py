"""数字果蝇的行为：行走、觅食、飞行。

三种行为共用同一套结构：

    世界/身体 -> 感觉神经元电流 -> 全脑 LIF 网络（18.5 万神经元）
                                        |
                        下行神经元左右群体活动 = 指令（速度/转向）
                                        |
                        运动模式发生器 -> 关节轨迹 -> MuJoCo 物理

连接组负责的是"决定往哪走、走多快、要不要伸喙"，模式发生器负责"腿怎么迈"。
这个分工是果蝇神经系统本身的分工，不是为了省事 —— 见 locomotion.py 顶部。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config, viz
from .brain import Brain
from .bridge import CommandBridge
from .calibrate import calibrated_params
from .connectome import load as load_connectome

BODY_LENGTH = 0.2885   # 模型实测体长，用于把速度换算成"体长/秒"


@dataclass
class Episode:
    """一次行为仿真的结果。"""
    name: str
    seconds: float
    frames: list = field(default_factory=list)
    trace: dict = field(default_factory=dict)
    video: Path | None = None
    summary: str = ""


def _orbit(body, center, radius: float = 0.55, kp: float = 1.5) -> float:
    """绕着一个点转圈的转向指令：偏离目标半径就往回修。"""
    pos = body.root_pos[:2]
    h = body.heading()
    r = np.asarray(center, dtype=np.float64) - pos
    d = float(np.linalg.norm(r))
    if d < 1e-6:
        return 0.0
    r /= d
    # 目标方向：绕圈切线 + 半径误差修正
    tangent = np.array([-r[1], r[0]])
    want = tangent + kp * (d - radius) * r
    want /= max(np.linalg.norm(want), 1e-9)
    err = float(h[0] * want[1] - h[1] * want[0])
    return float(np.clip(2.0 * err, -0.5, 0.5))


def _heading_hold(body, h0: np.ndarray, kp: float = 1.2,
                  limit: float = 0.35) -> float:
    """朝向保持反射：偏离初始朝向就反向修正。

    对应真实果蝇的旋转性运动反射（optomotor response）—— 看到视野整体旋转
    就反向转身。这是一条低层反射，不是连接组的输出，所以单独标出来。
    """
    h = body.heading()
    err = float(h0[0] * h[1] - h0[1] * h[0])
    return float(np.clip(kp * err, -limit, limit))


# ---------------------------------------------------------------------------
# 行走
# ---------------------------------------------------------------------------

def walk(seconds: float = 3.0, backend: str | None = None,
         camera: str = "track1", render_fps: int = 30,
         drive: float = 0.8, out: str | Path | None = None,
         verbose: bool = True) -> Episode:
    """连接组驱动的行走：脑决定速度与转向，模式发生器产生步态。"""
    from .body import FlyBody
    from .locomotion import PreprogrammedGait

    c = load_connectome()
    body = FlyBody(mode="walk")
    gait = PreprogrammedGait(body)
    brain = Brain(c.W, calibrated_params(), backend=backend, verbose=verbose)
    bridge = CommandBridge(c, dt_ms=brain.p.dt, verbose=verbose)

    if verbose:
        print(f"[body] 行走配置: 执行器 {body.n_act} 个（翅膀收拢）, "
              f"步频 {gait.freq_hz} Hz")

    dt_brain = brain.p.dt / 1000.0
    ctrl_every = 20                       # 每 10 ms 更新一次指令
    dt_ctrl = dt_brain * ctrl_every
    # 每个脑步只推进一个脑步的物理时间。按控制周期算子步会让物理多跑 20 倍。
    sub = max(1, int(round(dt_brain / body.timestep)))
    n_steps = int(seconds / dt_brain)
    render_every = max(1, int(round((1.0 / render_fps) / dt_brain)))

    h0 = body.heading().copy()
    ctrl = body.rest.copy()
    ext = np.zeros(brain.n, dtype=np.float32)
    frames, rec = [], {"speed": [], "turn": [], "dn_hz": [], "brain_hz": []}
    t0 = time.time()
    cmd_speed = cmd_turn = 0.0

    for i in range(n_steps):
        # 感觉输入：本体感觉 + 一个恒定的"行走驱动"，模拟让果蝇起步的背景输入
        bridge.sensory_current(brain.n, body=body, out=ext)
        ext[bridge.proprio] += drive
        spikes = brain.step(ext)

        if i % ctrl_every == 0:
            cmd = bridge.read_command(spikes)
            cmd_speed = cmd.speed
            # 转向 = 脑的指令 + 朝向保持反射
            cmd_turn = float(np.clip(
                0.3 * cmd.turn + _heading_hold(body, h0), -0.5, 0.5))
            gait.step(ctrl, dt_ctrl, speed=max(cmd_speed, 0.35),
                      turn=cmd_turn, adhesion=0.5)
            rec["speed"].append(cmd_speed)
            rec["turn"].append(cmd_turn)
            rec["dn_hz"].append((cmd.dn_left_hz + cmd.dn_right_hz) / 2)
            rec["brain_hz"].append(float(spikes.sum()) / brain.n / dt_brain)
        body.step(ctrl, n_substeps=sub)

        if i % render_every == 0:
            frames.append(body.render(camera))
        if verbose and i % max(1, n_steps // 10) == 0:
            d = body.displacement()
            print(f"\r  t={i * dt_brain:4.1f}s  速度指令 {cmd_speed:.2f} "
                  f"转向 {cmd_turn:+.2f}  位移 {np.linalg.norm(d[:2]):.2f}",
                  end="")
    if verbose:
        print()

    d = body.displacement()[:2]
    fwd = float(d @ h0)
    ep = Episode(name="walk", seconds=seconds, frames=frames, trace=rec)
    ep.summary = (
        f"  行走 {seconds:.1f}s（墙钟 {time.time() - t0:.0f}s）\n"
        f"  沿初始朝向前进 {fwd:.2f}  "
        f"= {fwd / seconds / BODY_LENGTH:.2f} 体长/秒\n"
        f"  全脑平均 {np.mean(rec['brain_hz']):.2f} Hz  "
        f"下行神经元平均 {np.mean(rec['dn_hz']):.1f} Hz  "
        f"翻倒={body.fallen()}")
    ep.video = _save(frames, out or config.OUTPUT_DIR / "walk.mp4", render_fps)
    body.close()
    return ep


# ---------------------------------------------------------------------------
# 觅食：闻着糖味走过去，够到了就伸喙
# ---------------------------------------------------------------------------

def forage(seconds: float = 8.0, backend: str | None = None,
           camera: str = "track1", render_fps: int = 30,
           food_xy=(2.2, 1.2), sigma: float = 1.6,
           out: str | Path | None = None, verbose: bool = True) -> Episode:
    """觅食：闻着糖味走过去，够到了就伸喙。

    **哪一段是连接组做的，哪一段不是，必须分清楚。**

    连接组负责：
        * 行进速度 —— 由下行神经元群体活动读出。
        * 伸喙决策 —— 口器接触到糖后，味觉输入经全脑网络传到伸喙运动神经元
          MN9，MN9 的发放驱动口器。这一段是真的由连接组算出来的，
          和 `sim --exp sugar_pe` 验证的是同一条通路。

    连接组做不到、因而用显式规则的：
        * 转向。双侧气味比较需要精确的增益匹配，均一 LIF + 未调节的突触权重
          支撑不了 —— `sim --exp steering` 定量地证明了这一点（总强度相同时，
          下行神经元读数对气味左右分布的分辨力是 0）。所以这里的转向用显式的
          双侧比较趋化规则，并在输出里标明。
    """
    from .body import FlyBody
    from .locomotion import PreprogrammedGait

    c = load_connectome()
    body = FlyBody(mode="forage")
    gait = PreprogrammedGait(body)
    brain = Brain(c.W, calibrated_params(), backend=backend, verbose=verbose)
    bridge = CommandBridge(c, dt_ms=brain.p.dt, verbose=verbose)
    if verbose:
        print("[forage] 转向 = 显式双侧比较趋化（非连接组，见 --exp steering）；"
              "速度与伸喙 = 连接组")

    food = np.asarray(food_xy, dtype=np.float64)
    antenna_offset = 0.12          # 左右触角相对中线的横向距离
    reach = 0.3                    # 够到食物的距离

    dt_brain = brain.p.dt / 1000.0
    ctrl_every = 20
    dt_ctrl = dt_brain * ctrl_every
    sub = max(1, int(round(dt_brain / body.timestep)))
    n_steps = int(seconds / dt_brain)
    render_every = max(1, int(round((1.0 / render_fps) / dt_brain)))

    proboscis = [body.act_index[n] for n in
                 ("rostrum", "haustellum", "labrum_left", "labrum_right")
                 if n in body.act_index]

    ext = np.zeros(brain.n, dtype=np.float32)
    ctrl = body.rest.copy()
    frames = []
    rec = {"dist": [], "odor_l": [], "odor_r": [], "turn": [],
           "mn9_hz": [], "brain_hz": [], "speed": []}
    window = np.zeros(brain.n, dtype=np.float32)
    reached_at = None
    t0 = time.time()
    cmd_turn = 0.0
    pe = 0.0

    def conc(p):
        return float(np.exp(-np.sum((food - p) ** 2) / (2 * sigma ** 2)))

    for i in range(n_steps):
        pos = body.root_pos[:2]
        h = body.heading()
        lat = np.array([-h[1], h[0]])
        dist = float(np.linalg.norm(food - pos))

        odor_l = conc(pos + antenna_offset * lat + 0.1 * h)
        odor_r = conc(pos - antenna_offset * lat + 0.1 * h)
        taste = 1.0 if dist < reach else 0.0
        if taste and reached_at is None:
            reached_at = i * dt_brain

        bridge.sensory_current(brain.n, body=body, odor_left=odor_l,
                               odor_right=odor_r, taste=taste, out=ext)
        ext[bridge.proprio] += 0.8
        spikes = brain.step(ext)
        window += spikes

        if i % ctrl_every == 0:
            cmd = bridge.read_command(spikes)
            # --- 转向：显式双侧比较（不是连接组） ---
            # 左侧浓度高 -> 向左转。turn > 0 是左转（见 PreprogrammedGait）。
            diff = (odor_l - odor_r) / max(odor_l + odor_r, 1e-6)
            cmd_turn = float(np.clip(8.0 * diff, -0.5, 0.5))
            # --- 速度：连接组的下行神经元活动 ---
            speed = 0.0 if taste else float(np.clip(cmd.speed, 0.35, 1.0))
            gait.step(ctrl, dt_ctrl, speed=speed, turn=cmd_turn, adhesion=0.5)

            # --- 伸喙：连接组算出来的 ---
            mn9 = bridge.mn9_rate(window, ctrl_every * dt_brain)
            window[:] = 0.0
            # 伸喙是肌肉动作，比脉冲慢：对 MN9 发放率做低通，避免口器抖动
            pe += 0.25 * (float(np.clip(mn9 / 60.0, 0.0, 1.0)) - pe)
            for k, a in enumerate(proboscis):
                lo, hi = body.ctrl_range[a]
                ctrl[a] = lo + (hi - lo) * (pe if k < 2 else 0.5 * pe)

            rec["dist"].append(dist)
            rec["odor_l"].append(odor_l)
            rec["odor_r"].append(odor_r)
            rec["turn"].append(cmd_turn)
            rec["mn9_hz"].append(mn9)
            rec["speed"].append(speed)
            rec["brain_hz"].append(float(spikes.sum()) / brain.n / dt_brain)
        body.step(ctrl, n_substeps=sub)

        if i % render_every == 0:
            frames.append(body.render(camera))
        if verbose and i % max(1, n_steps // 10) == 0:
            print(f"\r  t={i * dt_brain:4.1f}s  距糖源 {dist:5.2f}  "
                  f"气味 L{odor_l:.2f}/R{odor_r:.2f}  转向 {cmd_turn:+.2f}  "
                  f"MN9 {rec['mn9_hz'][-1] if rec['mn9_hz'] else 0:5.1f}Hz  "
                  f"伸喙 {pe:.2f}", end="")
    if verbose:
        print()

    d0 = float(np.linalg.norm(food - np.asarray(body._start_pos[:2])))
    d1 = float(np.linalg.norm(food - body.root_pos[:2]))
    peak_mn9 = max(rec["mn9_hz"]) if rec["mn9_hz"] else 0.0
    ep = Episode(name="forage", seconds=seconds, frames=frames, trace=rec)
    ep.summary = (
        f"  觅食 {seconds:.1f}s（墙钟 {time.time() - t0:.0f}s）\n"
        f"  到糖源距离 {d0:.2f} -> {d1:.2f}"
        + (f"，{reached_at:.1f}s 时够到" if reached_at else "，未够到") + "\n"
        f"  MN9 伸喙峰值 {peak_mn9:.1f} Hz（连接组算出）  "
        f"全脑平均 {np.mean(rec['brain_hz']):.2f} Hz  "
        f"平均速度指令 {np.mean(rec['speed']):.2f}")
    ep.video = _save(frames, out or config.OUTPUT_DIR / "forage.mp4",
                     render_fps)
    body.close()
    return ep


# ---------------------------------------------------------------------------
# 飞行（拴系）
# ---------------------------------------------------------------------------

def flight(seconds: float = 1.0, backend: str | None = None,
           camera: str = "flight_close", render_fps: int = 30,
           out: str | Path | None = None, verbose: bool = True) -> Episode:
    """拴系飞行：翅膀以 218 Hz 真实运动学拍动，连接组读出转向指令。

    **为什么是拴系而不是自由飞行。** 自由飞行的姿态稳定需要一个学出来的
    控制器：flybody 官方的飞行任务用的是训练好的强化学习策略，权重放在
    figshare 上（本机访问被对方拒绝，403）。只靠翅膀运动学加 PD 反馈，
    果蝇会一边产生升力一边翻滚 —— 实测如此，不是没试过。

    拴系飞行不是退而求其次：真实果蝇神经科学里，绝大多数飞行实验就是把果蝇
    固定住、让翅膀自由拍动，用**左右振幅差**作为转向意图的读数。这里做的
    完全是同一件事，只不过转向意图来自连接组。
    """
    from .body import FlyBody
    from .brainview import BrainView, draw_overlay, side_by_side
    from .locomotion import WingBeat
    from .neurons import NeuronIndex

    c = load_connectome()
    body = FlyBody(mode="flight")
    brain = Brain(c.W, calibrated_params(), backend=backend, verbose=verbose)
    bridge = CommandBridge(c, dt_ms=brain.p.dt, verbose=verbose)

    dt_ctrl = 2e-4
    sub = max(1, int(round(dt_ctrl / body.timestep)))
    wings = WingBeat(body, dt_ctrl=dt_ctrl)
    for a in wings.act:                      # 提高增益补偿近似翅拍模式
        body.model.actuator_gainprm[a][0] = 70.0
    if verbose:
        print(f"[body] 飞行配置: 执行器 {body.n_act} 个（腿收起）, "
              f"翅拍 {wings.base_freq:.0f} Hz, 物理步长 {body.timestep * 1e6:.0f} μs")
        print("[body] 拴系：躯干固定，只看翅膀运动学与转向指令")

    # 拴系：把躯干自由关节钉住（场景里那根细杆就是系杆）
    body.reset()
    tether_qpos = np.array(body.data.qpos[:7], dtype=np.float64)
    view = BrainView(c.meta, NeuronIndex(c).by_region(), size=(480, 560))

    dt_brain = brain.p.dt / 1000.0
    brain_every = max(1, int(round(dt_brain / dt_ctrl)))
    n_ctrl = int(seconds / dt_ctrl)
    render_every = max(1, int(round((1.0 / render_fps) / dt_ctrl)))

    ext = np.zeros(brain.n, dtype=np.float32)
    ctrl = body.rest.copy()
    frames = []
    rec = {"asym": [], "wing_l": [], "wing_r": [], "dn_l": [], "dn_r": []}
    spikes = np.zeros(brain.n, dtype=np.float32)
    t0 = time.time()
    asym = 0.0

    for i in range(n_ctrl):
        if i % brain_every == 0:
            # 飞行中的视觉/机械感觉背景输入
            bridge.sensory_current(brain.n, body=body, out=ext)
            ext[bridge.proprio] += 0.8
            spikes = brain.step(ext)
            cmd = bridge.read_command(spikes)
            asym = float(np.clip(0.6 * cmd.turn, -0.8, 0.8))
            rec["asym"].append(asym)
            rec["dn_l"].append(cmd.dn_left_hz)
            rec["dn_r"].append(cmd.dn_right_hz)

        wings.step(ctrl, freq_rel=0.0, asymmetry=asym, amplitude=1.0)
        body.step(ctrl, n_substeps=sub)
        # 维持拴系
        body.data.qpos[:7] = tether_qpos
        body.data.qvel[:6] = 0.0

        view.update(spikes, brain.p.dt)
        if i % render_every == 0:
            rec["wing_l"].append(float(body.data.qpos[wings.qadr[0]]))
            rec["wing_r"].append(float(body.data.qpos[wings.qadr[3]]))
            left = draw_overlay(
                body.render(camera), "数字果蝇 · 拴系飞行",
                [f"翅拍 {wings.base_freq:.0f} Hz　躯干抬头 47.5°",
                 f"左右翅振幅差（转向读数） {asym:+.3f}",
                 "躯干由系杆固定，翅膀自由拍动"])
            right = draw_overlay(
                view.render(), "全脑活动 · MaleCNS v1.0",
                [f"{brain.n:,} 神经元 · 白点 = 此刻发放",
                 f"下行神经元 左 {cmd.dn_left_hz:.1f} / "
                 f"右 {cmd.dn_right_hz:.1f} Hz"],
                view.labels)
            frames.append(side_by_side(left, right))

    ep = Episode(name="flight", seconds=seconds, frames=frames, trace=rec)
    ep.summary = (
        f"  拴系飞行 {seconds:.2f}s（墙钟 {time.time() - t0:.0f}s）\n"
        f"  翅拍频率 {wings.base_freq:.0f} Hz  "
        f"左翅冲程幅度 {np.ptp(rec['wing_l']):.2f} rad\n"
        f"  转向指令（左右振幅差）均值 {np.mean(rec['asym']):+.3f}  "
        f"范围 [{min(rec['asym']):+.2f}, {max(rec['asym']):+.2f}]")
    ep.video = _save(frames, out or config.OUTPUT_DIR / "flight.mp4",
                     render_fps)
    body.close()
    return ep


def _save(frames, path, fps) -> Path | None:
    if not frames:
        return None
    return viz.write_video(frames, path, fps=fps)


BEHAVIORS = {"walk": walk, "forage": forage, "flight": flight}


# ---------------------------------------------------------------------------
# 魔方
# ---------------------------------------------------------------------------

def cube(seconds: float = 20.0, backend: str | None = None,
         camera: str = "cube_cam", render_fps: int = 30,
         scramble: int = 8, seed: int = 7, move_secs: float = 0.35,
         out: str | Path | None = None, verbose: bool = True) -> Episode:
    """果蝇「拧」魔方 —— 网上那批病毒视频的复刻版。

    **必须先说清楚这是什么。** 网上那些"果蝇玩 Beat Saber / 解魔方 / 开车"
    的视频，做法都是把连接组的活动映射成操作指令，真正完成任务的是外部程序。
    做 Minecraft 那个的作者自己讲得很直白：这是"一种探索连接组的交互方式，
    不是意识的证据，也不是对活果蝇的完整重建"。这里做的是同一件事，
    只是把界线标出来：

        连接组负责 —— **什么时候拧**。下行神经元群体活动积累到阈值就触发
                      一次转动，所以转动的节奏完全由全脑网络的动力学决定。
        求解器负责 —— **拧哪一面**。解法由 IDA* 搜索得到（本项目自带的纯
                      Python 求解器），是最优解。

    另外还免费测了一件事：把下行神经元分成 6 个子群，取发放率最高的那个当作
    "果蝇自己想拧哪一面"，统计它和求解器的一致率。随机猜是 1/6 ≈ 16.7%。
    这个数字直接告诉你连接组对这个任务贡献了多少信息。
    """
    from .body import FlyBody
    from .brainview import BrainView, draw_overlay, side_by_side
    from .cube import Cube, solve
    from .cube_scene import CubeScene
    from .locomotion import PreprogrammedGait
    from .neurons import NeuronIndex

    FACES = ("U", "R", "F", "D", "L", "B")

    c = load_connectome()
    scene = CubeScene(size=0.55, pos=(0.55, 0.0, 0.28))
    body = FlyBody(mode="walk", cube_scene=scene)
    scene.bind(body.model, body.data)
    gait = PreprogrammedGait(body)
    brain = Brain(c.W, calibrated_params(), backend=backend, verbose=verbose)
    bridge = CommandBridge(c, dt_ms=brain.p.dt, verbose=verbose)

    # 打乱并求解
    seq = scene.state.scramble(scramble, seed=seed)
    if verbose:
        print(f"[cube] 打乱 {scramble} 步: {' '.join(seq)}")
        print("[cube] 求解中（IDA* + BFS 剪枝表）…")
    t_solve = time.time()
    solution, source = solve(scene.state.copy(), fallback=seq)
    if verbose:
        print(f"[cube] 解 {len(solution)} 步（{source}，"
              f"{time.time() - t_solve:.1f}s）: {' '.join(solution)}")
        print("[cube] 连接组决定「什么时候拧」，求解器决定「拧哪一面」")

    # 读出用的 6 个下行神经元子群，对应六个面
    dn = bridge.dn_all
    groups = [dn[i::6] for i in range(6)]

    # 右半屏：全脑三维点云，神经元发放时闪烁（网上那批视频的标志画面）
    view = BrainView(c.meta, NeuronIndex(c).by_region(), size=(480, 560))

    dt_brain = brain.p.dt / 1000.0
    sub = max(1, int(round(dt_brain / body.timestep)))
    n_steps = int(seconds / dt_brain)
    render_every = max(1, int(round((1.0 / render_fps) / dt_brain)))
    ctrl_every = 20
    dt_ctrl = dt_brain * ctrl_every

    ext = np.zeros(brain.n, dtype=np.float32)
    ctrl = body.rest.copy()
    frames = []
    rec = {"dn_hz": [], "brain_hz": [], "moves": [], "agree": []}
    window = np.zeros(brain.n, dtype=np.float32)
    # 下行活动的积累量，满一格触发一次转动。
    # 分母按实测的下行发放率（约 1.3 Hz）定，使得大约 1.2 秒拧一步 ——
    # 节奏完全跟着网络活动走：网络越活跃，果蝇拧得越快。
    charge = 0.0
    CHARGE_SCALE = 1.6
    CHARGE_PER_MOVE = 1.0
    move_i = 0
    solved_at = None
    t0 = time.time()

    for i in range(n_steps):
        bridge.sensory_current(brain.n, body=body, out=ext)
        bridge._drive(ext, bridge.proprio, 20.0)
        spikes = brain.step(ext)
        window += spikes

        if i % ctrl_every == 0:
            cmd = bridge.read_command(spikes)
            # 绕着魔方转圈，免得走出画面。速度仍然由连接组给。
            turn = _orbit(body, scene.origin[:2], radius=0.55)
            gait.step(ctrl, dt_ctrl, speed=float(np.clip(cmd.speed, 0.3, 1.0)),
                      turn=turn, adhesion=0.5)
            secs = ctrl_every * dt_brain
            dn_hz = float(window[dn].sum() / len(dn) / secs)
            rec["dn_hz"].append(dn_hz)
            rec["brain_hz"].append(
                float(window.sum() / brain.n / secs))

            # 连接组决定节奏：下行活动按发放率积累，满一格触发一次转动
            charge += dn_hz * secs / CHARGE_SCALE
            if (charge >= CHARGE_PER_MOVE and not scene.animating
                    and move_i < len(solution)):
                charge = 0.0
                mv = solution[move_i]
                # 果蝇"自己想拧哪一面"：六个下行子群里发放最多的那个
                rates = [float(window[g].sum() / len(g) / secs) for g in groups]
                proposed = FACES[int(np.argmax(rates))]
                rec["agree"].append(proposed == mv[0])
                rec["moves"].append(mv)
                scene.start_move(mv, duration=move_secs)
                move_i += 1
                if verbose:
                    print(f"\r  第 {move_i:2d}/{len(solution)} 步: {mv:3s} "
                          f"（果蝇想拧 {proposed}）  t={i * dt_brain:5.1f}s",
                          end="")
            window[:] = 0.0

        body.step(ctrl, n_substeps=sub)
        if scene.advance(dt_brain) and scene.state.is_solved():
            solved_at = i * dt_brain
        scene.write()
        view.update(spikes, brain.p.dt)

        if i % render_every == 0:
            left = draw_overlay(
                body.render(camera), "数字果蝇 · 拧魔方",
                [f"第 {move_i}/{len(solution)} 步"
                 + (f"　下一步 {solution[move_i]}"
                    if move_i < len(solution) else "　已复原 ✓"),
                 f"蓄力（下行神经元活动） {min(charge, 1.0) * 100:3.0f}%",
                 f"解法 {source}　果蝇选面一致 "
                 f"{sum(rec['agree'])}/{max(len(rec['agree']), 1)}"])
            right = draw_overlay(
                view.render(),
                "全脑活动 · MaleCNS v1.0",
                [f"{brain.n:,} 神经元 · {c.stats['n_edges'] / 1e6:.0f}M 突触边",
                 f"白点 = 此刻发放　全脑 "
                 f"{rec['brain_hz'][-1] if rec['brain_hz'] else 0:.2f} Hz",
                 f"下行神经元 "
                 f"{rec['dn_hz'][-1] if rec['dn_hz'] else 0:.1f} Hz"],
                view.labels)
            frames.append(side_by_side(left, right))
        if solved_at is not None and i * dt_brain > solved_at + 1.5:
            break
    if verbose:
        print()

    n_agree = sum(rec["agree"])
    n_moves = len(rec["agree"])
    ep = Episode(name="cube", seconds=seconds, frames=frames, trace=rec)
    ep.summary = (
        f"  魔方 {seconds:.0f}s（墙钟 {time.time() - t0:.0f}s）\n"
        f"  打乱 {scramble} 步 -> 解 {len(solution)} 步（{source}），"
        f"已执行 {len(rec['moves'])} 步"
        + (f"，{solved_at:.1f}s 时复原" if solved_at else "，未走完") + "\n"
        f"  下行神经元平均 {np.mean(rec['dn_hz']):.1f} Hz  "
        f"全脑平均 {np.mean(rec['brain_hz']):.2f} Hz\n"
        f"  果蝇选面与求解器一致 {n_agree}/{n_moves}"
        + (f" = {100 * n_agree / n_moves:.0f}%" if n_moves else "")
        + "（随机猜 17%）")
    ep.video = _save(frames, out or config.OUTPUT_DIR / "cube.mp4", render_fps)
    body.close()
    return ep


BEHAVIORS["cube"] = cube
