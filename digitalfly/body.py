"""果蝇身体：MuJoCo 物理仿真。

身体模型来自 flybody (Google DeepMind x HHMI Janelia, *Nature* 643, 2025) ——
解剖学精细的成年果蝇模型，6 条腿、翅膀、喙、触角，带流体力学。

这里用 flybody 自己的 `FruitFly` 装配类，而不是直接读 XML：因为不同行为需要
不同的身体配置，这些开关只有装配类才有。

    行走  use_legs=True,  use_wings=False  —— 翅膀收拢贴在腹部
    飞行  use_legs=False, use_wings=True   —— 腿收起，身体前倾 47.5°
    觅食  行走配置 + 口器(use_mouth)        —— 需要伸喙

翅膀一定要在行走时收起来。翅膀关节的活动范围很大（yaw 可达 ±3 rad），
只要给它非零控制量，翅膀就会被转到远离身体的位置，看上去像是"掉下来了"。
真实果蝇行走时翅膀是折叠贴合的，对应的做法就是 use_wings=False。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from . import config

# MuJoCo 的离屏渲染后端。NVIDIA 的 EGL 可用就走 GPU，否则退到 CPU 软渲染。
if "MUJOCO_GL" not in os.environ:
    _egl = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    os.environ["MUJOCO_GL"] = "egl" if _egl.exists() else "osmesa"

# flybody 不需要 pip 安装（它的 pyproject 会把 numpy 拖回 1.26），直接加到路径里
if str(config.FLYBODY_DIR) not in sys.path:
    sys.path.insert(0, str(config.FLYBODY_DIR))

# 每种行为对应的身体配置
MODES = {
    "walk": dict(use_legs=True, use_wings=False, use_mouth=False,
                 use_antennae=True, body_pitch_angle=0.0),
    "forage": dict(use_legs=True, use_wings=False, use_mouth=True,
                   use_antennae=True, body_pitch_angle=0.0),
    "flight": dict(use_legs=False, use_wings=True, use_mouth=False,
                   use_antennae=True, body_pitch_angle=47.5),
}

# 飞行时躯干的抬头角，度。取自 flybody 的 _BODY_PITCH_ANGLE。
FLIGHT_PITCH_DEG = 47.5

# 腿的命名：胸节 x 侧别。T1 前足、T2 中足、T3 后足。
SEGMENTS = ("T1", "T2", "T3")
SIDES = ("left", "right")
LEGS = [f"{s}_{d}" for s in SEGMENTS for d in SIDES]
# 每条腿的关节（由近端到远端）
LEG_JOINTS = ("coxa_abduct", "coxa_twist", "coxa", "femur_twist",
              "femur", "tibia", "tarsus", "tarsus2")


class FlyBody:
    """MuJoCo 果蝇身体。

    属性:
        n_act      执行器数量
        act_names  执行器名称
        act_index  名称 -> 下标，运动模式发生器按名字寻址关节
    """

    def __init__(self, mode: str = "walk", render_size=(480, 640),
                 floor: bool = True, cube_scene=None, **overrides):
        import mujoco
        from dm_control import mjcf
        from dm_control.locomotion.arenas import floors
        from flybody.fruitfly.fruitfly import FruitFly

        if mode not in MODES:
            raise ValueError(f"未知行为模式 {mode!r}，可选: {list(MODES)}")
        self.mode = mode
        self.mj = mujoco

        kwargs = {**MODES[mode], **overrides}
        if mode == "flight":
            from flybody.tasks.constants import (_FLY_CONTROL_TIMESTEP,
                                                 _FLY_PHYSICS_TIMESTEP)
            # 飞行要更细的物理步长，翅膀每秒拍 218 次
            kwargs.setdefault("physics_timestep", _FLY_PHYSICS_TIMESTEP)
            kwargs.setdefault("control_timestep", _FLY_CONTROL_TIMESTEP)
        self.fly = FruitFly(**kwargs)

        if mode == "flight":
            self._configure_flight()

        if floor:
            arena = floors.Floor(size=(20.0, 20.0))
            arena.add_free_entity(self.fly)
            root = arena.mjcf_model
        else:
            root = self.fly.mjcf_model

        if mode == "flight" and floor:
            # 拴系飞行的系杆 + 专用机位。真实果蝇飞行实验就是把果蝇粘在一根
            # 细杆上、让翅膀自由拍动，画面里画出来才不会被误认为是"掉下来了"。
            wb = root.worldbody
            wb.add("body", name="tether_rig", pos=[0, 0, 1.0]).add(
                "geom", type="capsule", fromto=[0, 0, 0.02, 0, 0, 0.9],
                size=0.012, rgba=[0.55, 0.57, 0.60, 1],
                contype=0, conaffinity=0, group=1, mass=0)
            wb.add("camera", name="flight_cam", pos=[-0.9, -1.5, 1.25],
                   xyaxes=[0.86, -0.51, 0, 0.16, 0.27, 0.95])
            wb.add("camera", name="flight_close", pos=[-0.45, -0.75, 1.12],
                   xyaxes=[0.86, -0.51, 0, 0.16, 0.27, 0.95])

        self.cube = cube_scene
        if cube_scene is not None:
            # 魔方的 26 个块是 mocap 体，每帧直接写位姿，不参与物理仿真。
            # MuJoCo 要求 mocap 体是 worldbody 的直接子节点。
            cube_scene.build(root.worldbody)

        self.physics = mjcf.Physics.from_mjcf_model(root)
        if mode == "flight":
            # arena 的 option 会覆盖 walker 的物理步长，这里强制设回来：
            # 翅膀每秒拍 218 次，步长太粗气动力算不准。
            from flybody.tasks.constants import _FLY_PHYSICS_TIMESTEP
            self.physics.model.ptr.opt.timestep = _FLY_PHYSICS_TIMESTEP

        self.model = self.physics.model.ptr
        self.data = self.physics.data.ptr
        self.h, self.w = render_size
        self._renderer = None
        # 自由视角的状态（方位角 / 仰角 / 距离），由 Web 端鼠标拖动驱动
        self._orbit_cam = None
        self._orbit = {"azimuth": 135.0, "elevation": -20.0, "distance": 1.4}

        n = self.model.nu
        self.act_names = [
            (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
             or f"act{i}").split("/")[-1] for i in range(n)]
        self.act_index = {nm: i for i, nm in enumerate(self.act_names)}
        self.camera_names = [
            (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
             or f"cam{i}").split("/")[-1] for i in range(self.model.ncam)]

        self._lo = np.array(self.model.actuator_ctrlrange[:, 0], np.float64)
        self._hi = np.array(self.model.actuator_ctrlrange[:, 1], np.float64)
        # ctrl=0 是这个身体的自然静息姿态（实测：全 0 时果蝇能稳稳站住）
        self.rest = np.clip(0.0, self._lo, self._hi)
        self.reset()

    def _configure_flight(self) -> None:
        """给翅膀装上空气动力学。

        光让翅膀按正确的运动学拍动是飞不起来的（实测：果蝇直接掉下去）。
        升力来自 MuJoCo 的椭球流体模型，必须显式打开并设好系数；翅膀执行器的
        增益、翅关节的刚度与阻尼也要按飞行参数设置。这些值全部取自 flybody 的
        `_WING_PARAMS`，和它自己的飞行任务保持一致。
        """
        from flybody.tasks.constants import _WING_PARAMS
        root = self.fly.mjcf_model

        # 翅膀执行器增益（yaw / roll / pitch）
        for i, dclass in enumerate(("yaw", "roll", "pitch")):
            root.find("default", dclass).general.gainprm[0] = \
                _WING_PARAMS["gainprm"][i]

        # 打开椭球流体模型 —— 升力的来源
        for geom in root.find_all("geom"):
            if geom.name and "fluid" in geom.name:
                geom.fluidshape = "ellipsoid"
                geom.fluidcoef = _WING_PARAMS["fluidcoef"]

        # 翅关节的刚度与阻尼
        wing_joint = root.find("default", "wing").joint
        wing_joint.stiffness = _WING_PARAMS["stiffness"]
        wing_joint.damping = _WING_PARAMS["damping"]

    # -- 基本属性 ----------------------------------------------------------
    @property
    def n_act(self) -> int:
        return self.model.nu

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def ctrl_range(self) -> np.ndarray:
        return np.stack([self._lo, self._hi], axis=1)

    @property
    def root_pos(self) -> np.ndarray:
        return np.array(self.data.qpos[:3], dtype=np.float64)

    @property
    def root_quat(self) -> np.ndarray:
        return np.array(self.data.qpos[3:7], dtype=np.float64)

    def heading(self) -> np.ndarray:
        """身体朝向在水平面上的单位向量。"""
        w, x, y, z = self.root_quat
        fwd = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z)])
        n = np.linalg.norm(fwd)
        return fwd / n if n > 1e-9 else np.array([1.0, 0.0])

    def leg_actuators(self, leg: str) -> dict[str, int]:
        """某条腿的关节名 -> 执行器下标。"""
        out = {}
        for j in LEG_JOINTS:
            name = f"{j}_{leg}"
            if name in self.act_index:
                out[j] = self.act_index[name]
        return out

    def claw_actuator(self, leg: str) -> int | None:
        return self.act_index.get(f"adhere_claw_{leg}")

    # -- 仿真 --------------------------------------------------------------
    def reset(self) -> None:
        self.physics.reset()
        if self.mode == "flight":
            # 起飞高度：让果蝇悬在空中，否则一开始就贴地
            self.data.qpos[2] += 1.0
            # 飞行姿态角要自己设。FruitFly 的 body_pitch_angle 参数在这里
            # 不起作用：把果蝇挂到 arena 上之后，自由关节的 qpos 会覆盖掉
            # 模型自带的初始姿态，physics.reset() 复位到的是单位四元数。
            # 真实果蝇飞行时躯干抬头约 47.5°（Muijres et al., Science 2014）。
            a = np.radians(FLIGHT_PITCH_DEG) / 2
            self.data.qpos[3:7] = [np.cos(a), 0.0, np.sin(a), 0.0]
        self.mj.mj_forward(self.model, self.data)
        self._start_pos = self.root_pos.copy()

    def step(self, ctrl: np.ndarray, n_substeps: int = 1) -> None:
        c = np.clip(np.asarray(ctrl, np.float64).reshape(-1)[:self.n_act],
                    self._lo, self._hi)
        self.data.ctrl[:len(c)] = c
        for _ in range(n_substeps):
            self.mj.mj_step(self.model, self.data)

    def displacement(self) -> np.ndarray:
        return self.root_pos - self._start_pos

    def fallen(self) -> bool:
        """躯干翻倒判定：身体 z 轴与世界 z 轴夹角过大。"""
        w, x, y, z = self.root_quat
        up_z = 1 - 2 * (x * x + y * y)
        return bool(up_z < 0.3)

    # -- 本体感觉：送回大脑的反馈 --------------------------------------------
    def proprioception(self) -> np.ndarray:
        qpos = np.asarray(self.data.qpos, dtype=np.float32)
        qvel = np.asarray(self.data.qvel, dtype=np.float32)
        return np.concatenate([
            np.tanh(qpos[7:]),          # 跳过自由关节的 7 维位姿
            np.tanh(qvel[6:] * 0.02),
            np.tanh(qpos[3:7]),         # 躯干四元数
        ]).astype(np.float32)

    # -- 渲染 --------------------------------------------------------------
    def set_orbit(self, azimuth: float | None = None,
                  elevation: float | None = None,
                  distance: float | None = None) -> dict:
        """设置自由视角（绕果蝇转）。Web 端鼠标拖动改的就是这三个数。"""
        o = self._orbit
        if azimuth is not None:
            o["azimuth"] = float(azimuth) % 360.0
        if elevation is not None:
            o["elevation"] = float(np.clip(elevation, -89.0, 89.0))
        if distance is not None:
            o["distance"] = float(np.clip(distance, 0.15, 20.0))
        return dict(o)

    @property
    def orbit(self) -> dict:
        return dict(self._orbit)

    def _orbit_camera(self):
        """跟随躯干的自由相机 —— 果蝇走到哪都在画面里。"""
        if self._orbit_cam is None:
            cam = self.mj.MjvCamera()
            body_id = -1
            for nm in ("thorax", "torso", "fly"):
                try:
                    body_id = self.mj.mj_name2id(
                        self.model, self.mj.mjtObj.mjOBJ_BODY, nm)
                except Exception:                        # noqa: BLE001
                    body_id = -1
                if body_id >= 0:
                    break
            if body_id >= 0:
                cam.type = self.mj.mjtCamera.mjCAMERA_TRACKING
                cam.trackbodyid = body_id
            else:
                cam.type = self.mj.mjtCamera.mjCAMERA_FREE
            self._orbit_cam = cam
        cam = self._orbit_cam
        o = self._orbit
        cam.azimuth = o["azimuth"]
        cam.elevation = o["elevation"]
        cam.distance = o["distance"]
        if cam.type == self.mj.mjtCamera.mjCAMERA_FREE:
            cam.lookat[:] = self.root_pos
        return cam

    def render(self, camera: str | int = -1) -> np.ndarray:
        if self._renderer is None:
            self._renderer = self.mj.Renderer(self.model, self.h, self.w)
        if camera == "orbit":
            self._renderer.update_scene(self.data, camera=self._orbit_camera())
            return self._renderer.render()
        cam = camera
        if isinstance(camera, str):
            cam = (self.camera_names.index(camera)
                   if camera in self.camera_names else -1)
        self._renderer.update_scene(self.data, camera=cam)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
