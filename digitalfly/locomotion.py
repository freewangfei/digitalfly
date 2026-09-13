"""运动模式发生器：产生步态与振翅的节律。

**为什么需要这一层，以及它和连接组的分工。**

果蝇的行走节律不是大脑逐个关节算出来的。腹神经索里的腿部神经节含有中枢模式
发生器 (CPG)，自己就能产生三角步态的相位关系；大脑通过下行神经元下达的是
"走快点""向左转"这类**指令**，不是关节角度。切断果蝇的脑，腹神经索仍然能产生
节律性的腿部运动 —— 这是经典实验结果。

所以这里的分工是：

    连接组脉冲网络  ->  下行神经元群体活动  ->  前进速度 / 转向 / 振翅幅度
                                                        |
    本模块的模式发生器  ->  六条腿 / 两只翅膀的具体关节角度轨迹

这一层的相位关系（三角步态、占空比、摆动抬腿）是运动学模型，不是从连接组里
读出来的 —— 连接组给不出这个。MaleCNS v1.0 有完整的腹神经索接线图，原则上
CPG 应该能从中涌现，但那需要神经元自身的内在振荡特性（离子通道动力学），
电镜重建里没有这些信息，本项目的 LIF 神经元也不具备。

flybody 官方的行走任务同样不从网络里生成步态，而是跟踪真实果蝇的参考轨迹。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from . import config

if str(config.FLYBODY_DIR) not in sys.path:
    sys.path.insert(0, str(config.FLYBODY_DIR))

# 三角步态：两组腿交替着地，每组三条，互差半个周期。
# 前左 / 中右 / 后左 是一组，前右 / 中左 / 后右 是另一组 ——
# 与 flygym 的 tripod 耦合相位矩阵一致。
TRIPOD_A = ("T1_left", "T2_right", "T3_left")
TRIPOD_B = ("T1_right", "T2_left", "T3_right")


# ---------------------------------------------------------------------------
# 行走：真实果蝇迈步运动学
# ---------------------------------------------------------------------------

# 数据文件的 7 个自由度顺序（来自 flygym 的 meta.json），
# 依次映射到 flybody 的执行器名
DOF_TO_ACTUATOR = ("coxa", "coxa_abduct", "coxa_twist",
                   "femur", "femur_twist", "tibia", "tarsus")

# 三角步态的相位分配：前左 + 中右 + 后左 为一组，另三条腿差 π
TRIPOD_PHASE = {
    "T1_left": 0.0, "T2_right": 0.0, "T3_left": 0.0,
    "T1_right": np.pi, "T2_left": np.pi, "T3_right": np.pi,
}
# 胸节 -> 数据里的腿位下标（F=前 0, M=中 1, H=后 2）
SEG_TO_POS = {"T1": 0, "T2": 1, "T3": 2}

# 每段轨迹是从哪一侧的腿上提取的（见 single_steps_flybody.meta.json 的 picks）。
# flygym 的说明是 flybody 的腿关节轴相对矢状面对称，六条腿可以直接复用同一条
# 轨迹；实测也确实是不镜像时前进最快（2.4 vs 2.0 体长/秒），而且转向指令的
# 符号是对的。镜像开关保留下来，`mirror=True` 可以对照。
SEG_SOURCE_SIDE = {"T1": "right", "T2": "left", "T3": "left"}
# 镜像时要取反的自由度下标（Coxa_roll, Coxa_yaw, Femur_roll）
MIRROR_DOFS = (1, 2, 4)

STEP_DATA = config.ROOT / "third_party" / "flygym_cpg" / "assets"


class PreprogrammedGait:
    """用真实果蝇的迈步运动学驱动 flybody 的六条腿。

    数据来自 NeuroMechFly v2 / flygym 项目 (NeLy-EPFL) 的
    `single_steps_flybody.npz`：他们把真实果蝇在球上行走的腿部运动学
    重定向到 flybody 模型上，抽出每条腿位（前/中/后足）一个完整步态周期的
    7 个关节角轨迹，以及各自的摆动相占比。

    这比自己按正弦拼一套步态可靠得多 —— 实测自己调的步态只能走到约
    0.5 体长/秒，而且姿态很勉强；真实运动学是被论文验证过的。

    相位 0 是后极点 (PEP)，周期布局：
        [0, swing_end]  摆动相，足离地
        [swing_end, 2π] 着地相，足着地并推进
    """

    NOMINAL_FREQ_HZ = 12.0     # 数据的名义步频

    # 步态配平。三段轨迹分别取自右前足、左中足、左后足（见 meta.json 的 picks），
    # 六条腿共用它们会留下一点左右不对称，turn=0 时果蝇会以约 20°/秒向右偏。
    # 实测 turn=+0.22 时 3 秒内朝向变化只有约 2°，把它作为零点。
    TURN_TRIM = 0.22

    def __init__(self, body, path=None, freq_hz: float | None = None,
                 mirror: bool = False):
        from scipy.interpolate import CubicSpline

        path = path or (STEP_DATA / "single_steps_flybody.npz")
        if not path.exists():
            raise FileNotFoundError(
                f"缺少迈步数据 {path}\n"
                "来自 NeLy-EPFL/flygym，用 setup.sh 获取")
        with np.load(path, allow_pickle=False) as npz:
            angles = npz["joint_angles"]          # (3, 7, n_bins)
            swing_frac = npz["swing_fractions"]   # (3,)

        self.body = body
        self.freq_hz = freq_hz or self.NOMINAL_FREQ_HZ
        n_bins = angles.shape[-1]
        # 周期样条：数据不含终点，补上首样本才能做周期边界条件
        grid = np.concatenate([np.linspace(0, 2 * np.pi, n_bins,
                                           endpoint=False), [2 * np.pi]])

        self._legs = {}
        for leg, phase0 in TRIPOD_PHASE.items():
            act = body.leg_actuators(leg)
            if not act:
                continue
            pos = SEG_TO_POS[leg.split("_")[0]]
            a = angles[pos]
            a_periodic = np.concatenate([a, a[:, :1]], axis=1)
            self._legs[leg] = {
                "spline": CubicSpline(grid, a_periodic, axis=1,
                                      bc_type="periodic"),
                # 与数据来源不同侧的腿要做镜像
                "mirror": mirror and leg.split("_")[1] != SEG_SOURCE_SIDE[
                    leg.split("_")[0]],
                "act": act,
                "claw": body.claw_actuator(leg),
                "phase0": phase0,
                "swing_end": float(swing_frac[pos]) * 2 * np.pi,
                # 中性姿态取相位 π 处，用于按幅度缩放
                "neutral": None,
            }
            neutral = self._legs[leg]["spline"](np.pi).copy()
            if self._legs[leg]["mirror"]:
                neutral[list(MIRROR_DOFS)] *= -1
            self._legs[leg]["neutral"] = neutral

        self.n_legs = len(self._legs)
        self.phase = 0.0

    def reset(self) -> None:
        self.phase = 0.0

    def step(self, ctrl: np.ndarray, dt: float,
             speed: float = 1.0, turn: float = 0.0,
             adhesion: float = 1.0) -> np.ndarray:
        """写入六条腿的关节角度。

            speed ∈ [0, 1]   0 = 站立不动
            turn  ∈ [-1, 1]  负 = 左转（左侧步幅缩小）
        """
        speed = float(np.clip(speed, 0.0, 1.0))
        # turn > 0 = 左转（实测确定的方向约定）；叠加配平后 turn=0 即直行
        turn = float(np.clip(turn + self.TURN_TRIM, -1.0, 1.0))
        if speed > 1e-3:
            self.phase = (self.phase + 2 * np.pi * self.freq_hz * dt *
                          speed) % (2 * np.pi)

        for leg, L in self._legs.items():
            ph = (self.phase + L["phase0"]) % (2 * np.pi)
            # 差速转向：内侧腿步幅缩小，外侧放大
            side = leg.split("_")[1]
            mag = speed * (1.0 - turn if side == "left" else 1.0 + turn)
            mag = float(np.clip(mag, 0.0, 1.6))

            raw = L["spline"](ph).copy()
            if L["mirror"]:
                raw[list(MIRROR_DOFS)] *= -1
            ang = L["neutral"] + mag * (raw - L["neutral"])
            for k, dof in enumerate(DOF_TO_ACTUATOR):
                i = L["act"].get(dof)
                if i is not None:
                    ctrl[i] = ang[k]
            if L["claw"] is not None:
                in_swing = ph < L["swing_end"]
                ctrl[L["claw"]] = 0.0 if in_swing else adhesion
        return ctrl


class WingBeat:
    """振翅发生器，用 flybody 自带的 WingBeatPatternGenerator。

    它按给定频率生成完整的振翅周期（yaw / roll / pitch 三个自由度），
    频率切换时保持相位连续。基础频率 218 Hz，取自 flybody 的常数表。

    **翅膀执行器是力执行器，不是位置伺服。** 直接写绝对角度是没有用的
    （实测：果蝇原地掉下去）。正确做法和 flybody 的飞行任务一致 ——
    喂"目标角 − 当前关节角"，相当于在力执行器上做一个比例控制器。
    """

    def __init__(self, body, dt_ctrl: float):
        import mujoco
        from flybody.tasks.constants import _WING_PARAMS
        from flybody.tasks.pattern_generators import WingBeatPatternGenerator

        self.body = body
        self.mj = mujoco
        self.base_freq = _WING_PARAMS["base_freq"]
        self.rel_range = _WING_PARAMS["rel_freq_range"]
        self.wpg = WingBeatPatternGenerator(dt_ctrl=dt_ctrl)

        # 执行器与对应关节的 qpos 地址（顺序：左 yaw/roll/pitch，右 yaw/roll/pitch）
        self.act, self.qadr = [], []
        m = body.model
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
                 for i in range(m.njnt)]
        for side in ("left", "right"):
            for axis in ("yaw", "roll", "pitch"):
                a = body.act_index.get(f"wing_{axis}_{side}")
                jn = next((n for n in names
                           if n and n.endswith(f"wing_{axis}_{side}")), None)
                if a is None or jn is None:
                    self.act, self.qadr = [], []
                    break
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
                self.act.append(a)
                self.qadr.append(int(m.jnt_qposadr[jid]))
        self.has_wings = len(self.act) == 6
        self.reset()

    def reset(self) -> None:
        """复位模式发生器，并把翅关节置到周期起点的角度与角速度。"""
        out = self.wpg.reset(initial_phase=0.0)
        if not self.has_wings:
            return
        qpos, qvel = (out if isinstance(out, tuple) else (out, None))
        d, m = self.body.data, self.body.model
        for k, adr in enumerate(self.qadr):
            d.qpos[adr] = float(np.ravel(qpos)[k])
            if qvel is not None:
                jid = int(np.searchsorted(m.jnt_qposadr, adr))
                d.qvel[int(m.jnt_dofadr[jid])] = float(np.ravel(qvel)[k])
        self.mj.mj_forward(m, d)

    def step(self, ctrl: np.ndarray, freq_rel: float = 0.0,
             asymmetry: float = 0.0, amplitude: float = 1.0) -> np.ndarray:
        """写入两只翅膀的控制量。

            freq_rel   ∈ [-1, 1]  相对基础频率的调整，控制升力
            asymmetry  ∈ [-1, 1]  左右振幅差，负 = 左转，正 = 右转
            amplitude  ∈ [0, 1.5] 整体振幅
        """
        if not self.has_wings:
            return ctrl
        freq = self.base_freq * (1 + self.rel_range * np.clip(freq_rel, -1, 1))
        target = np.ravel(self.wpg.step(ctrl_freq=freq))   # (3,) 或 (6,)
        if target.size == 3:
            target = np.concatenate([target, target])

        a = float(np.clip(asymmetry, -1, 1))
        gain = np.array([amplitude * (1 - a)] * 3 + [amplitude * (1 + a)] * 3)
        qpos = self.body.data.qpos
        for k, (act, adr) in enumerate(zip(self.act, self.qadr)):
            # 力执行器 + 比例控制：控制量 = 目标角 − 当前角
            ctrl[act] = target[k] * gain[k] - qpos[adr]
        return ctrl


class FlightController:
    """飞行控制：振翅产生升力 + 姿态稳定 + 高度保持。

    为什么需要姿态稳定：只让翅膀按正确运动学拍动，果蝇会一边产生升力一边翻滚
    （实测）。真实果蝇靠**平衡棒**（halteres，退化的后翅，充当陀螺仪）把身体
    角速度反馈到翅膀肌肉上，这是一条延迟仅几毫秒的反射弧，不经过大脑。
    这里用一个 PD 控制器实现同样的功能，并如实标注它是反射层而不是连接组的输出。

    分工和行走一致：
        连接组下行神经元  ->  期望爬升 / 期望转向（指令）
        本控制器          ->  翅膀的具体拍动与姿态校正（反射）
    """

    # 翅膀执行器增益。flybody 飞行任务用 18，那是配合他们真实翅拍数据的；
    # 本项目用的是 WingBeatPatternGenerator 的内置近似模式（真实模式数据在
    # figshare 上，本机访问被拒），pitch 跟踪只能到目标的 ~40%，攻角不足。
    # 提高增益补偿后才能产生足以悬停的升力（实测 18 会下坠，60 能维持高度）。
    WING_GAIN = 70.0

    def __init__(self, body, dt_ctrl: float, wing_gain: float | None = None):
        self.body = body
        self.dt = dt_ctrl
        self.wings = WingBeat(body, dt_ctrl=dt_ctrl)
        gain = self.WING_GAIN if wing_gain is None else wing_gain
        for a in self.wings.act:
            body.model.actuator_gainprm[a][0] = gain
        self.abdomen = body.act_index.get("abdomen")
        self.target_z = None

    def reset(self, target_z: float | None = None) -> None:
        self.wings.reset()
        self.target_z = (self.body.root_pos[2] if target_z is None
                         else target_z)

    def _attitude(self) -> tuple[float, float, float, np.ndarray]:
        """返回 (roll, pitch, yaw 的正弦近似, 角速度)。"""
        w, x, y, z = self.body.root_quat
        roll = 2 * (w * x + y * z)
        pitch = 2 * (w * y - z * x)
        yaw = 2 * (w * z + x * y)
        omega = np.asarray(self.body.data.qvel[3:6], dtype=np.float64)
        return float(roll), float(pitch), float(yaw), omega

    def step(self, ctrl: np.ndarray, climb: float = 0.0,
             turn: float = 0.0, amplitude: float = 1.0) -> np.ndarray:
        """
            climb ∈ [-1, 1]  期望爬升 / 下降
            turn  ∈ [-1, 1]  期望转向，负 = 左
        """
        roll, pitch, _, omega = self._attitude()

        # 平衡棒反射：角速度与倾角反馈到左右翅振幅差（横滚）
        kp, kd = 0.9, 0.06
        roll_corr = np.clip(-kp * roll - kd * omega[0], -0.6, 0.6)
        # 指令转向叠加在反射之上
        asym = float(np.clip(roll_corr + 0.35 * turn, -0.8, 0.8))

        # 高度保持：升力靠整体振幅，期望爬升叠加在保持项上
        z = self.body.root_pos[2]
        dz = (self.target_z - z) if self.target_z is not None else 0.0
        vz = float(self.body.data.qvel[2])
        lift = 1.0 + 0.35 * np.clip(dz, -1, 1) - 0.02 * vz + 0.25 * climb
        amp = float(np.clip(amplitude * lift, 0.5, 1.6))

        self.wings.step(ctrl, freq_rel=np.clip(climb, -1, 1),
                        asymmetry=asym, amplitude=amp)

        # 俯仰用腹部摆动来配平，这也是果蝇真实的做法
        if self.abdomen is not None:
            ctrl[self.abdomen] = float(np.clip(
                -1.2 * pitch - 0.08 * omega[1], -0.8, 0.6))
        return ctrl
