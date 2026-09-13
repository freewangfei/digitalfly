"""用前腿拨魔方 —— 让画面和网上那些视频一致。

之前魔方是自己转的：果蝇绕着它走圈，方块到点就转一格，两者没有任何接触。
这里补上缺的那一环 —— **果蝇站到魔方跟前，用一条前腿去拨要转的那一面，
方块跟着这一拨转动**。

分工写清楚：

* **什么时候拧** 来自连接组 —— 下行神经元（DN）的群体发放率积分到阈值才触发
  一次转动，网络越活跃拧得越快。这一段是真的。
* **拧哪一面** 来自 IDA* 求解器。DN 的六个子群也各投一票，投票和求解器的
  一致率会显示出来（实测约 17%，和随机持平 —— 这是个如实报告的阴性结果）。
* **前腿这一拨** 是运动学动画，不是物理接触求解。果蝇的跗节和魔方之间没有
  真实的摩擦驱动，是腿的姿态和面的转角按同一个相位一起走。网上那些视频也
  都是动画，这里只是把它做得一致。

T1 = 前胸足（prothoracic leg），也就是果蝇的前腿。
"""
from __future__ import annotations

import numpy as np

# 每条前腿在"伸手去拨"时各关节的目标姿态（相对各自活动范围的比例 0~1）。
# 数值是照着果蝇前腿够物体的姿势凑的：髋部外展抬起、股节前伸、胫节压下去。
REACH = {
    "coxa_abduct": 0.72,
    "coxa_twist": 0.50,
    "coxa": 0.30,
    "femur_twist": 0.50,
    "femur": 0.24,
    "tibia": 0.78,
    "tarsus": 0.62,
    "tarsus2": 0.55,
}
# 拨到一半时的姿态 —— 和 REACH 的差值就是"这一拨"的行程
STROKE = {
    "coxa_abduct": 0.55,
    "coxa_twist": 0.50,
    "coxa": 0.62,
    "femur_twist": 0.50,
    "femur": 0.45,
    "tibia": 0.55,
    "tarsus": 0.45,
    "tarsus2": 0.45,
}

# 哪一面用哪条腿。U/F/R 这些面在果蝇右前方时用右前腿，反之用左前腿。
FACE_SIDE = {"U": "right", "R": "right", "F": "right",
             "D": "left", "L": "left", "B": "left"}


class FrontLegReach:
    """把一条前腿的姿态按相位写进 ctrl。"""

    def __init__(self, body):
        self.body = body
        self.act: dict[str, dict[str, int]] = {"left": {}, "right": {}}
        for side in ("left", "right"):
            for joint in REACH:
                a = body.act_index.get(f"{joint}_T1_{side}")
                if a is not None:
                    self.act[side][joint] = a
        self.range = body.ctrl_range

    def available(self) -> bool:
        return bool(self.act["left"] and self.act["right"])

    def apply(self, ctrl: np.ndarray, side: str, phase: float,
              blend: float = 1.0) -> None:
        """phase 0->1 走完一次"伸手 + 拨动"，blend 是与步态姿态的混合比例。

        相位前 30% 伸出去，之后 70% 拨过去 —— 和面转动的角度同步。
        """
        acts = self.act.get(side)
        if not acts or blend <= 0:
            return
        t = float(np.clip(phase, 0.0, 1.0))
        if t < 0.3:
            u = t / 0.3                     # 伸手：从步态姿态过渡到 REACH
            frac = {k: REACH[k] for k in REACH}
            w = blend * u
        else:
            u = (t - 0.3) / 0.7             # 拨动：REACH -> STROKE
            frac = {k: REACH[k] + (STROKE[k] - REACH[k]) * _ease(u)
                    for k in REACH}
            w = blend
        for joint, a in acts.items():
            lo, hi = self.range[a]
            target = lo + (hi - lo) * float(np.clip(frac[joint], 0.0, 1.0))
            ctrl[a] = (1.0 - w) * ctrl[a] + w * target


def _ease(t: float) -> float:
    """两头慢中间快 —— 拨动看起来才像一下子推过去的。"""
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def approach(body, target_xy, stop_dist: float = 0.42) -> tuple:
    """走向魔方并在它跟前停下、正对着它。

    返回 (speed, turn, arrived)。arrived 之后才开始伸腿。
    """
    pos = np.asarray(body.root_pos[:2], dtype=np.float64)
    tgt = np.asarray(target_xy, dtype=np.float64)
    d = tgt - pos
    dist = float(np.linalg.norm(d))
    if dist < 1e-6:
        return 0.0, 0.0, True
    want = d / dist
    h = body.heading()
    # 叉积给出该往哪边转；turn > 0 = 左转（locomotion.py 的约定）
    cross = float(h[0] * want[1] - h[1] * want[0])
    dot = float(h[0] * want[0] + h[1] * want[1])
    turn = float(np.clip(2.0 * cross, -0.6, 0.6))
    arrived = dist <= stop_dist and dot > 0.85
    if arrived:
        return 0.0, float(np.clip(1.5 * cross, -0.3, 0.3)), True
    # 没对准方向就先转，别绕大圈
    speed = 0.0 if dot < 0.3 else float(np.clip(dist - stop_dist, 0.0, 1.0))
    return max(speed, 0.35) if dot >= 0.3 else 0.0, turn, False
