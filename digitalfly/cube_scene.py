"""把三阶魔方放进 MuJoCo 场景，并做转层动画。

魔方的 26 个块用 **mocap 体**表示：每帧直接写它们的位置与朝向，不走物理。
魔方不是被果蝇physically推动的，转动是运动学动画 —— 这一点在 README 和
运行输出里都会写明。

每个块自带六面贴纸（薄片），颜色由它的身份决定；块在魔方里的槽位和朝向
由 `cube.Cube` 的状态算出来。转层时把该层 9 个块绕对应轴连续旋转，
动画结束再把状态推进一步。
"""
from __future__ import annotations

import numpy as np

from .cube import (CENTER_SLOT, CORNER_COLORS, CORNER_SLOT, EDGE_COLORS,
                   EDGE_SLOT, FACE_AXIS, FACE_NORMAL, FACE_RGBA, Cube,
                   layer_slots)

STICKER_T = 0.055        # 贴纸厚度（占块宽的比例）
GAP = 0.06               # 块间缝隙


def _quat_from_axis_angle(axis, angle) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.array([1.0, 0, 0, 0])
    axis = axis / n
    s = np.sin(angle / 2)
    return np.array([np.cos(angle / 2), *(axis * s)])


def _quat_mul(a, b) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_to_mat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _mat_to_quat(m) -> np.ndarray:
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s,
                         (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
    if i == 0:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        return np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                         (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    if i == 1:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        return np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                         0.25 * s, (m[1, 2] + m[2, 1]) / s])
    s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
    return np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                     (m[1, 2] + m[2, 1]) / s, 0.25 * s])


def _rotation_from_normals(home, target) -> np.ndarray:
    """由"哪张贴纸朝哪个方向"解出块的旋转矩阵。

    块在魔方里的朝向必然是魔方旋转群（24 个元素）里的一个，也就是一个
    带符号的置换矩阵。之前用"把 home 方向转到 here 方向的最小旋转"是错的：
    那会得到非轴对齐的朝向，渲染出来每个块都是歪的。

    正确做法：块的每张贴纸在出生位置有一个法向 h_j，在当前槽位应当朝向
    t_j，于是旋转 R 满足 R @ h_j = t_j。三个法向互相正交且是单位向量，
    所以 H 是正交矩阵，R = T @ Hᵀ。
    """
    H = np.column_stack(home)
    T = np.column_stack(target)
    return T @ H.T


def _corner_rotation(cubie: int, slot: int, co: int) -> np.ndarray:
    home = [np.asarray(FACE_NORMAL[f], float) for f in CORNER_COLORS[cubie]]
    sf = CORNER_COLORS[slot]
    # 朝向 co 表示：块的第 j 张贴纸落在槽位的第 (j+co)%3 个facelet 上
    target = [np.asarray(FACE_NORMAL[sf[(j + co) % 3]], float)
              for j in range(3)]
    return _rotation_from_normals(home, target)


def _edge_rotation(cubie: int, slot: int, eo: int) -> np.ndarray:
    hf, sf = EDGE_COLORS[cubie], EDGE_COLORS[slot]
    order = (0, 1) if eo == 0 else (1, 0)
    h = [np.asarray(FACE_NORMAL[f], float) for f in hf]
    t = [np.asarray(FACE_NORMAL[sf[order[0]]], float),
         np.asarray(FACE_NORMAL[sf[order[1]]], float)]
    # 棱块只有两张贴纸，第三个轴取叉积补全
    return _rotation_from_normals(h + [np.cross(h[0], h[1])],
                                  t + [np.cross(t[0], t[1])])


class CubeScene:
    """魔方的几何、贴纸与转层动画。"""

    def __init__(self, size: float = 1.2, pos=(1.6, 0.0, 0.0),
                 name: str = "cube"):
        self.size = float(size)
        self.origin = np.asarray(pos, dtype=np.float64)
        self.name = name
        self.unit = self.size / 3.0          # 单块宽度
        self.state = Cube()

        # 26 个块：8 角 + 12 棱 + 6 心
        self.pieces = ([("corner", i) for i in range(8)] +
                       [("edge", i) for i in range(12)] +
                       [("center", f) for f in "URFDLB"])
        self.index = {p: i for i, p in enumerate(self.pieces)}

        self.anim_face = None
        self.anim_t = 0.0
        self.anim_dur = 0.25
        self.anim_turns = 1

    # -- 建模 ---------------------------------------------------------------
    def build(self, worldbody) -> None:
        """把 26 个块作为 mocap 体直接加到 worldbody 下。

        MuJoCo 要求 mocap 体必须是 worldbody 的直接子节点，所以不能包一层
        父 body，也不能用 mjcf 的 attach（那会多出一层 frame）。
        """
        h = self.unit / 2 * (1 - GAP)
        st = self.unit / 2 * STICKER_T
        # 一个专门框住"果蝇 + 魔方"的机位
        cx, cy, cz = self.origin
        worldbody.add("camera", name=f"{self.name}_cam",
                      pos=[cx - 1.1, cy - 1.5, cz + 0.9],
                      xyaxes=[0.80, -0.60, 0, 0.26, 0.35, 0.90])
        worldbody.add("camera", name=f"{self.name}_close",
                      pos=[cx - 0.55, cy - 0.75, cz + 0.42],
                      xyaxes=[0.80, -0.60, 0, 0.26, 0.35, 0.90])

        for kind, key in self.pieces:
            bn = f"{self.name}_{kind}_{key}"
            body = worldbody.add("body", name=bn, mocap=True,
                                 pos=[0, 0, 0])
            body.add("geom", type="box", size=[h, h, h],
                     rgba=[0.08, 0.08, 0.09, 1], contype=0, conaffinity=0,
                     group=1, mass=0)
            colors = (CORNER_COLORS[key] if kind == "corner"
                      else EDGE_COLORS[key] if kind == "edge" else (key,))
            for face in colors:
                n = np.asarray(FACE_NORMAL[face], dtype=np.float64)
                pos = n * (h + st)
                sz = [st if abs(n[k]) > 0.5 else h * 0.92 for k in range(3)]
                body.add("geom", type="box", size=sz, pos=pos,
                         rgba=list(FACE_RGBA[face]), contype=0,
                         conaffinity=0, group=1, mass=0)

    # -- 状态 -> 每块的位姿 --------------------------------------------------
    def piece_pose(self, kind: str, key) -> tuple[np.ndarray, np.ndarray]:
        """返回某个块此刻应当在的 (位置, 四元数)。"""
        st = self.state
        if kind == "corner":
            slot = st.cp.index(key)          # 这个块现在在哪个槽位
            R = _corner_rotation(key, slot, st.co[slot])
            pos = np.asarray(CORNER_SLOT[slot], float) * self.unit
        elif kind == "edge":
            slot = st.ep.index(key)
            R = _edge_rotation(key, slot, st.eo[slot])
            pos = np.asarray(EDGE_SLOT[slot], float) * self.unit
        else:
            R = np.eye(3)
            pos = np.asarray(CENTER_SLOT[key], float) * self.unit
        return pos, _mat_to_quat(R)

    def poses(self) -> dict:
        """所有块的世界位姿，考虑正在进行的转层动画。"""
        out = {}
        anim = None
        if self.anim_face is not None:
            frac = min(self.anim_t / self.anim_dur, 1.0)
            angle = (self.anim_turns * np.pi / 2) * _ease(frac)
            axis = np.asarray(FACE_AXIS[self.anim_face], dtype=np.float64)
            qa = _quat_from_axis_angle(axis, angle)
            anim = (set(layer_slots(self.anim_face)), qa)

        for kind, key in self.pieces:
            pos, q = self.piece_pose(kind, key)
            if anim is not None:
                slots, qa = anim
                slot = self._current_slot(kind, key)
                if slot in slots:
                    R = _quat_to_mat(qa)
                    pos = R @ pos
                    q = _quat_mul(qa, q)
            out[(kind, key)] = (self.origin + pos, q)
        return out

    def _current_slot(self, kind: str, key):
        if kind == "corner":
            return ("corner", self.state.cp.index(key))
        if kind == "edge":
            return ("edge", self.state.ep.index(key))
        return ("center", key)

    # -- 动画 ---------------------------------------------------------------
    def start_move(self, move: str, duration: float | None = None) -> None:
        self.anim_face = move[0]
        self.anim_turns = {"": 1, "2": 2, "'": 3}[move[1:]]
        if self.anim_turns == 3:
            self.anim_turns = -1
        self.anim_t = 0.0
        self.pending = move
        if duration is not None:
            self.anim_dur = duration

    @property
    def animating(self) -> bool:
        return self.anim_face is not None

    def advance(self, dt: float) -> bool:
        """推进动画。返回 True 表示这一步刚刚完成。"""
        if self.anim_face is None:
            return False
        self.anim_t += dt
        if self.anim_t >= self.anim_dur:
            self.state.apply(self.pending)
            self.anim_face = None
            self.anim_t = 0.0
            return True
        return False

    # -- 写进 MuJoCo --------------------------------------------------------
    def bind(self, model, data) -> None:
        """建立 mocap 体下标映射。"""
        import mujoco
        self.mocap_id = {}
        for kind, key in self.pieces:
            bn = f"{self.name}_{kind}_{key}"
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bn)
            if bid < 0:
                # 附到 arena 上时名字会带前缀
                for i in range(model.nbody):
                    nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                    if nm and nm.endswith(bn):
                        bid = i
                        break
            if bid < 0:
                raise RuntimeError(f"找不到魔方块 {bn}")
            self.mocap_id[(kind, key)] = int(model.body_mocapid[bid])
        self.model, self.data = model, data

    def write(self) -> None:
        for k, (pos, q) in self.poses().items():
            m = self.mocap_id[k]
            self.data.mocap_pos[m] = pos
            self.data.mocap_quat[m] = q


def _ease(t: float) -> float:
    """转层用的缓动，起止平滑一点，看着像手在拧。"""
    return t * t * (3 - 2 * t)


def _slot_rotation(home: np.ndarray, here: np.ndarray) -> np.ndarray:
    """把块从它的出生槽位转到当前槽位所需的旋转。

    只用位置定不出唯一的旋转（块还能绕自身轴转），但角块的朝向由 co 单独给，
    棱块由 eo 单独给，所以这里只需要一个把 home 方向带到 here 方向的最短旋转。
    """
    a = home / max(np.linalg.norm(home), 1e-12)
    b = here / max(np.linalg.norm(here), 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.array([1.0, 0, 0, 0])
        # 反向：绕任意垂直轴转 180°
        axis = np.array([1.0, 0, 0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0])
        axis = np.cross(a, axis)
        return _quat_from_axis_angle(axis, np.pi)
    s = np.sqrt((1 + c) * 2)
    return np.array([s / 2, *(v / s)])
