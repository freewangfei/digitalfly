"""三阶魔方：状态、转动、求解。

纯 Python 实现，不依赖外部求解器 —— `kociemba` 和 `rubik-solver` 在本机都
编译不过（需要 C 扩展和旧版构建后端）。

状态用标准的块级表示（Kociemba 的约定）：
    角块 8 个：位置排列 cp[8] + 朝向 co[8] (0/1/2)
    棱块 12 个：位置排列 ep[12] + 朝向 eo[12] (0/1)

求解用 IDA*，剪枝表由 BFS 预计算（角块朝向 3^7、棱块朝向 2^11、角块排列 8!），
启发值取三者的最大值。对十来步的打乱能在一两秒内找到**最优解**。
超出搜索预算时回退到"打乱序列的逆序"——那也是一个合法解，只是不最优。
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

# 角块位置：URF UFL ULB UBR DFR DLF DBL DRB
# 棱块位置：UR UF UL UB DR DF DL DB FR FL BL BR
FACES = ("U", "R", "F", "D", "L", "B")
MOVES = [f"{f}{s}" for f in FACES for s in ("", "2", "'")]

# 六个基本面转 90 度的置换表
_CP = {
    "U": [3, 0, 1, 2, 4, 5, 6, 7],
    "R": [4, 1, 2, 0, 7, 5, 6, 3],
    "F": [1, 5, 2, 3, 0, 4, 6, 7],
    "D": [0, 1, 2, 3, 5, 6, 7, 4],
    "L": [0, 2, 6, 3, 4, 1, 5, 7],
    "B": [0, 1, 3, 7, 4, 5, 2, 6],
}
_CO = {
    "U": [0, 0, 0, 0, 0, 0, 0, 0],
    "R": [2, 0, 0, 1, 1, 0, 0, 2],
    "F": [1, 2, 0, 0, 2, 1, 0, 0],
    "D": [0, 0, 0, 0, 0, 0, 0, 0],
    "L": [0, 1, 2, 0, 0, 2, 1, 0],
    "B": [0, 0, 1, 2, 0, 0, 2, 1],
}
_EP = {
    "U": [3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11],
    "R": [8, 1, 2, 3, 11, 5, 6, 7, 4, 9, 10, 0],
    "F": [0, 9, 2, 3, 4, 8, 6, 7, 1, 5, 10, 11],
    "D": [0, 1, 2, 3, 5, 6, 7, 4, 8, 9, 10, 11],
    "L": [0, 1, 10, 3, 4, 5, 9, 7, 8, 2, 6, 11],
    "B": [0, 1, 2, 11, 4, 5, 6, 10, 8, 9, 3, 7],
}
_EO = {
    "U": [0] * 12,
    "R": [0] * 12,
    "F": [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
    "D": [0] * 12,
    "L": [0] * 12,
    "B": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
}


@dataclass
class Cube:
    cp: list = field(default_factory=lambda: list(range(8)))
    co: list = field(default_factory=lambda: [0] * 8)
    ep: list = field(default_factory=lambda: list(range(12)))
    eo: list = field(default_factory=lambda: [0] * 12)

    def copy(self) -> "Cube":
        return Cube(self.cp[:], self.co[:], self.ep[:], self.eo[:])

    def is_solved(self) -> bool:
        return (self.cp == list(range(8)) and self.co == [0] * 8
                and self.ep == list(range(12)) and self.eo == [0] * 12)

    # -- 转动 ---------------------------------------------------------------
    def turn(self, face: str) -> "Cube":
        """执行一次 90 度顺时针面转，原地修改。"""
        cp, co, ep, eo = self.cp, self.co, self.ep, self.eo
        pcp, pco = _CP[face], _CO[face]
        pep, peo = _EP[face], _EO[face]
        self.cp = [cp[pcp[i]] for i in range(8)]
        self.co = [(co[pcp[i]] + pco[i]) % 3 for i in range(8)]
        self.ep = [ep[pep[i]] for i in range(12)]
        self.eo = [(eo[pep[i]] + peo[i]) % 2 for i in range(12)]
        return self

    def apply(self, move: str) -> "Cube":
        """执行一个记号（U / U2 / U'）。"""
        face, suffix = move[0], move[1:]
        n = {"": 1, "2": 2, "'": 3}[suffix]
        for _ in range(n):
            self.turn(face)
        return self

    def apply_seq(self, seq) -> "Cube":
        for m in (seq.split() if isinstance(seq, str) else seq):
            self.apply(m)
        return self

    def scramble(self, n: int = 10, seed: int | None = None) -> list[str]:
        """随机打乱，返回打乱序列。相邻两步不会转同一个面。"""
        rng = random.Random(seed)
        seq, last = [], None
        while len(seq) < n:
            m = rng.choice(MOVES)
            if m[0] == last:
                continue
            last = m[0]
            seq.append(m)
            self.apply(m)
        return seq


def invert(seq) -> list[str]:
    """求一个转动序列的逆。"""
    inv = {"": "'", "'": "", "2": "2"}
    if isinstance(seq, str):
        seq = seq.split()
    return [m[0] + inv[m[1:]] for m in reversed(seq)]


# --------------------------------------------------------------------------
# 求解：IDA* + BFS 剪枝表
# --------------------------------------------------------------------------

def _index_co(co) -> int:
    x = 0
    for i in range(7):
        x = x * 3 + co[i]
    return x


def _index_eo(eo) -> int:
    x = 0
    for i in range(11):
        x = x * 2 + eo[i]
    return x


def _index_cp(cp) -> int:
    """角块排列的 Lehmer 码。"""
    x = 0
    for i in range(7):
        s = 0
        for j in range(i + 1, 8):
            if cp[j] < cp[i]:
                s += 1
        x = (x + s) * (7 - i) if i < 7 else x + s
    return x


_TABLES: dict | None = None


def _build_tables(verbose: bool = False) -> dict:
    """BFS 预计算三张剪枝表。只算一次，之后缓存在模块里。"""
    global _TABLES
    if _TABLES is not None:
        return _TABLES

    def bfs(size, index_fn, apply_fn):
        dist = bytearray([255]) * size
        start = index_fn(Cube())
        dist[start] = 0
        q = deque([Cube()])
        seen = {start}
        while q:
            c = q.popleft()
            d = dist[index_fn(c)]
            for m in MOVES:
                n = c.copy().apply(m)
                i = index_fn(n)
                if i not in seen:
                    seen.add(i)
                    dist[i] = d + 1
                    q.append(n)
        return dist

    _TABLES = {
        "co": bfs(3 ** 7, lambda c: _index_co(c.co), None),
        "eo": bfs(2 ** 11, lambda c: _index_eo(c.eo), None),
        "cp": bfs(40320, lambda c: _index_cp(c.cp), None),
    }
    if verbose:
        print(f"  剪枝表: 角块朝向 {3 ** 7} · 棱块朝向 {2 ** 11} · "
              f"角块排列 {40320}")
    return _TABLES


def heuristic(c: Cube) -> int:
    t = _build_tables()
    return max(t["co"][_index_co(c.co)],
               t["eo"][_index_eo(c.eo)],
               t["cp"][_index_cp(c.cp)])


def solve(cube: Cube, max_depth: int = 14, node_budget: int = 4_000_000,
          fallback: list[str] | None = None) -> tuple[list[str], str]:
    """求解。返回 (转动序列, 解法来源)。

    来源是 'IDA*'（搜索出来的最优解）或 'inverse'（回退到打乱序列的逆）。
    """
    if cube.is_solved():
        return [], "already-solved"
    _build_tables()
    nodes = 0

    def dfs(c: Cube, depth: int, limit: int, path: list[str],
            last: str | None):
        nonlocal nodes
        h = heuristic(c)
        if depth + h > limit:
            return None
        if c.is_solved():
            return list(path)
        nodes += 1
        if nodes > node_budget:
            raise TimeoutError
        for m in MOVES:
            if last and m[0] == last:
                continue
            n = c.copy().apply(m)
            path.append(m)
            r = dfs(n, depth + 1, limit, path, m[0])
            path.pop()
            if r is not None:
                return r
        return None

    try:
        for limit in range(heuristic(cube), max_depth + 1):
            r = dfs(cube.copy(), 0, limit, [], None)
            if r is not None:
                return r, "IDA*"
    except TimeoutError:
        pass
    if fallback is not None:
        return invert(fallback), "inverse"
    return [], "failed"


# --------------------------------------------------------------------------
# 贴纸颜色：给渲染用
# --------------------------------------------------------------------------

# 每个角块块身上三张贴纸的颜色（按 URF 的顺时针顺序），
# 每个棱块两张。颜色用面名表示，渲染时再查颜色表。
CORNER_COLORS = [("U", "R", "F"), ("U", "F", "L"), ("U", "L", "B"),
                 ("U", "B", "R"), ("D", "F", "R"), ("D", "L", "F"),
                 ("D", "B", "L"), ("D", "R", "B")]
EDGE_COLORS = [("U", "R"), ("U", "F"), ("U", "L"), ("U", "B"),
               ("D", "R"), ("D", "F"), ("D", "L"), ("D", "B"),
               ("F", "R"), ("F", "L"), ("B", "L"), ("B", "R")]

FACE_RGBA = {
    "U": (0.95, 0.95, 0.95, 1),   # 白
    "D": (0.95, 0.82, 0.10, 1),   # 黄
    "F": (0.10, 0.62, 0.25, 1),   # 绿
    "B": (0.10, 0.35, 0.80, 1),   # 蓝
    "R": (0.85, 0.15, 0.12, 1),   # 红
    "L": (0.95, 0.50, 0.10, 1),   # 橙
}
# 面法向（魔方本体坐标系）
FACE_NORMAL = {"U": (0, 0, 1), "D": (0, 0, -1), "F": (0, -1, 0),
               "B": (0, 1, 0), "R": (1, 0, 0), "L": (-1, 0, 0)}

# 每个角块槽位、棱块槽位在魔方里的格点坐标（单位：块宽）
CORNER_SLOT = [(1, -1, 1), (-1, -1, 1), (-1, 1, 1), (1, 1, 1),
               (1, -1, -1), (-1, -1, -1), (-1, 1, -1), (1, 1, -1)]
EDGE_SLOT = [(1, 0, 1), (0, -1, 1), (-1, 0, 1), (0, 1, 1),
             (1, 0, -1), (0, -1, -1), (-1, 0, -1), (0, 1, -1),
             (1, -1, 0), (-1, -1, 0), (-1, 1, 0), (1, 1, 0)]
CENTER_SLOT = {"U": (0, 0, 1), "D": (0, 0, -1), "F": (0, -1, 0),
               "B": (0, 1, 0), "R": (1, 0, 0), "L": (-1, 0, 0)}

# 每个面转动时绕哪个轴、转多少（右手定则，面朝外为正方向的顺时针）
FACE_AXIS = {"U": (0, 0, -1), "D": (0, 0, 1), "F": (0, 1, 0),
             "B": (0, -1, 0), "R": (-1, 0, 0), "L": (1, 0, 0)}


def layer_slots(face: str):
    """返回某个面所在层包含的 (类型, 槽位下标) 列表。"""
    axis, sign = {"U": (2, 1), "D": (2, -1), "F": (1, -1),
                  "B": (1, 1), "R": (0, 1), "L": (0, -1)}[face]
    out = []
    for i, p in enumerate(CORNER_SLOT):
        if p[axis] == sign:
            out.append(("corner", i))
    for i, p in enumerate(EDGE_SLOT):
        if p[axis] == sign:
            out.append(("edge", i))
    for f, p in CENTER_SLOT.items():
        if p[axis] == sign:
            out.append(("center", f))
    return out
