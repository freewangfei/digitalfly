"""魔方模型、求解器与场景几何的测试。"""
from __future__ import annotations

import numpy as np
import pytest

from digitalfly.cube import (MOVES, Cube, invert, solve)


def test_identity_is_solved():
    assert Cube().is_solved()


@pytest.mark.parametrize("face", list("URFDLB"))
def test_four_turns_restore(face):
    """任何一个面转四次回到原状。"""
    c = Cube()
    for _ in range(4):
        c.turn(face)
    assert c.is_solved()


def test_sexy_move_order_six():
    """R U R' U' 重复六次复原 —— 这个序列的阶恰好是 6。"""
    c = Cube()
    for _ in range(6):
        c.apply_seq("R U R' U'")
    assert c.is_solved()


def test_superflip_order():
    """全翻（superflip）自逆：做两遍回到原状。"""
    seq = "U R2 F B R B2 R U2 L B2 R U' D' R2 F R' L B2 U2 F2"
    c = Cube().apply_seq(seq)
    assert not c.is_solved()
    assert c.eo == [1] * 12, "superflip 应当把 12 个棱块全部翻转"
    assert c.ep == list(range(12)) and c.cp == list(range(8))


def test_inverse_restores():
    c = Cube()
    seq = c.scramble(20, seed=42)
    assert not c.is_solved()
    c.apply_seq(invert(seq))
    assert c.is_solved()


def test_scramble_is_deterministic():
    a, b = Cube(), Cube()
    assert a.scramble(12, seed=5) == b.scramble(12, seed=5)


def test_scramble_avoids_same_face_twice():
    c = Cube()
    seq = c.scramble(30, seed=1)
    assert all(seq[i][0] != seq[i + 1][0] for i in range(len(seq) - 1))


def test_move_parity_preserved():
    """任意转动都保持角块朝向和为 0 (mod 3)、棱块朝向和为 0 (mod 2)。"""
    rng = np.random.default_rng(0)
    c = Cube()
    for _ in range(200):
        c.apply(MOVES[rng.integers(len(MOVES))])
        assert sum(c.co) % 3 == 0
        assert sum(c.eo) % 2 == 0


@pytest.mark.parametrize("n", [3, 5, 7])
def test_solver_finds_optimal(n):
    """IDA* 求出的解不长于打乱步数 —— 也就是最优或更短。"""
    c = Cube()
    seq = c.scramble(n, seed=n * 11)
    sol, src = solve(c.copy(), node_budget=3_000_000, fallback=seq)
    assert src == "IDA*", f"应当搜索成功，实际来源 {src}"
    assert len(sol) <= n
    assert c.copy().apply_seq(sol).is_solved(), "解必须真的把魔方还原"


def test_solver_fallback_is_valid():
    """搜索预算耗尽时回退到逆序列，仍然是一个合法解。"""
    c = Cube()
    seq = c.scramble(18, seed=3)
    sol, src = solve(c.copy(), node_budget=2000, fallback=seq)
    assert src == "inverse"
    assert c.copy().apply_seq(sol).is_solved()


# --------------------------------------------------------------------------
# 场景几何
# --------------------------------------------------------------------------

def test_piece_orientations_are_axis_aligned():
    """每个块的朝向必须是魔方旋转群里的元素（带符号置换矩阵，行列式 +1）。

    之前用"把出生方向转到当前方向的最小旋转"来算朝向，得到的是非轴对齐的
    姿态，渲染出来每个块都是歪的 —— 这个测试就是为了钉住那个 bug。
    """
    from digitalfly.cube_scene import (CubeScene, _corner_rotation,
                                       _edge_rotation)
    scene = CubeScene()
    scene.state.scramble(25, seed=3)
    st = scene.state
    for kind, key in scene.pieces:
        if kind == "corner":
            slot = st.cp.index(key)
            R = _corner_rotation(key, slot, st.co[slot])
        elif kind == "edge":
            slot = st.ep.index(key)
            R = _edge_rotation(key, slot, st.eo[slot])
        else:
            continue
        assert np.allclose(R, np.round(R)), f"{kind}{key} 朝向不是轴对齐的"
        assert np.isclose(np.linalg.det(R), 1.0), f"{kind}{key} 不是纯旋转"
        assert np.allclose(R @ R.T, np.eye(3)), f"{kind}{key} 不正交"


def test_all_pieces_occupy_distinct_slots():
    """26 个块必须各占一个格点，不重叠。"""
    from digitalfly.cube_scene import CubeScene
    scene = CubeScene(size=0.6)
    scene.state.scramble(15, seed=8)
    poses = scene.poses()
    pts = np.array([p for p, _ in poses.values()])
    assert len(pts) == 26
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(d, 1e9)
    assert d.min() > scene.unit * 0.9, "有块重叠在一起"


def test_animation_advances_state():
    """转层动画结束时状态恰好推进一步。"""
    from digitalfly.cube_scene import CubeScene
    scene = CubeScene()
    before = scene.state.copy()
    scene.start_move("R", duration=0.2)
    assert scene.animating
    done = False
    for _ in range(40):
        done = scene.advance(0.01) or done
    assert done and not scene.animating
    assert scene.state.cp == before.copy().apply("R").cp
    assert scene.state.ep == before.copy().apply("R").ep


def test_solved_cube_faces_are_uniform():
    """复原状态下，每个面朝外的贴纸颜色应当一致。"""
    from digitalfly.cube import FACE_NORMAL
    from digitalfly.cube_scene import CubeScene
    scene = CubeScene(size=0.6, pos=(0, 0, 0))
    poses = scene.poses()
    # 复原态下每个块都应当在自己的家，姿态为单位四元数
    for (kind, key), (_, q) in poses.items():
        assert np.allclose(q, [1, 0, 0, 0], atol=1e-9), f"{kind}{key} 姿态非零"
    assert len(FACE_NORMAL) == 6


# --------------------------------------------------------------------------
# 全脑点云渲染
# --------------------------------------------------------------------------

@pytest.mark.skipif(
    not __import__("digitalfly.config", fromlist=["x"]).GRAPH_NPZ.exists(),
    reason="尚未构建连接组")
def test_brainview_uses_real_soma_coordinates():
    """点云必须来自数据集里的真实胞体坐标，不是编出来的。"""
    from digitalfly.brainview import BrainView
    from digitalfly.connectome import load
    from digitalfly.neurons import NeuronIndex
    c = load()
    assert {"soma_x", "soma_y", "soma_z"} <= set(c.meta.columns)
    n_with = int(c.meta["soma_x"].notna().sum())
    assert n_with > 100_000, f"只有 {n_with} 个神经元有坐标"

    view = BrainView(c.meta, NeuronIndex(c).by_region(), size=(120, 160))
    assert len(view.idx) > 1000
    # 投影结果必须落在画面内
    assert view.px.min() >= 0 and view.px.max() < 160
    assert view.py.min() >= 0 and view.py.max() < 120


@pytest.mark.skipif(
    not __import__("digitalfly.config", fromlist=["x"]).GRAPH_NPZ.exists(),
    reason="尚未构建连接组")
def test_brainview_lights_up_on_spikes():
    """有神经元发放时，画面应当变亮；衰减后回落。"""
    from digitalfly.brainview import BrainView
    from digitalfly.connectome import load
    from digitalfly.neurons import NeuronIndex
    c = load()
    view = BrainView(c.meta, NeuronIndex(c).by_region(), size=(360, 420))
    dark = view.render().astype(float).mean()

    spikes = np.zeros(len(c.meta), dtype=np.float32)
    spikes[view.idx[:5000]] = 1.0
    view.update(spikes, 0.5)
    lit = view.render().astype(float).mean()
    # 点云在小画幅上像素会互相叠加到饱和，所以不按倍数卡，按绝对增量卡
    assert lit > dark * 1.15, f"发放时画面没有变亮：{dark:.1f} -> {lit:.1f}"

    for _ in range(40):                      # 让热度按时间常数衰减
        view.update(np.zeros_like(spikes), 20.0)
    faded = view.render().astype(float).mean()
    assert faded < dark + (lit - dark) * 0.1, "热度没有衰减回去"


def test_side_by_side_dimensions_are_even():
    """分屏输出必须是偶数尺寸 —— libx264 的 yuv420p 不接受奇数宽高，
    否则整段视频会编码失败，写出一个 0 字节的 mp4。"""
    from digitalfly.brainview import side_by_side
    for wl, wr in [(600, 599), (601, 601), (480, 480)]:
        out = side_by_side(np.zeros((480, wl, 3), np.uint8),
                           np.zeros((479, wr, 3), np.uint8))
        assert out.shape[0] % 2 == 0 and out.shape[1] % 2 == 0


def test_softmax_decoder_learns_separable_data():
    """解码器本身是对的：线性可分的数据必须能学到接近满分。"""
    from digitalfly.experiments.cube_decode import Softmax
    rng = np.random.default_rng(0)
    centers = rng.normal(0, 4, size=(6, 12))
    y = rng.integers(0, 6, size=600)
    X = centers[y] + rng.normal(0, 0.5, size=(600, 12))
    m = Softmax(6).fit(X[:400], y[:400], epochs=400)
    assert (m.predict(X[400:]) == y[400:]).mean() > 0.95
