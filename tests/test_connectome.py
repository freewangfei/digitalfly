"""连接组构建与神经元分组的集成测试。

需要先跑 `python cli.py download && python cli.py build`；
数据不在就整体跳过，不会让测试套失败。
"""
from __future__ import annotations

import numpy as np
import pytest

from digitalfly import config

pytestmark = pytest.mark.skipif(
    not config.GRAPH_NPZ.exists(),
    reason="尚未构建连接组，先运行 python cli.py build")


@pytest.fixture(scope="module")
def c():
    from digitalfly.connectome import load
    return load()


@pytest.fixture(scope="module")
def idx(c):
    from digitalfly.neurons import NeuronIndex
    return NeuronIndex(c)


def test_scale_matches_publication(c):
    """网络规模应当和论文报告的量级一致。

    本地会比 neuPrint 的 :Neuron 略多：neuPrint 对孤立片段额外设了突触数阈值，
    本地只要求"有标注 + 在连接表中出现 + 非胶质 / 非 Unimportant"。
    允许 ±10% 的口径差。
    """
    n = c.stats["n_neurons"]
    assert 0.90 * config.EXPECTED_NEURONS <= n <= 1.10 * config.EXPECTED_NEURONS
    assert c.stats["n_edges"] > 1_000_000
    # 论文报告约 1.25 亿突触，本地构建应当落在同一量级且接近
    assert 1.1e8 < c.stats["n_synapses"] < 1.4e8


def test_matrix_shape_and_dtype(c):
    n = c.stats["n_neurons"]
    assert c.W.shape == (n, n)
    assert c.W.dtype == np.float32
    assert len(c.meta) == n


def test_signs_present(c):
    """兴奋与抑制都必须存在，且抑制性不应是少数派里的少数派。"""
    exc, inh = c.stats["excitatory_edges"], c.stats["inhibitory_edges"]
    assert exc > 0 and inh > 0
    assert 0.05 < inh / (exc + inh) < 0.6


def test_sign_follows_presynaptic_nt(c):
    """同一个突触前神经元发出的所有边，符号必须一致 —— 戴尔原则。"""
    rng = np.random.default_rng(0)
    Wc = c.W.tocsc()
    nz = np.flatnonzero(np.diff(Wc.indptr) > 1)
    for j in rng.choice(nz, size=min(200, len(nz)), replace=False):
        col = Wc.data[Wc.indptr[j]:Wc.indptr[j + 1]]
        signs = np.unique(np.sign(col[col != 0]))
        assert len(signs) <= 1, f"神经元 {j} 同时发出兴奋和抑制突触"


def test_no_self_loops_dominate(c):
    """自连接应当极少（电镜重建里偶有，但不该是主流）。"""
    diag = c.W.diagonal()
    assert np.count_nonzero(diag) < 0.02 * c.W.shape[0]


def test_interfaces_exist(idx):
    """数字果蝇的输入输出接口必须都能找到。"""
    assert len(idx.descending()) > 500,      "下行神经元"
    assert len(idx.motor()) > 300,           "运动神经元"
    assert len(idx.sensory()) > 5_000,       "感觉神经元"
    assert len(idx.photoreceptors()) > 1_000, "光感受器"
    assert len(idx.proprioceptive()) > 500,  "本体感觉神经元"
    assert len(idx.gustatory()) > 500,       "味觉神经元"
    assert len(idx.proboscis_motor()) >= 1,  "MN9 伸喙运动神经元"
    assert len(idx.sugar_pathway()) >= 1,    "糖味通路"
    assert len(idx.bitter_pathway()) >= 1,   "苦味通路"


def test_regions_partition(idx, c):
    """脑区分组必须是对全体神经元的一个划分：不重不漏。"""
    groups = idx.by_region()
    allv = np.concatenate(list(groups.values()))
    assert len(allv) == len(np.unique(allv)), "脑区分组有重叠"
    assert len(allv) == c.stats["n_neurons"], "脑区分组没覆盖全部神经元"


def test_index_of_roundtrip(c, idx):
    """bodyId <-> 矩阵下标的往返映射一致。"""
    sample = c.meta["bodyId"].to_numpy()[:100]
    back = c.index_of(sample)
    assert np.array_equal(c.meta.iloc[back]["bodyId"].to_numpy(), sample)


def test_sugar_pathway_reaches_mn9(c, idx):
    """糖味通路到伸喙运动神经元之间必须在图上真的连通（3 跳以内）。

    这是行为能不能发生的结构前提 —— 如果图上根本不通，
    仿真出来的任何响应都只能是噪声。
    """
    src = idx.sugar_pathway()
    dst = set(idx.proboscis_motor().tolist())
    A = (c.W != 0)
    reached = set(src.tolist())
    frontier = src
    for _ in range(3):
        nxt = np.unique(A[:, frontier].nonzero()[0])   # 下游 = 行
        if dst & set(nxt.tolist()):
            return
        frontier = np.setdiff1d(nxt, list(reached))
        reached |= set(frontier.tolist())
        if len(frontier) == 0:
            break
    pytest.fail("糖味通路在 3 跳内到不了 MN9")
