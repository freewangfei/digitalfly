"""蘑菇体可塑性与嗅觉联想学习的测试。"""
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
def mb(c):
    from digitalfly.brain import Brain
    from digitalfly.calibrate import calibrated_params
    from digitalfly.plasticity import MushroomBody
    brain = Brain(c.W.copy(), calibrated_params(), backend="scipy",
                  verbose=False)
    return MushroomBody(c, brain, verbose=False)


def test_learning_circuit_present(c):
    """蘑菇体这条学习回路必须完整 —— 缺任何一环都学不了。"""
    from digitalfly.neurons import NeuronIndex
    idx = NeuronIndex(c)
    assert len(idx.where(**{"class": "Kenyon_Cell"})) > 3000
    assert len(idx.where(**{"class": "MBON"})) > 50
    assert len(idx.where(**{"class": "DAN"})) > 100
    # 嗅觉进蘑菇体的通路（触角叶投射神经元）
    assert len(idx.where(**{"class": "ALPN"})) > 500


def test_plastic_synapses_are_kc_to_mbon(mb, c):
    """可塑的必须且只能是 KC→MBON 那批突触。"""
    assert len(mb.slots) > 10_000
    kc = set(mb.kc.tolist())
    mbon = set(mb.mbon.tolist())
    assert set(mb.syn_pre.tolist()) <= kc, "突触前混进了非 Kenyon 细胞"
    assert set(mb.syn_post.tolist()) <= mbon, "突触后混进了非 MBON"
    # 抽查：矩阵里对应位置的确是这些神经元对
    W = mb.brain.W
    k = min(200, len(mb.slots))
    for i in np.random.default_rng(0).choice(len(mb.slots), k, replace=False):
        assert W.indices[mb.slots[i]] == mb.syn_pre[i]


def test_teach_only_depresses(mb):
    """学习只能压低权重，不能抬高，也不能变号。"""
    mb.restore()
    w_before = mb.brain.W.data[mb.slots].copy()
    kc_act = np.zeros(len(mb.kc), dtype=np.float32)
    kc_act[:200] = 1.0
    da = np.ones(len(mb.mbon))
    mb.teach(da, eta=0.5, kc_source=kc_act)
    w_after = mb.brain.W.data[mb.slots]

    assert np.all(np.abs(w_after) <= np.abs(w_before) + 1e-6), "出现了增强"
    assert np.all(np.sign(w_after) * np.sign(w_before) >= 0), "权重变号了"
    assert np.abs(w_after).sum() < np.abs(w_before).sum(), "什么都没压低"
    mb.restore()


def test_teach_is_specific_to_active_kc(mb):
    """只有活动的 Kenyon 细胞的突触被改，其余不动。"""
    mb.restore()
    active = np.zeros(len(mb.kc), dtype=np.float32)
    active[:100] = 1.0
    w0 = mb.brain.W.data[mb.slots].copy()
    mb.teach(np.ones(len(mb.mbon)), eta=0.5, kc_source=active)
    changed = np.abs(mb.brain.W.data[mb.slots] - w0) > 1e-9
    pre_changed = set(mb.pre_local[changed].tolist())
    assert pre_changed and pre_changed <= set(range(100)), \
        "改到了不活动的 Kenyon 细胞"
    mb.restore()


def test_teach_is_specific_to_dopamine_compartment(mb):
    """只有收到多巴胺的区室的突触被改。"""
    mb.restore()
    da = np.zeros(len(mb.mbon))
    da[:5] = 1.0
    kc_act = np.ones(len(mb.kc), dtype=np.float32)
    w0 = mb.brain.W.data[mb.slots].copy()
    mb.teach(da, eta=0.5, kc_source=kc_act)
    changed = np.abs(mb.brain.W.data[mb.slots] - w0) > 1e-9
    post_changed = set(mb.post_local[changed].tolist())
    assert post_changed and post_changed <= set(range(5)), \
        "改到了没收到多巴胺的区室"
    mb.restore()


def test_restore_and_recover(mb):
    mb.restore()
    assert mb.strength() == pytest.approx(1.0, abs=1e-6)
    kc_act = np.ones(len(mb.kc), dtype=np.float32)
    mb.teach(np.ones(len(mb.mbon)), eta=0.4, kc_source=kc_act)
    weakened = mb.strength()
    assert weakened < 0.95
    for _ in range(50):
        mb.recover(0.1)
    assert mb.strength() > weakened, "恢复机制没起作用"
    mb.restore()
    assert mb.strength() == pytest.approx(1.0, abs=1e-6)


def test_sparse_code_is_sparse_and_normalised(mb):
    """稀疏编码要真的稀疏，并且按基线归一后不同刺激能分开。"""
    rng = np.random.default_rng(0)
    n = mb.brain.n
    base = rng.random(len(mb.kc)) * 100          # 固有响应差异很大
    stim = []
    for _ in range(4):
        full = np.zeros(n, dtype=np.float32)
        full[mb.kc] = base + rng.random(len(mb.kc)) * 10
        stim.append(full)

    # 不归一：选出来的几乎都是固有最强的那批，不同刺激高度重叠
    mb.kc_mu = None
    raw = [mb.sparse_code(s) > 0 for s in stim]
    ov_raw = np.mean([(raw[i] & raw[j]).sum() / max(raw[i].sum(), 1)
                      for i in range(4) for j in range(4) if i < j])

    mb.set_baseline([s[mb.kc] for s in stim])
    nrm = [mb.sparse_code(s) > 0 for s in stim]
    ov_nrm = np.mean([(nrm[i] & nrm[j]).sum() / max(nrm[i].sum(), 1)
                      for i in range(4) for j in range(4) if i < j])

    frac = float(nrm[0].sum()) / len(mb.kc)
    assert 0.02 < frac < 0.10, f"稀疏度 {frac:.3f} 不在 5% 附近"
    assert ov_nrm < ov_raw * 0.5, \
        f"按基线归一没有降低重叠：{ov_raw:.2f} -> {ov_nrm:.2f}"
    mb.kc_mu = None


def test_conditioning_is_odour_specific(c):
    """完整的条件化实验：配对的气味被显著压低，未配对的基本不变。"""
    from digitalfly.experiments import conditioning
    r = conditioning.run(c, trials=8, verbose=False)
    assert r["paired_change_pct"] < -10, \
        f"配对气味只变了 {r['paired_change_pct']:.1f}%"
    assert abs(r["paired_change_pct"]) > 2 * abs(r["control_change_pct"]), \
        (f"学习没有气味特异性：配对 {r['paired_change_pct']:.1f}% "
         f"vs 未配对 {r['control_change_pct']:.1f}%")
    assert 0.5 < r["synapse_strength"] < 1.0
    assert r["specific"]
