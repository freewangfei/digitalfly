"""LIF 引擎的正确性测试 —— 不依赖连接组数据，用小网络对拍解析解。"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from digitalfly.brain import Brain, LIFParams, Stimulus


def _empty(n: int) -> sp.csr_matrix:
    return sp.csr_matrix((n, n), dtype=np.float32)


def test_no_input_no_spikes():
    """没有输入时，膜电位停在静息值，不应该有自发发放。"""
    b = Brain(_empty(10), verbose=False)
    rec = b.run(100.0)
    assert rec.counts.sum() == 0
    assert np.allclose(b.v, b.p.v_rest)


def test_threshold_crossing():
    """注入恰好跨过阈值的电流，应当在第一步就发放。"""
    p = LIFParams()
    b = Brain(_empty(4), p, verbose=False)
    need = p.v_th - p.v_rest            # 7 mV
    stim = Stimulus(4).add([0], need + 0.1)
    s = b.step(stim(0.0, 4))
    assert s[0] == 1.0
    assert s[1:].sum() == 0


def test_subthreshold_decay():
    """阈下输入撤掉后，膜电位按 exp(-dt/tau) 衰减回静息值。"""
    p = LIFParams()
    b = Brain(_empty(2), p, verbose=False)
    b.step(np.array([3.0, 0.0], dtype=np.float32))
    v1 = float(b.v[0])
    b.step(None)
    v2 = float(b.v[0])
    expected = p.v_rest + (v1 - p.v_rest) * p.decay
    assert v2 == pytest.approx(expected, abs=1e-4)


def test_refractory_period():
    """不应期内即使持续强刺激也不能再次发放。"""
    p = LIFParams()
    n_ref = p.refractory_steps
    b = Brain(_empty(1), p, verbose=False)
    stim = np.array([50.0], dtype=np.float32)
    fired = [float(b.step(stim)[0]) for _ in range(n_ref + 2)]
    assert fired[0] == 1.0
    assert sum(fired[1:n_ref]) == 0, "不应期内不应发放"
    assert fired[n_ref] == 1.0, "不应期结束后应恢复发放"


def test_excitatory_propagation():
    """兴奋性突触应当把发放传到下游。"""
    p = LIFParams()
    n_syn = int(np.ceil((p.v_th - p.v_rest) / p.epsp_mv)) + 1
    W = sp.csr_matrix(([float(n_syn)], ([1], [0])), shape=(2, 2),
                      dtype=np.float32)
    b = Brain(W, p, verbose=False)
    b.step(np.array([10.0, 0.0], dtype=np.float32))   # 0 号发放
    assert b.spikes[0] == 1.0
    s = b.step(None)                                   # 传到 1 号
    assert s[1] == 1.0


def test_inhibitory_blocks():
    """抑制性突触应当阻止下游发放。"""
    p = LIFParams()
    n_syn = int(np.ceil((p.v_th - p.v_rest) / p.epsp_mv)) + 1
    # 0 -> 2 兴奋, 1 -> 2 强抑制
    W = sp.csr_matrix(
        ([float(n_syn), -float(4 * n_syn)], ([2, 2], [0, 1])),
        shape=(3, 3), dtype=np.float32)
    b = Brain(W, p, verbose=False)
    b.step(np.array([10.0, 10.0, 0.0], dtype=np.float32))
    s = b.step(None)
    assert s[2] == 0.0


def test_ablation_silences():
    """被切除的神经元不再发放。"""
    b = Brain(_empty(3), verbose=False)
    b.ablate(np.array([1]))
    s = b.step(np.array([10.0, 10.0, 10.0], dtype=np.float32))
    assert s[0] == 1.0 and s[2] == 1.0
    assert s[1] == 0.0


def test_stimulus_window():
    """刺激只在指定时间窗内生效。"""
    b = Brain(_empty(1), verbose=False)
    stim = Stimulus(1).add([0], 10.0, start=5.0, stop=10.0)
    assert stim(0.0, 1)[0] == 0.0
    assert stim(6.0, 1)[0] == 10.0
    assert stim(12.0, 1)[0] == 0.0


def test_record_rates():
    """发放率统计单位正确 (Hz)。"""
    p = LIFParams()
    b = Brain(_empty(2), p, verbose=False)
    stim = Stimulus(2).add([0], 50.0)
    rec = b.run(1000.0, stimulus=stim, record=np.array([0, 1]))
    # 相邻两次发放至少间隔 refractory_steps 步
    max_hz = 1000.0 / (p.refractory_steps * p.dt)
    hz = rec.rates_hz()[0]
    assert 0 < hz <= max_hz
    assert rec.rates_hz()[1] == 0
    assert rec.raster.shape == (2000, 2)


@pytest.mark.parametrize("backend", ["scipy", "torch-cpu", "torch-cuda"])
def test_backends_agree(backend):
    """不同后端在同一网络上应给出一致结果。"""
    if backend.startswith("torch"):
        torch = pytest.importorskip("torch")
        if backend == "torch-cuda" and not torch.cuda.is_available():
            pytest.skip("无可用 CUDA")
    rng = np.random.default_rng(0)
    n = 200
    W = sp.random(n, n, density=0.05, format="csr", dtype=np.float32,
                  random_state=0) * 60
    W.data *= rng.choice([1.0, -1.0], size=W.nnz).astype(np.float32)

    ext = np.zeros(n, dtype=np.float32)
    ext[:10] = 8.0
    counts = {}
    for be in ("scipy", backend):
        b = Brain(W, backend=be, verbose=False)
        tot = 0
        for _ in range(200):
            tot += int(b.step(ext).sum())
        counts[be] = tot
    assert counts["scipy"] == counts[backend]
