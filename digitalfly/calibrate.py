"""标定突触强度，让全脑网络工作在生理合理的区间。

**为什么需要这一步。**

Shiu et al. (*Nature* 2024) 在 FlyWire（成年雌性脑，约 13 万神经元、5 千万突触）
上用的单突触 EPSP 是 0.275 mV。直接把这个数搬到 MaleCNS v1.0 上会出问题 ——
本数据集含腹神经索，18.5 万神经元、1.25 亿突触，平均每个神经元收到约 677 个突触，
是 FlyWire 的 1.8 倍。在 0.275 mV 下，只要刺激 6 个神经元，整个网络就会进入
约 48 Hz 的自持放电，撤除刺激后也停不下来 —— 也就是癫痫样发作，不是果蝇的大脑。

这不是哪一方做错了：0.275 mV 本来就是在特定数据集上标定出来的经验值，
换数据集必须重新标。

**标定准则：不自持。**

真实大脑的一个硬性事实是，刺激撤除后活动会平息。所以这里用二分法找出
**最大的、撤除刺激后活动仍能衰减到静默的突触强度**。这个点是网络的临界点：
再强一点就失控，再弱一点信号传不出去。取它的 0.95 倍作为工作点，留一点余量。

准则本身不涉及任何行为学结果，不会把想要的结论"标定"进去 —— 糖味能不能激活
伸喙运动神经元，是标定完成之后才去检验的独立问题。
"""
from __future__ import annotations

import json

import numpy as np

from . import config
from .brain import Brain, LIFParams, Stimulus
from .connectome import Connectome

CALIB_JSON = config.BUILD_DIR / "calibration.json"

# 判定"已平息"的阈值：撤除刺激后全网平均发放率低于此值 (Hz)
QUIET_HZ = 0.1
SAFETY = 0.95


def _probe(c: Connectome, epsp: float, probe: np.ndarray,
           amplitude: float, backend: str | None,
           stim_ms: float = 200.0, settle_ms: float = 250.0) -> tuple:
    """给一段刺激再撤除，返回 (刺激期发放率, 撤除后发放率)，单位 Hz。"""
    p = LIFParams(epsp_mv=epsp)
    b = Brain(c.W, p, backend=backend, verbose=False)
    t_on, t_off = 50.0, 50.0 + stim_ms
    stim = Stimulus(b.n).add(probe, amplitude, t_on, t_off)
    rec = b.run(t_off + settle_ms, stimulus=stim)

    dt = p.dt
    to_hz = lambda a, z: float(
        rec.total[int(a / dt):int(z / dt)].mean() / b.n / (dt / 1000))
    # 撤除后留 50ms 余波不计，只看真正衰减之后的水平
    return to_hz(t_on, t_off), to_hz(t_off + 50, t_off + settle_ms)


def calibrate(c: Connectome,
              probe: np.ndarray | None = None,
              amplitude: float = 3.0,
              lo: float = 0.001, hi: float = 0.5,
              iters: int = 14,
              backend: str | None = None,
              verbose: bool = True) -> dict:
    """二分搜索临界突触强度。"""
    from .neurons import NeuronIndex

    if probe is None:
        # 用嗅觉感受神经元当探针：数量适中、位于网络上游，
        # 且与后面要检验的味觉->伸喙通路无关，避免把结论标进去。
        probe = NeuronIndex(c).olfactory()
    log = print if verbose else (lambda *a, **k: None)
    log(f"  标定探针: {len(probe):,} 个嗅觉神经元，刺激强度 {amplitude} mV/步")
    log(f"  准则: 撤除刺激后全网发放率必须衰减到 < {QUIET_HZ} Hz")

    # 先确认区间两端确实跨越了临界点
    on_hi, off_hi = _probe(c, hi, probe, amplitude, backend)
    if off_hi < QUIET_HZ:
        log(f"  上界 {hi} mV 也不自持，直接采用")
        return _result(hi * SAFETY, hi, on_hi, off_hi, len(probe), amplitude)

    for i in range(iters):
        mid = (lo + hi) / 2
        on, off = _probe(c, mid, probe, amplitude, backend)
        sustains = off >= QUIET_HZ
        log(f"    [{i + 1:2d}/{iters}] epsp={mid:7.4f} mV  刺激中 {on:7.2f} Hz"
            f"  撤除后 {off:7.2f} Hz  {'自持(过强)' if sustains else '平息(可用)'}")
        if sustains:
            hi = mid
        else:
            lo = mid

    critical = lo
    on, off = _probe(c, critical, probe, amplitude, backend)
    chosen = critical * SAFETY
    log(f"  临界强度 {critical:.4f} mV -> 工作点 {chosen:.4f} mV "
        f"(留 {100 * (1 - SAFETY):.0f}% 余量)")
    return _result(chosen, critical, on, off, len(probe), amplitude)


def _result(chosen, critical, on, off, n_probe, amplitude) -> dict:
    return {
        "epsp_mv": float(chosen),
        "critical_epsp_mv": float(critical),
        "on_rate_hz": float(on),
        "off_rate_hz": float(off),
        "quiet_threshold_hz": QUIET_HZ,
        "safety_factor": SAFETY,
        "n_probe_neurons": int(n_probe),
        "probe_amplitude_mv": float(amplitude),
        "reference_epsp_mv": 0.275,
        "reference": "Shiu et al., Nature 634 (2024), FlyWire",
    }


def save(result: dict) -> None:
    config.ensure_dirs()
    CALIB_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2))


def load() -> dict | None:
    if CALIB_JSON.exists():
        return json.loads(CALIB_JSON.read_text())
    return None


def calibrated_params(**overrides) -> LIFParams:
    """读取标定结果构造 LIFParams；没标定过就用文献默认值并给出提示。"""
    r = load()
    if r is None:
        print("[calibrate] 尚未标定，使用 Shiu et al. 的 0.275 mV。"
              "在本数据集上这会让网络自持放电，建议先运行: "
              "python cli.py calibrate")
        return LIFParams(**overrides)
    return LIFParams(epsp_mv=r["epsp_mv"], **overrides)
