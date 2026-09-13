"""糖味 -> 伸喙 (PER)：连接组仿真的关键验证实验。

思路来自 Shiu et al., *Nature* 634 (2024) 在 FlyWire 上做的经典验证：
只给连接组和 LIF 动力学，不做任何训练或调参，刺激糖味通路，看伸喙运动神经元
MN9 是否被激活；再刺激苦味通路作阴性对照 —— 苦味在真实果蝇里抑制伸喙。

如果连接组 + 简单动力学真的承载了行为，这两组应当分离；如果只是随机网络，
两组不会有差别。这是检验"数字果蝇"是不是真的接上了生物学的第一道关。
"""
from __future__ import annotations

import numpy as np

from ..brain import Brain, Stimulus
from ..calibrate import calibrated_params
from ..connectome import Connectome
from ..neurons import NeuronIndex


def run(c: Connectome,
        duration_ms: float = 1000.0,
        stim_start: float = 200.0,
        stim_stop: float = 800.0,
        amplitude: float = 10.0,
        backend: str | None = None,
        verbose: bool = True) -> dict:
    """跑糖味组、苦味组、空白对照三组，返回 MN9 发放率对比。"""
    idx = NeuronIndex(c)

    groups = {
        "sugar": idx.sugar_pathway(),
        "bitter": idx.bitter_pathway(),
        "grn": idx.labellar_grns(),      # 一级味觉输入，作正向对照
        "none": np.array([], dtype=np.int64),
    }
    readout = idx.proboscis_motor()          # MN9
    if len(readout) == 0:
        raise RuntimeError("未找到 MN9，连接组可能构建有误")

    if verbose:
        print(f"  读数神经元 MN9: {len(readout)} 个 "
              f"(bodyId {list(c.meta.iloc[readout]['bodyId'])})")
        for name, g in groups.items():
            if len(g):
                types = c.meta.iloc[g]["type"].value_counts().to_dict()
                print(f"  刺激组 {name}: {len(g)} 个神经元 {types}")

    results = {}
    for name, stim_idx in groups.items():
        brain = Brain(c.W, calibrated_params(), backend=backend,
                      verbose=verbose and name == "sugar")
        stim = Stimulus(brain.n)
        if len(stim_idx):
            stim.add(stim_idx, amplitude, stim_start, stim_stop)

        rec = brain.run(duration_ms, stimulus=stim, record=readout,
                        progress=verbose)

        # 分别统计刺激前 / 刺激中的 MN9 发放率
        dt = brain.p.dt
        raster = rec.raster                                # (T, n_readout)
        pre_w = slice(0, int(stim_start / dt))
        on_w = slice(int(stim_start / dt), int(stim_stop / dt))
        to_hz = lambda w: float(
            raster[w].sum() / len(readout) / (
                (w.stop - w.start) * dt / 1000.0))

        results[name] = {
            "mn9_baseline_hz": to_hz(pre_w),
            "mn9_stim_hz": to_hz(on_w),
            "network_total_spikes": float(rec.total.sum()),
            "mean_network_rate_hz": float(
                rec.total.mean() / brain.n / (dt / 1000.0)),
            "raster": raster,
            "total_trace": rec.total,
            "rates_hz": rec.rates_hz(),
        }
        if verbose:
            r = results[name]
            print(f"  [{name:6s}] MN9 基线 {r['mn9_baseline_hz']:6.1f} Hz"
                  f"  -> 刺激期 {r['mn9_stim_hz']:6.1f} Hz"
                  f"   全网平均 {r['mean_network_rate_hz']:.2f} Hz")

    sugar = results["sugar"]["mn9_stim_hz"]
    bitter = results["bitter"]["mn9_stim_hz"]
    base = results["none"]["mn9_stim_hz"]
    results["verdict"] = {
        "sugar_drives_mn9": bool(sugar > max(base, 1.0) * 1.5),
        "bitter_does_not": bool(bitter <= sugar),
        "sugar_hz": sugar, "bitter_hz": bitter, "spontaneous_hz": base,
    }
    if verbose:
        v = results["verdict"]
        print(f"\n  结论: 糖味 {sugar:.1f} Hz vs 苦味 {bitter:.1f} Hz "
              f"vs 无刺激 {base:.1f} Hz")
        print(f"        糖味驱动伸喙 = {v['sugar_drives_mn9']}; "
              f"苦味不高于糖味 = {v['bitter_does_not']}")
    return results
