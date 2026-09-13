"""切除实验：把某条通路打掉，看下游还剩多少响应。

这是连接组仿真最有用的一类操作 —— 在真实果蝇身上要用遗传学工具花几个月做的
功能失活实验，在数字果蝇上是一行代码，而且可以穷举。
"""
from __future__ import annotations

import numpy as np

from ..brain import Brain, Stimulus
from ..calibrate import calibrated_params
from ..connectome import Connectome
from ..neurons import NeuronIndex


def run(c: Connectome,
        stim_group: str = "sugar_pathway",
        readout_group: str = "proboscis_motor",
        ablate_groups: list[str] | None = None,
        duration_ms: float = 800.0,
        amplitude: float = 3.0,
        backend: str | None = None,
        verbose: bool = True) -> dict:
    """先测完整网络的响应，再逐个切除指定群体重测。

    ablate_groups 里的名字是 NeuronIndex 上的方法名。
    """
    idx = NeuronIndex(c)
    stim_idx = getattr(idx, stim_group)()
    readout = getattr(idx, readout_group)()
    ablate_groups = ablate_groups or []

    def trial(ablate: np.ndarray | None, label: str) -> float:
        brain = Brain(c.W, calibrated_params(), backend=backend, verbose=False)
        if ablate is not None and len(ablate):
            brain.ablate(ablate)
        stim = Stimulus(brain.n).add(stim_idx, amplitude, 100.0, duration_ms)
        rec = brain.run(duration_ms, stimulus=stim, record=readout)
        hz = float(rec.counts[readout].sum() / len(readout) /
                   (duration_ms / 1000.0))
        if verbose:
            print(f"    {label:<32s} 读数 {hz:7.1f} Hz")
        return hz

    if verbose:
        print(f"  刺激 {stim_group} ({len(stim_idx)} 个) "
              f"-> 读数 {readout_group} ({len(readout)} 个)")
    intact = trial(None, "完整网络")

    results = {"intact_hz": intact, "ablations": {}}
    for name in ablate_groups:
        group = getattr(idx, name)()
        hz = trial(group, f"切除 {name} ({len(group)})")
        results["ablations"][name] = {
            "n": int(len(group)),
            "hz": hz,
            "remaining_fraction": (hz / intact) if intact > 0 else float("nan"),
        }
    return results
