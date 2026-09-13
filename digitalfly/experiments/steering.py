"""能不能从连接组读出"气味在左还是在右"？—— 一个阴性结果。

果蝇趋化的行为学基础是双侧比较：两根触角闻到的浓度不同，就朝浓的一侧转。
如果连接组 + LIF 动力学足以支撑这个计算，那么给左右嗅觉神经元不同强度的输入，
下行神经元的左右活动差应当随之改变。

这个实验检验这一点，结论是**做不到**：

  * 用全部 1,314 个下行神经元按解剖学左右分群，读数的动态范围只有约 0.03，
    而且被"气味总强度"主导，跟左右分布无关。
  * 换成功能性筛选（两次探针试验，挑出差分响应最大的 40 个下行神经元），
    在探针条件下对比度很漂亮（35 Hz），但把气味强度稍微一改就完全失效 ——
    典型的过拟合，不是真实的侧别编码。

这个结果和文献是一致的：Shiu et al. (*Nature* 2024) 证明的是连接组能复现
**反射式**的感觉运动映射（味觉 -> 伸喙），而不是精细的导航计算。均一的 LIF
神经元加未经调节的突触权重，不足以支撑双侧比较这种需要精确增益匹配的运算。

所以 behaviors.forage 里的转向用的是显式趋化规则（并明确标注），
而味觉 -> 伸喙那一段是真的由连接组决定的。
"""
from __future__ import annotations

import numpy as np

from ..brain import Brain
from ..bridge import CommandBridge
from ..calibrate import calibrated_params
from ..connectome import Connectome


def run(c: Connectome, backend: str | None = None,
        verbose: bool = True) -> dict:
    br = CommandBridge(c, verbose=False)
    brain = Brain(c.W, calibrated_params(), backend=backend, verbose=verbose)

    def measure(odor_l: float, odor_r: float, groups, ms: float = 800.0):
        brain.reset()
        ext = np.zeros(brain.n, dtype=np.float32)
        br.sensory_current(brain.n, odor_left=odor_l, odor_right=odor_r,
                           out=ext)
        ext[br.proprio] += 0.8
        counts = np.zeros(brain.n, dtype=np.float32)
        for _ in range(int(ms / brain.p.dt)):
            counts += brain.step(ext)
        secs = ms / 1000.0
        return [float(counts[g].mean() / secs) if len(g) else 0.0
                for g in groups]

    # 三组气味条件，总强度相同，只有左右分布不同 —— 这样"总强度"不会混进来
    conditions = [("气味偏左", 1.0, 0.3),
                  ("左右对称", 0.65, 0.65),
                  ("气味偏右", 0.3, 1.0)]

    results = {"anatomical": [], "functional": []}
    if verbose:
        print("  读数一：全部下行神经元按解剖学左右分群")
    for name, ol, orr in conditions:
        l, r = measure(ol, orr, [br.dn_left, br.dn_right])
        norm = (l - r) / max(l + r, 1e-6)
        results["anatomical"].append((name, l, r, norm))
        if verbose:
            print(f"    {name}: 左 {l:6.1f} 右 {r:6.1f} Hz  归一化差 {norm:+.3f}")

    if verbose:
        print("  读数二：功能性筛选出的转向下行神经元")
    br.identify_steering_dns(brain, cache=False, verbose=verbose)
    for name, ol, orr in conditions:
        l, r = measure(ol, orr, [br.dn_turn_left, br.dn_turn_right])
        norm = (l - r) / max(l + r, 1e-6)
        results["functional"].append((name, l, r, norm))
        if verbose:
            print(f"    {name}: 左 {l:6.1f} 右 {r:6.1f} Hz  归一化差 {norm:+.3f}")

    def spread(rows):
        vals = [x[3] for x in rows]
        return max(vals) - min(vals), vals[0] - vals[-1]

    a_spread, a_signed = spread(results["anatomical"])
    f_spread, f_signed = spread(results["functional"])
    results["verdict"] = {
        "anatomical_range": a_spread,
        "anatomical_left_minus_right": a_signed,
        "functional_range": f_spread,
        "functional_left_minus_right": f_signed,
        # 要算"能编码侧别"，偏左和偏右之间必须有明确且单调的差
        "encodes_laterality": bool(abs(f_signed) > 0.05 and
                                   abs(a_signed) > 0.05),
    }
    if verbose:
        v = results["verdict"]
        print(f"\n  偏左与偏右之间的读数差：解剖学分群 {a_signed:+.3f}，"
              f"功能性分群 {f_signed:+.3f}")
        print(f"  结论：连接组能否编码气味侧别 = {v['encodes_laterality']}")
        if not v["encodes_laterality"]:
            print("  在总强度相同的条件下，两种读数都分辨不出气味在左还是在右。")
    return results
