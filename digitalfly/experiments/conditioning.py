"""嗅觉联想学习 —— 果蝇用自己的突触学，没有外挂分类器。

这是本项目里唯一一处**学习真正发生在连接组内部**的地方：改的是 KC→MBON 那
61,210 条突触的权重，用的是果蝇自己的学习规则，没有反向传播，没有外部模型。

范式就是教科书里的果蝇嗅觉条件化：

    给气味 A，同时激活多巴胺能神经元（相当于电击）
      -> 那一刻活动的 Kenyon 细胞到"惩罚区室"MBON 的突触被压低
      -> 之后再闻到 A，该区室 MBON 的输出下降，行为倾向随之改变
    气味 B 没有配对过，应当基本不受影响 —— 这就是学习的**特异性**

规则出自 Hige et al. (*Neuron* 2015) 与 Aso & Rubin (*eLife* 2016)。

**两处模型补充，必须标出来：**

1. **稀疏编码是显式加的。** 真实果蝇里 APL 那个巨大的 GABA 能神经元把 KC
   压到约 5% 活跃，这是学习具有气味特异性的前提。本重建里 KC 收到的兴奋是
   抑制的 7.3 倍（184 万 vs 25 万），APL 只有 2 个神经元，压不住 64 万条
   KC→KC 递归兴奋 —— 实测任何输入都把全部 4,064 个 KC 点亮到 145 Hz。
   所以这里显式取前 5%，替 APL 做它在真实果蝇里做的事。

2. **稀疏化要按每个 KC 自己的基线来。** 直接取"响应最强的前 5%"选出来的是
   固有输入最强的那批细胞，跟当前是什么气味无关 —— 实测两种气味的 KC 编码
   重叠 73%，比输入本身的重叠（25%）还高，等于网络在增加相关性。按每个 KC
   的均值和标准差归一之后重叠降到 2%。这对应 KC 的适应与增益控制。

学习规则本身、可塑突触的位置和数量、教学信号的通路，全部来自连接组。
"""
from __future__ import annotations

import numpy as np

from ..brain import Brain
from ..bridge import CommandBridge
from ..calibrate import calibrated_params
from ..connectome import Connectome
from ..neurons import NeuronIndex
from ..plasticity import MushroomBody


class OlfactoryConditioning:
    """一次条件化实验的完整装置。"""

    def __init__(self, c: Connectome, n_odours: int = 2,
                 backend: str | None = None, seed: int = 0,
                 verbose: bool = True):
        self.c = c
        self.idx = NeuronIndex(c)
        # 用 W 的副本 —— 学习会改权重，别污染磁盘上的连接组
        self.brain = Brain(c.W.copy(), calibrated_params(), backend=backend,
                           verbose=verbose)
        self.bridge = CommandBridge(c, dt_ms=self.brain.p.dt, verbose=False)
        self.mb = MushroomBody(c, self.brain, verbose=verbose)

        # 一种气味 = 触角叶投射神经元的一个子集。
        # 真实果蝇里一种气味激活一组特定的嗅小球，这里用固定的随机子集代表。
        rng = np.random.default_rng(seed)
        alpn = self.idx.where(**{"class": "ALPN"})
        self.names = [chr(ord("A") + i) for i in range(n_odours)]
        self.odours = {n: rng.choice(alpn, size=len(alpn) // 4, replace=False)
                       for n in self.names}
        self.alpn = alpn

        # 惩罚区室 / 对照区室：把 MBON 对半分。真实果蝇里区室由解剖学划定，
        # 这里做的是最简划分，够用来看学习的特异性。
        h = len(self.mb.mbon) // 2
        self.punish = np.zeros(len(self.mb.mbon))
        self.punish[:h] = 1.0
        self.half = h

        self.codes: dict = {}
        self._calibrate_codes()
        if verbose:
            ov = self.code_overlap()
            print(f"[条件化] {n_odours} 种气味，每种 "
                  f"{len(self.odours[self.names[0]])} 个 ALPN")
            print(f"[条件化] KC 稀疏编码平均重叠 {100 * ov:.0f}%")

    # -- 呈现 ---------------------------------------------------------------
    def present(self, name: str, ms: float = 180.0, rate: float = 25.0,
                gate: bool = False) -> tuple:
        """给一种气味，返回 (脉冲计数, 秒数)。

        gate=True 时只让该气味的稀疏 KC 编码参与（替代 APL 的全局抑制），
        MBON 才会落在分级区间 —— 不门控的话 MBON 顶在 300 Hz 饱和，
        压低突触也推不动输出。
        """
        b, mb = self.brain, self.mb
        b.reset()
        if gate and name in self.codes:
            off = mb.kc[self.codes[name] == 0]
            self._set_gate(off)
        else:
            self._set_gate(np.array([], dtype=np.int64))

        ext = np.zeros(b.n, dtype=np.float32)
        counts = np.zeros(b.n, dtype=np.float32)
        n = int(ms / b.p.dt)
        for i in range(n):
            ext.fill(0.0)
            self.bridge._drive(ext, self.odours[name], rate)
            s = b.step(ext)
            if i >= n // 3:
                counts += s
        return counts, (n - n // 3) * b.p.dt / 1000.0

    def _set_gate(self, off: np.ndarray) -> None:
        b = self.brain
        b._ablated[:] = False
        if len(off):
            b._ablated[off] = True
        if b.backend.startswith("torch"):
            b.ablated_t[:] = False
            if len(off):
                b.ablated_t[b.torch.from_numpy(
                    off.astype(np.int64)).to(b.dev)] = True

    def _calibrate_codes(self) -> None:
        """先测每种气味的 KC 响应，定基线，再定各自的稀疏编码。"""
        raw = {n: self.present(n)[0][self.mb.kc] for n in self.names}
        self.mb.set_baseline([raw[n] for n in self.names])
        for n in self.names:
            full = np.zeros(self.brain.n, dtype=np.float32)
            full[self.mb.kc] = raw[n]
            self.codes[n] = self.mb.sparse_code(full)

    def code_overlap(self) -> float:
        ks = [self.codes[n] > 0 for n in self.names]
        vals = [float((ks[i] & ks[j]).sum()) / max(float(ks[i].sum()), 1)
                for i in range(len(ks)) for j in range(len(ks)) if i < j]
        return float(np.mean(vals)) if vals else 0.0

    # -- 读数 ---------------------------------------------------------------
    def readout(self, name: str, repeats: int = 3) -> tuple:
        """返回 (惩罚区室 MBON 发放率, 对照区室发放率)。

        取多次呈现的平均：输入是速率编码的随机门控，单次读数噪声不小，
        会让"特异性"这个结论在不同随机种子下跳来跳去。
        """
        pun, ctl = [], []
        for _ in range(repeats):
            counts, secs = self.present(name, gate=True)
            r = self.mb.mbon_rate(counts, secs)
            pun.append(float(r[:self.half].mean()))
            ctl.append(float(r[self.half:].mean()))
        return float(np.mean(pun)), float(np.mean(ctl))

    # -- 训练 ---------------------------------------------------------------
    def pair(self, name: str, eta: float = 0.5) -> dict:
        """把一种气味和多巴胺配对一次（一个训练回合）。"""
        self.present(name, gate=True)
        return self.mb.teach(self.punish, eta=eta,
                             kc_source=self.codes[name])

    def forget(self, rate: float = 0.05) -> None:
        self.mb.recover(rate)

    def reset(self) -> None:
        self.mb.restore()


def run(c: Connectome, trials: int = 10, backend: str | None = None,
        seed: int = 0, verbose: bool = True) -> dict:
    """完整实验：训练前后测两种气味，看学习是否具有特异性。"""
    exp = OlfactoryConditioning(c, n_odours=2, backend=backend, seed=seed,
                                verbose=verbose)
    before = {n: exp.readout(n) for n in exp.names}
    if verbose:
        print("\n  训练前")
        for n in exp.names:
            print("    气味 %s: 惩罚区室 %6.1f Hz  对照区室 %6.1f Hz"
                  % (n, before[n][0], before[n][1]))
        print(f"\n  配对训练：气味 {exp.names[0]} + 多巴胺，{trials} 回合")

    for _ in range(trials):
        exp.pair(exp.names[0])

    after = {n: exp.readout(n) for n in exp.names}
    strength = exp.mb.strength()
    paired, control = exp.names[0], exp.names[1]

    def change(n):
        b0 = before[n][0]
        return 100 * (after[n][0] - b0) / max(b0, 1e-9)

    d_paired, d_control = change(paired), change(control)
    # 特异性 = 配对气味的降幅明显大于未配对的。两个都是负数，
    # 所以要比绝对值，不能直接比大小。
    specific = bool(d_paired < -10
                    and abs(d_paired) > 2 * abs(d_control))

    if verbose:
        print("\n  训练后")
        for n in exp.names:
            print("    气味 %s: 惩罚区室 %6.1f Hz  对照区室 %6.1f Hz"
                  % (n, after[n][0], after[n][1]))
        print(f"\n  可塑突触剩余强度 {100 * strength:.1f}%"
              f"（共 {len(exp.mb.slots):,} 条 KC→MBON）")
        print(f"  配对气味 {paired} 惩罚区室 {d_paired:+.1f}%")
        print(f"  未配对   {control} 惩罚区室 {d_control:+.1f}%")
        if abs(d_control) > 1e-6:
            print(f"  特异性倍数 {abs(d_paired) / abs(d_control):.1f}x")
        print(f"\n  学习是否具有气味特异性 = {specific}")
        if specific:
            print("  这是果蝇自己的突触变了 —— 没有外部分类器，没有反向传播。")

    return {
        "before": before, "after": after,
        "paired": paired, "control": control,
        "paired_change_pct": d_paired, "control_change_pct": d_control,
        "synapse_strength": strength,
        "n_plastic_synapses": int(len(exp.mb.slots)),
        "code_overlap": exp.code_overlap(),
        "specific": specific,
    }
