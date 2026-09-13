"""果蝇的视觉系统能不能分辨数字和字母？

和 `cube_decode` 问的是同一类问题，但这一次输入走的是**果蝇真正用来看东西的
那条路**，所以先验完全不同：

  * 魔方那个实验里，局面是用一个人为约定的随机投影塞进网络的 —— 果蝇没有
    魔方感受器，那个编码本身不携带任何结构。结论是阴性。
  * 这里的图像是按**数据集自带的六角视柱坐标**一柱一柱铺到复眼上的，驱动的是
    L1（ON）和 L2（OFF）—— 光感受器在板层的两个主要靶标。视叶有 10.5 万个
    神经元，是这套连接组里最大的一块，而且它就是演化出来做这件事的。

流程：
  1. 生成刺激。每个字符渲染成灰度图，加随机的平移 / 缩放 / 旋转扰动，
     这样解码器学到的不能是"某几个像素亮"，必须是形状。
  2. 呈现并记录。图像 -> 视柱 -> L1/L2 目标发放率 -> 全脑网络跑一段，
     记下视觉投射神经元 (VPN) 按类型分群的发放率作为特征。
     VPN 是视叶通向中央脑的输出通道，346 种类型里包括 LC12、LC17、LC10
     这些已知的特征检测器。
  3. 解码并对照。多项逻辑回归 + 留出集，和四条基线比：
       随机猜 · 最频繁类别 · **打乱标签**（纯过拟合能到多少）
       · **像素基线**（直接拿图像本身解码，给出这套刺激的上限）

像素基线是这个实验里最重要的一条：它把"任务本身有多容易"和"网络保留了多少
信息"分开。网络准确率接近像素基线，说明视叶把形状信息基本传下来了；
接近打乱对照，说明信息丢光了。
"""
from __future__ import annotations

import time

import numpy as np

from ..brain import Brain
from ..bridge import CommandBridge
from ..calibrate import calibrated_params
from ..connectome import Connectome
from ..vision import DIGITS, LETTERS, Retina, render_char
from .cube_decode import Softmax


# 被直接驱动的输入层，读出时必须排除 —— 读它们等于把注入的图像原样读回来
DRIVEN_TYPES = ("L1", "L2", "L3", "L5")


def _readout_groups(c, idx, layer: str = "column",
                    min_size: int = 4) -> tuple:
    """构造解码特征的神经元分群。

    layer="column"  按视柱分群的髓质神经元（排除被驱动的 L1/L2）。
                    保留视网膜拓扑，是视觉信息最直接的下游表示。
    layer="vpn"     视觉投射神经元按类型分群。这是视叶通向中央脑的输出，
                    读出难度更高，能看出信息在视叶里损失了多少。
    """
    meta = c.meta
    types = meta["type"].astype(str).to_numpy()
    if layer == "vpn":
        sel = idx.where(superclass="visual_projection")
        groups, names = [], []
        for t in np.unique(types[sel]):
            g = sel[types[sel] == t]
            if len(g) >= min_size:
                groups.append(g)
                names.append(t)
        return groups, names

    ol = idx.where(superclass="ol_intrinsic")
    ol = ol[~np.isin(types[ol], DRIVEN_TYPES)]
    h1 = meta["hex1"].to_numpy()
    h2 = meta["hex2"].to_numpy()
    side = idx.side().to_numpy()
    key = np.array([f"{a}|{x}|{y}" for a, x, y in
                    zip(side[ol], h1[ol], h2[ol])])
    valid = ~np.array(["nan" in k or "None" in k for k in key])
    ol, key = ol[valid], key[valid]
    groups, names = [], []
    for u in np.unique(key):
        g = ol[key == u]
        if len(g) >= min_size:
            groups.append(g)
            names.append(u)
    return groups, names


def collect(c: Connectome, charset: str = DIGITS, reps: int = 30,
            settle_ms: float = 50.0, probe_ms: float = 120.0,
            grid: int = 24, peak_hz: float = 2000.0,
            layer: str = "column",
            backend: str | None = None, seed: int = 0,
            verbose: bool = True) -> tuple:
    """采集 (VPN 特征, 字符标签, 原始像素) 数据集。"""
    from ..neurons import NeuronIndex

    rng = np.random.default_rng(seed)
    brain = Brain(c.W, calibrated_params(), backend=backend, verbose=verbose)
    bridge = CommandBridge(c, dt_ms=brain.p.dt, verbose=False)
    idx = NeuronIndex(c)

    eyes = [Retina(c, side=s, size=grid) for s in ("L", "R")]
    groups, names = _readout_groups(c, idx, layer=layer)
    if verbose:
        print(f"  刺激: {len(charset)} 类 x {reps} 次 = "
              f"{len(charset) * reps} 个试次")
        print(f"  输入: 双眼 {sum(len(e.on_idx) for e in eyes)} 个 L1(ON) + "
              f"{sum(len(e.off_idx) for e in eyes)} 个 L2(OFF)，"
              f"{eyes[1].n_columns} 个视柱/眼")
        label = ("视柱（排除被驱动的 L1/L2）" if layer == "column"
                 else "类视觉投射神经元")
        print(f"  读出: {len(groups)} {label}")

    n_settle = int(settle_ms / brain.p.dt)
    n_probe = int(probe_ms / brain.p.dt)
    ext = np.zeros(brain.n, dtype=np.float32)

    X, y, P = [], [], []
    trials = [(ci, ch) for ci, ch in enumerate(charset) for _ in range(reps)]
    rng.shuffle(trials)
    t0 = time.time()

    for k, (label, ch) in enumerate(trials):
        # 每次都换一个扰动：平移 ±2 像素、缩放 0.85~1.15、旋转 ±12°
        img = render_char(
            ch, size=grid,
            jitter=(rng.integers(-2, 3), rng.integers(-2, 3)),
            scale=float(rng.uniform(0.85, 1.15)),
            rotate=float(rng.uniform(-12, 12)))

        brain.reset()
        bridge.reset()
        counts = np.zeros(brain.n, dtype=np.float32)
        for i in range(n_settle + n_probe):
            ext.fill(0.0)
            for eye in eyes:
                eye.present(img, ext, bridge, peak_hz=peak_hz)
            s = brain.step(ext)
            if i >= n_settle:
                counts += s
        secs = n_probe * brain.p.dt / 1000.0
        X.append([counts[g].sum() / len(g) / secs for g in groups])
        y.append(label)
        P.append(img.ravel())
        if verbose and (k + 1) % 25 == 0:
            print(f"\r  采集 {k + 1}/{len(trials)}  "
                  f"用时 {time.time() - t0:.0f}s", end="")
    if verbose:
        print()
    return np.asarray(X), np.asarray(y), np.asarray(P), names


def run(c: Connectome, charset: str = DIGITS, reps: int = 30,
        layer: str = "column", backend: str | None = None, seed: int = 0,
        verbose: bool = True) -> dict:
    X, y, P, names = collect(c, charset=charset, reps=reps, layer=layer,
                             backend=backend, seed=seed, verbose=verbose)
    k = len(charset)
    rng = np.random.default_rng(seed + 1)
    order = rng.permutation(len(X))
    X, y, P = X[order], y[order], P[order]
    cut = int(len(X) * 0.7)

    def acc(feat, ytr, yte):
        m = Softmax(k, l2=2.0).fit(feat[:cut], ytr, epochs=400, lr=0.6)
        return float((m.predict(feat[cut:]) == yte).mean())

    real = acc(X, y[:cut], y[cut:])
    pixel = acc(P, y[:cut], y[cut:])
    shuf = np.asarray([
        acc(X, rng.permutation(y[:cut]), rng.permutation(y[cut:]))
        for _ in range(10)])
    major = float((y[cut:] == np.bincount(y[:cut], minlength=k).argmax()
                   ).mean())
    chance = 1.0 / k

    decodes = bool(real > shuf.mean() + 3 * shuf.std() and real > chance * 2)
    out = {
        "charset": charset, "n_samples": int(len(X)),
        "n_features": X.shape[1], "n_classes": k,
        "accuracy": real, "pixel_baseline": pixel, "chance": chance,
        "majority": major, "shuffled_mean": float(shuf.mean()),
        "shuffled_std": float(shuf.std()), "decodes": decodes,
        "retained": real / pixel if pixel > 0 else float("nan"),
    }
    if verbose:
        print(f"\n  样本 {out['n_samples']}（训练 {cut} / 测试 "
              f"{len(X) - cut}）· 特征 {X.shape[1]} 类 VPN 的发放率")
        print(f"  网络解码准确率   {100 * real:5.1f}%")
        print(f"  像素基线         {100 * pixel:5.1f}%   "
              f"<- 直接看图像能到多少（任务上限）")
        print(f"  打乱标签对照     {100 * shuf.mean():5.1f}% "
              f"± {100 * shuf.std():.1f}%   <- 纯过拟合能到多少")
        print(f"  最频繁类别       {100 * major:5.1f}%")
        print(f"  随机             {100 * chance:5.1f}%")
        print(f"\n  果蝇视觉系统能否分辨这 {k} 个字符 = {decodes}")
        if decodes:
            print(f"  网络保留了像素信息的 {100 * out['retained']:.0f}%。")
    return out
