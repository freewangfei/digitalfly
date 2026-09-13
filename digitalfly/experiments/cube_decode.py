"""连接组能不能预测魔方的下一步？—— 一个定量实验。

A 方案（`behave --what cube`）里，连接组只决定"什么时候拧"，"拧哪一面"来自
求解器。这里问的是更硬的问题：**把魔方状态喂进全脑网络，网络的活动里有没有
关于"该拧哪一面"的信息？**

做法是标准的神经解码实验：

  1. 采集数据。随机生成一批魔方局面，每个局面用求解器算出正确的下一步（6 个
     面之一，作为标签）。把局面编码成感觉输入喂给全脑网络，跑一段时间，
     记下读出群体的发放率作为特征。
  2. 训练解码器。用多项逻辑回归（自己实现，纯 numpy）在训练集上拟合
     特征 -> 下一面。
  3. 在留出的测试集上看准确率，和三条基线比：
       随机猜 1/6 ≈ 16.7%
       最频繁类别（数据里出现最多的那一面）
       **打乱标签对照** —— 同样的特征、打乱过的标签，重跑整个流程。
     最后这条最关键：它给出"在这个样本量和特征维度下，纯靠过拟合能到多少"。
     只有真实准确率明显高过打乱对照，才能说网络里有信息。

魔方局面怎么喂进果蝇的脑：果蝇没有魔方感受器，所以这必然是人为约定的编码。
这里用一个**固定的随机投影**把 54 个贴纸的颜色映射到视觉神经元（光感受器）
的发放率上 —— 每个局面都用同一个投影，所以局面的差异会忠实地传进网络。
这个约定本身不携带"该拧哪面"的信息，读不读得出来完全取决于网络。
"""
from __future__ import annotations

import time

import numpy as np

from ..brain import Brain
from ..bridge import CommandBridge
from ..calibrate import calibrated_params
from ..connectome import Connectome
from ..cube import FACES, Cube, solve

N_FACES = 6


def _facelets(c: Cube) -> np.ndarray:
    """把魔方状态编成一个定长向量（角块/棱块的位置与朝向的独热拼接）。"""
    v = np.zeros(8 * 8 + 8 * 3 + 12 * 12 + 12 * 2, dtype=np.float32)
    o = 0
    for i, p in enumerate(c.cp):
        v[o + i * 8 + p] = 1
    o += 64
    for i, x in enumerate(c.co):
        v[o + i * 3 + x] = 1
    o += 24
    for i, p in enumerate(c.ep):
        v[o + i * 12 + p] = 1
    o += 144
    for i, x in enumerate(c.eo):
        v[o + i * 2 + x] = 1
    return v


class Softmax:
    """多项逻辑回归。纯 numpy，带 L2 正则。"""

    def __init__(self, n_class: int, l2: float = 1.0):
        self.k = n_class
        self.l2 = l2

    def fit(self, X, y, epochs: int = 300, lr: float = 0.5):
        n, d = X.shape
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        self.W = np.zeros((d, self.k))
        self.b = np.zeros(self.k)
        Y = np.eye(self.k)[y]
        for _ in range(epochs):
            P = self._soft(Z @ self.W + self.b)
            g = (P - Y) / n
            self.W -= lr * (Z.T @ g + self.l2 / n * self.W)
            self.b -= lr * g.sum(0)
        return self

    @staticmethod
    def _soft(a):
        a = a - a.max(axis=1, keepdims=True)
        e = np.exp(a)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        Z = (X - self.mu) / self.sd
        return np.argmax(Z @ self.W + self.b, axis=1)


def collect(c: Connectome, n_trials: int = 240, scramble: int = 6,
            settle_ms: float = 60.0, probe_ms: float = 120.0,
            backend: str | None = None, seed: int = 0,
            verbose: bool = True) -> tuple:
    """采集 (神经特征, 正确下一步) 数据集。"""
    rng = np.random.default_rng(seed)
    brain = Brain(c.W, calibrated_params(), backend=backend, verbose=verbose)
    bridge = CommandBridge(c, dt_ms=brain.p.dt, verbose=False)

    # 读出群体：下行神经元 + 运动神经元，按固定分组切成若干子群
    readout = np.concatenate([bridge.dn_all, bridge.motor])
    n_groups = 48
    groups = [readout[i::n_groups] for i in range(n_groups)]

    # 魔方状态 -> 视觉神经元发放率的固定随机投影
    photo = bridge.idx.photoreceptors()
    n_in = 64 + 24 + 144 + 24
    proj = rng.normal(0, 1, size=(n_in, 96)).astype(np.float32)
    chans = [photo[i::96] for i in range(96)]

    X, y = [], []
    t0 = time.time()
    n_settle = int(settle_ms / brain.p.dt)
    n_probe = int(probe_ms / brain.p.dt)
    ext = np.zeros(brain.n, dtype=np.float32)

    for t in range(n_trials):
        cube = Cube()
        seq = cube.scramble(scramble, seed=int(rng.integers(1 << 30)))
        sol, src = solve(cube.copy(), node_budget=400_000, fallback=seq)
        if not sol:
            continue
        label = FACES.index(sol[0][0])

        # 局面 -> 各视觉通道的目标发放率
        rates = np.tanh(_facelets(cube) @ proj) * 40.0 + 45.0

        brain.reset()
        bridge.reset()
        counts = np.zeros(brain.n, dtype=np.float32)
        for i in range(n_settle + n_probe):
            ext.fill(0.0)
            for ch, r in zip(chans, rates):
                bridge._drive(ext, ch, float(max(r, 0.0)))
            bridge._drive(ext, bridge.proprio, 20.0)
            s = brain.step(ext)
            if i >= n_settle:
                counts += s
        secs = n_probe * brain.p.dt / 1000.0
        feat = np.array([counts[g].sum() / len(g) / secs for g in groups])
        X.append(feat)
        y.append(label)
        if verbose and (t + 1) % 20 == 0:
            print(f"\r  采集 {t + 1}/{n_trials} 局  "
                  f"用时 {time.time() - t0:.0f}s", end="")
    if verbose:
        print()
    return np.asarray(X), np.asarray(y), n_groups


def run(c: Connectome, n_trials: int = 240, scramble: int = 6,
        backend: str | None = None, seed: int = 0,
        verbose: bool = True) -> dict:
    X, y, n_groups = collect(c, n_trials=n_trials, scramble=scramble,
                             backend=backend, seed=seed, verbose=verbose)
    if len(X) < 40:
        raise RuntimeError("有效样本太少，无法做解码")

    rng = np.random.default_rng(seed + 1)
    order = rng.permutation(len(X))
    X, y = X[order], y[order]
    cut = int(len(X) * 0.7)
    Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]

    def acc(Xtr, ytr, Xte, yte):
        m = Softmax(N_FACES).fit(Xtr, ytr)
        return float((m.predict(Xte) == yte).mean())

    real = acc(Xtr, ytr, Xte, yte)
    # 打乱标签对照：跑 10 次取分布，量出"纯过拟合能到多少"
    shuf = []
    for k in range(10):
        yp = rng.permutation(ytr)
        shuf.append(acc(Xtr, yp, Xte, rng.permutation(yte)))
    shuf = np.asarray(shuf)
    major = float((yte == np.bincount(ytr, minlength=N_FACES).argmax()).mean())

    verdict = bool(real > shuf.mean() + 2 * shuf.std() and real > 1.0 / 6 * 1.5)
    out = {
        "n_samples": int(len(X)), "n_features": int(n_groups),
        "accuracy": real, "chance": 1.0 / N_FACES,
        "majority": major,
        "shuffled_mean": float(shuf.mean()), "shuffled_std": float(shuf.std()),
        "decodes": verdict,
    }
    if verbose:
        print(f"\n  样本 {out['n_samples']}（训练 {cut} / 测试 "
              f"{len(X) - cut}）· 特征 {n_groups} 个读出子群的发放率")
        print(f"  解码准确率      {100 * real:5.1f}%")
        print(f"  随机基线        {100 * out['chance']:5.1f}%")
        print(f"  最频繁类别基线   {100 * major:5.1f}%")
        print(f"  打乱标签对照     {100 * shuf.mean():5.1f}% "
              f"± {100 * shuf.std():.1f}%   <- 纯过拟合能到的水平")
        print(f"\n  结论：连接组的活动里能否解码出「该拧哪一面」 = "
              f"{out['decodes']}")
        if not verdict:
            print("  准确率没有超出打乱标签对照，说明网络活动里没有关于")
            print("  正确解法的可用信息 —— 和 --exp steering 的结论一致。")
    return out
