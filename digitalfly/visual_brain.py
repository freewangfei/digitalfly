"""脑内视觉判决 —— 判断"这是几"的那一步，发生在连接组内部。

这个模块存在的理由，是 `handwriting.py` 里那个识别器**并不满足**这个要求：
那里前面是果蝇的脑，但最后拍板的是一个多项逻辑回归，权重存在磁盘上的
`handwriting.npz` 里，不在连接组里。那是个外挂分类器。

这里换一种做法：

    图像 ──► 六角视柱 ──► L1/L2（真实的板层神经元，被电流直接驱动）
                              │  ★ 可塑突触 —— 连接组里真实存在的那些
                              ▼
                      髓质宽场无长突细胞（Dm6/Dm19/Dm17/Dm15/Dm1）
                              │  按细胞分成 N 组，一组代表一个字符
                              ▼
                        哪一组发放最多，答案就是几

**分类器的参数就是 `W` 里的元素本身。** 没有第二套权重，没有反向传播。
学习用的是蘑菇体那条规则的同一形式（教学信号门控的双向突触可塑性，
Hige et al. *Neuron* 2015；Cohn et al. *Cell* 2015；Aso & Rubin *eLife* 2016）：
答错时，把"当时正在活动的 L1/L2"到正确组的突触增强、到错误获胜组的压低。

### 为什么判决层是髓质，而不是 LC

按生物学，果蝇真正的视觉判决神经元是小叶柱状细胞（LC）—— Wu et al.
(*eLife* 2016) 光遗传激活单个 LC 类型就能诱发特定行为。本项目最初就是奔着
LC 去的，实测结果是 **LC4 + LC12 共 624 个细胞，在标定工作点上一个都不发放
（0/624，峰值 0.0 Hz）**。原因是视网膜→髓质→LC 隔了两个突触，而这套重建在
标定后只能可靠传递大约一个突触 —— 这是本项目反复撞到的同一堵墙。

所以判决层退到一个突触之内的髓质。这是数据驱动的选择，不是偏好：实测除被
直接驱动的 L1/L2 之外，全脑只有 2,011 个神经元发放超过 1 Hz。判决只能发生在
真的会响应的地方。具体选哪些髓质细胞，见下面 DECISION_TYPES 的注释 —— 简短说，
必须是**宽场**的（汇聚整个视野），不能是单柱的。

### 两处必须标出来的模型补充

1. **哪一组代表哪个字符是任意指定的。** 这一点和真实果蝇一致 —— MBON 的效价
   也不是天生的，是由哪个 DAN 教它决定的。这里由教学信号指定。
2. **组增益按各组自己的基线归一。** 各组细胞数和固有发放率不同，不归一的话
   argmax 永远选中固有最强的那组。基线取一批刺激上的平均发放率，对应真实
   神经元的适应与增益控制。这是 10 个标量，不是一套权重矩阵。
"""
from __future__ import annotations

import numpy as np

# 判决层用髓质的**宽场无长突细胞**（Dm 类）。两条实测理由，缺一不可：
#
# 1. **汇聚度。** 一个 Dm17 从 204 个视柱收输入，Dm19 是 130，Dm6 是 44.5；
#    相比之下单柱细胞 Tm1 平均每胞只有 0.6 条来自 L1/L2 的突触 —— 绝大多数 Tm1
#    根本没有可塑输入。判决要对**整张图**做，判决神经元就必须汇聚整个视野；
#    用单柱细胞的话每组只看得到视野的 1/N，各组比的是图像的不同部分，argmax
#    本身就没有意义。最初选 Tm1 就是栽在这里，4 类任务卡在 42.5%。
# 2. **信噪比。** 实测 Dm19 发放 113.8 Hz、Dm17 90.9 Hz、Dm6 48.5 Hz，跨字符
#    调制 13~20%；Tm1 只有 2.2 Hz、调制 8%。发放率高 20~50 倍，泊松噪声小得多。
DECISION_TYPES = ("Dm6", "Dm19", "Dm17", "Dm15", "Dm1")


class VisualDecision:
    """把"这是几"的判决放进连接组里。"""

    def __init__(self, c, brain, retina, charset: str = "0123456789",
                 types=DECISION_TYPES, seed: int = 0, verbose: bool = True):
        self.c = c
        self.brain = brain
        self.retina = retina
        self.charset = charset
        n_cls = len(charset)

        typ = np.array([str(v) for v in c.meta["type"].values])
        # 被视网膜直接驱动的 L1/L2 —— 可塑突触的突触前
        self.driven = np.unique(np.concatenate([retina.on_idx,
                                                retina.off_idx]))
        drv = np.zeros(brain.W.shape[1], dtype=bool)
        drv[self.driven] = True

        pool = np.flatnonzero(np.isin(typ, list(types)))
        pool = pool[~np.isin(pool, self.driven)]
        if len(pool) < n_cls * 5:
            raise RuntimeError(f"可用判决神经元太少：{len(pool)}")

        # 随机分成 N 组。**必须随机、不能按空间分** —— 按视野位置分组的话每组
        # 只看得到视野的一小块，等于在比"哪个角落亮"。随机分组让每组都铺满整个
        # 视野，判决才是对整张图做的。
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(pool))
        self.groups = [np.sort(pool[order[i::n_cls]]) for i in range(n_cls)]
        self.cells = np.concatenate(self.groups)
        self.group_of = np.zeros(brain.W.shape[0], dtype=np.int32) - 1
        for g, ix in enumerate(self.groups):
            self.group_of[ix] = g

        # 在 CSR 里定位 L1/L2 → 判决神经元 的那些元素。W 按 post 行存。
        W = brain.W
        slots, posts, pres = [], [], []
        for row in self.cells:
            lo, hi = W.indptr[row], W.indptr[row + 1]
            sel = np.flatnonzero(drv[W.indices[lo:hi]])
            if len(sel):
                p = lo + sel
                slots.append(p)
                posts.append(np.full(len(p), row))
                pres.append(W.indices[p])
        if not slots:
            raise RuntimeError("判决神经元收不到任何视网膜输入")
        self.slots = np.concatenate(slots)
        self.syn_post = np.concatenate(posts)
        self.syn_pre = np.concatenate(pres)
        self.w0 = W.data[self.slots].copy()
        self.slot_group = self.group_of[self.syn_post]

        self.gain = np.ones(n_cls)
        self.pre_mu = None
        self.pre_sd = None
        self.n_teach = 0
        if verbose:
            print(f"[脑内判决] 判决神经元 {len(self.cells)} 个"
                  f"（{'/'.join(types)}），分 {n_cls} 组，"
                  f"每组约 {len(self.cells) // n_cls} 个")
            print(f"[脑内判决] 可塑突触 {len(self.slots):,} 条"
                  f"（L1/L2 → 判决神经元），全部来自连接组")

    # -- 呈现与判决 -----------------------------------------------------------
    def present(self, image, bridge, ms: float = 180.0,
                peak_hz: float = 60.0) -> np.ndarray:
        """把一张图呈现给复眼，返回各判决组的发放率 (Hz)。"""
        b = self.brain
        b.reset()
        ext = np.zeros(b.n, dtype=np.float32)
        counts = np.zeros(b.n, dtype=np.float32)
        n = int(ms / b.p.dt)
        warm = n // 3
        for i in range(n):
            ext.fill(0.0)
            self.retina.present(image, ext, bridge, peak_hz=peak_hz)
            s = b.step(ext)
            if i >= warm:
                counts += s
        secs = (n - warm) * b.p.dt / 1000.0
        self.last_counts = counts
        return self.group_rates(counts, secs)

    def group_rates(self, counts, seconds: float) -> np.ndarray:
        """每组的平均发放率。"""
        return np.array([counts[ix].sum() / (len(ix) * max(seconds, 1e-9))
                         for ix in self.groups])

    def set_gain(self, baseline_rates) -> None:
        """按各组自己的基线定增益（见模块文档第 2 点）。"""
        base = np.mean(np.asarray(baseline_rates, dtype=np.float64), axis=0)
        self.gain = 1.0 / np.maximum(base, 1e-6)

    def decide(self, rates) -> tuple:
        """返回 (字符, 各组的归一化得分)。"""
        score = np.asarray(rates) * self.gain
        return self.charset[int(np.argmax(score))], score

    def update_gain(self, rates, tau: float = 0.05) -> None:
        """稳态增益调节 —— 各组增益跟着自己的长期平均发放率慢慢走。

        权重改了之后各组的基础发放率会漂移，固定增益会过时，argmax 就偏向漂高
        的那一组。真实神经元靠稳态可塑性维持自己的平均发放率，这里做同一件事。
        """
        r = np.asarray(rates, dtype=np.float64)
        if getattr(self, "_run_mean", None) is None:
            self._run_mean = r.copy()
        else:
            self._run_mean = (1 - tau) * self._run_mean + tau * r
        self.gain = 1.0 / np.maximum(self._run_mean, 1e-6)

    # -- 学习 -----------------------------------------------------------------
    # 增强的上限，相对原始强度。真实突触不会无限增强；取 3 倍是给学习留够动态
    # 范围，又不让单条突触主导判决。
    CEILING = 3.0

    def teach(self, true_char: str, pred_char: str, eta: float = 0.08,
              recover: float = 0.02) -> dict:
        """双向的、教学信号门控的可塑性。

        答错时：正确那一组的活跃突触**增强**，错误获胜那一组的活跃突触压低；
        两边都只作用在"当时活动的"突触前（L1/L2）。答对时只做缓慢恢复。

        **为什么必须双向。** 最初这里只做抑制（照搬蘑菇体那条规则），权重被锁在
        [0, w0] 里，正确类永远超不过原始强度 —— 系统只能单调收缩，区分度得从一个
        不断衰减的底子里刻出来。实测 4 类任务卡在 42.5%（随机 25%），突触总强度
        一路掉到 0.87 还在降。改成双向，动态范围才打开。

        双向有文献依据，不是为了好看：Cohn et al. (*Cell* 2015)、Aso & Rubin
        (*eLife* 2016) 都表明 MB 区室的多巴胺可塑性是双向的，DAN 既能压低也能
        增强，方向由区室和 DAN 类型决定。
        """
        if pred_char == true_char:
            self._recover_group(self.charset.index(true_char), recover)
            return {"changed": 0, "correct": True}

        act = self._presyn_activity()
        W = self.brain.W
        n_ch = 0
        for g, sign in ((self.charset.index(true_char), +1.0),
                        (self.charset.index(pred_char), -1.0)):
            sel = (self.slot_group == g) & (act > 0)
            if not np.any(sel):
                continue
            s = self.slots[sel]
            cur = W.data[s]
            new = cur * (1.0 + sign * eta * act[sel])
            # 有界：不越过上限，也不穿过 0 变号
            hi = np.abs(self.w0[sel]) * self.CEILING
            W.data[s] = (np.sign(cur) * np.clip(np.abs(new), 0.0, hi)
                         ).astype(W.data.dtype)
            n_ch += int(sel.sum())
        self.n_teach += 1
        self._sync()
        return {"changed": n_ch, "correct": False}

    def _presyn_activity(self) -> np.ndarray:
        """每条可塑突触的突触前活动（0~1），取自刚才那次呈现。

        **必须按各突触前自己的基线归一。** 抑制只作用在"当时活动的"突触前，
        可手写图里约 80% 的像素是黑的，L2（OFF 通道）被暗度驱动 —— 不归一的话
        几乎每个 L1/L2 对每个字符都在活动，这个门就选不出任何东西，抑制退化成
        无差别的全局衰减。实测：不归一时四组突触强度掉到 0.60/0.59/0.58/0.64，
        完全没有区分度，测试准确率从 40.6% 反而掉到 31.2%。

        归一之后，只有**相对于自己常态异常活跃**的突触前才算数，抑制才具有
        刺激特异性。这和蘑菇体那边 KC 稀疏编码要按基线归一是同一件事，生物学
        上对应神经元的适应与增益控制。
        """
        cnt = getattr(self, "last_counts", None)
        if cnt is None:
            return np.zeros(len(self.slots))
        a = cnt[self.syn_pre].astype(np.float64)
        if getattr(self, "pre_mu", None) is not None:
            a = (a - self.pre_mu) / self.pre_sd
            a = np.maximum(a, 0.0)
        peak = a.max()
        return a / peak if peak > 0 else a

    def set_presyn_baseline(self, counts_list) -> None:
        """记录一批刺激下每条突触的突触前平均活动，作为归一基线。"""
        A = np.asarray([cnt[self.syn_pre] for cnt in counts_list],
                       dtype=np.float64)
        self.pre_mu = A.mean(axis=0)
        self.pre_sd = A.std(axis=0) + 1e-6

    def _recover_group(self, g: int, rate: float) -> None:
        if rate <= 0:
            return
        sel = self.slot_group == g
        W = self.brain.W
        s = self.slots[sel]
        cur = W.data[s]
        W.data[s] = (cur + rate * (self.w0[sel] - cur)).astype(W.data.dtype)
        self._sync()

    def normalise(self) -> None:
        """突触缩放：把每组可塑突触的总强度拉回原始值。

        双向可塑性如果不加约束，增强会赢 —— 实测 10 类训练里总强度一路涨到
        1.378 还在升，判决被整体漂移主导，测试准确率在 13~23% 之间震荡。
        真实神经元靠**突触缩放**（synaptic scaling，Turrigiano 1998）维持总输入
        恒定：按比例缩放全部输入突触，保留它们之间的相对权重，也就保留学到的
        东西。这里做的就是这件事，按组做。
        """
        W = self.brain.W
        for g in range(len(self.charset)):
            sel = self.slot_group == g
            s_ = self.slots[sel]
            cur = np.abs(W.data[s_]).sum()
            tgt = np.abs(self.w0[sel]).sum()
            if cur > 1e-9:
                W.data[s_] = (W.data[s_] * (tgt / cur)).astype(W.data.dtype)
        self._sync()

    def restore(self) -> None:
        self.brain.W.data[self.slots] = self.w0
        self.n_teach = 0
        self._sync()

    def strength(self) -> float:
        cur = np.abs(self.brain.W.data[self.slots]).sum()
        return float(cur / np.abs(self.w0).sum())

    def group_strength(self) -> np.ndarray:
        """每组可塑突触的剩余强度 —— 学习的痕迹长什么样。"""
        return np.array([
            float(np.abs(self.brain.W.data[self.slots[self.slot_group == g]]
                         ).sum()
                  / max(np.abs(self.w0[self.slot_group == g]).sum(), 1e-9))
            for g in range(len(self.charset))])

    # -- 存取 -----------------------------------------------------------------
    def save(self, path) -> None:
        """只存突触权重 —— 学到的东西就是这些数。"""
        np.savez_compressed(
            path, slots=self.slots, weights=self.brain.W.data[self.slots],
            w0=self.w0, gain=self.gain, charset=self.charset,
            groups=np.array([len(g) for g in self.groups]),
            n_teach=self.n_teach)

    def load(self, path) -> None:
        d = np.load(path, allow_pickle=False)
        if not np.array_equal(d["slots"], self.slots):
            raise RuntimeError("突触位置对不上 —— 连接组或分组种子变了")
        self.brain.W.data[self.slots] = d["weights"]
        self.gain = d["gain"]
        self.n_teach = int(d["n_teach"])
        self._sync()

    def _sync(self) -> None:
        """改完 scipy 那份要同步到 GPU 后端，否则学了但网络没变。"""
        b = self.brain
        if not b.backend.startswith("torch"):
            return
        if not hasattr(self, "_tslots"):
            coo = b.Wt.indices().cpu().numpy()
            order = np.lexsort((coo[1], coo[0]))
            key_coo = coo[0][order].astype(np.int64) * b.n + coo[1][order]
            key = self.syn_post.astype(np.int64) * b.n + self.syn_pre
            self._tslots = b.torch.from_numpy(
                order[np.searchsorted(key_coo, key)].astype(np.int64)).to(b.dev)
        vals = b.Wt.values()
        vals[self._tslots] = b.torch.from_numpy(
            b.W.data[self.slots].astype(np.float32)).to(vals.device)
