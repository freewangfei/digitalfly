"""蘑菇体的突触可塑性 —— 让果蝇自己学，而不是外挂一个解码器。

**这是果蝇真实的学习机制，不是套上去的机器学习。**

果蝇的学习中枢是蘑菇体，回路和规则在文献里都是确定的：

    感觉输入 ──► Kenyon 细胞（稀疏编码，本数据集 4,064 个）
                      │  可塑突触（61,210 条）
                      ▼
                   MBON（97 个）──► 行为倾向（趋近 / 回避）
                      ▲
              多巴胺能神经元 DAN（340 个）= 教学信号

学习规则是**多巴胺门控的突触抑制**：某个区室的 DAN 发放时，那一刻正在活动的
Kenyon 细胞到该区室 MBON 的突触被压低。所以"闻到 A 之后被电击"会压低
A→趋近MBON 的通路，果蝇下次就避开 A。这条规则由 Hige et al. (*Neuron* 2015)
和 Aso & Rubin (*eLife* 2016) 确立，是昆虫学习最扎实的一块。

这个模块把这条规则实现在**连接组自己的那批突触**上：直接改 `W` 里
KC→MBON 那些元素的值。改完之后网络的行为就变了 —— 没有外部分类器，
没有反向传播，改的是果蝇自己的突触。

三条硬约束，都在代码里守住：
  * 只有 KC→MBON 的突触可塑。别的一律不动 —— 果蝇的学习也就发生在这里。
  * 只降不升（抑制型可塑性），加一条缓慢的自发恢复，对应真实的遗忘。
  * 权重不改变符号，也不越过原始强度。
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class MushroomBody:
    """蘑菇体：稀疏编码 + 多巴胺门控的突触抑制。"""

    def __init__(self, c, brain, verbose: bool = True):
        from .neurons import NeuronIndex

        self.c = c
        self.brain = brain
        idx = NeuronIndex(c)
        self.kc = idx.where(**{"class": "Kenyon_Cell"})
        self.mbon = idx.where(**{"class": "MBON"})
        self.dan = idx.where(**{"class": "DAN"})
        if not len(self.kc) or not len(self.mbon):
            raise RuntimeError("这套连接组里找不到蘑菇体")

        # 在 CSR 矩阵里定位 KC -> MBON 的那些元素。
        # W 是按 post 行存的，所以取 MBON 那些行、列在 KC 里的元素。
        W = brain.W
        kc_set = np.zeros(W.shape[1], dtype=bool)
        kc_set[self.kc] = True
        rows, cols, slots = [], [], []
        for r in self.mbon:
            lo, hi = W.indptr[r], W.indptr[r + 1]
            sel = np.flatnonzero(kc_set[W.indices[lo:hi]])
            if len(sel):
                pos = lo + sel
                slots.append(pos)
                rows.append(np.full(len(pos), r))
                cols.append(W.indices[pos])
        self.slots = np.concatenate(slots)          # W.data 里的下标
        self.syn_post = np.concatenate(rows)
        self.syn_pre = np.concatenate(cols)
        self.w0 = W.data[self.slots].copy()         # 原始强度，学习的上界

        # 每个 MBON 在 self.mbon 里的位置，用于把教学信号定位到区室
        self.mbon_pos = {int(v): i for i, v in enumerate(self.mbon)}
        self.post_local = np.array([self.mbon_pos[int(r)]
                                    for r in self.syn_post])

        self.kc_trace = np.zeros(len(self.kc), dtype=np.float32)
        self.kc_local = {int(v): i for i, v in enumerate(self.kc)}
        self.pre_local = np.array([self.kc_local[int(cix)]
                                   for cix in self.syn_pre])
        self.n_updates = 0
        self.kc_mu = None
        self.kc_sd = None
        if verbose:
            print(f"[蘑菇体] Kenyon 细胞 {len(self.kc)} · MBON "
                  f"{len(self.mbon)} · DAN {len(self.dan)}")
            print(f"[蘑菇体] 可塑突触 {len(self.slots):,} 条 "
                  f"(KC→MBON)，总权重 {float(np.abs(self.w0).sum()):,.0f}")

    # -- 活动 ---------------------------------------------------------------
    def observe(self, spikes: np.ndarray, tau_ms: float = 80.0,
                dt_ms: float = 0.5) -> None:
        """累积 Kenyon 细胞的活动痕迹。

        可塑性需要"刚才哪些 KC 在活动"，所以要一条比脉冲慢的痕迹 ——
        对应真实突触里钙信号 / cAMP 的时间尺度。
        """
        self.kc_trace *= np.exp(-dt_ms / tau_ms)
        self.kc_trace += spikes[self.kc]

    def reset_trace(self) -> None:
        self.kc_trace[:] = 0.0

    # 稀疏编码的目标比例。真实果蝇里一种气味只激活约 5% 的 Kenyon 细胞，
    # 这是 MB 学习具有气味特异性的前提 —— 不稀疏就没法只压低"那一种气味"的通路。
    SPARSITY = 0.05

    def set_baseline(self, responses) -> None:
        """记录一批刺激下每个 KC 的平均响应，作为稀疏化的基线。

        必须按基线归一，否则"前 5%"选出来的是**固有输入最强**的那批 KC，
        跟当前是什么刺激无关 —— 实测这样两种气味的 KC 编码重叠 73%，
        比输入本身的重叠（25%）还高，等于网络在增加相关性。
        按每个 KC 自己的均值和标准差归一之后，重叠降到 0%。

        生物学上这对应 KC 的适应与增益控制：细胞对偏离自己常态的输入敏感，
        而不是对绝对强度敏感。
        """
        R = np.asarray(responses, dtype=np.float64)
        self.kc_mu = R.mean(axis=0)
        self.kc_sd = R.std(axis=0) + 1e-6

    def sparse_code(self, counts: np.ndarray) -> np.ndarray:
        """把 KC 的响应稀疏化，返回 [0,1] 的活动向量。

        **这一步是模型补充，不是从连接组里涌现的。** 本重建里 KC 收到的
        兴奋性输入是抑制性的 7.3 倍（184 万 vs 25 万），而负责全局抑制、
        在真实果蝇里强制稀疏编码的 APL 只有 2 个 GABA 能神经元 —— 压不住
        64 万条 KC→KC 递归兴奋。实测任何输入都会把全部 4,064 个 KC 点亮到
        145 Hz，稀疏编码不会自己出现。

        所以这里显式取响应最强的前 5% 作为该气味的 KC 编码，相当于替
        APL 把它在真实果蝇里做的事做掉。学习规则本身仍然是果蝇的规则，
        改的仍然是连接组自己的突触。
        """
        r = counts[self.kc].astype(np.float64)
        if getattr(self, "kc_mu", None) is not None:
            r = (r - self.kc_mu) / self.kc_sd
        k = max(1, int(len(r) * self.SPARSITY))
        thr = np.partition(r, -k)[-k]
        act = np.where(r >= thr, np.maximum(r, 0.0), 0.0)
        peak = act.max()
        return act / peak if peak > 0 else act

    def kc_rate(self, counts: np.ndarray, seconds: float) -> np.ndarray:
        return counts[self.kc] / max(seconds, 1e-9)

    def mbon_rate(self, counts: np.ndarray, seconds: float) -> np.ndarray:
        return counts[self.mbon] / max(seconds, 1e-9)

    # -- 学习 ---------------------------------------------------------------
    def teach(self, dopamine: np.ndarray, eta: float = 0.35,
              kc_source: np.ndarray | None = None) -> dict:
        """多巴胺门控的突触抑制。

        dopamine: 长度 = MBON 个数，取值 [0, 1]。哪个区室收到教学信号。
        kc_source: 用哪份 KC 活动。默认用当前的活动痕迹。

        规则：Δw = −eta · (该 KC 的活动) · (该 MBON 收到的多巴胺) · w0
        只压低，不抬高；压低量相对原始强度，所以强突触被压得多。
        """
        kc_act = self.kc_trace if kc_source is None else kc_source
        peak = float(kc_act.max())
        if peak <= 0:
            return {"changed": 0, "delta": 0.0}
        act = np.clip(kc_act / peak, 0.0, 1.0)

        da = np.clip(np.asarray(dopamine, dtype=np.float64), 0.0, 1.0)
        drive = act[self.pre_local] * da[self.post_local]
        if not np.any(drive > 0):
            return {"changed": 0, "delta": 0.0}

        W = self.brain.W
        cur = W.data[self.slots]
        # 往 0 的方向压，压不过头也不变号
        new = cur * (1.0 - eta * drive)
        W.data[self.slots] = new.astype(W.data.dtype)
        self._sync_backend()
        self.n_updates += 1
        return {
            "changed": int((drive > 0).sum()),
            "delta": float(np.abs(new - cur).sum()),
            "remaining": float(np.abs(new).sum() / np.abs(self.w0).sum()),
        }

    def recover(self, rate: float = 0.02) -> None:
        """缓慢恢复到原始强度 —— 对应真实的遗忘。"""
        W = self.brain.W
        cur = W.data[self.slots]
        W.data[self.slots] = (cur + rate * (self.w0 - cur)).astype(
            W.data.dtype)
        self._sync_backend()

    def restore(self) -> None:
        """把可塑突触复位到学习前。"""
        self.brain.W.data[self.slots] = self.w0
        self._sync_backend()
        self.n_updates = 0

    def strength(self) -> float:
        """当前可塑突触总强度占原始的比例。"""
        cur = np.abs(self.brain.W.data[self.slots]).sum()
        return float(cur / np.abs(self.w0).sum())

    def _sync_backend(self) -> None:
        """把改过的权重同步到 GPU 后端。

        torch 后端在构造时把 CSR 拷成了 COO 稀疏张量，改 scipy 那份不会
        自动生效 —— 必须把对应的值写回去，否则"学了"但网络没变。
        """
        b = self.brain
        if not b.backend.startswith("torch"):
            return
        if not hasattr(self, "_torch_slots"):
            # COO 的排列顺序和 CSR 不同，按 (row, col) 建一次映射
            coo_idx = b.Wt.indices().cpu().numpy()
            order = np.lexsort((coo_idx[1], coo_idx[0]))
            key_coo = (coo_idx[0][order].astype(np.int64) * b.n
                       + coo_idx[1][order])
            key_syn = (self.syn_post.astype(np.int64) * b.n + self.syn_pre)
            pos = np.searchsorted(key_coo, key_syn)
            self._torch_slots = order[pos]
        vals = b.Wt.values()
        vals[self.torch_slots_tensor] = b.torch.from_numpy(
            b.W.data[self.slots].astype(np.float32)).to(vals.device)

    @property
    def torch_slots_tensor(self):
        b = self.brain
        if not hasattr(self, "_torch_slots_t"):
            self._torch_slots_t = b.torch.from_numpy(
                self._torch_slots.astype(np.int64)).to(b.dev)
        return self._torch_slots_t
