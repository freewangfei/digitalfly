"""全脑泄漏积分发放（LIF）脉冲网络引擎。

神经元模型与参数取自 Shiu et al., *Nature* (2024) 用 FlyWire 连接组做全脑仿真
的那一套 —— 该参数集能以约 95% 的准确率复现真实果蝇的行为输出，是目前
"连接组直接驱动行为" 最有实证支持的简化模型。

    膜电位:   tau_m * dV/dt = -(V - V_rest) + R * I
    发放:     V >= V_th  -> 发放脉冲, V <- V_reset, 进入不应期
    突触:     一次突触事件使突触后膜电位阶跃 EPSP_MV 毫伏（乘以突触计数与符号）

后端按可用性自动降级: torch-cuda -> torch-cpu -> scipy。
降级会明确打印，不静默发生。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


@dataclass
class LIFParams:
    """Shiu et al. 2024 的 LIF 参数（单位: mV / ms）。"""
    v_rest: float = -52.0
    v_th: float = -45.0
    v_reset: float = -52.0
    tau_m: float = 20.0          # 膜时间常数
    refractory: float = 2.2      # 不应期
    epsp_mv: float = 0.275       # 单个突触的膜电位阶跃
    dt: float = 0.5              # 仿真步长

    @property
    def decay(self) -> float:
        """一步之后膜电位向静息值衰减的系数。"""
        return float(np.exp(-self.dt / self.tau_m))

    @property
    def refractory_steps(self) -> int:
        """不应期折合多少个时间步。

        用 ceil 而不是 round：宁可略长于 2.2 ms，也不能让神经元提前恢复发放 ——
        取整偏短会把最高发放率抬到生理上不可能的水平。
        """
        return int(np.ceil(self.refractory / self.dt))


def _select_backend(prefer: str | None = None) -> str:
    """返回 'torch-cuda' / 'torch-cpu' / 'scipy'，并打印实际选中的后端。"""
    order = ["torch-cuda", "torch-cpu", "scipy"]
    if prefer:
        order = [prefer] + [b for b in order if b != prefer]
    for backend in order:
        if backend == "scipy":
            return "scipy"
        try:
            import torch
        except ImportError:
            continue
        if backend == "torch-cuda":
            if torch.cuda.is_available():
                return "torch-cuda"
        else:
            return "torch-cpu"
    return "scipy"


class Brain:
    """连接组驱动的全脑脉冲网络。

    W 是 CSR 稀疏矩阵，W[post, pre] 为带符号突触计数。
    每步做一次稀疏乘 W @ spikes 得到突触输入。
    """

    def __init__(self, W: sp.csr_matrix, params: LIFParams | None = None,
                 backend: str | None = None, seed: int = 0,
                 verbose: bool = True):
        self.W = W.tocsr()
        self.p = params or LIFParams()
        self.n = W.shape[0]
        self.rng = np.random.default_rng(seed)
        self.backend = _select_backend(backend)
        self.verbose = verbose
        self._ablated = np.zeros(self.n, dtype=bool)

        if verbose:
            print(f"[brain] 后端 = {self.backend}  神经元 {self.n:,}  "
                  f"突触边 {W.nnz:,}")
            if self.backend == "scipy":
                print("[brain] 提示: 未检测到可用的 PyTorch/CUDA，使用 scipy 后端"
                      "（能跑，但比 GPU 慢一个量级）")

        if self.backend.startswith("torch"):
            import torch
            self.torch = torch
            self.dev = torch.device(
                "cuda" if self.backend == "torch-cuda" else "cpu")
            coo = self.W.tocoo()
            idx = torch.from_numpy(
                np.vstack([coo.row, coo.col]).astype(np.int64))
            val = torch.from_numpy(coo.data.astype(np.float32))
            self.Wt = torch.sparse_coo_tensor(
                idx, val, (self.n, self.n), device=self.dev).coalesce()
        self.reset()

    # -- 状态 --------------------------------------------------------------
    def reset(self) -> None:
        p = self.p
        self.v = np.full(self.n, p.v_rest, dtype=np.float32)
        self.refrac = np.zeros(self.n, dtype=np.int16)
        self.spikes = np.zeros(self.n, dtype=np.float32)
        self.t_ms = 0.0
        self.n_steps = 0
        if self.backend.startswith("torch"):
            t = self.torch
            self.v_t = t.from_numpy(self.v).to(self.dev)
            self.refrac_t = t.zeros(self.n, dtype=t.int16, device=self.dev)
            self.spikes_t = t.zeros(self.n, dtype=t.float32, device=self.dev)
            self.ablated_t = t.zeros(self.n, dtype=t.bool, device=self.dev)

    def ablate(self, indices: np.ndarray) -> None:
        """把指定神经元永久静默（模拟切除 / 基因沉默实验）。"""
        self._ablated[indices] = True
        if self.backend.startswith("torch"):
            self.ablated_t[self.torch.from_numpy(
                np.asarray(indices, dtype=np.int64)).to(self.dev)] = True

    # -- 仿真 --------------------------------------------------------------
    def step(self, ext_current: np.ndarray | None = None) -> np.ndarray:
        """推进一个时间步。ext_current 是外部注入的膜电位增量 (mV)。

        返回本步的脉冲向量 (float32, 0/1)。
        """
        if self.backend.startswith("torch"):
            out = self._step_torch(ext_current)
        else:
            out = self._step_scipy(ext_current)
        self.n_steps += 1
        self.t_ms += self.p.dt
        return out

    def _step_scipy(self, ext: np.ndarray | None) -> np.ndarray:
        p = self.p
        syn = self.W @ self.spikes            # 带符号突触计数
        dv = syn * p.epsp_mv
        if ext is not None:
            dv = dv + ext

        active = self.refrac <= 0
        self.v = np.where(
            active,
            p.v_rest + (self.v - p.v_rest) * p.decay + dv,
            p.v_reset,
        ).astype(np.float32)

        fired = (self.v >= p.v_th) & active & (~self._ablated)
        self.v[fired] = p.v_reset
        self.refrac[fired] = p.refractory_steps
        np.subtract(self.refrac, 1, out=self.refrac, where=self.refrac > 0)

        self.spikes = fired.astype(np.float32)
        return self.spikes

    def _step_torch(self, ext: np.ndarray | None) -> np.ndarray:
        t, p = self.torch, self.p
        syn = t.sparse.mm(self.Wt, self.spikes_t.unsqueeze(1)).squeeze(1)
        dv = syn * p.epsp_mv
        if ext is not None:
            dv = dv + t.from_numpy(np.asarray(ext, dtype=np.float32)).to(self.dev)

        active = self.refrac_t <= 0
        decayed = p.v_rest + (self.v_t - p.v_rest) * p.decay + dv
        self.v_t = t.where(active, decayed,
                           t.full_like(self.v_t, p.v_reset))

        fired = (self.v_t >= p.v_th) & active & (~self.ablated_t)
        self.v_t = t.where(fired, t.full_like(self.v_t, p.v_reset), self.v_t)
        # 刚发放的置成 refractory_steps - 1：本步已经过去了，和 scipy 后端
        # 先赋值再统一自减的效果保持一致，否则会多阻塞一步。
        self.refrac_t = t.where(
            fired,
            t.full_like(self.refrac_t, p.refractory_steps - 1),
            t.clamp(self.refrac_t - 1, min=0))
        self.spikes_t = fired.to(t.float32)
        self.spikes = self.spikes_t.detach().cpu().numpy()
        return self.spikes

    # -- 便捷接口 -----------------------------------------------------------
    def run(self, duration_ms: float,
            stimulus: "Stimulus | None" = None,
            record: np.ndarray | None = None,
            progress: bool = False) -> "SpikeRecord":
        """连续仿真一段时间，返回脉冲记录。

        record: 要逐步记录的神经元下标；None 表示只记录每步总发放数。
        """
        n_steps = int(round(duration_ms / self.p.dt))
        rec = SpikeRecord(dt=self.p.dt, indices=record)
        t0 = time.time()
        for i in range(n_steps):
            ext = stimulus(self.t_ms, self.n) if stimulus else None
            s = self.step(ext)
            rec.append(s)
            if progress and i % max(1, n_steps // 20) == 0:
                print(f"\r  仿真 {100 * i / n_steps:5.1f}%  "
                      f"t={self.t_ms:7.1f}ms  发放 {int(s.sum()):5d}", end="")
        if progress:
            print(f"\r  仿真完成 {duration_ms:.0f}ms，"
                  f"墙钟 {time.time() - t0:.1f}s" + " " * 20)
        rec.finish()
        return rec


@dataclass
class SpikeRecord:
    """脉冲记录。总是存每步总发放数；可选地存指定神经元的脉冲栅格。"""
    dt: float
    indices: np.ndarray | None = None
    total: list = field(default_factory=list)
    raster: list = field(default_factory=list)
    counts: np.ndarray | None = None   # 每个神经元的累计发放数

    def append(self, spikes: np.ndarray) -> None:
        if self.counts is None:
            self.counts = np.zeros(len(spikes), dtype=np.int32)
        self.counts += spikes.astype(np.int32)
        self.total.append(float(spikes.sum()))
        if self.indices is not None:
            self.raster.append(spikes[self.indices].astype(np.uint8))

    def finish(self) -> None:
        self.total = np.asarray(self.total, dtype=np.float32)
        if self.indices is not None:
            self.raster = np.asarray(self.raster, dtype=np.uint8)  # (T, k)

    @property
    def duration_ms(self) -> float:
        return len(self.total) * self.dt

    def rates_hz(self) -> np.ndarray:
        """每个神经元的平均发放率 (Hz)。"""
        return self.counts / (self.duration_ms / 1000.0)


class Stimulus:
    """把外部输入编码成注入电流（膜电位增量, mV/步）。

    用法:
        stim = Stimulus(n)
        stim.add(sugar_grn_indices, amplitude=2.0, start=100, stop=600)
    """

    def __init__(self, n: int):
        self.n = n
        self.events: list[tuple[np.ndarray, float, float, float]] = []
        self._static = np.zeros(n, dtype=np.float32)

    def add(self, indices, amplitude: float,
            start: float = 0.0, stop: float = float("inf")) -> "Stimulus":
        self.events.append((np.asarray(indices, dtype=np.int64),
                            float(amplitude), float(start), float(stop)))
        return self

    def __call__(self, t_ms: float, n: int) -> np.ndarray:
        out = self._static
        out.fill(0.0)
        for idx, amp, start, stop in self.events:
            if start <= t_ms < stop:
                out[idx] += amp
        return out
