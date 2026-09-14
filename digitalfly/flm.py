"""FLM —— 把果蝇连接组接到语言模型上。

思路来自 nftechie 的 [flm](https://github.com/nftechie/flm)：把 MaleCNS 连接组当成一个
**冻结的储备池**（reservoir），token 嵌入驱动整张图，读出图的状态去修正语言模型
下一个 token 的 logits。只训练那个很小的适配器，连接组和语言模型都不动。

    token ──► 输入投影 ──► 连接组递归 x = tanh(W·(αx + β·in)) ──► 池化
                                                                    │
                语言模型 logits ◄── 有界修正 ◄── 适配器（仅此部分可训练）

**这里的 W 不是脉冲仿真。** 和本项目其他部分不同，这一层用的是抽象的数值状态，
不是动作电位：没有递质符号、没有多巴胺、没有生物学时间。它衡量的是"这个拓扑
结构作为一个固定的非线性变换，能不能给语言模型提供有用的信息"。

**必须说在前面的两件事：**

1. 语言能力全部来自预训练的语言模型。连接组不理解语言，一只真果蝇也不会。
2. 原作者做过参数量匹配的**直接输入对照组**，结果是对照组略好 —— 也就是说
   目前没有证据表明果蝇的解剖结构带来了优势。本实现同样提供这个对照
   （`--control`），并如实报告两边的数字。不做"果蝇会说话了"这种声明。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config

LFM_DIR = config.DATA_DIR / "lfm"
FLM_NPZ = config.BUILD_DIR / "flm_adapter.npz"

# 储备池递归的两个系数：保留多少旧状态、注入多少新输入
ALPHA, BETA = 0.6, 0.4


class Reservoir:
    """连接组作为固定的非线性储备池。"""

    def __init__(self, c, n_probe: int = 2048, seed: int = 0,
                 verbose: bool = True):
        import scipy.sparse as sp

        self.n = c.W.shape[0]
        # 按**入度归一**的接触数，取绝对值 —— 这一层不用递质符号（见模块文档）
        W = abs(c.W).astype(np.float32).tocsr()
        deg = np.asarray(W.sum(axis=1)).ravel()
        inv = np.where(deg > 0, 1.0 / np.maximum(deg, 1e-9), 0.0)
        self.W = sp.diags(inv.astype(np.float32)) @ W

        rng = np.random.default_rng(seed)
        # 输入投影打到哪些神经元、读出从哪些神经元池化 —— 固定的随机稀疏投影
        self.in_idx = rng.choice(self.n, size=n_probe, replace=False)
        self.out_idx = rng.choice(self.n, size=n_probe, replace=False)
        self.n_probe = n_probe
        if verbose:
            print(f"[FLM] 储备池 {self.n:,} 个节点 · {self.W.nnz:,} 条边 · "
                  f"输入/读出各 {n_probe} 个探针")

    def run(self, drive: np.ndarray, steps: int = 3) -> np.ndarray:
        """drive 是长度 n_probe 的输入，返回长度 n_probe 的状态读数。"""
        x = np.zeros(self.n, dtype=np.float32)
        inp = np.zeros(self.n, dtype=np.float32)
        inp[self.in_idx] = drive.astype(np.float32)
        for _ in range(steps):
            x = np.tanh(self.W @ (ALPHA * x + BETA * inp))
        return x[self.out_idx]

    def run_batch(self, drives: np.ndarray, steps: int = 3) -> np.ndarray:
        """一次跑一批。drives: (B, n_probe) -> (B, n_probe)。"""
        B = drives.shape[0]
        X = np.zeros((self.n, B), dtype=np.float32)
        I = np.zeros((self.n, B), dtype=np.float32)
        I[self.in_idx] = drives.T.astype(np.float32)
        for _ in range(steps):
            X = np.tanh(self.W @ (ALPHA * X + BETA * I))
        return X[self.out_idx].T


class FlyAdapter:
    """读出储备池状态，给语言模型的 logits 加一个**有界**修正。

    有界很重要：修正量经过 tanh 再乘一个小系数，所以它只能微调语言模型的
    选择，不可能凭空制造语言能力。去掉连接组，这个修正就精确地消失。
    """

    SCALE = 0.5          # 修正的最大幅度（logit 单位）

    def __init__(self, n_probe: int, vocab: int, hidden: int = 64,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.A = (rng.normal(0, 0.05, (n_probe, hidden))).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.B = (rng.normal(0, 0.02, (hidden, vocab))).astype(np.float32)
        self.n_params = self.A.size + self.b1.size + self.B.size

    def forward(self, state: np.ndarray) -> np.ndarray:
        h = np.tanh(state @ self.A + self.b1)
        return self.SCALE * np.tanh(h @ self.B)

    def save(self, path) -> None:
        np.savez_compressed(path, A=self.A, b1=self.b1, B=self.B)

    def load(self, path) -> None:
        d = np.load(path)
        self.A, self.b1, self.B = d["A"], d["b1"], d["B"]


def token_drive(token_ids, embed: np.ndarray, in_proj: np.ndarray
                ) -> np.ndarray:
    """把一段 token 的嵌入压成储备池的输入向量。"""
    e = embed[np.asarray(token_ids)]
    # 近期的 token 权重更大 —— 简单的指数加权
    w = np.exp(np.linspace(-2.0, 0.0, len(e))).astype(np.float32)
    v = (e * w[:, None]).sum(0) / max(w.sum(), 1e-9)
    return np.tanh(v @ in_proj)


def load_lm(device: str = "cuda"):
    """加载冻结的语言模型。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not (LFM_DIR / "model.safetensors").exists():
        raise RuntimeError(
            f"语言模型没下载。先跑：python cli.py flm-download")
    tok = AutoTokenizer.from_pretrained(str(LFM_DIR))
    dev = device if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        str(LFM_DIR), dtype=torch.float32).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model, dev


def download(verbose: bool = True) -> None:
    """从 hf-mirror 下载冻结的语言模型（huggingface.co 在本机不可达）。"""
    import subprocess

    LFM_DIR.mkdir(parents=True, exist_ok=True)
    base = ("https://hf-mirror.com/LiquidAI/LFM2.5-1.2B-Instruct"
            "/resolve/main")
    for f in ("config.json", "generation_config.json", "tokenizer.json",
              "tokenizer_config.json", "special_tokens_map.json",
              "model.safetensors"):
        out = LFM_DIR / f
        if out.exists() and out.stat().st_size > 0:
            if verbose:
                print(f"  已有 {f}")
            continue
        if verbose:
            print(f"  下载 {f} ...")
        # -4 是必须的：本机 IPv6 出站会在 TLS 阶段断掉
        subprocess.run(["curl", "-sL", "-4", "--http1.1", "-C", "-",
                        "--retry", "5", "-o", str(out), f"{base}/{f}"],
                       check=True)
    if verbose:
        print(f"[FLM] 语言模型就绪 -> {LFM_DIR}")


def chat(prompt: str, max_new: int = 60, device: str = "cuda",
         use_fly: bool = True, verbose: bool = True) -> str:
    """用（可选的）果蝇修正生成一段回复。

    **说清楚：语言能力全部来自那个预训练模型。** 果蝇适配器只给 logits 加一个
    幅度受限的修正（最大 ±0.5），而且实测它并不比打乱拓扑的对照更好。这是个
    演示，不是"果蝇会说话"。
    """
    import torch

    tok, model, dev = load_lm(device)
    ids = tok(prompt, return_tensors="pt").to(dev)
    adapter = res = proj = None
    if use_fly and FLM_NPZ.exists():
        from .connectome import load
        d = np.load(FLM_NPZ)
        c = load()
        res = Reservoir(c, n_probe=int(d["A"].shape[0]), verbose=False)
        adapter = FlyAdapter(int(d["A"].shape[0]), int(d["B"].shape[1]),
                             hidden=int(d["A"].shape[1]))
        adapter.A, adapter.b1, adapter.B = d["A"], d["b1"], d["B"]
        proj = d["proj"]
        if verbose:
            print(f"[FLM] 果蝇适配器已载入（{adapter.n_params:,} 参数）")
    elif use_fly and verbose:
        print("[FLM] 还没有训练好的适配器，先跑 python cli.py sim --exp flm")

    embed = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    cur = ids["input_ids"]
    for _ in range(max_new):
        with torch.no_grad():
            lg = model(cur).logits[0, -1].float()
        if adapter is not None:
            ctx = cur[0, -5:].cpu().numpy()
            e = embed[ctx]
            w = np.exp(np.linspace(-2.0, 0.0, len(e))).astype(np.float32)
            drive = np.tanh(((e * w[:, None]).sum(0) / w.sum()) @ proj)
            corr = adapter.forward(res.run(drive))
            lg = lg + torch.tensor(corr, device=lg.device)
        nxt = int(lg.argmax())
        cur = torch.cat([cur, torch.tensor([[nxt]], device=cur.device)], 1)
        if nxt == tok.eos_token_id:
            break
    return tok.decode(cur[0], skip_special_tokens=True)
