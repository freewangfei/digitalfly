"""训练 FLM 适配器，并和参数量匹配的对照组对比。

**对照组是这个实验的全部意义所在。** 只训一个适配器然后说"果蝇帮到了语言模型"
是没有说服力的 —— 适配器自己就有几十万参数，它当然能学到东西。要证明*连接组*
有贡献，必须有一个参数量完全相同、但把连接组换掉的对照：

  * `fly`      token 嵌入 → 连接组储备池 → 适配器 → 修正 logits
  * `shuffled` 同一张图**打乱边**之后的储备池（保留度分布，破坏拓扑）
  * `direct`   跳过储备池，token 嵌入直接进适配器（参数量匹配）

如果 fly 不明显好过 shuffled 和 direct，那就是没有证据表明果蝇的拓扑有用 ——
原作者报的就是这个结果（对照组略好）。这里如实跑、如实报。
"""
from __future__ import annotations

import time

import numpy as np

from ..flm import ALPHA, BETA, FlyAdapter, Reservoir


class ShuffledReservoir(Reservoir):
    """把边打乱的对照储备池：保留每个节点的入度，破坏具体连接。"""

    def __init__(self, c, n_probe: int = 2048, seed: int = 0,
                 verbose: bool = True):
        super().__init__(c, n_probe=n_probe, seed=seed, verbose=False)
        rng = np.random.default_rng(seed + 99)
        W = self.W.tocoo()
        # 打乱突触前，保持每行（突触后）的边数和权重不变
        cols = rng.permutation(W.col)
        import scipy.sparse as sp
        self.W = sp.csr_matrix((W.data, (W.row, cols)), shape=W.shape)
        if verbose:
            print(f"[FLM] 打乱对照：同样 {self.W.nnz:,} 条边，拓扑已破坏")


def build_corpus(tok, n: int = 220, seed: int = 0) -> list:
    """一批短句。每条切成 (前文, 下一个 token) 的预测样本。"""
    rng = np.random.default_rng(seed)
    subjects = ["The fly", "A neuron", "The brain", "This circuit",
                "The connectome", "A synapse", "The retina", "One cell"]
    verbs = ["responds to", "encodes", "connects to", "inhibits",
             "drives", "integrates", "projects to", "learns from"]
    objects = ["light", "odour", "movement", "sugar", "the mushroom body",
               "descending neurons", "the optic lobe", "a bitter taste"]
    tails = ["during flight.", "in the dark.", "after training.",
             "over many trials.", "at rest.", "when hungry."]
    out = []
    for _ in range(n):
        s = (f"{rng.choice(subjects)} {rng.choice(verbs)} "
             f"{rng.choice(objects)} {rng.choice(tails)}")
        ids = tok(s, return_tensors=None)["input_ids"]
        if len(ids) >= 6:
            out.append(ids)
    return out


def _samples(corpus, ctx: int = 5):
    """(前文 token 列表, 目标 token) 对。"""
    for ids in corpus:
        for i in range(ctx, len(ids)):
            yield ids[i - ctx:i], ids[i]


def run(c, mode: str = "all", epochs: int = 6, n_probe: int = 1024,
        hidden: int = 48, lr: float = 0.05, device: str = "cuda",
        verbose: bool = True) -> dict:
    import torch

    from ..flm import load_lm

    tok, model, dev = load_lm(device)
    embed = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    vocab = embed.shape[0]
    corpus = build_corpus(tok)
    pairs = list(_samples(corpus))
    rng = np.random.default_rng(0)
    rng.shuffle(pairs)
    cut = int(0.75 * len(pairs))
    train, test = pairs[:cut], pairs[cut:]
    if verbose:
        print(f"[FLM] 语料 {len(corpus)} 句 -> 样本 {len(pairs)} 条"
              f"（训练 {len(train)} / 测试 {len(test)}）")

    # 语言模型对每条样本的基线 logits，只算一次
    def base_logits(ps):
        outs = []
        for i in range(0, len(ps), 32):
            batch = [p[0] for p in ps[i:i + 32]]
            t = torch.tensor(batch, device=dev)
            with torch.no_grad():
                lg = model(t).logits[:, -1, :].float().cpu().numpy()
            outs.append(lg)
        return np.concatenate(outs)

    t0 = time.time()
    L_tr, L_te = base_logits(train), base_logits(test)
    if verbose:
        print(f"[FLM] 基线 logits 算完 {time.time() - t0:.0f}s")

    # 固定的输入投影：嵌入维 -> 探针数
    proj = rng.normal(0, 1.0 / np.sqrt(embed.shape[1]),
                      (embed.shape[1], n_probe)).astype(np.float32)

    def drives(ps):
        D = np.zeros((len(ps), n_probe), dtype=np.float32)
        for k, (ids, _) in enumerate(ps):
            e = embed[np.asarray(ids)]
            w = np.exp(np.linspace(-2.0, 0.0, len(e))).astype(np.float32)
            D[k] = np.tanh(((e * w[:, None]).sum(0) / w.sum()) @ proj)
        return D

    D_tr, D_te = drives(train), drives(test)
    y_tr = np.array([p[1] for p in train])
    y_te = np.array([p[1] for p in test])

    modes = (["fly", "shuffled", "direct"] if mode == "all" else [mode])
    results = {}
    for m in modes:
        if m == "direct":
            S_tr, S_te = D_tr, D_te          # 跳过储备池，参数量相同
        else:
            R = (Reservoir(c, n_probe=n_probe, verbose=verbose) if m == "fly"
                 else ShuffledReservoir(c, n_probe=n_probe, verbose=verbose))
            t1 = time.time()
            S_tr, S_te = R.run_batch(D_tr), R.run_batch(D_te)
            if verbose:
                print(f"[FLM] {m} 储备池跑完 {time.time() - t1:.0f}s")
        results[m] = _fit(S_tr, y_tr, L_tr, S_te, y_te, L_te, vocab,
                          hidden, epochs, lr, m, verbose)
        if m == "fly":
            from ..flm import FLM_NPZ
            FLM_NPZ.parent.mkdir(parents=True, exist_ok=True)
            w = results[m].pop("weights")
            np.savez_compressed(FLM_NPZ, A=w[0], b1=w[1], B=w[2], proj=proj)
            if verbose:
                print(f"[FLM] 适配器已保存 -> {FLM_NPZ}")

    base = _nll(L_te, y_te)
    if verbose:
        print(f"\n  语言模型基线 NLL {base:.4f}")
        for m in modes:
            r = results[m]
            print(f"  {m:9s} NLL {r['nll']:.4f}  改善 {r['gain']:+.4f}  "
                  f"参数 {r['n_params']:,}")
        if "fly" in results and "direct" in results:
            d = results["fly"]["gain"] - results["direct"]["gain"]
            print(f"\n  果蝇 vs 直接输入对照：{d:+.4f}")
            print("  " + ("果蝇拓扑更好" if d > 0.002 else
                          "没有证据表明果蝇拓扑带来优势 —— 和原作者的结论一致"))
    results["baseline_nll"] = base
    return results


def _nll(logits, y) -> float:
    z = logits - logits.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=1))
    return float(-(z[np.arange(len(y)), y] - lse).mean())


def _fit(S_tr, y_tr, L_tr, S_te, y_te, L_te, vocab, hidden, epochs, lr,
         name, verbose):
    """只训适配器。语言模型和储备池全程不动。"""
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ad = FlyAdapter(S_tr.shape[1], vocab, hidden=hidden)
    A = torch.tensor(ad.A, device=dev, requires_grad=True)
    b1 = torch.tensor(ad.b1, device=dev, requires_grad=True)
    B = torch.tensor(ad.B, device=dev, requires_grad=True)
    opt = torch.optim.Adam([A, b1, B], lr=lr)
    St = torch.tensor(S_tr, device=dev)
    Lt = torch.tensor(L_tr, device=dev)
    yt = torch.tensor(y_tr, device=dev, dtype=torch.long)
    Se = torch.tensor(S_te, device=dev)
    Le = torch.tensor(L_te, device=dev)
    ye = torch.tensor(y_te, device=dev, dtype=torch.long)

    def fwd(S, L):
        h = torch.tanh(S @ A + b1)
        return L + FlyAdapter.SCALE * torch.tanh(h @ B)

    best = float("inf")
    for ep in range(epochs):
        for i in range(0, len(St), 64):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(
                fwd(St[i:i + 64], Lt[i:i + 64]), yt[i:i + 64])
            loss.backward()
            opt.step()
        with torch.no_grad():
            te = float(torch.nn.functional.cross_entropy(fwd(Se, Le), ye))
        best = min(best, te)
        if verbose:
            print(f"    {name} 第 {ep + 1} 轮 测试 NLL {te:.4f}")
    return {"nll": best, "gain": _nll(L_te, y_te) - best,
            "n_params": ad.n_params,
            "weights": (A.detach().cpu().numpy(), b1.detach().cpu().numpy(),
                        B.detach().cpu().numpy())}
