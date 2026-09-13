"""把 MaleCNS v1.0 的扁平连接组表构建成一张带符号的稀疏突触矩阵。

流程:
    body-annotations.feather      神经元身份（类型 / class / 侧别 / 脑区）
    body-neurotransmitters.feather 每个神经元的递质预测 -> 突触符号
    connectome-weights.feather    (pre, post, weight) 边表 -> 稀疏矩阵
                 |
                 v
    data/build/graph.npz  +  data/build/meta.parquet   (一次构建，秒级加载)

符号约定遵循果蝇电生理的主流结论：
    ACh (乙酰胆碱)   -> 兴奋  (+1)
    GABA            -> 抑制  (-1)
    Glu (谷氨酸)     -> 抑制  (-1)   果蝇中谷氨酸主要作用于 GluCl 氯通道
    组胺             -> 抑制  (-1)   光感受器递质，作用于 HisCl 氯通道
    DA / 5-HT / OA   -> 兴奋  (+1)   见下
    预测不可靠的      -> 兴奋  (+1)   果蝇中枢以胆碱能为主

单胺类（多巴胺 / 血清素 / 章胺）为什么按兴奋处理：它们主要通过 G 蛋白偶联受体
起调质作用，严格说不是快速突触传递。但如果把它们的权重置零，等于把这些神经元
从网络里整个删掉 —— 本数据集里 545 个神经元会因此完全失去输出，其中就包括
Yao & Scott (2022) 鉴定的全部糖味和苦味通路神经元（它们的递质预测是血清素）。
连接组仿真的通行做法是把非 GABA / 非谷氨酸的一律当兴奋处理，这里沿用该做法。
想验证单胺类的影响，可以用 modulator_sign=0.0 重建并对比。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import scipy.sparse as sp

from . import config

# 递质 -> 符号
NT_SIGN = {
    "acetylcholine": 1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "histamine": -1.0,      # 光感受器用组胺，作用于氯通道，抑制性
}
# 数据里表示"预测不可靠"的标签
UNKNOWN_LABELS = {"unclear", "unknown", "none", "nan", "", "null"}
MODULATORS = {"dopamine", "serotonin", "octopamine"}

# 未知递质的兜底符号。连接组里有一部分神经元没有可靠递质预测，
# 全部丢弃会把网络打散，这里按果蝇中枢胆碱能占多数的事实给正号，
# 并在统计里单独报出这部分占比。
UNKNOWN_SIGN = 1.0


def _pick(cols: list[str], *candidates: str) -> str:
    """在列名里挑第一个命中的候选，找不到就报错并列出可选项。"""
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    raise KeyError(f"未找到列 {candidates}，现有列: {cols}")


@dataclass
class Connectome:
    """全脑突触网络。

    W: CSR 稀疏矩阵，W[post, pre] = 带符号的突触计数。
       按 post 行存储，因为仿真里每步要算 W @ spikes。
    meta: 每个神经元一行，索引即矩阵下标。
    """
    W: sp.csr_matrix
    meta: pd.DataFrame
    stats: dict

    @property
    def n(self) -> int:
        return self.W.shape[0]

    def index_of(self, body_ids) -> np.ndarray:
        """bodyId -> 矩阵下标（不在网络里的会被丢弃）。"""
        s = self.meta.reset_index().set_index("bodyId")["index"]
        ids = pd.Index(np.asarray(body_ids, dtype=np.int64))
        hit = ids.intersection(s.index)
        return s.loc[hit].to_numpy()


# --------------------------------------------------------------------------
# 构建
# --------------------------------------------------------------------------

def build(modulator_sign: float = UNKNOWN_SIGN,
          verbose: bool = True) -> Connectome:
    """构建全脑带符号突触矩阵。

    modulator_sign: 单胺类神经元（DA / 5-HT / OA）突触的符号。
        默认 +1（当兴奋处理）；设为 0.0 可以把它们从网络里摘掉做对照实验。
    """
    t0 = time.time()
    log = print if verbose else (lambda *a, **k: None)

    # --- 1. 标注 ---------------------------------------------------------
    ann_cols = ["bodyId", "type", "class", "subclass", "superclass",
                "somaSide", "rootSide", "somaNeuromere", "entryNerve",
                "exitNerve", "receptorType", "status", "instance",
                # synonyms 收录了文献里已鉴定神经元的身份（如 Yao & Scott 2022
                # 的糖味/苦味通路），是定位味觉通路的依据
                "synonyms", "flywireType", "hemibrainType",
                # 胞体的三维坐标（14.2 万个神经元有），用来画全脑点云
                "somaLocation"]
    ann = pd.read_feather(config.raw_path("annotations"), columns=ann_cols)
    log(f"  标注表: {len(ann):,} 行")

    # --- 2. 递质 ---------------------------------------------------------
    nt_tbl = feather.read_table(config.raw_path("neurotransmitters"))
    nt_cols = list(nt_tbl.schema.names)
    nt_body = _pick(nt_cols, "bodyId", "body_id", "bodyid", "body")
    nt_pred = _pick(nt_cols, "consensus_nt", "consensusNt", "predicted_nt",
                    "predictedNt", "nt")
    want = [nt_body, nt_pred]
    # ground_truth 是实验确证的递质，比预测可靠，有就优先用
    has_gt = "ground_truth" in nt_cols
    if has_gt:
        want.append("ground_truth")
    nt = nt_tbl.select(want).to_pandas()
    del nt_tbl
    nt = nt.rename(columns={nt_body: "bodyId", nt_pred: "nt"})
    if has_gt:
        gt = nt["ground_truth"]
        nt["nt"] = gt.where(gt.notna() & (gt.astype(str) != ""), nt["nt"])
        n_gt = int(gt.notna().sum())
        nt = nt.drop(columns=["ground_truth"])
    else:
        n_gt = 0
    nt["nt"] = nt["nt"].astype(str).str.lower().str.strip()
    nt = nt.drop_duplicates(subset="bodyId")
    log(f"  递质表: {len(nt):,} 个体，{nt['nt'].nunique()} 种标签"
        f"（其中 {n_gt:,} 条有实验真值）")

    # 把 somaLocation 拆成三列，缺失的置为 NaN
    def _xyz(v, k):
        return (float(v[k]) if v is not None and len(v) == 3 else np.nan)
    loc = ann.pop("somaLocation")
    for k, axis in enumerate("xyz"):
        ann[f"soma_{axis}"] = [_xyz(v, k) for v in loc]
    log(f"  胞体坐标: {int(ann['soma_x'].notna().sum()):,} 个神经元有")

    meta = ann.merge(nt, on="bodyId", how="left")
    meta["nt"] = meta["nt"].fillna("unknown")
    meta.loc[meta["nt"].isin(UNKNOWN_LABELS), "nt"] = "unknown"

    is_modulator = meta["nt"].isin(MODULATORS)
    sign = meta["nt"].map(NT_SIGN)
    unknown_mask = sign.isna() & ~is_modulator
    sign = sign.where(~is_modulator, modulator_sign)
    sign = sign.fillna(UNKNOWN_SIGN)
    meta["sign"] = sign.astype(np.float32)
    log(f"  突触符号: 兴奋 {int((sign > 0).sum()):,} 个神经元, "
        f"抑制 {int((sign < 0).sum()):,} 个, 静默 {int((sign == 0).sum()):,} 个"
        f"  (其中单胺类 {int(is_modulator.sum()):,} 个按 {modulator_sign:+.0f} 处理,"
        f" 递质未知 {int(unknown_mask.sum()):,} 个按 {UNKNOWN_SIGN:+.0f} 处理)")

    # --- 3. 边表 ---------------------------------------------------------
    wt_path = config.raw_path("weights")
    wt_tbl = feather.read_table(wt_path)
    cols = list(wt_tbl.schema.names)
    c_pre = _pick(cols, "bodyId_pre", "bodyid_pre", "pre", "pre_id", "body_pre")
    c_post = _pick(cols, "bodyId_post", "bodyid_post", "post", "post_id", "body_post")
    c_w = _pick(cols, "weight", "count", "syn_count", "n")
    log(f"  边表: {wt_tbl.num_rows:,} 行  列 -> pre={c_pre} post={c_post} w={c_w}")

    pre = wt_tbl.column(c_pre).to_numpy().astype(np.int64)
    post = wt_tbl.column(c_post).to_numpy().astype(np.int64)
    w = wt_tbl.column(c_w).to_numpy().astype(np.float32)
    del wt_tbl

    # --- 4. 建立连续索引 ---------------------------------------------------
    # 胶质细胞不产生动作电位，不能当成 LIF 单元放进网络；
    # Unimportant 是校对流程里明确标记为"不值得纳入分析"的体，neuPrint 的
    # :Neuron 集合同样把它们排除在外。
    drop = meta["status"].isin(["Glia", "Unimportant"])
    n_glia = int((meta["status"] == "Glia").sum())
    n_unimportant = int((meta["status"] == "Unimportant").sum())
    meta = meta[~drop].copy()
    log(f"  排除胶质细胞 {n_glia:,} 个、Unimportant {n_unimportant:,} 个")

    # 只保留既有标注、又真的出现在边表里的神经元
    in_graph = pd.unique(np.concatenate([pre, post]))
    meta = meta[meta["bodyId"].isin(in_graph)].copy()
    meta = meta.drop_duplicates(subset="bodyId").sort_values("bodyId")
    meta = meta.reset_index(drop=True)
    body_ids = meta["bodyId"].to_numpy()

    lut = pd.Series(np.arange(len(body_ids), dtype=np.int32), index=body_ids)
    pre_ix = lut.reindex(pre).to_numpy()
    post_ix = lut.reindex(post).to_numpy()
    keep = ~(np.isnan(pre_ix) | np.isnan(post_ix))
    dropped = int((~keep).sum())
    pre_ix = pre_ix[keep].astype(np.int32)
    post_ix = post_ix[keep].astype(np.int32)
    w = w[keep]
    log(f"  神经元: {len(body_ids):,}   边: {len(w):,}"
        f"   (丢弃 {dropped:,} 条端点无标注的边)")

    # --- 5. 带符号稀疏矩阵 -------------------------------------------------
    signs = meta["sign"].to_numpy(np.float32)[pre_ix]
    signed_w = (w * signs).astype(np.float32)

    n = len(body_ids)
    W = sp.csr_matrix((signed_w, (post_ix, pre_ix)), shape=(n, n),
                      dtype=np.float32)
    W.sum_duplicates()

    stats = {
        "n_neurons": n,
        "n_edges": int(W.nnz),
        "n_synapses": float(np.abs(w).sum()),
        "excitatory_edges": int((W.data > 0).sum()),
        "inhibitory_edges": int((W.data < 0).sum()),
        "silent_edges": int((W.data == 0).sum()),
        "unknown_nt_neurons": int((meta["nt"] == "unknown").sum()),
        "excluded_glia": n_glia,
        "excluded_unimportant": n_unimportant,
        "status_counts": meta["status"].value_counts(dropna=False)
                             .head(8).to_dict(),
        "nt_counts": meta["nt"].value_counts().to_dict(),
        "modulator_sign": modulator_sign,
        "n_modulator_neurons": int(is_modulator.sum()),
        "build_seconds": round(time.time() - t0, 1),
        "dataset": config.DATASET,
    }
    log(f"  完成，用时 {stats['build_seconds']}s")
    return Connectome(W=W, meta=meta, stats=stats)


def save(c: Connectome) -> None:
    config.ensure_dirs()
    sp.save_npz(config.GRAPH_NPZ, c.W, compressed=False)
    c.meta.to_parquet(config.META_PARQUET)
    (config.BUILD_DIR / "stats.json").write_text(
        json.dumps(c.stats, ensure_ascii=False, indent=2))


def load() -> Connectome:
    if not config.GRAPH_NPZ.exists():
        raise FileNotFoundError(
            f"{config.GRAPH_NPZ} 不存在，请先运行: python cli.py build")
    W = sp.load_npz(config.GRAPH_NPZ).tocsr()
    meta = pd.read_parquet(config.META_PARQUET)
    stats = json.loads((config.BUILD_DIR / "stats.json").read_text())
    return Connectome(W=W, meta=meta, stats=stats)
