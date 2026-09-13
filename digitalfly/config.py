"""全局路径与数据源配置。"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DIGITALFLY_DATA", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
BUILD_DIR = DATA_DIR / "build"
OUTPUT_DIR = Path(os.environ.get("DIGITALFLY_OUT", ROOT / "outputs"))
FLYBODY_DIR = ROOT / "third_party" / "flybody"

# --- MaleCNS v1.0 连接组 -------------------------------------------------
# HHMI Janelia FlyEM + University of Cambridge + Google Research, Cell (2026)
# 数据集 v1.0 发布于 2026-06-08，CC-BY 4.0。
# 主页 male-cns.janelia.org 只有 IPv6 记录，这里直接用其 Google Storage 对象 URL。
DATASET = "male-cns:v1.0"
GCS_BASE = ("https://storage.googleapis.com/flyem-male-cns/v1.0"
            "/connectome-data/flat-connectome")
NEUPRINT_SERVER = "https://neuprint.janelia.org"

# name -> (远端文件名, 官方公布大小的字节数下界, 说明)
# 大小用作完整性下界校验；官方页面给的是四舍五入后的 MB/GB。
FILES = {
    "weights": (
        "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        1_000_000_000,
        "神经元对之间的连接权重（突触计数）—— 全脑网络的骨架",
    ),
    "annotations": (
        "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        10_000_000,
        "神经元标注：细胞类型 / class / 侧别 / 所属脑区",
    ),
    "neurotransmitters": (
        "body-neurotransmitters-male-cns-v1.0.feather",
        35_000_000,
        "每个神经元的递质预测 —— 决定突触是兴奋还是抑制",
    ),
}

# 论文报告的规模，用于自检对账
EXPECTED_NEURONS = 176_422

GRAPH_NPZ = BUILD_DIR / "graph.npz"
META_PARQUET = BUILD_DIR / "meta.parquet"


def ensure_dirs() -> None:
    for d in (RAW_DIR, BUILD_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def raw_path(name: str) -> Path:
    return RAW_DIR / FILES[name][0]


def url_for(name: str) -> str:
    return f"{GCS_BASE}/{FILES[name][0]}"
