"""可视化：脉冲栅格图、脑区活动热图、行为视频。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 中文标签在无字体的服务器上会变豆腐块，统一用英文出图
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def raster_plot(results: dict, out: Path | str | None = None,
                stim_window: tuple[float, float] = (200, 800),
                dt: float = 0.5) -> Path:
    """糖味 / 苦味 / 空白三组的 MN9 栅格图 + 全网活动曲线。"""
    plt = _mpl()
    names = [n for n in ("sugar", "bitter", "grn", "none") if n in results]
    # 底排是同一个量（全网发放数），必须共用一把尺子 —— 各自缩放会把
    # "苦味几乎什么都没发生"画得和"糖味大量放电"一样高。
    fig, axes = plt.subplots(2, len(names), figsize=(4.0 * len(names), 5.2),
                             sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": [1, 1.3]})
    for ax in axes[1, 1:]:
        ax.sharey(axes[1, 0])
    titles = {"sugar": "Sugar SEL", "bitter": "Bitter SEL",
              "grn": "Labellar GRNs", "none": "No stimulus"}
    # 分类色槽：正/负对照用可区分的两色，其余用中性
    colors = {"sugar": "#199e70", "bitter": "#d95926",
              "grn": "#3987e5", "none": "#898781"}

    for j, name in enumerate(names):
        r = results[name]
        raster = np.asarray(r["raster"])          # (T, k)
        t = np.arange(raster.shape[0]) * dt

        ax = axes[0, j]
        for k in range(raster.shape[1]):
            ts = t[raster[:, k] > 0]
            ax.vlines(ts, k + 0.1, k + 0.9, color=colors[name], lw=1.2)
        ax.axvspan(*stim_window, color="#FFC10733", zorder=0)
        ax.set_ylim(0, max(raster.shape[1], 1))
        ax.set_yticks([k + 0.5 for k in range(raster.shape[1])])
        ax.set_yticklabels([f"MN9-{k + 1}" for k in range(raster.shape[1])],
                           fontsize=9)
        ax.tick_params(axis="y", length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(f"{titles[name]}\nMN9 {r['mn9_stim_hz']:.1f} Hz during stim",
                     fontsize=10)
        if j == 0:
            ax.set_ylabel("proboscis extension\nmotor neurons", fontsize=9)

        ax2 = axes[1, j]
        trace = np.asarray(r["total_trace"])
        ax2.plot(t, trace, color=colors[name], lw=0.7)
        ax2.axvspan(*stim_window, color="#FFC10733", zorder=0)
        ax2.set_xlabel("time (ms)")
        for side in ("top", "right"):
            ax2.spines[side].set_visible(False)
        if j == 0:
            ax2.set_ylabel("whole-brain spikes / step")
        else:
            ax2.tick_params(labelleft=False)

    fig.suptitle("Digital fly — connectome-driven taste response "
                 "(MaleCNS v1.0)", fontsize=12)
    fig.tight_layout()
    out = Path(out or config.OUTPUT_DIR / "sugar_pe_raster.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def region_activity_plot(rates_by_group: dict[str, np.ndarray],
                         out: Path | str | None = None,
                         title: str = "Whole-brain activity by region") -> Path:
    """各解剖区域的平均发放率条形图。"""
    plt = _mpl()
    names = list(rates_by_group)
    vals = [float(np.mean(v)) if len(v) else 0.0
            for v in rates_by_group.values()]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(range(len(names)), vals, color="#1565C0")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("mean firing rate (Hz)")
    ax.set_title(title, fontsize=11)
    ax.invert_yaxis()
    fig.tight_layout()
    out = Path(out or config.OUTPUT_DIR / "region_activity.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def write_video(frames: list[np.ndarray], out: Path | str,
                fps: int = 30) -> Path:
    """把渲染帧写成 mp4。"""
    import imageio.v2 as imageio
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, frames, fps=fps, quality=8,
                     macro_block_size=1)
    return out
