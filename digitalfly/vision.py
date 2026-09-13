"""视觉输入：把图像呈现到果蝇的复眼上。

**呈现的位置是有解剖依据的。** MaleCNS v1.0 的标注表给了 23,720 个视叶柱状
神经元的六角视柱坐标 (`assignedOlHex1/2`) —— 每只眼约 880 个视柱，这就是
数据集自带的视网膜拓扑图。图像不是随便投影进网络的，而是按这套坐标一柱一柱
铺上去的。

驱动的是 **L1 和 L2**：它们是光感受器在板层的两个主要突触后靶标，分别对应
ON 和 OFF 通道（Joesch et al., Nature 2010）。光感受器本身在数据集里没有
视柱坐标 —— 它们的胞体在视网膜里，不在成像体积内，6,098 个里只有 28 个
有坐标 —— 所以从板层这一层进比从光感受器进更实在。

    图像像素 -> 每个视柱的亮度 -> L1（ON）/ L2（OFF）的目标发放率

六角坐标是轴向坐标 (axial)，转成平面直角坐标才能和图像对齐：
    x = h1 + h2 / 2
    y = h2 * sqrt(3) / 2
"""
from __future__ import annotations

import numpy as np

# 板层的两个主要通道。L1 = ON（亮增强），L2 = OFF（暗增强）。
ON_TYPES = ("L1",)
OFF_TYPES = ("L2",)
# 需要更强的输入时可以把 L3/L5 也算上
EXTRA_TYPES = ("L3", "L5")


class Retina:
    """果蝇复眼的一侧：视柱坐标 -> 图像像素 -> L1/L2 的目标发放率。"""

    def __init__(self, c, side: str = "R", size: int = 28,
                 use_extra: bool = False):
        from .neurons import NeuronIndex

        self.c = c
        self.side = side
        self.size = size
        meta = c.meta
        if "hex1" not in meta.columns:
            raise RuntimeError(
                "meta 里没有视柱坐标，请重新运行 python cli.py build")

        # 包围盒必须先算：L1 和 L2 要落在同一套像素网格上
        self._x0, self._xs, self._y0, self._ys = _fit_bounds(c)

        idx = NeuronIndex(c)
        sides = idx.side().to_numpy()
        has_hex = meta["hex1"].notna().to_numpy()

        on = list(ON_TYPES) + (list(EXTRA_TYPES) if use_extra else [])
        types = meta["type"].astype(str).to_numpy()
        sel_on = has_hex & (sides == side) & np.isin(types, on)
        sel_off = has_hex & (sides == side) & np.isin(types, list(OFF_TYPES))

        self.on_idx = np.flatnonzero(sel_on)
        self.off_idx = np.flatnonzero(sel_off)
        if not len(self.on_idx) or not len(self.off_idx):
            raise RuntimeError(f"{side} 眼没找到 L1/L2 柱状神经元")

        h1 = meta["hex1"].to_numpy(dtype=float)
        h2 = meta["hex2"].to_numpy(dtype=float)
        self.on_px = self._pixel_of(h1[self.on_idx], h2[self.on_idx])
        self.off_px = self._pixel_of(h1[self.off_idx], h2[self.off_idx])

        self.n_columns = len(
            {(a, b) for a, b in zip(h1[sel_on | sel_off],
                                    h2[sel_on | sel_off])})

    def _pixel_of(self, h1, h2) -> np.ndarray:
        """六角轴向坐标 -> 图像像素下标。"""
        x = h1 + h2 / 2.0
        y = h2 * np.sqrt(3) / 2.0
        # 归一化到 [0, size-1]，用全体视柱的包围盒（不是当前子集），
        # 保证 L1 和 L2 落在同一套像素网格上
        gx = (x - self._x0) / max(self._xs, 1e-9)
        gy = (y - self._y0) / max(self._ys, 1e-9)
        col = np.clip((gx * (self.size - 1)).round(), 0, self.size - 1)
        row = np.clip(((1 - gy) * (self.size - 1)).round(), 0, self.size - 1)
        return (row * self.size + col).astype(np.int64)

    def present(self, image: np.ndarray, ext: np.ndarray,
                bridge, peak_hz: float = 90.0) -> np.ndarray:
        """把一张灰度图（size x size，取值 [0,1]）呈现到这只眼上。

        L1 编码亮度，L2 编码暗度 —— 对应真实果蝇的 ON / OFF 通道。
        """
        img = np.asarray(image, dtype=np.float32).ravel()
        bright = np.clip(img, 0, 1)
        dark = 1.0 - bright
        for idx, px, drive in ((self.on_idx, self.on_px, bright),
                               (self.off_idx, self.off_px, dark)):
            rates = drive[px] * peak_hz
            hit = rates > 0
            if hit.any():
                bridge._drive_rates(ext, idx[hit], rates[hit])
        return ext


def _fit_bounds(c) -> tuple:
    """全体视柱在平面上的包围盒。"""
    m = c.meta
    ok = m["hex1"].notna().to_numpy()
    h1 = m["hex1"].to_numpy(dtype=float)[ok]
    h2 = m["hex2"].to_numpy(dtype=float)[ok]
    x = h1 + h2 / 2.0
    y = h2 * np.sqrt(3) / 2.0
    return float(x.min()), float(np.ptp(x)), float(y.min()), float(np.ptp(y))


# --------------------------------------------------------------------------
# 刺激：数字与字母
# --------------------------------------------------------------------------

DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

_FONT_PATHS = (
    "/usr/local/share/fonts/NotoSansCJKSC-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def render_char(ch: str, size: int = 28, jitter=(0, 0),
                scale: float = 1.0, rotate: float = 0.0) -> np.ndarray:
    """把一个字符画成 size x size 的灰度图，取值 [0, 1]，白字黑底。"""
    from PIL import Image, ImageDraw, ImageFont

    big = size * 4
    im = Image.new("L", (big, big), 0)
    d = ImageDraw.Draw(im)
    font = None
    for p in _FONT_PATHS:
        try:
            font = ImageFont.truetype(p, int(big * 0.72 * scale))
            break
        except Exception:                                # noqa: BLE001
            continue
    if font is None:                                     # pragma: no cover
        font = ImageFont.load_default()
    box = d.textbbox((0, 0), ch, font=font)
    d.text(((big - box[2] - box[0]) / 2, (big - box[3] - box[1]) / 2),
           ch, fill=255, font=font)
    if rotate:
        im = im.rotate(rotate, resample=Image.BILINEAR, fillcolor=0)
    if any(jitter):
        im = im.transform(im.size, Image.AFFINE,
                          (1, 0, jitter[0] * 4, 0, 1, jitter[1] * 4),
                          fillcolor=0)
    im = im.resize((size, size), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0
