"""全脑三维点云渲染：神经元发放时闪烁。

网上那批视频的画面都是同一个格式 —— 分屏，一边是任务，一边是实时的神经活动
读出（3D 神经元渲染，发放的细胞闪一下，标上神经元群的名字）。这个模块负责
右半边那一块。

点云不是编出来的：MaleCNS v1.0 的标注表里有 14.2 万个神经元的胞体三维坐标
(`somaLocation`)，这里直接用它们，按脑区着色。

实现上不走 matplotlib —— 每帧都要画十几万个点，必须快。做法是预先把三维坐标
投影成二维像素坐标，每帧只做一次 `np.add.at` 累加，再按亮度上色。
一帧大约 3 毫秒。
"""
from __future__ import annotations

import numpy as np

# 脑区配色。与 Web 控制台的分类色槽同源（暗色版，已过 CVD 校验）。
REGION_COLORS = {
    "视叶 Optic lobe": (0.22, 0.53, 0.90),
    "中央脑 Central brain": (0.85, 0.35, 0.15),
    "腹神经索 VNC": (0.10, 0.62, 0.44),
    "下行 Descending": (0.98, 0.70, 0.10),
    "上行 Ascending": (0.84, 0.32, 0.51),
    "其他 Unassigned": (0.45, 0.45, 0.43),
}
BG = np.array([0.04, 0.04, 0.045])

# 中文标签要用 CJK 字体，PIL 的默认位图字体画中文是一串方块
_FONT_CANDIDATES = (
    "/usr/local/share/fonts/NotoSansCJKSC-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
_FONTS: dict = {}


def font(size: int = 13):
    """按字号取字体，取不到就退回 PIL 自带的位图字体。"""
    if size in _FONTS:
        return _FONTS[size]
    from PIL import ImageFont
    for path in _FONT_CANDIDATES:
        try:
            _FONTS[size] = ImageFont.truetype(path, size)
            return _FONTS[size]
        except Exception:                                # noqa: BLE001
            continue
    _FONTS[size] = ImageFont.load_default()
    return _FONTS[size]


def _rot(yaw: float, pitch: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    return np.array([[cy, 0, sy],
                     [sy * sp, cp, -cy * sp],
                     [-sy * cp, sp, cy * cp]])


class BrainView:
    """把全脑脉冲活动画成一张三维点云图。"""

    def __init__(self, meta, regions: dict, size=(360, 480),
                 yaw: float = 0.0, pitch: float = 0.9,
                 decay_ms: float = 60.0, max_points: int = 60000,
                 seed: int = 0):
        self.h, self.w = size
        self.decay_ms = decay_ms

        xyz = meta[["soma_x", "soma_y", "soma_z"]].to_numpy(dtype=np.float64)
        has = np.isfinite(xyz).all(axis=1)
        idx = np.flatnonzero(has)
        # 神经元太多就抽样，保证渲染速度
        if len(idx) > max_points:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(idx, size=max_points, replace=False))
        self.idx = idx
        pts = xyz[idx]

        # 归一化到单位立方体再投影
        c = pts.mean(axis=0)
        scale = np.abs(pts - c).max()
        p = (pts - c) / scale
        self.project(yaw, pitch, p)

        # 每个点按脑区着色
        color = np.tile(np.array(REGION_COLORS["其他 Unassigned"]),
                        (len(idx), 1))
        pos_in_sample = {v: i for i, v in enumerate(idx)}
        self.group_rows = {}
        for name, members in regions.items():
            rows = [pos_in_sample[m] for m in members if m in pos_in_sample]
            if not rows:
                continue
            rows = np.asarray(rows, dtype=np.int64)
            self.group_rows[name] = rows
            color[rows] = REGION_COLORS.get(
                name, REGION_COLORS["其他 Unassigned"])
        self.color = color.astype(np.float32)
        self.heat = np.zeros(len(idx), dtype=np.float32)

        self._p = p          # 留着，换视角时重投影
        self.labels: list[tuple[str, tuple[int, int], tuple]] = []
        self._make_labels(regions)

    def project(self, yaw: float, pitch: float, p=None) -> None:
        p = self._p if p is None else p
        q = p @ _rot(yaw, pitch).T
        # 自动适配：按投影后的实际包围盒缩放，保证整个中枢神经系统都在画面里。
        # 用固定放大系数会把腹神经索切出去 —— 脑 + 腹神经索是细长的。
        x, z = q[:, 0], q[:, 2]
        cx, cz = (x.min() + x.max()) / 2, (z.min() + z.max()) / 2
        span = max(float(np.ptp(x)), float(np.ptp(z))) or 1.0
        k = 0.86 / span
        u = ((x - cx) * k + 0.5) * (self.w - 1)
        v = (1 - ((z - cz) * k + 0.5)) * (self.h - 1)
        self.px = np.clip(u, 0, self.w - 1).astype(np.int32)
        self.py = np.clip(v, 0, self.h - 1).astype(np.int32)
        self.flat = self.py * self.w + self.px
        # 远近：用深度做一点明暗，让点云有立体感
        d = q[:, 1]
        self.depth = ((d - d.min()) / max(float(np.ptp(d)), 1e-9) * 0.55 + 0.45
                      ).astype(np.float32)
        self.yaw, self.pitch = yaw, pitch

    def _make_labels(self, regions) -> None:
        """每个脑区的标签锚点（取该区点云的中位位置），并避免叠在一起。"""
        raw = []
        for name, rows in self.group_rows.items():
            if len(rows) < 50:
                continue
            raw.append((name, int(np.median(self.px[rows])),
                        int(np.median(self.py[rows]))))
        # 按纵坐标排开，靠得太近就往下推，免得几个标签糊成一团
        raw.sort(key=lambda t: t[2])
        min_gap, last = 17, -1e9
        self.labels = []
        for name, x, y in raw:
            y = max(y, last + min_gap)
            last = y
            self.labels.append((name, (x, int(np.clip(y, 8, self.h - 10))),
                                REGION_COLORS.get(
                                    name, REGION_COLORS["其他 Unassigned"])))

    # -- 每步更新 -----------------------------------------------------------
    def update(self, spikes: np.ndarray, dt_ms: float) -> None:
        """把这一步的脉冲累到热度上，并按时间常数衰减。"""
        self.heat *= np.exp(-dt_ms / self.decay_ms)
        s = spikes[self.idx]
        nz = s > 0
        if nz.any():
            self.heat[nz] += 1.0

    def render(self, spin: float = 0.0) -> np.ndarray:
        """画出一帧 (h, w, 3) uint8。"""
        if spin:
            self.project(self.yaw + spin, self.pitch)

        img = np.tile(BG, (self.h * self.w, 1)).astype(np.float32)
        # 基底：所有神经元都画出来，暗淡，按深度分层
        base = self.color * (0.16 * self.depth)[:, None]
        np.add.at(img, self.flat, base)
        # 发放的神经元：亮起来，越近期越白
        hot = np.clip(self.heat, 0, 1.5)[:, None]
        live = self.color * hot * 1.6 + hot * 0.55
        np.add.at(img, self.flat, live)

        img = img.reshape(self.h, self.w, 3)
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def draw_overlay(img: np.ndarray, title: str, lines,
                 labels=None, title_size: int = 15,
                 line_size: int = 13) -> np.ndarray:
    """在图上叠加标题、若干行文字和脑区标签。"""
    from PIL import Image, ImageDraw

    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    if labels:
        for name, (x, y), rgb in labels:
            col = tuple(int(min(1.0, c * 1.3) * 255) for c in rgb)
            d.ellipse([x - 3, y - 3, x + 3, y + 3], outline=col, width=1)
            d.text((x + 8, y - 7), name, fill=col, font=font(line_size - 1))
    d.text((12, 10), title, fill=(240, 240, 235), font=font(title_size))
    for i, t in enumerate(lines):
        d.text((12, 12 + title_size + 6 + i * (line_size + 4)), t,
               fill=(178, 178, 170), font=font(line_size))
    return np.asarray(im)


def side_by_side(left: np.ndarray, right: np.ndarray,
                 gap_px: int = 4) -> np.ndarray:
    """把任务画面和脑活动画面拼成分屏。

    输出的宽高都补成偶数：libx264 的 yuv420p 要求偶数尺寸，
    奇数宽会让整个编码失败（写出一个 0 字节的 mp4）。
    """
    h = max(left.shape[0], right.shape[0])

    def pad_h(a):
        if a.shape[0] == h:
            return a
        out = np.zeros((h, a.shape[1], 3), dtype=a.dtype)
        out[:a.shape[0]] = a
        return out

    gap = np.full((h, gap_px, 3), 30, dtype=np.uint8)
    img = np.concatenate([pad_h(left), gap, pad_h(right)], axis=1)
    ph, pw = img.shape[0] % 2, img.shape[1] % 2
    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)))
    return img
