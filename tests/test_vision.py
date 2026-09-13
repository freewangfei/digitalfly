"""视觉输入与手写识别的测试。"""
from __future__ import annotations

import numpy as np
import pytest

from digitalfly import config

pytestmark = pytest.mark.skipif(
    not config.GRAPH_NPZ.exists(),
    reason="尚未构建连接组，先运行 python cli.py build")


@pytest.fixture(scope="module")
def c():
    from digitalfly.connectome import load
    return load()


def test_meta_has_retinotopic_columns(c):
    """视柱坐标必须在 meta 里 —— 视觉输入全靠它定位。"""
    assert {"hex1", "hex2"} <= set(c.meta.columns)
    n = int(c.meta["hex1"].notna().sum())
    assert n > 20_000, f"只有 {n} 个神经元有视柱坐标"
    cols = c.meta.dropna(subset=["hex1"]).groupby(["hex1", "hex2"]).ngroups
    assert 600 < cols < 1200, f"视柱数 {cols} 不在合理范围"


def test_retina_covers_both_eyes(c):
    """两只眼都要能建出视网膜映射，且 L1/L2 数量相当。"""
    from digitalfly.vision import Retina
    for side in ("L", "R"):
        r = Retina(c, side=side, size=24)
        assert len(r.on_idx) > 500, f"{side} 眼 L1 只有 {len(r.on_idx)} 个"
        assert len(r.off_idx) > 500
        # ON / OFF 应当一一对应到视柱，数量不该差太多
        assert abs(len(r.on_idx) - len(r.off_idx)) / len(r.on_idx) < 0.1
        assert r.n_columns > 700


def test_retina_pixels_in_range(c):
    from digitalfly.vision import Retina
    r = Retina(c, side="R", size=24)
    for px in (r.on_px, r.off_px):
        assert px.min() >= 0 and px.max() < 24 * 24


def test_retina_preserves_shape(c):
    """把字符按视柱采样再画回网格，应当还认得出是同一个字符。

    这是视觉输入正确性的核心检验：如果六角坐标到像素的映射错了，
    重建出来的图和原图就不相关了。
    """
    from digitalfly.vision import Retina, render_char
    size = 24
    r = Retina(c, side="R", size=size)

    def reconstruct(img):
        v = img.ravel()[r.on_px]
        acc = np.zeros(size * size)
        cnt = np.zeros(size * size)
        np.add.at(acc, r.on_px, v)
        np.add.at(cnt, r.on_px, 1)
        return acc / np.maximum(cnt, 1)

    same, cross = [], []
    chars = "0257AEHK"
    imgs = {ch: render_char(ch, size) for ch in chars}
    recs = {ch: reconstruct(imgs[ch]) for ch in chars}
    for a in chars:
        ia = imgs[a].ravel()[r.on_px]
        for b in chars:
            rb = recs[b][r.on_px]
            cc = float(np.corrcoef(ia, rb)[0, 1])
            (same if a == b else cross).append(cc)
    assert np.mean(same) > 0.9, f"重建相关性只有 {np.mean(same):.2f}"
    assert np.mean(same) > np.mean(cross) + 0.3, "不同字符区分不开"


def test_render_char_is_normalised():
    from digitalfly.vision import render_char
    img = render_char("8", 24)
    assert img.shape == (24, 24)
    assert 0.0 <= img.min() and img.max() <= 1.0
    assert img.max() > 0.8, "字符应当有明显笔画"
    assert img.mean() < 0.5, "应当是白字黑底，不是反过来"


def test_render_char_jitter_changes_image():
    from digitalfly.vision import render_char
    a = render_char("5", 24)
    b = render_char("5", 24, jitter=(3, -3))
    cc = render_char("5", 24, rotate=15)
    assert not np.allclose(a, b)
    assert not np.allclose(a, cc)


def test_driven_types_excluded_from_readout(c):
    """读出层必须排除被直接驱动的 L1/L2 —— 否则等于把输入原样读回来。"""
    from digitalfly.experiments.vision_decode import (DRIVEN_TYPES,
                                                      _readout_groups)
    from digitalfly.neurons import NeuronIndex
    groups, names = _readout_groups(c, NeuronIndex(c), layer="column")
    assert len(groups) > 500
    allidx = np.concatenate(groups)
    types = c.meta.iloc[allidx]["type"].astype(str).to_numpy()
    assert not np.isin(types, DRIVEN_TYPES).any(), "读出层混进了被驱动的神经元"


def test_vpn_readout_groups(c):
    from digitalfly.experiments.vision_decode import _readout_groups
    from digitalfly.neurons import NeuronIndex
    groups, names = _readout_groups(c, NeuronIndex(c), layer="vpn")
    assert len(groups) > 50
    assert all(len(g) >= 4 for g in groups)


def test_mnist_loader_if_present():
    """MNIST 在本地就检查一下格式对不对。"""
    from digitalfly.handwriting import MNIST_DIR, MNIST_FILES, load_mnist
    if not (MNIST_DIR / MNIST_FILES["train_x"]).exists():
        pytest.skip("本地没有 MNIST")
    x, y = load_mnist("train")
    assert x.shape[0] == y.shape[0] == 60000
    assert x.shape[1:] == (28, 28)
    assert 0.0 <= x.min() and x.max() <= 1.0
    assert set(np.unique(y).tolist()) == set(range(10))


def test_resize_keeps_range():
    from digitalfly.handwriting import resize_to
    img = np.random.default_rng(0).random((28, 28)).astype(np.float32)
    out = resize_to(img, 24)
    assert out.shape == (24, 24)
    assert 0.0 <= out.min() and out.max() <= 1.0


@pytest.mark.skipif(
    not (config.BUILD_DIR / "handwriting.npz").exists(),
    reason="尚未训练解码器，先运行 python cli.py train-vision")
def test_trained_model_beats_shuffled():
    """训练好的模型，留出集准确率必须明显高过打乱标签对照。"""
    m = np.load(config.BUILD_DIR / "handwriting.npz", allow_pickle=False)
    acc = float(m["accuracy"])
    shuf = float(m["shuffled_mean"])
    n = len(m["charset"])
    assert acc > shuf + 0.2, f"准确率 {acc:.2f} 没有明显高过对照 {shuf:.2f}"
    assert acc > 3.0 / n, "准确率没到随机的三倍"
