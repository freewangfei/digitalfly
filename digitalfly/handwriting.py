"""手写字符识别：训练一个读出果蝇视觉系统的解码器，并保存下来。

**这里做的是什么，不是什么。**

做的是神经解码 —— 和真实实验里"从猴子 V1 的群体活动解码出光栅朝向"是同一类
操作：把刺激呈现给感觉系统，记录群体响应，训练一个线性读出。**果蝇本身不认识
数字**，是我们在读它的视觉表征。

不是的：这不是"果蝇学会了识字"。突触权重一个都没变，网络里没有任何学习发生。
学习只发生在最外面那一层线性解码器上。

数据用真实手写样本：
    数字  MNIST（Google GCS 镜像，9.9 MB）
    字母  EMNIST letters（NIST，可选，561 MB）
    取不到就退回字体渲染 + 随机形变。

流程和 `experiments/vision_decode` 一致，区别是把训练好的解码器存到
`data/build/handwriting.npz`，Web 控制台的手写识别就用它。
"""
from __future__ import annotations

import gzip
import struct
import subprocess
import time
from pathlib import Path

import numpy as np

from . import config
from .brain import Brain
from .bridge import CommandBridge
from .calibrate import calibrated_params
from .experiments.cube_decode import Softmax
from .experiments.vision_decode import _readout_groups
from .neurons import NeuronIndex
from .vision import Retina, render_char

MNIST_DIR = config.DATA_DIR / "mnist"
MNIST_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist"
MNIST_FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}
MODEL_PATH = config.BUILD_DIR / "handwriting.npz"

DIGIT_LABELS = "0123456789"
LETTER_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------

def download_mnist(verbose: bool = True) -> bool:
    """下载 MNIST。走 curl -4，理由见 tools/ipv4.py。"""
    MNIST_DIR.mkdir(parents=True, exist_ok=True)
    ok = True
    for name in MNIST_FILES.values():
        dest = MNIST_DIR / name
        if dest.exists() and dest.stat().st_size > 1000:
            continue
        if verbose:
            print(f"  下载 {name}")
        r = subprocess.run(
            ["curl", "-sS", "-4", "--http1.1", "--retry", "5",
             "--retry-all-errors", "-o", str(dest), f"{MNIST_BASE}/{name}"],
            capture_output=True, text=True, timeout=600)
        ok = ok and r.returncode == 0 and dest.exists()
    return ok


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic == 2051:
            rows, cols = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows,
                                                                   cols)
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_mnist(split: str = "train") -> tuple:
    x = _read_idx(MNIST_DIR / MNIST_FILES[f"{split}_x"])
    y = _read_idx(MNIST_DIR / MNIST_FILES[f"{split}_y"])
    return x.astype(np.float32) / 255.0, y.astype(np.int64)


def resize_to(img: np.ndarray, size: int) -> np.ndarray:
    """把手写样本缩放到复眼的网格尺寸。"""
    from PIL import Image
    if img.shape[0] == size:
        return img
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.resize((size, size), Image.LANCZOS),
                      dtype=np.float32) / 255.0


def sample_stimuli(charset: str, reps: int, grid: int, rng,
                   source: str = "auto", verbose: bool = True) -> tuple:
    """取刺激图像与标签。返回 (图像列表, 标签, 数据来源说明)。"""
    use_mnist = (source in ("auto", "mnist") and charset == DIGIT_LABELS
                 and (MNIST_DIR / MNIST_FILES["train_x"]).exists())
    if use_mnist:
        x, y = load_mnist("train")
        imgs, labs = [], []
        for cls in range(10):
            pool = np.flatnonzero(y == cls)
            pick = rng.choice(pool, size=reps, replace=False)
            for p in pick:
                imgs.append(resize_to(x[p], grid))
                labs.append(cls)
        if verbose:
            print(f"  样本来源: MNIST 真实手写数字（{len(imgs)} 张）")
        return imgs, np.asarray(labs), "MNIST"

    imgs, labs = [], []
    for ci, ch in enumerate(charset):
        for _ in range(reps):
            imgs.append(render_char(
                ch, size=grid,
                jitter=(int(rng.integers(-2, 3)), int(rng.integers(-2, 3))),
                scale=float(rng.uniform(0.85, 1.15)),
                rotate=float(rng.uniform(-14, 14))))
            labs.append(ci)
    if verbose:
        print(f"  样本来源: 字体渲染 + 随机形变（{len(imgs)} 张）")
    return imgs, np.asarray(labs), "rendered"


# --------------------------------------------------------------------------
# 呈现给果蝇的视觉系统
# --------------------------------------------------------------------------

class FlyEncoder:
    """把图像喂给果蝇的复眼，取出视柱级的群体响应。"""

    def __init__(self, c, grid: int = 24, layer: str = "column",
                 peak_hz: float = 2000.0, settle_ms: float = 50.0,
                 probe_ms: float = 180.0, backend: str | None = None,
                 verbose: bool = True):
        self.c = c
        self.grid = grid
        self.peak_hz = peak_hz
        self.brain = Brain(c.W, calibrated_params(), backend=backend,
                           verbose=verbose)
        self.bridge = CommandBridge(c, dt_ms=self.brain.p.dt, verbose=False)
        idx = NeuronIndex(c)
        self.eyes = [Retina(c, side=s, size=grid) for s in ("L", "R")]
        self.groups, self.names = _readout_groups(c, idx, layer=layer)
        self.n_settle = int(settle_ms / self.brain.p.dt)
        self.n_probe = int(probe_ms / self.brain.p.dt)
        self.secs = self.n_probe * self.brain.p.dt / 1000.0
        self._ext = np.zeros(self.brain.n, dtype=np.float32)
        if verbose:
            print(f"  复眼: {self.eyes[1].n_columns} 个视柱/眼，"
                  f"读出 {len(self.groups)} 个视柱群")

    def encode(self, img: np.ndarray, keep_spikes: bool = False) -> tuple:
        """呈现一张图，返回 (特征向量, 全脑平均发放率, 可选的脉冲计数)。"""
        self.brain.reset()
        self.bridge.reset()
        counts = np.zeros(self.brain.n, dtype=np.float32)
        total = 0.0
        for i in range(self.n_settle + self.n_probe):
            self._ext.fill(0.0)
            for eye in self.eyes:
                eye.present(img, self._ext, self.bridge, peak_hz=self.peak_hz)
            s = self.brain.step(self._ext)
            total += float(s.sum())
            if i >= self.n_settle:
                counts += s
        feat = np.array([counts[g].sum() / len(g) / self.secs
                         for g in self.groups])
        steps = self.n_settle + self.n_probe
        rate = total / steps / self.brain.n / (self.brain.p.dt / 1000.0)
        return feat, rate, (counts if keep_spikes else None)


# --------------------------------------------------------------------------
# 训练与推理
# --------------------------------------------------------------------------

def train(c, charset: str = DIGIT_LABELS, reps: int = 40, grid: int = 24,
          layer: str = "column", peak_hz: float = 2000.0,
          settle_ms: float = 50.0, probe_ms: float = 180.0,
          backend: str | None = None, seed: int = 0,
          save: bool = True, verbose: bool = True) -> dict:
    """采集响应、训练解码器、保存。"""
    rng = np.random.default_rng(seed)
    enc = FlyEncoder(c, grid=grid, layer=layer, peak_hz=peak_hz,
                     settle_ms=settle_ms, probe_ms=probe_ms,
                     backend=backend, verbose=verbose)
    imgs, labels, source = sample_stimuli(charset, reps, grid, rng,
                                          verbose=verbose)

    order = rng.permutation(len(imgs))
    imgs = [imgs[i] for i in order]
    labels = labels[order]

    X, P, rates = [], [], []
    t0 = time.time()
    for k, img in enumerate(imgs):
        f, r, _ = enc.encode(img)
        X.append(f)
        P.append(img.ravel())
        rates.append(r)
        if verbose and (k + 1) % 25 == 0:
            print(f"\r  采集 {k + 1}/{len(imgs)}  "
                  f"用时 {time.time() - t0:.0f}s", end="")
    if verbose:
        print()
    X, P = np.asarray(X), np.asarray(P)
    n_class = len(charset)
    cut = int(len(X) * 0.75)

    model = Softmax(n_class, l2=2.0).fit(X[:cut], labels[:cut],
                                         epochs=1200, lr=0.8)
    acc = float((model.predict(X[cut:]) == labels[cut:]).mean())
    pix = Softmax(n_class, l2=2.0).fit(P[:cut], labels[:cut],
                                       epochs=1200, lr=0.8)
    pixel_acc = float((pix.predict(P[cut:]) == labels[cut:]).mean())
    shuf = []
    for _ in range(8):
        m = Softmax(n_class, l2=2.0).fit(
            X[:cut], rng.permutation(labels[:cut]), epochs=1200, lr=0.8)
        shuf.append(float((m.predict(X[cut:])
                           == rng.permutation(labels[cut:])).mean()))
    shuf = np.asarray(shuf)

    out = {
        "charset": charset, "source": source, "n_samples": len(X),
        "accuracy": acc, "pixel_baseline": pixel_acc,
        "shuffled_mean": float(shuf.mean()), "shuffled_std": float(shuf.std()),
        "chance": 1.0 / n_class, "mean_brain_hz": float(np.mean(rates)),
        "grid": grid, "layer": layer, "n_features": X.shape[1],
    }
    if verbose:
        print(f"\n  样本 {len(X)}（训练 {cut} / 测试 {len(X) - cut}）"
              f"· 来源 {source}")
        print(f"  果蝇视觉解码准确率  {100 * acc:5.1f}%")
        print(f"  像素基线            {100 * pixel_acc:5.1f}%   <- 任务上限")
        print(f"  打乱标签对照        {100 * shuf.mean():5.1f}% "
              f"± {100 * shuf.std():.1f}%")
        print(f"  随机                {100 * out['chance']:5.1f}%")
        print(f"  全脑平均发放率      {out['mean_brain_hz']:.2f} Hz")

    if save:
        config.ensure_dirs()
        # 编码器的配置必须和权重存在一起。之前只存了 grid/layer，
        # 改了 probe_ms 之后推理端的特征尺度就和训练时对不上了 ——
        # 归一化之后 logits 饱和，每张图都预测同一个类。
        np.savez(MODEL_PATH, W=model.W, b=model.b, mu=model.mu, sd=model.sd,
                 charset=np.array(list(charset)), grid=grid, layer=layer,
                 peak_hz=peak_hz, settle_ms=settle_ms, probe_ms=probe_ms,
                 accuracy=acc, pixel_baseline=pixel_acc,
                 shuffled_mean=float(shuf.mean()), source=source)
        if verbose:
            print(f"  解码器已保存 -> {MODEL_PATH}")
    return out


ONLINE_PATH = config.BUILD_DIR / "handwriting_online.npz"


class Recognizer:
    """加载训练好的解码器，对一张手写图做识别，并支持在线反馈学习。

    **在线学习改的是什么。** 只有最外层线性解码器的权重会动，
    连接组的突触权重一个都不变 —— 果蝇没有在学，是读出层在适应你的笔迹。
    这和脑机接口里解码器随使用者调整是同一回事。

    基础模型（离线用 MNIST 训的）保留不动，在线增量单独存一份，
    随时可以丢掉重来。
    """

    def __init__(self, c, backend: str | None = None, verbose: bool = True):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"{MODEL_PATH} 不存在，请先运行: "
                "python cli.py train-vision")
        d = np.load(MODEL_PATH, allow_pickle=False)
        self.charset = "".join(d["charset"].tolist())
        self.grid = int(d["grid"])
        self.meta = {
            "accuracy": float(d["accuracy"]),
            "pixel_baseline": float(d["pixel_baseline"]),
            "shuffled_mean": float(d["shuffled_mean"]),
            "source": str(d["source"]),
        }
        self.W, self.b = d["W"], d["b"]
        self.mu, self.sd = d["mu"], d["sd"]
        keys = set(d.files)

        def cfg(name, default):
            return float(d[name]) if name in keys else default

        self.enc = FlyEncoder(
            c, grid=self.grid, layer=str(d["layer"]),
            peak_hz=cfg("peak_hz", 2000.0),
            settle_ms=cfg("settle_ms", 50.0),
            probe_ms=cfg("probe_ms", 180.0),
            backend=backend, verbose=verbose)

        # 基础权重留一份底，随时可以恢复
        self.W0, self.b0 = self.W.copy(), self.b.copy()
        self.last = None           # 最近一次识别的特征，反馈时用
        self.log = []              # 用户反馈记录
        self.buffer_X: list = []   # 反馈样本的特征
        self.buffer_y: list = []   # 反馈样本的正确标签
        self._load_online()

    # -- 在线反馈学习 -------------------------------------------------------
    def _load_online(self) -> None:
        if not ONLINE_PATH.exists():
            return
        try:
            d = np.load(ONLINE_PATH, allow_pickle=False)
            if d["X"].shape[1] == self.W.shape[0]:
                self.buffer_X = list(d["X"])
                self.buffer_y = list(d["y"])
                self.log = [{"correct": bool(v)} for v in d["hits"]]
                self._refit()
        except Exception:                                # noqa: BLE001
            pass

    def _save_online(self) -> None:
        config.ensure_dirs()
        # 存的是反馈样本本身，不是权重 —— 权重每次都从基础模型重算，
        # 这样换了基础模型也能直接复用历史反馈。
        np.savez(ONLINE_PATH,
                 X=np.asarray(self.buffer_X, dtype=np.float32),
                 y=np.asarray(self.buffer_y, dtype=np.int64),
                 hits=np.array([bool(r["correct"]) for r in self.log]))

    # 在线学习用**回放缓冲 + 从基础权重重拟合**，不是单样本 SGD。
    #
    # 单样本梯度下降在这里是行不通的：特征有 1770 维，一个样本一学就把离线
    # 用 2500 张 MNIST 训出来的模型带偏 —— 实测反馈 10 次之后，留出集准确率
    # 从 80% 掉到 57%，别的数字开始被高置信度认错，典型的灾难性干扰。
    # 加了弹性锚定也压不住。
    #
    # 现在的做法：把每次反馈的 (特征, 正确标签) 存进缓冲，每次都**从基础权重
    # 出发**、只在缓冲上做少量梯度步。基础模型永远是起点，所以不会漂走；
    # 缓冲越大，对你的笔迹适应得越多。
    REPLAY_LR = 0.25
    REPLAY_STEPS = 60

    def feedback(self, true_char: str) -> dict:
        """告诉它刚才那张图真正是什么。

        用的是最近一次 `recognise` 缓存下来的神经特征，不需要重跑仿真。
        """
        if self.last is None:
            return {"ok": False, "error": "还没有可反馈的识别结果"}
        if true_char not in self.charset:
            return {"ok": False, "error": f"字符 {true_char} 不在字符集里"}

        before = self.last["prediction"]
        self.buffer_X.append(self.last["feat"])
        self.buffer_y.append(self.charset.index(true_char))
        self.log.append({"correct": before == true_char, "true": true_char,
                         "before": before})
        self._refit()

        p = self._probs(self.last["feat"])
        k = self.charset.index(true_char)
        after = self.charset[int(np.argmax(p))]
        self._save_online()
        n = len(self.log)
        hits = sum(1 for r in self.log if r["correct"])
        return {
            "ok": True, "true": true_char, "before": before, "after": after,
            "confidence": float(p[k]), "n_feedback": n,
            "buffer": len(self.buffer_y),
            "user_accuracy": hits / n if n else 0.0,
        }

    def _probs(self, feat: np.ndarray) -> np.ndarray:
        z = (feat - self.mu) / self.sd @ self.W + self.b
        z = z - z.max()
        p = np.exp(z)
        return p / p.sum()

    def _refit(self) -> None:
        """从基础权重出发，在反馈缓冲上做少量梯度步。"""
        if not self.buffer_y:
            self.W, self.b = self.W0.copy(), self.b0.copy()
            return
        Z = (np.asarray(self.buffer_X) - self.mu) / self.sd
        Y = np.eye(len(self.charset))[np.asarray(self.buffer_y)]
        W, b = self.W0.copy(), self.b0.copy()
        n = len(Y)
        for _ in range(self.REPLAY_STEPS):
            z = Z @ W + b
            z = z - z.max(axis=1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(axis=1, keepdims=True)
            g = (p - Y) / n
            W -= self.REPLAY_LR * Z.T @ g
            b -= self.REPLAY_LR * g.sum(0)
        self.W, self.b = W, b

    def reset_online(self) -> dict:
        """丢掉在线学到的东西，回到离线训练的基础模型。"""
        self.W, self.b = self.W0.copy(), self.b0.copy()
        self.log = []
        self.buffer_X, self.buffer_y = [], []
        if ONLINE_PATH.exists():
            ONLINE_PATH.unlink()
        return {"ok": True}

    def recognise(self, img: np.ndarray, keep_spikes: bool = False) -> dict:
        """返回预测字符、概率分布与网络状态。"""
        img = resize_to(np.asarray(img, dtype=np.float32), self.grid)
        feat, rate, spikes = self.enc.encode(img, keep_spikes=keep_spikes)
        z = (feat - self.mu) / self.sd @ self.W + self.b
        z = z - z.max()
        p = np.exp(z)
        p /= p.sum()
        order = np.argsort(-p)[:3]
        out = {
            "prediction": self.charset[int(order[0])],
            "confidence": float(p[order[0]]),
            "top3": [{"char": self.charset[int(i)], "p": float(p[i])}
                     for i in order],
            "brain_hz": rate,
            "probs": {self.charset[i]: float(p[i]) for i in range(len(p))},
        }
        # 缓存特征，供反馈时就地更新读出层（不用重跑仿真）
        self.last = {"feat": feat, "prediction": out["prediction"]}
        n = len(self.log)
        out["n_feedback"] = n
        out["user_accuracy"] = (sum(1 for r in self.log if r["correct"]) / n
                                if n else None)
        if keep_spikes:
            out["_spikes"] = spikes
        return out
