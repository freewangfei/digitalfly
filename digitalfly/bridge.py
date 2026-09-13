"""脑—体接口：连接组的脉冲活动 <-> 身体的行为指令与感觉输入。

**这一层要讲清楚哪些是数据、哪些是工程假设。**

来自真实数据的部分
    * 谁是下行神经元 (DN)、运动神经元、感觉神经元，在左侧还是右侧
      —— MaleCNS v1.0 的官方标注 (superclass / somaSide)。
    * 神经元之间怎么连、突触是兴奋还是抑制 —— 连接组本身。
    * 转向由左右两侧下行神经元活动的不对称性编码 —— 这是果蝇行为学的
      既有结论（如 DNa02 等已鉴定的转向指令神经元），不是这里编出来的。

属于工程假设的部分
    * 群体发放率到"前进速度""转向角速度"的**定量标度**。连接组给出谁连谁，
      给不出每赫兹对应多少厘米每秒。
    * 感觉刺激强度到注入电流的标度。

**为什么下行神经元接的是指令而不是关节角。** 果蝇的行走节律由腹神经索的中枢
模式发生器产生，大脑下达的是"走快点""向左转"这类指令。之前把下行神经元活动
直接映射成关节目标角，果蝇会当场瘫倒 —— 那个接法在生物学上就是错的。
现在的分工是：

    连接组 -> DN 左右群体发放率 -> (前进速度, 转向, 爬升)  [本模块]
                                            |
                              运动模式发生器 -> 关节轨迹    [locomotion.py]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .connectome import Connectome
from .neurons import NeuronIndex


@dataclass
class Command:
    """大脑下达给身体的指令。"""
    speed: float = 0.0        # [0, 1]  前进速度
    turn: float = 0.0         # [-1, 1] 负 = 左转
    climb: float = 0.0        # [-1, 1] 飞行时的爬升
    dn_left_hz: float = 0.0
    dn_right_hz: float = 0.0


@dataclass
class BridgeReport:
    n_dn: int
    n_dn_left: int
    n_dn_right: int
    n_motor: int
    n_proprio: int
    n_olfactory: int
    n_food_orns: int
    n_gustatory: int

    def summary(self) -> str:
        return (
            f"  下行神经元 {self.n_dn} 个（左 {self.n_dn_left} / "
            f"右 {self.n_dn_right}）-> 前进速度与转向指令\n"
            f"  感觉输入: 食物气味通道 {self.n_food_orns}/{self.n_olfactory} 嗅觉 · "
            f"味觉 {self.n_gustatory} · "
            f"本体感觉 {self.n_proprio}\n"
            f"  运动神经元 {self.n_motor} 个（作为读数，不直接驱动关节）")


class CommandBridge:
    """把连接组活动读成行为指令，把世界读成注入电流。"""

    # 群体发放率归一化的参考值：DN 群体达到这个率算"全速指令"。
    # 实测在标定后的工作点上，行走时下行神经元群体大约 3-8 Hz，
    # 取 8 Hz 作参考，指令才能覆盖整个 [0,1] 区间。
    REF_DN_HZ = 8.0

    def __init__(self, c: Connectome, dt_ms: float = 0.5,
                 tau_ms: float = 60.0, verbose: bool = True):
        self.c = c
        self.idx = NeuronIndex(c)
        self.dt_ms = dt_ms
        # 肌肉与行为指令都是低通的，不会跟着单个脉冲抖
        self.alpha = float(1.0 - np.exp(-dt_ms / tau_ms))
        self._ema_ref = self.REF_DN_HZ * dt_ms / 1000.0

        dn = self.idx.descending()
        self.dn_left, self.dn_right = self.idx.split_lr(dn)
        self.dn_all = dn

        self.motor = self.idx.motor()
        self.proprio = self.idx.proprioceptive()
        self.olfactory = self.idx.olfactory()
        self.gustatory = self.idx.gustatory()
        # 味觉刺激只注入唇瓣刚毛 GRN（真正的一级味觉输入，163 个）+ 糖味通路。
        # 把全部 1,428 个味觉神经元一起激活会把全脑推进爆发态（实测 18 Hz）。
        self.labellar = self.idx.labellar_grns()
        self.sugar = self.idx.sugar_pathway()
        self.mn9 = self.idx.proboscis_motor()

        # 气味只注入对食物气味响应的那几类嗅觉通道，而不是全部嗅觉神经元。
        # 注意感觉神经元的侧别在 rootSide 而不是 somaSide，见 NeuronIndex.side。
        self.food_orns = self.idx.food_odor_orns()
        self.olf_left, self.olf_right = self.idx.split_lr(self.food_orns)
        self.gus_left, self.gus_right = self.idx.split_lr(self.gustatory)

        # 左右群体大小本身就不对称（嗅觉左 884 / 右 1344，是重建与标注的产物，
        # 不是生物学）。注入电流按群体大小归一，保证"同样浓度 = 同样总驱动"。
        n_ref = max(len(self.olf_left), len(self.olf_right), 1)
        self.olf_gain_l = n_ref / max(len(self.olf_left), 1)
        self.olf_gain_r = n_ref / max(len(self.olf_right), 1)

        self.rng = np.random.default_rng(0)
        self.rate_l = 0.0
        self.rate_r = 0.0
        # 网络静息时左右下行神经元活动就有固定差值，转向必须相对这个基线来读
        self.turn_baseline = 0.0
        # 功能性鉴定出的转向下行神经元群，由 identify_steering_dns 填充；
        # 没鉴定过就退回解剖学左右分群
        self.dn_turn_left = self.dn_left
        self.dn_turn_right = self.dn_right
        self.rate_turn_l = 0.0
        self.rate_turn_r = 0.0
        self.report = BridgeReport(
            n_dn=len(dn), n_dn_left=len(self.dn_left),
            n_dn_right=len(self.dn_right), n_motor=len(self.motor),
            n_proprio=len(self.proprio), n_olfactory=len(self.olfactory),
            n_food_orns=len(self.food_orns),
            n_gustatory=len(self.gustatory))
        if verbose:
            print("[bridge] 脑-体接口")
            print(self.report.summary())

    def reset(self) -> None:
        self.rate_l = self.rate_r = 0.0
        self.rate_turn_l = self.rate_turn_r = 0.0

    def identify_steering_dns(self, brain, k: int = 40, probe_ms: float = 700.0,
                              cache: bool = True, verbose: bool = True
                              ) -> tuple[np.ndarray, np.ndarray]:
        """从模型本身找出真正编码"气味在左还是在右"的下行神经元。

        为什么不能直接用全部下行神经元的左右群体差：连接组重建出来的左右
        神经元数目本来就不对称，而且绝大多数下行神经元跟气味侧别无关，
        把它们平均进来会把信号冲掉 —— 实测全群读数只有 0.03 的动态范围，
        而挑出来的子群体有 35 Hz 的对比度。

        做法是两次探针试验（气味偏左 / 偏右），比较每个下行神经元的发放率，
        取差值最大的 k 个作为"左转群"、最小的 k 个作为"右转群"。
        这就是神经科学里功能性鉴定指令神经元的标准做法。
        """
        import json

        from . import config
        cache_file = config.BUILD_DIR / "steering_dns.json"
        if cache and cache_file.exists():
            d = json.loads(cache_file.read_text())
            if d.get("k") == k and d.get("n_dn") == len(self.dn_all):
                self.dn_turn_left = np.asarray(d["left"], dtype=np.int64)
                self.dn_turn_right = np.asarray(d["right"], dtype=np.int64)
                if verbose:
                    print(f"[bridge] 转向下行神经元（缓存）: "
                          f"左转群 {len(self.dn_turn_left)} / "
                          f"右转群 {len(self.dn_turn_right)}")
                return self.dn_turn_left, self.dn_turn_right

        def probe(odor_l, odor_r):
            brain.reset()
            ext = np.zeros(brain.n, dtype=np.float32)
            self.sensory_current(brain.n, odor_left=odor_l,
                                 odor_right=odor_r, out=ext)
            ext[self.proprio] += 0.8
            counts = np.zeros(brain.n, dtype=np.float32)
            n = int(probe_ms / self.dt_ms)
            for _ in range(n):
                counts += brain.step(ext)
            return counts[self.dn_all] / (probe_ms / 1000.0)

        diff = probe(1.0, 0.2) - probe(0.2, 1.0)
        order = np.argsort(-diff)
        self.dn_turn_left = self.dn_all[order[:k]]
        self.dn_turn_right = self.dn_all[order[-k:]]
        brain.reset()
        self.reset()
        if verbose:
            print(f"[bridge] 转向下行神经元: 左转群 {k} 个 "
                  f"(左侧气味使其多发放 {diff[order[:k]].mean():+.1f} Hz), "
                  f"右转群 {k} 个 ({diff[order[-k:]].mean():+.1f} Hz)")
        if cache:
            config.ensure_dirs()
            cache_file.write_text(json.dumps({
                "k": k, "n_dn": len(self.dn_all),
                "left": self.dn_turn_left.tolist(),
                "right": self.dn_turn_right.tolist(),
                "contrast_hz": float(diff[order[:k]].mean()
                                     - diff[order[-k:]].mean()),
            }))
        return self.dn_turn_left, self.dn_turn_right

    def calibrate_turn_baseline(self, brain, seconds: float = 0.6,
                                drive: float = 0.8,
                                verbose: bool = True) -> float:
        """在对称输入下跑一段，把左右下行活动的固定差值记为基线。"""
        n = brain.n
        ext = np.zeros(n, dtype=np.float32)
        ext[self.proprio] += drive
        if len(self.olf_left):
            ext[self.olf_left] += 1.0 * self.olf_gain_l
        if len(self.olf_right):
            ext[self.olf_right] += 1.0 * self.olf_gain_r
        self.turn_baseline = 0.0
        self.reset()
        vals = []
        for i in range(int(seconds * 1000 / self.dt_ms)):
            cmd = self.read_command(brain.step(ext))
            if i > int(0.3 * 1000 / self.dt_ms):
                vals.append(cmd.turn)
        self.turn_baseline = float(np.mean(vals)) if vals else 0.0
        brain.reset()
        self.reset()
        if verbose:
            print(f"[bridge] 转向基线校准: {self.turn_baseline:+.3f}"
                  "（静息时左右下行活动的固定差，已扣除）")
        return self.turn_baseline

    # -- 下行通路：脑 -> 指令 -------------------------------------------------
    def read_command(self, spikes: np.ndarray) -> Command:
        a = self.alpha
        l = spikes[self.dn_left].mean() if len(self.dn_left) else 0.0
        r = spikes[self.dn_right].mean() if len(self.dn_right) else 0.0
        self.rate_l += a * (float(l) - self.rate_l)
        self.rate_r += a * (float(r) - self.rate_r)

        tl = (spikes[self.dn_turn_left].mean()
              if len(self.dn_turn_left) else 0.0)
        tr = (spikes[self.dn_turn_right].mean()
              if len(self.dn_turn_right) else 0.0)
        self.rate_turn_l += a * (float(tl) - self.rate_turn_l)
        self.rate_turn_r += a * (float(tr) - self.rate_turn_r)

        nl = self.rate_l / self._ema_ref
        nr = self.rate_r / self._ema_ref
        speed = float(np.clip(0.5 * (nl + nr), 0.0, 1.0))

        ml = self.rate_turn_l / self._ema_ref
        mr = self.rate_turn_r / self._ema_ref
        denom = max(ml + mr, 1e-3)
        raw_turn = (ml - mr) / denom
        # 扣掉静息基线：网络本身的左右不对称是重建的产物，
        # 有行为意义的是"相对基线的偏离"
        turn = float(np.clip(raw_turn - self.turn_baseline, -1.0, 1.0))
        return Command(speed=speed, turn=turn,
                       dn_left_hz=self.rate_l / (self.dt_ms / 1000.0),
                       dn_right_hz=self.rate_r / (self.dt_ms / 1000.0))

    def mn9_rate(self, window_counts: np.ndarray, seconds: float) -> float:
        if not len(self.mn9) or seconds <= 0:
            return 0.0
        return float(window_counts[self.mn9].sum() / len(self.mn9) / seconds)

    # -- 上行通路：世界与身体 -> 脑 -------------------------------------------
    #
    # 感觉输入用"目标发放率"编码，而不是直接给一个持续电流。
    #
    # 原因：LIF 神经元没有适应机制，持续注入电流会让它饱和 —— 膜电位稳态是
    # I/(1-decay) = 40 x I，只要电流超过 0.18 mV，神经元就会以不应期允许的
    # 最高频率（约 400 Hz）发放，而且再怎么加大刺激也不会变。实测把气味强度
    # 从 0.5 调到 4.0，全脑活动纹丝不动地停在 11 Hz。
    #
    # 真实感受神经元的发放率是 10-200 Hz。所以这里按 p = rate x dt 的概率
    # 逐步门控一个阈上电流，让目标群体的平均发放率就等于设定值。

    SPIKE_KICK = 12.0        # 一次阈上刺激的幅度 (mV)，保证必定发放

    def _drive(self, ext: np.ndarray, idx: np.ndarray, rate_hz: float,
               gain: float = 1.0) -> None:
        if not len(idx) or rate_hz <= 0:
            return
        p = min(1.0, rate_hz * gain * self.dt_ms / 1000.0)
        hit = self.rng.random(len(idx)) < p
        if hit.any():
            ext[idx[hit]] += self.SPIKE_KICK

    def sensory_current(self, n: int, body=None,
                        odor_left: float = 0.0, odor_right: float = 0.0,
                        taste: float = 0.0,
                        proprio_rate: float = 20.0,
                        odor_rate: float = 10.0,
                        taste_rate: float = 2000.0,
                        out: np.ndarray | None = None) -> np.ndarray:
        """把感觉输入编码成注入电流 (mV/步)。

            odor_left / odor_right ∈ [0, 1]  左右触角处的气味浓度
            taste                  ∈ [0, 1]  口器接触到的糖浓度
            *_rate                           该通道在满刺激下的目标发放率 (Hz)

        气味只注入对食物气味响应的那几类嗅觉通道（约 280 个），
        味觉只注入 163 个唇瓣刚毛 GRN 和糖味通路 —— 真实果蝇闻到一种气味也
        只激活少数几类嗅觉神经元，不是全部两千多个一起放电。

        两个默认值的来历：
          * `taste_rate=2000` 对应饱和驱动（糖味通路实测 397 Hz，已是不应期
            上限）。实测必须到这个强度 MN9 才会响应 —— MN9 的总输入权重是
            6083，需要大量同步输入才能跨过阈值。此时全脑只有 1.7 Hz，
            也就是说这条反应是**特异**的，不是全脑普遍兴奋的副产品。
          * `odor_rate=10` 是中等气味下真实 ORN 的发放率量级。要注意此时
            全脑会升到约 8 Hz，高于果蝇的生理水平 —— 这是均一 LIF 模型的
            已知局限：没有适应机制，也没有触角叶的增益控制与侧向抑制。
        """
        ext = np.zeros(n, dtype=np.float32) if out is None else out
        if out is not None:
            ext.fill(0.0)

        if body is not None and len(self.proprio):
            # 本体感觉：用关节速度的整体幅度当作"腿在动"的信号
            qvel = np.asarray(body.data.qvel[6:], dtype=np.float32)
            drive = float(np.tanh(np.abs(qvel).mean() * 0.05))
            self._drive(ext, self.proprio, proprio_rate * drive)

        self._drive(ext, self.olf_left,
                    odor_rate * float(np.clip(odor_left, 0, 1)),
                    self.olf_gain_l)
        self._drive(ext, self.olf_right,
                    odor_rate * float(np.clip(odor_right, 0, 1)),
                    self.olf_gain_r)

        if taste > 0:
            t = float(np.clip(taste, 0, 1))
            self._drive(ext, self.labellar, taste_rate * t)
            self._drive(ext, self.sugar, taste_rate * t)
        return ext
