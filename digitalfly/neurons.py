"""按生物学身份定位神经元群体。

数字果蝇的"接口"全在这里：哪些神经元是感觉输入口，哪些是通向身体的输出口。
所有分组都直接来自 MaleCNS v1.0 的官方标注字段（superclass / class / subclass /
type / synonyms），不做任何猜测；凡是需要推断的地方都写明依据。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .connectome import Connectome

# MaleCNS 的 superclass 取值（按数量）：
#   ol_intrinsic 视叶内在 / cb_intrinsic 中央脑内在 / vnc_intrinsic 腹神经索内在
#   visual_projection 视觉投射 / vnc_sensory / ol_sensory / cb_sensory 感觉
#   ascending_neuron 上行 / descending_neuron 下行 / vnc_motor / cb_motor 运动
SENSORY_SUPERCLASSES = ("vnc_sensory", "ol_sensory", "cb_sensory",
                        "sensory_ascending")
MOTOR_SUPERCLASSES = ("vnc_motor", "cb_motor", "vnc_efferent")


class NeuronIndex:
    """在 Connectome.meta 上做群体查询，返回矩阵下标。"""

    def __init__(self, c: Connectome):
        self.c = c
        self.meta = c.meta
        self._s = self.meta.reset_index()[["index", "bodyId"]]

    # -- 基础查询 ----------------------------------------------------------
    def where(self, **kw) -> np.ndarray:
        """字段精确匹配。值可以是标量或可迭代对象。

            idx.where(superclass='descending_neuron')
            idx.where(type=['GNG540', 'GNG550'])
        """
        mask = pd.Series(True, index=self.meta.index)
        for col, val in kw.items():
            series = self.meta[col]
            if isinstance(val, (list, tuple, set, np.ndarray, pd.Index)):
                mask &= series.isin(list(val))
            else:
                mask &= (series == val)
        return np.flatnonzero(mask.to_numpy())

    def matching(self, column: str, pattern: str) -> np.ndarray:
        """字段正则匹配（大小写不敏感）。"""
        m = self.meta[column].astype(str).str.contains(
            pattern, case=False, na=False, regex=True)
        return np.flatnonzero(m.to_numpy())

    def side(self, indices: np.ndarray | None = None) -> pd.Series:
        """解析神经元的左右侧别，返回 'L' / 'R' / None 的 Series。

        MaleCNS 的侧别信息散在三个字段里，而且不同类群用的不是同一个：
            运动神经元      somaSide 填得全，rootSide 基本是空的
            感觉神经元      正好相反，侧别在 rootSide
            两者都缺的      instance 名字里带 _L / _R 后缀
        所以按 somaSide -> rootSide -> instance 后缀依次回退。
        """
        m = self.meta if indices is None else self.meta.iloc[indices]

        def norm(col):
            v = m[col].astype(str).str.upper().str.strip().str[0]
            return v.where(v.isin(["L", "R"]))

        side = norm("somaSide")
        side = side.fillna(norm("rootSide"))
        suffix = m["instance"].astype(str).str.extract(
            r"_([LR])$", expand=False)
        return side.fillna(suffix)

    def split_lr(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """把一群神经元按左右分成两组（下标是相对全网的）。"""
        s = self.side(indices).to_numpy()
        return indices[s == "L"], indices[s == "R"]

    def describe(self, indices: np.ndarray, n: int = 10) -> pd.DataFrame:
        return self.meta.iloc[indices][
            ["bodyId", "type", "class", "subclass", "superclass", "nt"]].head(n)

    # -- 输出口：通向身体 ---------------------------------------------------
    def descending(self) -> np.ndarray:
        """下行神经元 (DN)：从脑发往腹神经索的指令通路，约 1300 个。

        MaleCNS 首次把脑与腹神经索放在同一图谱上并保留颈部连接，
        这批神经元就是"脑 -> 身体"的物理接口。
        """
        return self.where(superclass="descending_neuron")

    def motor(self) -> np.ndarray:
        """运动神经元：直接支配肌肉的最终输出，约 800 个。"""
        return self.where(superclass=list(MOTOR_SUPERCLASSES))

    def leg_motor(self) -> np.ndarray:
        """腿部运动神经元（按 VNC 神经节区分：前/中/后胸节）。"""
        mn = self.motor()
        sub = self.meta.iloc[mn]["somaNeuromere"].astype(str)
        return mn[sub.str.match(r"(LegNp|T[123])", na=False).to_numpy()]

    def ascending(self) -> np.ndarray:
        """上行神经元：把身体状态送回脑的反馈通路。"""
        return self.where(superclass="ascending_neuron")

    # -- 输入口：来自世界 ---------------------------------------------------
    def sensory(self) -> np.ndarray:
        return self.where(superclass=list(SENSORY_SUPERCLASSES))

    def photoreceptors(self) -> np.ndarray:
        """视叶感觉神经元（光感受器 R1-R8）。"""
        return self.where(superclass="ol_sensory")

    def proprioceptive(self) -> np.ndarray:
        """本体感觉神经元：关节角度与肌肉张力，闭环控制的反馈来源。"""
        return self.where(**{"class": "mechanosensory_proprioceptive"})

    def mechanosensory(self) -> np.ndarray:
        return self.matching("class", r"^mechanosensory")

    def gustatory(self) -> np.ndarray:
        return self.where(**{"class": "gustatory"})

    def olfactory(self) -> np.ndarray:
        return self.where(**{"class": "olfactory"})

    # 对食物气味（醋酸、发酵水果）有响应的嗅觉通道。MaleCNS 的嗅觉神经元
    # 按嗅小球命名，下面这几个是文献里公认的诱食性通道：
    #   DM1 = Or42b（对醋酸的吸引最强）· DM4 = Or59b · DM2 = Or22a
    #   VM2 = Or43b · VA2 = Or92a
    # 真实果蝇闻到一种气味只激活少数几类嗅觉神经元，不是全部两千多个一起放电 ——
    # 按整群注入会把全脑推进爆发态（实测 14 Hz）。
    FOOD_ODOR_GLOMERULI = ("DM1", "DM4", "DM2", "VM2", "VA2")

    def food_odor_orns(self) -> np.ndarray:
        """对食物气味响应的嗅觉感受神经元。"""
        types = [f"ORN_{g}" for g in self.FOOD_ODOR_GLOMERULI]
        return self.where(type=types)

    # -- 味觉通路：糖 vs 苦（伸喙实验的正/负对照）----------------------------
    # MaleCNS 的标注没有直接给 Gr64f 之类的受体身份，但 synonyms 字段收录了
    # Yao & Scott (2022) 鉴定出的食管下神经节味觉通路神经元，这是文献确证的身份。
    def sugar_pathway(self) -> np.ndarray:
        """糖味通路二级神经元 (Sugar SEL PN / LN)。"""
        return self.matching("synonyms", r"Sugar SEL")

    def bitter_pathway(self) -> np.ndarray:
        """苦味通路神经元 (Bitter-SEL)。苦味抑制伸喙，作阴性对照。"""
        return self.matching("synonyms", r"Bitter-SEL")

    def labellar_grns(self) -> np.ndarray:
        """唇瓣刚毛味觉感受神经元 —— 味觉的一级输入。"""
        return self.where(**{"class": "gustatory", "subclass": "labellar bristle"})

    def sugar_grns(self, min_weight: float = 5.0) -> np.ndarray:
        """数据驱动地找出糖味一级 GRN：

        取唇瓣刚毛 GRN 中，向糖味通路二级神经元 (Sugar SEL) 发出强连接的那些。
        这里不靠命名猜测，而是用连接组本身的接线来判定身份。
        """
        grns = self.labellar_grns()
        targets = self.sugar_pathway()
        if len(grns) == 0 or len(targets) == 0:
            return np.array([], dtype=np.int64)
        # W[post, pre] -> 取 Sugar SEL 行、GRN 列
        sub = np.asarray(
            np.abs(self.c.W[targets][:, grns]).sum(axis=0)).ravel()
        return grns[sub >= min_weight]

    # -- 伸喙的输出读数 -----------------------------------------------------
    def proboscis_motor(self) -> np.ndarray:
        """伸喙运动神经元 MN9。

        MN9 支配 rostrum protractor 肌，是果蝇伸喙反射 (PER) 的经典读数，
        也是 Shiu et al. (Nature 2024) 全脑仿真里用的输出指标。
        """
        return self.where(type="MN9")

    # -- 脑区 ---------------------------------------------------------------
    def by_region(self) -> dict[str, np.ndarray]:
        """按粗粒度解剖区域分组，用于全脑活动热图。"""
        sc = self.meta["superclass"].astype(str)
        groups = {
            "视叶 Optic lobe": sc.isin(["ol_intrinsic", "ol_sensory",
                                        "visual_projection",
                                        "visual_centrifugal"]),
            "中央脑 Central brain": sc.isin(["cb_intrinsic", "cb_sensory",
                                             "cb_motor", "cb_endocrine"]),
            "腹神经索 VNC": sc.isin(["vnc_intrinsic", "vnc_sensory",
                                     "vnc_motor", "vnc_efferent", "vnc_tbc"]),
            "下行 Descending": sc == "descending_neuron",
            "上行 Ascending": sc.isin(["ascending_neuron", "sensory_ascending"]),
        }
        out = {k: np.flatnonzero(v.to_numpy()) for k, v in groups.items()}
        assigned = np.concatenate([v for v in out.values()]) if out else []
        rest = np.setdiff1d(np.arange(len(self.meta)), assigned)
        if len(rest):
            out["其他 Unassigned"] = rest
        return out

    def by_class(self, min_size: int = 50) -> dict[str, np.ndarray]:
        """按 class 字段分组（olfactory / gustatory / Kenyon_Cell / CX ...）。"""
        out = {}
        for cls, grp in self.meta.groupby("class"):
            if len(grp) >= min_size and cls is not None:
                out[str(cls)] = grp.index.to_numpy()
        return out
