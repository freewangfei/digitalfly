"""标定三角步态的运动学参数。

步态的相位结构（三角步态、摆动/着地划分）是给定的模型，但具体幅度、步频、
占空比在这个身体模型上取多少才能稳定前进，只能实测。这个脚本做随机搜索，
目标是 2 秒内的前进距离，同时要求不翻倒、不严重侧偏。

跑法: python tools/tune_gait.py [试验次数]
结果打印出来，手工填回 locomotion.GaitParams。
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from digitalfly.body import FlyBody              # noqa: E402
from digitalfly.locomotion import TRIPOD_A       # noqa: E402

LEGS = ["T1_left", "T1_right", "T2_left", "T2_right", "T3_left", "T3_right"]
T_SIM = 2.0
SUB = 20


def rollout(body, p: dict, seconds: float = T_SIM) -> tuple:
    body.reset()
    ctrl = body.rest.copy()
    dt = body.timestep * SUB
    phase = 0.0
    legs = {l: (body.leg_actuators(l), body.claw_actuator(l),
                0.0 if l in TRIPOD_A else 0.5) for l in LEGS}
    seg = {"T1": 0, "T2": 1, "T3": 2}

    for _ in range(int(seconds / dt)):
        phase = (phase + p["freq"] * dt) % 1.0
        for leg, (a, claw, off) in legs.items():
            s = seg[leg.split("_")[0]]
            ph = (phase + off) % 1.0
            A = p["stroke"] * p["seg_scale"][s]
            if ph < p["duty"]:
                u = ph / p["duty"]
                coxa = 2 * A * (1 - u)
                femur = 0.0
                tibia = p["tibia"]
                adhere = p["adhesion"]
            else:
                u = (ph - p["duty"]) / (1 - p["duty"])
                coxa = 2 * A * u
                femur = p["lift"] * np.sin(np.pi * u)
                tibia = 0.0
                adhere = 0.0
            if "coxa" in a:
                ctrl[a["coxa"]] = body.rest[a["coxa"]] + coxa
            if "femur" in a:
                ctrl[a["femur"]] = body.rest[a["femur"]] + femur
            if "tibia" in a:
                ctrl[a["tibia"]] = body.rest[a["tibia"]] + tibia
            if claw is not None:
                ctrl[claw] = adhere
        body.step(ctrl, n_substeps=SUB)
        # 物理发散保护：某些参数组合会让求解器炸掉并卡住，提前判失败
        qvel = body.data.qvel
        if not np.all(np.isfinite(qvel)) or np.abs(qvel).max() > 5e3:
            return 0.0, 0.0, 0.0, True

    d = body.displacement()
    if not np.all(np.isfinite(d)):
        return 0.0, 0.0, 0.0, True
    return float(d[0]), float(d[1]), float(np.linalg.norm(d[:2])), body.fallen()


def score(fwd, lat, dist, fallen) -> float:
    if fallen:
        return -1.0
    # 前进为主，侧偏扣分
    return fwd - 0.5 * abs(lat)


def main(n_trials: int = 200) -> None:
    rng = np.random.default_rng(0)
    body = FlyBody(mode="walk", render_size=(96, 128))
    best, best_p = -1e9, None
    for i in range(n_trials):
        p = {
            "freq": float(rng.uniform(4, 18)),
            "stroke": float(rng.uniform(0.2, 1.1)),
            "lift": float(rng.uniform(0.1, 0.9)),
            "duty": float(rng.uniform(0.45, 0.8)),
            "tibia": float(rng.uniform(-0.5, 0.5)),
            "adhesion": float(rng.uniform(0.0, 1.0)),
            "seg_scale": tuple(rng.uniform(0.6, 1.4, 3)),
        }
        fwd, lat, dist, fallen = rollout(body, p)
        sc = score(fwd, lat, dist, fallen)
        if sc > best:
            best, best_p = sc, p
            print(f"[{i:3d}] 得分 {sc:+.4f}  前进 {fwd:+.3f} 侧偏 {lat:+.3f} "
                  f"freq={p['freq']:.1f} stroke={p['stroke']:.2f} "
                  f"lift={p['lift']:.2f} duty={p['duty']:.2f} "
                  f"tibia={p['tibia']:+.2f} adh={p['adhesion']:.2f} "
                  f"seg={np.round(p['seg_scale'], 2)}")
    print("\n最优参数:")
    for k, v in best_p.items():
        print(f"  {k} = {np.round(v, 4) if not isinstance(v, float) else round(v, 4)}")
    fwd, lat, dist, fallen = rollout(body, best_p, seconds=3.0)
    bl = 0.2885
    print(f"\n3 秒复测: 前进 {fwd:.3f} ({fwd / 3 / bl:.2f} 体长/秒) "
          f"侧偏 {lat:+.3f} 翻倒={fallen}")
    body.close()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
