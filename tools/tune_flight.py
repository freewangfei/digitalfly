"""标定飞行控制器：让果蝇能稳定悬停，而不是一边产生升力一边翻滚。

目标是悬停 —— 高度维持在起始高度附近、姿态不翻、水平不乱窜。
悬停站得住，前飞和转向就只是在稳定点上叠加指令。

跑法: python tools/tune_flight.py [试验次数]
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from digitalfly.body import FlyBody              # noqa: E402
from digitalfly.locomotion import WingBeat       # noqa: E402

T_SIM = 1.2


def rollout(body, wings, p: dict, seconds: float = T_SIM) -> dict:
    for a in wings.act:
        body.model.actuator_gainprm[a][0] = p["gain"]
    body.reset()
    wings.reset()
    ctrl = body.rest.copy()
    dt = 2e-4
    sub = int(round(dt / body.timestep))
    abdomen = body.act_index.get("abdomen")
    z0 = body.root_pos[2]

    zs, tilts = [], []
    for _ in range(int(seconds / dt)):
        q = body.root_quat
        w, x, y, z = q
        roll = 2 * (w * x + y * z)
        pitch = 2 * (w * y - z * x)
        up_z = 1 - 2 * (x * x + y * y)
        omega = np.asarray(body.data.qvel[3:6])

        asym = float(np.clip(-p["kp_roll"] * roll - p["kd_roll"] * omega[0],
                             -0.8, 0.8))
        dz = z0 - body.root_pos[2]
        vz = float(body.data.qvel[2])
        amp = float(np.clip(p["amp"] + p["kp_z"] * np.clip(dz, -1, 1)
                            - p["kd_z"] * vz, 0.4, 1.8))
        wings.step(ctrl, freq_rel=0.0, asymmetry=asym, amplitude=amp)
        if abdomen is not None:
            ctrl[abdomen] = float(np.clip(
                -p["kp_pitch"] * pitch - p["kd_pitch"] * omega[1], -0.8, 0.6))
        body.step(ctrl, n_substeps=sub)

        qvel = body.data.qvel
        if not np.all(np.isfinite(qvel)) or np.abs(qvel).max() > 1e4:
            return {"ok": False}
        zs.append(float(body.root_pos[2]))
        tilts.append(float(up_z))

    d = body.displacement()
    if not np.all(np.isfinite(d)):
        return {"ok": False}
    return {
        "ok": True,
        "z_err": float(np.mean(np.abs(np.array(zs) - z0))),
        "min_up": float(np.min(tilts)),
        "drift": float(np.linalg.norm(d[:2])),
        "final_z": float(zs[-1]),
        "z0": float(z0),
    }


def score(r: dict) -> float:
    if not r["ok"]:
        return -1e3
    # 悬停：高度误差小、不翻（up_z 接近 1）、水平漂移小
    return -(3.0 * r["z_err"] + 6.0 * (1.0 - r["min_up"]) + 0.5 * r["drift"])


def main(n_trials: int = 120) -> None:
    rng = np.random.default_rng(0)
    body = FlyBody(mode="flight", render_size=(96, 128))
    wings = WingBeat(body, dt_ctrl=2e-4)
    best, best_p = -1e9, None
    for i in range(n_trials):
        p = {
            "gain": float(rng.uniform(30, 140)),
            "amp": float(rng.uniform(0.7, 1.3)),
            "kp_roll": float(rng.uniform(0.0, 4.0)),
            "kd_roll": float(rng.uniform(0.0, 0.5)),
            "kp_pitch": float(rng.uniform(0.0, 4.0)),
            "kd_pitch": float(rng.uniform(0.0, 0.5)),
            "kp_z": float(rng.uniform(0.0, 1.0)),
            "kd_z": float(rng.uniform(0.0, 0.3)),
        }
        r = rollout(body, wings, p)
        sc = score(r)
        if sc > best:
            best, best_p = sc, p
            print(f"[{i:3d}] 得分 {sc:+8.3f}  高度误差 {r.get('z_err', 0):.3f} "
                  f"最小直立度 {r.get('min_up', 0):.2f} 漂移 {r.get('drift', 0):.2f}"
                  f"  gain={p['gain']:.0f} amp={p['amp']:.2f} "
                  f"roll=({p['kp_roll']:.2f},{p['kd_roll']:.3f}) "
                  f"pitch=({p['kp_pitch']:.2f},{p['kd_pitch']:.3f}) "
                  f"z=({p['kp_z']:.2f},{p['kd_z']:.3f})", flush=True)
    print("\n最优参数:")
    for k, v in best_p.items():
        print(f"  {k} = {round(v, 4)}")
    r = rollout(body, wings, best_p, seconds=2.0)
    print(f"\n2 秒复测: 高度 {r['z0']:.2f} -> {r['final_z']:.2f}  "
          f"平均高度误差 {r['z_err']:.3f}  最小直立度 {r['min_up']:.2f}  "
          f"水平漂移 {r['drift']:.2f}")
    body.close()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
