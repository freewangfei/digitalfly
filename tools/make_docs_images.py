"""生成 README 里用的配图，输出到 docs/images/。

跑法: python tools/make_docs_images.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from digitalfly import config                      # noqa: E402
from digitalfly.body import FlyBody                # noqa: E402
from digitalfly.brainview import (BrainView, draw_overlay,  # noqa: E402
                                  side_by_side)
from digitalfly.connectome import load             # noqa: E402
from digitalfly.cube_scene import CubeScene        # noqa: E402
from digitalfly.locomotion import (PreprogrammedGait,  # noqa: E402
                                   WingBeat)
from digitalfly.neurons import NeuronIndex         # noqa: E402

OUT = config.ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)
SIZE = (480, 620)


def save(name: str, img: np.ndarray) -> None:
    import PIL.Image as I
    p = OUT / name
    I.fromarray(img).save(p, optimize=True)
    print(f"  {p.relative_to(config.ROOT)}  {img.shape[1]}x{img.shape[0]}")


def strip(frames, k: int = 4) -> np.ndarray:
    sel = [frames[int(i * (len(frames) - 1) / (k - 1))] for i in range(k)]
    return np.concatenate(sel, axis=1)


def walk_strip():
    """行走：真实果蝇迈步运动学驱动的三角步态。"""
    import mujoco
    body = FlyBody(mode="walk", render_size=SIZE)
    gait = PreprogrammedGait(body)
    ctrl = body.rest.copy()
    dt, sub = body.timestep, 20
    frames = []
    n = int(1.8 / (dt * sub))
    for i in range(n):
        gait.step(ctrl, dt * sub, speed=1.0, turn=0.0, adhesion=0.5)
        body.step(ctrl, n_substeps=sub)
        if i % (n // 4) == 0:
            mujoco.mj_forward(body.model, body.data)
            frames.append(body.render("track1"))
    save("walk.jpg", strip(frames))
    body.close()


def rest_pose():
    """静息姿态：翅膀折叠贴在腹部，六足站立。"""
    import mujoco
    body = FlyBody(mode="walk", render_size=SIZE)
    body.reset()
    for _ in range(3000):
        body.step(body.rest)
    mujoco.mj_forward(body.model, body.data)
    imgs = [body.render(c) for c in ("hero", "side", "back")]
    save("rest_pose.jpg", np.concatenate(imgs, axis=1))
    body.close()


def flight_strip():
    """拴系飞行：躯干抬头 47.5°，翅膀 218 Hz 拍动。"""
    import mujoco
    body = FlyBody(mode="flight", render_size=SIZE)
    dt, sub = body.timestep, 4
    wings = WingBeat(body, dt_ctrl=dt * sub)
    for a in wings.act:
        body.model.actuator_gainprm[a][0] = 70.0
    body.reset()
    wings.reset()
    ctrl = body.rest.copy()
    tether = np.array(body.data.qpos[:7])
    frames = []
    for i in range(340):
        wings.step(ctrl, amplitude=1.0)
        body.step(ctrl, n_substeps=sub)
        body.data.qpos[:7] = tether
        body.data.qvel[:6] = 0.0
        if i % 85 == 0:
            mujoco.mj_forward(body.model, body.data)
            frames.append(body.render("flight_close"))
    save("flight.jpg", strip(frames))
    body.close()


def cube_strip(c):
    """魔方：从打乱到复原。"""
    import mujoco
    from digitalfly.cube import solve
    scene = CubeScene(size=0.55, pos=(0.55, 0.0, 0.28))
    body = FlyBody(mode="walk", render_size=SIZE, cube_scene=scene)
    scene.bind(body.model, body.data)
    seq = scene.state.scramble(8, seed=7)
    solution, _ = solve(scene.state.copy(), fallback=seq)

    frames = []
    scene.write()
    mujoco.mj_forward(body.model, body.data)
    frames.append(body.render("cube_close"))
    for mv in solution:
        scene.start_move(mv, duration=0.25)
        for _ in range(30):
            scene.advance(0.01)
            scene.write()
        scene.write()
        mujoco.mj_forward(body.model, body.data)
        frames.append(body.render("cube_close"))
    save("cube.jpg", strip(frames, k=5))
    body.close()


def brain_view(c):
    """全脑三维点云：真实胞体坐标，发放时闪烁。"""
    idx = NeuronIndex(c)
    view = BrainView(c.meta, idx.by_region(), size=(620, 700))
    rng = np.random.default_rng(0)
    spikes = np.zeros(len(c.meta), dtype=np.float32)
    spikes[rng.choice(len(spikes), 9000, replace=False)] = 1
    view.update(spikes, 0.5)
    img = draw_overlay(
        view.render(), "全脑活动 · MaleCNS v1.0  Whole-brain activity",
        [f"{len(c.meta):,} 神经元 neurons · "
         f"{c.stats['n_edges'] / 1e6:.0f}M 突触边 signed synaptic edges",
         "白点 = 此刻正在发放  white = spiking now",
         "坐标来自数据集的 somaLocation 字段（14.2 万个神经元）"],
        view.labels)
    save("brainview.jpg", img)


def split_demo(c):
    """分屏：任务 + 全脑活动，网上那批视频的格式。"""
    import mujoco
    from digitalfly.cube import solve
    scene = CubeScene(size=0.55, pos=(0.55, 0.0, 0.28))
    body = FlyBody(mode="walk", render_size=(480, 600), cube_scene=scene)
    scene.bind(body.model, body.data)
    seq = scene.state.scramble(8, seed=7)
    solution, src = solve(scene.state.copy(), fallback=seq)
    for mv in solution[:3]:
        scene.state.apply(mv)
    scene.start_move(solution[3], duration=0.25)
    for _ in range(12):
        scene.advance(0.01)
    scene.write()
    mujoco.mj_forward(body.model, body.data)

    view = BrainView(c.meta, NeuronIndex(c).by_region(), size=(480, 560))
    rng = np.random.default_rng(1)
    spikes = np.zeros(len(c.meta), dtype=np.float32)
    spikes[rng.choice(len(spikes), 7000, replace=False)] = 1
    view.update(spikes, 0.5)

    left = draw_overlay(body.render("cube_close"), "数字果蝇 · 拧魔方",
                        [f"第 4/{len(solution)} 步　下一步 {solution[3]}",
                         "蓄力（下行神经元活动） 100%",
                         f"解法 {src}　果蝇选面一致 1/3"])
    right = draw_overlay(view.render(), "全脑活动 · MaleCNS v1.0",
                         [f"{len(c.meta):,} 神经元 · 白点 = 此刻发放",
                          "下行神经元 4.5 Hz"], view.labels)
    save("split_cube.jpg", side_by_side(left, right))
    body.close()


def main() -> None:
    print("生成配图 ->", OUT)
    c = load()
    rest_pose()
    walk_strip()
    flight_strip()
    brain_view(c)
    cube_strip(c)
    split_demo(c)
    print("完成")


if __name__ == "__main__":
    main()
