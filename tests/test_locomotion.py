"""运动层测试：身体配置、步态、脑-体接口。

需要 flybody 模型和迈步数据（setup.sh 会取），缺了就整体跳过。
"""
from __future__ import annotations

import numpy as np
import pytest

from digitalfly import config

FLYBODY_OK = (config.FLYBODY_DIR / "flybody" / "fruitfly" / "assets"
              / "floor.xml").exists()
STEPS_OK = (config.ROOT / "third_party" / "flygym_cpg" / "assets"
            / "single_steps_flybody.npz").exists()

pytestmark = pytest.mark.skipif(
    not (FLYBODY_OK and STEPS_OK),
    reason="缺少 flybody 模型或迈步数据，先运行 setup.sh")


@pytest.fixture(scope="module")
def walk_body():
    from digitalfly.body import FlyBody
    b = FlyBody(mode="walk", render_size=(96, 128))
    yield b
    b.close()


def test_walk_config_has_no_wing_actuators(walk_body):
    """行走配置必须收起翅膀。

    翅膀关节行程很大，只要给非零控制量就会被转到远离身体的位置，
    看上去像是掉了下来 —— 真实果蝇行走时翅膀是折叠贴合的。
    """
    assert not any("wing" in n for n in walk_body.act_names)
    assert any("coxa_T1_left" == n for n in walk_body.act_names)
    assert walk_body.claw_actuator("T1_left") is not None


def test_rest_pose_is_stable(walk_body):
    """控制量全 0 时果蝇应当稳稳站住 —— 这是步态叠加的基线姿态。"""
    walk_body.reset()
    for _ in range(3000):
        walk_body.step(walk_body.rest)
    assert not walk_body.fallen()
    assert abs(float(walk_body.data.qvel[2])) < 1.0


def test_gait_loads_real_kinematics(walk_body):
    """六条腿都要拿到真实果蝇的迈步轨迹。"""
    from digitalfly.locomotion import PreprogrammedGait
    g = PreprogrammedGait(walk_body)
    assert g.n_legs == 6
    for leg, L in g._legs.items():
        assert L["neutral"].shape == (7,)
        assert 0 < L["swing_end"] < 2 * np.pi


def test_gait_tripod_phasing(walk_body):
    """三角步态：同组的腿同相，异组差半周期。"""
    from digitalfly.locomotion import TRIPOD_A, TRIPOD_B, PreprogrammedGait
    g = PreprogrammedGait(walk_body)
    for leg in TRIPOD_A:
        assert g._legs[leg]["phase0"] == pytest.approx(0.0)
    for leg in TRIPOD_B:
        assert g._legs[leg]["phase0"] == pytest.approx(np.pi)


def test_gait_walks_forward(walk_body):
    """给满速指令，果蝇应当明显前进且不翻倒。

    真实果蝇自由行走约 1-3 体长/秒，这里要求至少 1 体长/秒。
    """
    from digitalfly.locomotion import PreprogrammedGait
    g = PreprogrammedGait(walk_body)
    walk_body.reset()
    ctrl = walk_body.rest.copy()
    dt, sub, T = walk_body.timestep, 20, 2.0
    h0 = walk_body.heading().copy()
    for _ in range(int(T / (dt * sub))):
        g.step(ctrl, dt * sub, speed=1.0, turn=0.0, adhesion=0.5)
        walk_body.step(ctrl, n_substeps=sub)
    fwd = float(walk_body.displacement()[:2] @ h0)
    assert not walk_body.fallen()
    assert fwd / T / 0.2885 > 1.0, f"只有 {fwd / T / 0.2885:.2f} 体长/秒"


def test_gait_turn_sign(walk_body):
    """转向方向约定：turn > 0 向左转，turn < 0 向右转。"""
    from digitalfly.locomotion import PreprogrammedGait

    def heading_change(turn):
        g = PreprogrammedGait(walk_body)
        walk_body.reset()
        ctrl = walk_body.rest.copy()
        dt, sub, T = walk_body.timestep, 20, 1.5
        h0 = walk_body.heading().copy()
        for _ in range(int(T / (dt * sub))):
            g.step(ctrl, dt * sub, speed=0.8, turn=turn, adhesion=0.5)
            walk_body.step(ctrl, n_substeps=sub)
        h = walk_body.heading()
        return np.degrees(np.arctan2(h0[0] * h[1] - h0[1] * h[0], h0 @ h))

    left, right = heading_change(+0.4), heading_change(-0.4)
    assert left > 10, f"turn=+0.4 应当明显左转，实际 {left:.1f}°"
    assert right < -10, f"turn=-0.4 应当明显右转，实际 {right:.1f}°"


def test_gait_trim_goes_straight(walk_body):
    """配平之后 turn=0 应当基本走直线。"""
    from digitalfly.locomotion import PreprogrammedGait
    g = PreprogrammedGait(walk_body)
    walk_body.reset()
    ctrl = walk_body.rest.copy()
    dt, sub, T = walk_body.timestep, 20, 3.0
    h0 = walk_body.heading().copy()
    for _ in range(int(T / (dt * sub))):
        g.step(ctrl, dt * sub, speed=0.8, turn=0.0, adhesion=0.5)
        walk_body.step(ctrl, n_substeps=sub)
    h = walk_body.heading()
    drift = abs(np.degrees(np.arctan2(h0[0] * h[1] - h0[1] * h[0], h0 @ h)))
    assert drift < 15, f"3 秒内偏了 {drift:.1f}°"


def test_flight_config_has_wings_no_legs():
    """飞行配置：翅膀可动、腿收起、翅膀带空气动力学。"""
    from digitalfly.body import FlyBody
    b = FlyBody(mode="flight", render_size=(96, 128))
    try:
        assert any("wing_yaw_left" == n for n in b.act_names)
        assert not any("coxa" in n for n in b.act_names)
        # 椭球流体模型必须已经打开，否则翅膀产生不了升力
        fluid = np.asarray(b.model.geom_fluid).reshape(b.model.ngeom, -1)
        assert int((np.abs(fluid).sum(axis=1) > 0).sum()) == 2
    finally:
        b.close()


@pytest.mark.skipif(not config.GRAPH_NPZ.exists(),
                    reason="尚未构建连接组")
def test_bridge_lateralises_populations():
    """脑-体接口必须能把下行、嗅觉、味觉神经元分出左右。

    侧别字段在 MaleCNS 里是分散的：运动神经元在 somaSide，
    感觉神经元在 rootSide —— 只看一个字段会得到空集。
    """
    from digitalfly.bridge import CommandBridge
    from digitalfly.connectome import load
    br = CommandBridge(load(), verbose=False)
    assert len(br.dn_left) > 100 and len(br.dn_right) > 100
    assert len(br.olf_left) > 10 and len(br.olf_right) > 10
    assert len(br.food_orns) > 50
    assert len(br.labellar) > 50
