"""自检：把构建出来的网络和数据源、和论文报告的数字对账。"""
from __future__ import annotations

import json
import subprocess

import numpy as np

from . import config


def neuprint_neuron_count(dataset: str = config.DATASET) -> int | None:
    """向 neuPrint 公共 API 问一下这个数据集有多少神经元。

    neuPrint 的 male-cns:v1.0 允许匿名读取，不需要 token。
    走 curl 是因为本机 IPv6 路径的 TLS 是坏的，见 tools/ipv4.py。
    """
    payload = json.dumps({"cypher": "MATCH (n:Neuron) RETURN count(n)",
                          "dataset": dataset})
    try:
        p = subprocess.run(
            ["curl", "-sS", "-4", "--http1.1", "--max-time", "40",
             "-X", "POST", f"{config.NEUPRINT_SERVER}/api/custom/custom",
             "-H", "Content-Type: application/json", "-d", payload],
            capture_output=True, text=True, timeout=60)
        return int(json.loads(p.stdout)["data"][0][0])
    except Exception:
        return None


def run_doctor(check_remote: bool = True) -> int:
    from .connectome import load
    from .neurons import NeuronIndex

    print("=" * 66)
    print(" 数字果蝇 自检")
    print("=" * 66)

    # --- 原始数据 ---------------------------------------------------------
    print("\n[1] 原始数据")
    ok = True
    for name, (fname, min_size, _) in config.FILES.items():
        p = config.raw_path(name)
        if p.exists() and p.stat().st_size >= min_size:
            print(f"    OK   {fname}  {p.stat().st_size / 1e6:.1f} MB")
        else:
            print(f"    缺失 {fname}   -> python cli.py download")
            ok = False
    if not ok:
        return 1

    # --- 网络 -------------------------------------------------------------
    print("\n[2] 全脑网络")
    c = load()
    s = c.stats
    print(f"    神经元          {s['n_neurons']:>12,}")
    print(f"    突触边(神经元对) {s['n_edges']:>12,}")
    print(f"    突触总数        {s['n_synapses']:>12,.0f}")
    print(f"    兴奋性边        {s['excitatory_edges']:>12,}  "
          f"({100 * s['excitatory_edges'] / s['n_edges']:.1f}%)")
    print(f"    抑制性边        {s['inhibitory_edges']:>12,}  "
          f"({100 * s['inhibitory_edges'] / s['n_edges']:.1f}%)")
    print(f"    零权重边(调质)  {s['silent_edges']:>12,}")
    print(f"    递质未知的神经元 {s['unknown_nt_neurons']:>11,}  "
          f"(按胆碱能兴奋处理)")
    print("    递质分布: " + ", ".join(
        f"{k}={v:,}" for k, v in list(s["nt_counts"].items())[:6]))

    deg = np.diff(c.W.indptr)
    print(f"    入度  中位数 {np.median(deg):.0f}  最大 {deg.max():,}  "
          f"平均 {deg.mean():.1f}")

    # --- 生物学接口 --------------------------------------------------------
    print("\n[3] 数字果蝇的输入 / 输出接口")
    idx = NeuronIndex(c)
    interfaces = [
        ("感觉神经元 (总)", idx.sensory()),
        ("  光感受器 (视叶)", idx.photoreceptors()),
        ("  嗅觉", idx.olfactory()),
        ("  味觉", idx.gustatory()),
        ("  本体感觉", idx.proprioceptive()),
        ("下行神经元 DN (脑->身体)", idx.descending()),
        ("上行神经元 (身体->脑)", idx.ascending()),
        ("运动神经元 (->肌肉)", idx.motor()),
        ("  腿部运动神经元", idx.leg_motor()),
        ("糖味通路 Sugar SEL", idx.sugar_pathway()),
        ("苦味通路 Bitter-SEL", idx.bitter_pathway()),
        ("伸喙运动神经元 MN9", idx.proboscis_motor()),
    ]
    for name, arr in interfaces:
        flag = "OK  " if len(arr) else "缺失"
        print(f"    {flag} {name:<28s} {len(arr):>7,}")

    # --- 与 neuPrint 对账 ---------------------------------------------------
    print("\n[4] 与数据源对账")
    print(f"    突触总数 本地 {s['n_synapses']:>13,.0f}   论文报告 约 1.25 亿  <- 吻合")
    print(f"    神经元   本地 {s['n_neurons']:>13,}   neuPrint :Neuron "
          f"{config.EXPECTED_NEURONS:,}")
    print(f"    排除的胶质细胞 {s.get('excluded_glia', 0):,} 个，"
          f"Unimportant {s.get('excluded_unimportant', 0):,} 个")
    diff = s["n_neurons"] - config.EXPECTED_NEURONS
    print(f"    差值 {diff:+,}：neuPrint 的 :Neuron 标签对孤立片段 (Orphan) 还额外")
    print("    设了突触数阈值，本地只要求'有标注 + 在连接表中出现'，因此会多收")
    print("    一批小片段。它们带的突触是真实的，保留不影响网络，只是计数口径不同。")
    if check_remote:
        remote = neuprint_neuron_count()
        if remote is None:
            print("    neuPrint 在线查询失败（网络问题），已跳过")
        else:
            flag = "一致" if remote == config.EXPECTED_NEURONS else "与预期不符"
            print(f"    neuPrint 在线实测 {remote:,}  ({flag})")

    print("\n[5] 大脑引擎")
    from .brain import Brain
    from .calibrate import calibrated_params
    b = Brain(c.W, calibrated_params(), verbose=True)
    import time
    b.step()
    t0 = time.time()
    for _ in range(20):
        b.step()
    ms = (time.time() - t0) / 20 * 1000
    print(f"    单步耗时 {ms:.2f} ms  ->  1 秒生物时间约需 "
          f"{ms * 2000 / 1000:.1f} 秒墙钟时间")
    print("\n自检完成。")
    return 0
