#!/usr/bin/env python
"""数字果蝇命令行入口。

    python cli.py download          下载 MaleCNS v1.0 连接组
    python cli.py build             构建全脑带符号突触网络
    python cli.py calibrate         标定突触强度（让网络不进入自持放电）
    python cli.py doctor            自检：数据规模、递质分布、群体接口、与 neuPrint 对账
    python cli.py sim --exp sugar_pe  跑糖味->伸喙实验
    python cli.py behave --what walk|forage|flight|cube  行为仿真，输出 mp4
    python cli.py web --port 8080     启动交互式可视化
"""
from __future__ import annotations

import argparse
import sys


def cmd_download(args) -> int:
    from digitalfly.download import download_all
    download_all(args.only, force=args.force)
    return 0


def cmd_build(args) -> int:
    from digitalfly import connectome
    print("构建全脑网络（首次需要几分钟，之后秒级加载）")
    c = connectome.build(modulator_sign=args.modulator_sign)
    connectome.save(c)
    print(f"\n已保存 -> {c.stats['n_neurons']:,} 神经元, "
          f"{c.stats['n_edges']:,} 条带符号突触边")
    return 0


def cmd_calibrate(args) -> int:
    from digitalfly import calibrate, connectome
    c = connectome.load()
    print(f"标定突触强度: {c.stats['n_neurons']:,} 神经元, "
          f"{c.stats['n_edges']:,} 边\n")
    r = calibrate.calibrate(c, backend=args.backend)
    calibrate.save(r)
    print(f"\n已保存 -> {calibrate.CALIB_JSON}")
    print(f"  工作点 EPSP = {r['epsp_mv']:.4f} mV/突触 "
          f"(文献值 {r['reference_epsp_mv']} mV 来自 {r['reference']})")
    return 0


def cmd_doctor(args) -> int:
    from digitalfly.doctor import run_doctor
    return run_doctor(check_remote=not args.offline)


def cmd_sim(args) -> int:
    from digitalfly import connectome, viz
    from digitalfly.experiments import sugar_pe

    c = connectome.load()
    print(f"载入连接组: {c.stats['n_neurons']:,} 神经元, "
          f"{c.stats['n_edges']:,} 边\n")
    if args.exp == "sugar_pe":
        res = sugar_pe.run(c, duration_ms=args.duration,
                           amplitude=args.amplitude, backend=args.backend)
        out = viz.raster_plot(res)
        print(f"\n图已保存 -> {out}")
        return 0 if res["verdict"]["sugar_drives_mn9"] else 1

    if args.exp == "cube_decode":
        from digitalfly.experiments import cube_decode
        cube_decode.run(c, n_trials=args.trials, backend=args.backend)
        return 0

    if args.exp == "steering":
        from digitalfly.experiments import steering
        res = steering.run(c, backend=args.backend)
        return 0 if not res["verdict"]["encodes_laterality"] else 0

    if args.exp == "ablation":
        from digitalfly.experiments import ablation
        res = ablation.run(
            c, stim_group="labellar_grns", readout_group="proboscis_motor",
            ablate_groups=["sugar_pathway", "bitter_pathway", "descending"],
            amplitude=args.amplitude, backend=args.backend)
        print("\n  相对完整网络的剩余响应：")
        for k, v in res["ablations"].items():
            print(f"    切除 {k:16s} n={v['n']:<6d} "
                  f"MN9 {v['hz']:6.1f} Hz  ({100 * v['remaining_fraction']:.0f}%)")
        return 0

    print(f"未知实验: {args.exp}")
    return 2


def cmd_behave(args) -> int:
    from digitalfly.behaviors import BEHAVIORS
    fn = BEHAVIORS[args.what]
    kw = dict(seconds=args.seconds, backend=args.backend,
              camera=args.camera, out=args.out)
    if args.what == "cube":
        if args.scramble is not None:
            kw["scramble"] = args.scramble
        if args.camera == "track1":
            kw["camera"] = "cube_cam"
    
    if args.seconds is None:
        kw.pop("seconds")
    ep = fn(**{k: v for k, v in kw.items() if v is not None})
    print()
    print(ep.summary)
    if ep.video:
        print(f"  视频 -> {ep.video}  ({len(ep.frames)} 帧)")
    return 0


def cmd_web(args) -> int:
    from digitalfly.web.server import main as web_main
    return web_main(host=args.host, port=args.port, backend=args.backend,
                    behavior=args.behavior)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dfly", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="下载连接组数据")
    d.add_argument("--only", nargs="*", default=None,
                   choices=["weights", "annotations", "neurotransmitters"])
    d.add_argument("--force", action="store_true", help="忽略已有文件重新下载")
    d.set_defaults(func=cmd_download)

    b = sub.add_parser("build", help="构建全脑网络")
    b.add_argument("--modulator-sign", type=float, default=1.0,
                   dest="modulator_sign",
                   help="单胺类(DA/5-HT/OA)突触的符号，默认 +1 按兴奋处理；"
                        "设 0 可把它们从网络中摘掉做对照")
    b.set_defaults(func=cmd_build)

    cal = sub.add_parser("calibrate", help="标定突触强度")
    cal.add_argument("--backend", default=None)
    cal.set_defaults(func=cmd_calibrate)

    doc = sub.add_parser("doctor", help="自检")
    doc.add_argument("--offline", action="store_true",
                     help="跳过与 neuPrint 服务器的对账")
    doc.set_defaults(func=cmd_doctor)

    s = sub.add_parser("sim", help="跑大脑仿真实验")
    s.add_argument("--exp", default="sugar_pe",
                   choices=["sugar_pe", "ablation", "steering",
                            "cube_decode"])
    s.add_argument("--trials", type=int, default=300,
                   help="cube_decode 采集多少局（每局约 0.7 秒）")
    s.add_argument("--duration", type=float, default=1000.0, help="毫秒")
    s.add_argument("--amplitude", type=float, default=10.0,
                   help="刺激强度 mV/步，默认 10 相当于光遗传学全激活")
    s.add_argument("--backend", default=None,
                   choices=["torch-cuda", "torch-cpu", "scipy"])
    s.set_defaults(func=cmd_sim)

    w = sub.add_parser("behave", help="行为仿真：行走 / 觅食 / 飞行")
    w.add_argument("--what", default="walk",
                   choices=["walk", "forage", "flight", "cube"])
    w.add_argument("--scramble", type=int, default=None,
                   help="魔方打乱步数（只对 --what cube 有效）")
    w.add_argument("--seconds", type=float, default=None)
    w.add_argument("--out", default=None)
    w.add_argument("--camera", default="track1")
    w.add_argument("--backend", default=None)
    w.set_defaults(func=cmd_behave)

    web = sub.add_parser("web", help="交互式可视化")
    web.add_argument("--host", default="0.0.0.0")
    web.add_argument("--port", type=int, default=8080)
    web.add_argument("--backend", default=None)
    web.add_argument("--behavior", default="walk",
                     choices=["walk", "forage", "flight", "cube"],
                     help="启动时的行为，页面上可随时切换")
    web.set_defaults(func=cmd_web)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    sys.exit(args.func(args))
