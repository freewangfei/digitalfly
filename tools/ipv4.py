"""强制 IPv4 的通用启动器。

本机有一条 IPv6 默认路由，TCP 能连上但 TLS 握手会被中途拒绝
(`SSLV3_ALERT_ILLEGAL_PARAMETER`)，导致 pip/conda 大约一半的请求瞬间失败。
Python 的 socket 只在 TCP 层失败时才回退到下一个地址族，这里 TCP 是通的，
所以不会自愈。

这个包装器在进程内把 `socket.getaddrinfo` 过滤成只返回 IPv4 结果，
然后把控制权交给目标模块 —— 不需要改动 /etc/gai.conf 之类的系统配置。

用法:
    python tools/ipv4.py pip install numpy
    python tools/ipv4.py conda create -n foo python=3.12
    python tools/ipv4.py -m some.module arg1
"""
from __future__ import annotations

import runpy
import socket
import sys

_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_only(*args, **kwargs):
    res = _orig_getaddrinfo(*args, **kwargs)
    v4 = [r for r in res if r[0] == socket.AF_INET]
    return v4 or res  # 万一某个主机真的只有 IPv6，别把路彻底堵死


socket.getaddrinfo = _ipv4_only

def _run_pip(args: list[str]) -> int:
    from pip._internal.cli.main import main as pip_main
    return pip_main(args)


def _run_conda(args: list[str]) -> int:
    from conda.cli.main import main_subshell
    return main_subshell(*args) or 0


_ENTRYPOINTS = {"pip": _run_pip, "conda": _run_conda}


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    target, rest = sys.argv[1], sys.argv[2:]

    if target == "-m":
        if not rest:
            print("用法: python tools/ipv4.py -m <module> [args...]")
            return 2
        sys.argv = [rest[0], *rest[1:]]
        runpy.run_module(rest[0], run_name="__main__", alter_sys=True)
        return 0

    if target not in _ENTRYPOINTS:
        print(f"未知目标 {target!r}，支持: {', '.join(_ENTRYPOINTS)} 或 -m <module>")
        return 2

    sys.argv = [target, *rest]
    return _ENTRYPOINTS[target](rest)


if __name__ == "__main__":
    sys.exit(main())
