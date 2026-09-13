"""从 Google Storage 下载 MaleCNS v1.0 连接组数据。

传输层用 curl 而不是 requests，原因是实测：本机有一条 IPv6 默认路由，
TCP 能连上但 TLS 握手会被中途拒绝（`SSLV3_ALERT_ILLEGAL_PARAMETER`），
表现为大约一半的请求瞬间失败。强制 IPv4 后成功率 10/10。
Python 的 socket 只在 TCP 层失败时才回退到下一个地址族，而这里 TCP 是通的，
所以 requests 无法自愈 —— 用 curl 的 `-4` 直接绕开。
"""
from __future__ import annotations

import concurrent.futures as cf
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import config

# 超过这个大小就分片并发下载。
# 实测单条连接到 GCS 被限在 100~500 KB/s，4 条并发能聚合到 ~1.7 MB/s，
# 所以阈值压得比较低，让中等大小的文件也走并发。
PARALLEL_THRESHOLD = 8 << 20
N_CONN = 8


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def _curl(args: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["curl", "-sS", "-4", "--http1.1", "--retry", "10", "--retry-all-errors",
         "--retry-delay", "2", "--connect-timeout", "30", *args],
        capture_output=True, text=True, timeout=timeout,
    )


def remote_size(url: str) -> int | None:
    """HEAD 取 Content-Length。"""
    p = _curl(["-I", "-w", "%{size_upload}", url], timeout=120)
    for line in p.stdout.splitlines():
        if line.lower().startswith("content-length:"):
            return int(line.split(":", 1)[1].strip())
    return None


def _fetch_range(url: str, start: int, end: int, part: Path) -> int:
    """下载 [start, end] 字节区间到 part 文件，支持续传。"""
    want = end - start + 1
    have = part.stat().st_size if part.exists() else 0
    if have >= want:
        return want
    p = _curl(["-o", str(part), "-C", "-", "-r", f"{start}-{end}", url])
    if p.returncode != 0:
        raise RuntimeError(f"分片 {start}-{end} 下载失败: {p.stderr.strip()}")
    got = part.stat().st_size
    if got != want:
        raise RuntimeError(f"分片 {start}-{end} 大小不符: {got} != {want}")
    return got


def _download_parallel(url: str, dest: Path, total: int) -> None:
    parts_dir = dest.parent / f".{dest.name}.parts"
    parts_dir.mkdir(exist_ok=True)
    size = (total + N_CONN - 1) // N_CONN
    jobs = []
    for i in range(N_CONN):
        s = i * size
        e = min(s + size - 1, total - 1)
        if s <= e:
            jobs.append((i, s, e, parts_dir / f"part{i:02d}"))

    t0 = time.time()
    start_bytes = sum(p.stat().st_size for _, _, _, p in jobs if p.exists())

    def _progress() -> None:
        done = sum(p.stat().st_size for _, _, _, p in jobs if p.exists())
        rate = (done - start_bytes) / max(time.time() - t0, 1e-6)
        sys.stdout.write(
            f"\r    {100 * done / total:5.1f}%  {_human(done)}/{_human(total)}"
            f"  {_human(rate)}/s  ({N_CONN} 路并发)   ")
        sys.stdout.flush()

    with cf.ThreadPoolExecutor(max_workers=N_CONN) as pool:
        futs = [pool.submit(_fetch_range, url, s, e, p) for _, s, e, p in jobs]
        while not all(f.done() for f in futs):
            _progress()
            time.sleep(2.0)
        for f in futs:
            f.result()          # 抛出分片里的异常
    _progress()
    sys.stdout.write("\n")

    with open(dest, "wb") as out:
        for _, _, _, p in jobs:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out, 1 << 22)
    shutil.rmtree(parts_dir)


def download_one(name: str, force: bool = False) -> Path:
    """下载单个数据文件，返回本地路径。已完整下载则跳过。"""
    fname, min_size, desc = config.FILES[name]
    url = config.url_for(name)
    dest = config.raw_path(name)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if force and dest.exists():
        dest.unlink()

    total = remote_size(url)
    have = dest.stat().st_size if dest.exists() else 0
    if total is not None and have == total:
        print(f"  [已完整] {fname}  ({_human(have)})")
        return dest
    if total is None:
        raise RuntimeError(f"无法获取 {fname} 的远端大小，请检查网络")

    print(f"  [下载] {fname}  {_human(total)}\n         {desc}")
    t0 = time.time()
    if total >= PARALLEL_THRESHOLD:
        _download_parallel(url, dest, total)
    else:
        p = _curl(["-o", str(dest), "-C", "-", url])
        if p.returncode != 0:
            raise RuntimeError(f"下载失败: {p.stderr.strip()}")

    got = dest.stat().st_size
    if got != total:
        raise RuntimeError(f"{fname} 下载不完整: {got} != {total}")
    if got < min_size:
        raise RuntimeError(f"{fname} 大小异常: {_human(got)} < 预期下界 {_human(min_size)}")
    print(f"    完成 {_human(got)}  用时 {time.time() - t0:.0f}s")
    return dest


def download_all(names: list[str] | None = None, force: bool = False) -> None:
    config.ensure_dirs()
    names = names or list(config.FILES)
    print("MaleCNS v1.0 连接组 (HHMI Janelia / Cambridge / Google Research, CC-BY 4.0)")
    print(f"来源: {config.GCS_BASE}\n")
    for n in names:
        download_one(n, force=force)
    print("\n全部数据就绪 ->", config.RAW_DIR)
