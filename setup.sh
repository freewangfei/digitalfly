#!/usr/bin/env bash
# 数字果蝇 —— 环境搭建
#
# 建一个独立的 conda 环境 digitalfly (Python 3.12)。
# 用独立环境是为了不污染 base，也避免 mujoco/dm_control 拖动 base 的依赖版本。
#
# 注意：本机 IPv6 默认路由的 TLS 握手是坏的（详见 tools/ipv4.py），
# 所以所有 pip/conda 网络操作都经过 tools/ipv4.py 强制走 IPv4。
set -euo pipefail

ENV_NAME=digitalfly
PY_VER=3.12
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="$(conda info --base)"
IPV4="${CONDA_BASE}/bin/python ${ROOT}/tools/ipv4.py"

echo "==> [1/5] 创建 conda 环境 ${ENV_NAME} (python ${PY_VER})"
if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "    环境已存在，跳过创建"
else
    $IPV4 conda create -n "${ENV_NAME}" "python=${PY_VER}" -y
fi

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
PIP="python ${ROOT}/tools/ipv4.py pip"

echo "==> [2/5] 安装基础依赖"
$PIP install -q --upgrade pip
$PIP install -q numpy scipy pandas pyarrow flask pillow matplotlib \
                imageio imageio-ffmpeg mediapy h5py tqdm pytest requests

echo "==> [3/5] 安装 MuJoCo + dm_control（身体模型运行时）"
# dm_control 1.0.45 + mujoco 3.13 是实测可配的一组；
# 更老的 dm_control 不认 mujoco 新增的字段，更新的又要求更高版本的 mujoco。
$PIP install -q "mujoco>=3.13" "dm_control==1.0.45"

echo "==> [4/5] 获取 flybody 果蝇身体模型 (Google DeepMind x HHMI Janelia)"
FLYBODY_DIR="${ROOT}/third_party/flybody"
if [ -f "${FLYBODY_DIR}/pyproject.toml" ]; then
    echo "    已存在，跳过下载"
else
    # 用 curl -4 拉 tarball 而不是 git clone：git 走 libcurl 但没法强制 IPv4，
    # 而本机的 IPv6 路径 TLS 是坏的，git clone 会永久卡住。
    echo "    下载 tarball (~90MB)"
    curl -4 -sS --http1.1 --retry 5 --retry-all-errors -L \
        -o /tmp/flybody.tar.gz \
        https://codeload.github.com/TuragaLab/flybody/tar.gz/refs/heads/main
    mkdir -p "${FLYBODY_DIR}"
    tar -xzf /tmp/flybody.tar.gz -C "${FLYBODY_DIR}" --strip-components=1
    rm -f /tmp/flybody.tar.gz
fi
# 不需要 pip install flybody：我们只用它的 MuJoCo 模型，body.py 直接按路径加载
# assets/floor.xml，不 import flybody 这个包。它的 pyproject 还会把 numpy 拖回
# 1.26.4，装了反而破坏上面的环境。
echo "    模型文件: ${FLYBODY_DIR}/flybody/fruitfly/assets/floor.xml"

echo "==> [5/5] 获取真实果蝇迈步运动学 (NeLy-EPFL/flygym)"
# NeuroMechFly v2 / flygym 把真实果蝇在球上行走的腿部运动学重定向到了
# flybody 模型上。用它的数据比自己按正弦拼一套步态可靠得多：
# 自调步态只能走到约 0.5 体长/秒且姿态勉强，真实运动学能到 2.6 体长/秒。
# 只取数据文件，不装 flygym 这个包（它会把 mujoco 钉回 3.9，和 dm_control 冲突）。
CPG_DIR="${ROOT}/third_party/flygym_cpg"
if [ -f "${CPG_DIR}/assets/single_steps_flybody.npz" ]; then
    echo "    已存在，跳过下载"
else
    mkdir -p "${CPG_DIR}/assets"
    BASE="https://raw.githubusercontent.com/NeLy-EPFL/flygym/HEAD/src/flygym_demo/complex_terrain"
    for f in cpg_controller.py preprogrammed.py \
             assets/single_steps_flybody.npz \
             assets/single_steps_flybody.meta.json \
             assets/flybody_step_selection.json; do
        curl -4 -sS --http1.1 --retry 5 --retry-all-errors -L \
             -o "${CPG_DIR}/${f}" "${BASE}/${f}"
    done
    echo "    迈步数据 -> ${CPG_DIR}/assets/single_steps_flybody.npz"
fi

echo
echo "==> 可选：安装 CUDA 版 PyTorch 以加速全脑仿真"
echo "    (装不上也没关系，大脑引擎会自动降级到 scipy 后端)"
set +e
$PIP install torch
set -e

echo
echo "==================================================="
echo " 环境就绪。使用方式："
echo "   conda activate ${ENV_NAME}"
echo "   cd ${ROOT}"
echo "   python cli.py download   # 下载 MaleCNS v1.0 连接组"
echo "   python cli.py build      # 构建全脑网络"
echo "   python cli.py calibrate  # 标定突触强度"
echo "   python cli.py doctor     # 自检"
echo "   python cli.py behave --what walk|forage|flight   # 行为仿真"
echo "==================================================="
