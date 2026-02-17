#!/bin/bash
# ============================================================
# OThink-R1 环境搭建脚本
# 使用 conda + uv 创建虚拟环境
#
# 使用方法:
#   chmod +x setup_env.sh
#   bash setup_env.sh
#
# 工作流程:
#   1. conda create 创建基础环境（提供 Python 解释器）
#   2. 在 conda 环境内用 uv sync 创建 .venv 并安装所有依赖
#   3. A100 集群: rsync 项目过去（不含 .venv），运行本脚本即可
# ============================================================

set -e

# ==================== 配置区 ====================
CONDA_ENV_NAME="othink-r1"
PYTHON_VERSION="3.11"
# ================================================

echo "=========================================="
echo "  OThink-R1 环境搭建 (conda + uv)"
echo "=========================================="

# ---------- Step 1: 创建 Conda 环境 ----------
echo ""
echo "[1/4] 创建 Conda 环境: ${CONDA_ENV_NAME} (Python ${PYTHON_VERSION})"

if conda env list | grep -qw "${CONDA_ENV_NAME}"; then
    echo "  ⚠️  环境 '${CONDA_ENV_NAME}' 已存在，跳过创建"
else
    conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} -y
    echo "  ✅ Conda 环境创建完成"
fi

# ---------- Step 2: 激活 Conda 环境 ----------
echo ""
echo "[2/4] 激活 Conda 环境"

eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV_NAME}

echo "  ✅ 当前 Python: $(which python)"
echo "  ✅ Python 版本: $(python --version)"

# ---------- Step 3: 安装 uv ----------
echo ""
echo "[3/4] 检查/安装 uv"

if command -v uv &> /dev/null; then
    echo "  ✅ uv 已安装: $(uv --version)"
else
    echo "  📦 正在安装 uv..."
    pip install uv
    echo "  ✅ uv 安装完成: $(uv --version)"
fi

# ---------- Step 4: uv sync ----------
echo ""
echo "[4/4] 创建 .venv 虚拟环境并安装依赖 (uv sync)"

if [ ! -f "pyproject.toml" ]; then
    echo "  ❌ 错误: 未找到 pyproject.toml，请确保在项目根目录运行此脚本"
    exit 1
fi

# uv sync 会自动:
#   1. 基于当前 conda 环境的 Python 创建 .venv
#   2. 根据 pyproject.toml 解析依赖
#   3. 生成/更新 uv.lock 锁文件
#   4. 安装所有依赖到 .venv
uv sync

echo ""
echo "=========================================="
echo "  ✅ 环境搭建完成！"
echo "=========================================="
echo ""
echo "使用方式（二选一）:"
echo ""
echo "  方式1 - 手动激活后运行:"
echo "    conda activate ${CONDA_ENV_NAME}"
echo "    source .venv/bin/activate"
echo "    python training.py ..."
echo ""
echo "  方式2 - 用 uv run 自动运行（推荐）:"
echo "    conda activate ${CONDA_ENV_NAME}"
echo "    uv run python training.py ..."
echo ""
echo "验证安装:"
echo "    conda activate ${CONDA_ENV_NAME}"
echo "    uv run python -c \"import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')\""
echo "    uv run python -c \"import vllm; print(f'vLLM: {vllm.__version__}')\""
echo ""
echo "同步到 A100 集群:"
echo "    rsync -avz --exclude '.venv' ./ user@a100:/path/to/project/"
echo "    # 在 A100 上运行: bash setup_env.sh"
echo ""