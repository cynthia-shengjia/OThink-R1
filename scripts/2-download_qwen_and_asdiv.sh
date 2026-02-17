#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${PROJECT_ROOT}/models"
DATA_DIR="${PROJECT_ROOT}/datasets"

echo "=========================================="
echo "  下载模型和数据集 (via ModelScope)"
echo "=========================================="

# Step 1: 创建目录
echo "[1/5] 创建目录结构"
mkdir -p "${MODEL_DIR}"
mkdir -p "${DATA_DIR}"
echo "  ✅ 目录创建完成"

# Step 2: 安装 modelscope
echo ""
echo "[2/5] 检查/安装 modelscope"
if uv run python -c "import modelscope" 2>/dev/null; then
    echo "  ✅ modelscope 已安装"
else
    uv pip install modelscope
    echo "  ✅ modelscope 安装完成"
fi

# Step 3: 下载 Qwen3-0.6B
echo ""
echo "[3/5] 下载 Qwen3-0.6B 模型"
if [ -d "${MODEL_DIR}/Qwen3-0.6B" ] && [ -f "${MODEL_DIR}/Qwen3-0.6B/config.json" ]; then
    echo "  ⚠️  模型已存在，跳过下载"
else
    uv run modelscope download \
        --model Qwen/Qwen3-0.6B \
        --local_dir "${MODEL_DIR}/Qwen3-0.6B"
    echo "  ✅ Qwen3-0.6B 下载完成"
fi

# Step 4: 下载 ASDIV
echo ""
echo "[4/5] 下载 ASDIV 数据集"
if [ -d "${DATA_DIR}/ASDIV" ]; then
    echo "  ⚠️  数据集已存在，跳过下载"
else
    uv run modelscope download \
        --dataset EleutherAI/asdiv \
        --local_dir "${DATA_DIR}/ASDIV"
    echo "  ✅ ASDIV 下载完成"
fi
