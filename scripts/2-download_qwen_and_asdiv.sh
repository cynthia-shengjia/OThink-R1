#!/bin/bash
set -e
eval "$(conda shell.bash hook)"
conda activate othink-r1

export HF_ENDPOINT=https://hf-mirror.com


PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${PROJECT_ROOT}/models"
DATA_DIR="${PROJECT_ROOT}/datasets"

echo "=========================================="
echo "  下载模型和数据集 (via HuggingFace)"
echo "=========================================="

# Step 1: 创建目录

echo "[1/5] 创建目录结构"
mkdir -p "${MODEL_DIR}"
mkdir -p "${DATA_DIR}"
echo "  ✅ 目录创建完成"

# Step 2: 安装 huggingface_hub

echo ""
echo "[2/5] 检查/安装 huggingface_hub"
if uv run python -c "import huggingface_hub" 2>/dev/null; then
    echo "  ✅ huggingface_hub 已安装"
else
    uv pip install huggingface_hub
    echo "  ✅ huggingface_hub 安装完成"
fi

# Step 3: 下载 Qwen2.5-0.5B

echo ""
echo "[3/5] 下载 Qwen2.5-0.5B-Instruct 模型"
if [ -d "${MODEL_DIR}/Qwen2.5-0.5B-Instruct" ] && [ -f "${MODEL_DIR}/Qwen2.5-0.5B-Instruct/config.json" ]; then
    echo "  ⚠️  模型已存在，跳过下载"
else
    uv run huggingface-cli download \
        Qwen/Qwen2.5-0.5B-Instruct \
        --local-dir "${MODEL_DIR}/Qwen2.5-0.5B-Instruct"
    echo "  ✅ Qwen2.5-0.5B-Instruct 下载完成"
fi

# Step 4: 下载 ASDIV

echo ""
echo "[4/5] 下载 ASDIV 数据集"
if [ -d "${DATA_DIR}/ASDIV" ]; then
    echo "  ⚠️  数据集已存在，跳过下载"
else
    uv run huggingface-cli download \
        EleutherAI/asdiv \
        --repo-type dataset \
        --local-dir "${DATA_DIR}/ASDIV"
    echo "  ✅ ASDIV 下载完成"
fi