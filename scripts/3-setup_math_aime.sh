#!/bin/bash
set -e
eval "$(conda shell.bash hook)"
conda activate othink-r1

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/datasets"

echo "=========================================="
echo "  下载 MATH + AIME 数据集"
echo "=========================================="

# 设置 HF 镜像
export HF_ENDPOINT=https://hf-mirror.com

mkdir -p "${DATA_DIR}"

# MATH
if [ -d "${DATA_DIR}/MATH" ]; then
    echo "  ⚠️  MATH 已存在，跳过"
else
    echo "  📦 下载 MATH..."
    uv run huggingface-cli download \
        --repo-type dataset \
        DigitalLearningGmbH/MATH-lighteval \
        --local-dir "${DATA_DIR}/MATH"
    echo "  ✅ MATH 下载完成"
fi

# AIME
if [ -d "${DATA_DIR}/AIME" ]; then
    echo "  ⚠️  AIME 已存在，跳过"
else
    echo "  📦 下载 AIME..."
    uv run huggingface-cli download \
        --repo-type dataset \
        AI-MO/aimo-validation-aime \
        --local-dir "${DATA_DIR}/AIME"
    echo "  ✅ AIME 下载完成"
fi

echo "  ✅ 数据集下载完成"

# 检查文件
echo ""
echo "  MATH 文件:"
find "${DATA_DIR}/MATH" -name "*.parquet" -o -name "*.json" -o -name "*.jsonl" | head -10
echo ""
echo "  AIME 文件:"
find "${DATA_DIR}/AIME" -name "*.parquet" -o -name "*.json" -o -name "*.jsonl" | head -10
