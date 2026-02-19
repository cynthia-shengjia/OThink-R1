#!/bin/bash
# 下载 LiveCodeBench 数据集
set -e
eval "$(conda shell.bash hook)"
conda activate othink-r1

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/datasets/livecodebench/code_generation_lite"

export HF_ENDPOINT=https://hf-mirror.com

if [ -d "${DATA_DIR}" ] && [ "$(ls -A ${DATA_DIR} 2>/dev/null)" ]; then
    echo "  数据集已存在: ${DATA_DIR}"
else
    echo "  下载 LiveCodeBench 数据集..."
    mkdir -p "${DATA_DIR}"
    uv run huggingface-cli download \
        --repo-type dataset \
        livecodebench/code_generation_lite \
        --local-dir "${DATA_DIR}" \
        --local-dir-use-symlinks False \
        --resume-download
    echo "  下载完成: ${DATA_DIR}"
fi
