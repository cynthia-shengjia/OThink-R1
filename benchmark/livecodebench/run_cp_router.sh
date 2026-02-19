#!/bin/bash
# ============================================================
# CP-Router x LiveCodeBench 运行脚本
#
# 用法:
#   bash benchmark/livecodebench/run_cp_router.sh \
#       --model_path ./models/Qwen2.5-0.5B-Instruct
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_PATH=""
GPU_IDS="0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift 2;;
        --gpu_ids) GPU_IDS="$2"; shift 2;;
        *) echo "Unknown: $1"; exit 1;;
    esac
done

if [ -z "${MODEL_PATH}" ]; then
    echo "❌ 请指定: --model_path /path/to/model"
    exit 1
fi

if [[ "${MODEL_PATH}" != /* ]]; then
    MODEL_PATH="$(cd "$(dirname "${MODEL_PATH}")" && pwd)/$(basename "${MODEL_PATH}")"
fi

eval "$(conda shell.bash hook)"
conda activate othink-r1

export HF_DATASETS_TRUST_REMOTE_CODE=1

echo "=========================================="
echo "  CP-Router x LiveCodeBench"
echo "  模型: ${MODEL_PATH}"
echo "=========================================="

uv run python "${SCRIPT_DIR}/cp_router_lcb.py" --model_path "${MODEL_PATH}"
