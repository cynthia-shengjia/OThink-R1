#!/bin/bash
# ============================================================
# DEER × LiveCodeBench 运行脚本
#
# 用法:
#   bash benchmark/livecodebench/run_deer_lcb.sh \
#       --model /path/to/model \
#       --gpu_ids 0 \
#       --threshold 0.95
# ============================================================
set -e
eval "$(conda shell.bash hook)"
conda activate othink-r1

MODEL_PATH=""
GPU_IDS="0"
THRESHOLD=0.95
MAX_LEN=16384
THINK_RATIO=0.87
POLICY="avg1"
TEMPERATURE=0.0
MAX_SAMPLES=""
LOCAL_DATA=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL_PATH="$2"; shift 2;;
        --gpu_ids) GPU_IDS="$2"; shift 2;;
        --threshold) THRESHOLD="$2"; shift 2;;
        --max_len) MAX_LEN="$2"; shift 2;;
        --think_ratio) THINK_RATIO="$2"; shift 2;;
        --policy) POLICY="$2"; shift 2;;
        --temperature) TEMPERATURE="$2"; shift 2;;
        --max_samples) MAX_SAMPLES="$2"; shift 2;;
        --local_data) LOCAL_DATA="$2"; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

if [ -z "${MODEL_PATH}" ]; then
    echo "❌ 请指定模型路径: --model /path/to/model"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 解析相对路径
if [[ "${MODEL_PATH}" != /* ]]; then
    MODEL_PATH="$(cd "$(dirname "${MODEL_PATH}")" && pwd)/$(basename "${MODEL_PATH}")"
fi

DEER_ARGS=(
    --model_name_or_path "${MODEL_PATH}"
    --threshold "${THRESHOLD}"
    --max_len "${MAX_LEN}"
    --think_ratio "${THINK_RATIO}"
    --policy "${POLICY}"
    --temperature "${TEMPERATURE}"
    --gpu_ids "${GPU_IDS}"
)

# 数据集路径
if [ -n "${LOCAL_DATA}" ]; then
    DEER_ARGS+=(--dataset_path "${LOCAL_DATA}")
elif [ -d "${PROJECT_ROOT}/datasets/livecodebench/code_generation_lite" ]; then
    DEER_ARGS+=(--dataset_path "${PROJECT_ROOT}/datasets/livecodebench/code_generation_lite")
fi

if [ -n "${MAX_SAMPLES}" ]; then
    DEER_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

cd "${PROJECT_ROOT}"
uv run python "${SCRIPT_DIR}/deer_lcb.py" "${DEER_ARGS[@]}"
