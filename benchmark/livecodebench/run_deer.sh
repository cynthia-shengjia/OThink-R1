#!/bin/bash
# ============================================================
# DEER x LiveCodeBench 运行脚本
#
# 用法:
#   bash benchmark/livecodebench/run_deer.sh \
#       --model_path ./models/Qwen2.5-0.5B-Instruct \
#       --threshold 0.95
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_PATH=""
GPU_IDS="0"
THRESHOLD=0.95
MAX_LEN=16384
THINK_RATIO=0.87
POLICY="avg1"
TEMPERATURE=0.0
MAX_SAMPLES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift 2;;
        --gpu_ids) GPU_IDS="$2"; shift 2;;
        --threshold) THRESHOLD="$2"; shift 2;;
        --max_len) MAX_LEN="$2"; shift 2;;
        --think_ratio) THINK_RATIO="$2"; shift 2;;
        --policy) POLICY="$2"; shift 2;;
        --temperature) TEMPERATURE="$2"; shift 2;;
        --max_samples) MAX_SAMPLES="$2"; shift 2;;
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

if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "❌ 模型不存在: ${MODEL_PATH}"; exit 1
fi

DATASET_DIR="${PROJECT_ROOT}/datasets/livecodebench/code_generation_lite"
if [ ! -d "${DATASET_DIR}" ]; then
    echo "❌ 数据集不存在: ${DATASET_DIR}"; exit 1
fi

LCB_SRC="${SCRIPT_DIR}/LiveCodeBench"
if [ ! -d "${LCB_SRC}/lcb_runner" ]; then
    echo "❌ LCB 源码不存在: ${LCB_SRC}"; exit 1
fi

eval "$(conda shell.bash hook)"
conda activate othink-r1

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTHONPATH="${LCB_SRC}:${PYTHONPATH:-}"

ARGS=(
    --model_path "${MODEL_PATH}"
    --dataset_path "${DATASET_DIR}"
    --threshold "${THRESHOLD}"
    --max_len "${MAX_LEN}"
    --think_ratio "${THINK_RATIO}"
    --policy "${POLICY}"
    --temperature "${TEMPERATURE}"
    --gpu_ids "${GPU_IDS}"
)
if [ -n "${MAX_SAMPLES}" ]; then
    ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

echo "=========================================="
echo "  DEER x LiveCodeBench"
echo "  模型: ${MODEL_PATH}"
echo "  阈值: ${THRESHOLD}"
echo "  GPU: ${GPU_IDS}"
echo "=========================================="

uv run python "${SCRIPT_DIR}/deer_lcb.py" "${ARGS[@]}"
