#!/bin/bash
# ============================================================
# LiveCodeBench 标准评测入口
# ============================================================

set -e
eval "$(conda shell.bash hook)"
conda activate othink-r1

MODEL=""
MODEL_PATH=""
GPU_IDS="0"
MAX_TOKENS=16289
TEMPERATURE=0.9
CODEGEN_N=1
N=1
RELEASE_VERSION="release_v5"
SCENARIO="codegeneration"
STOP_WORDS="None"
LOCAL_DATA_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2;;
        --model_path) MODEL_PATH="$2"; shift 2;;
        --gpu_ids) GPU_IDS="$2"; shift 2;;
        --max_tokens) MAX_TOKENS="$2"; shift 2;;
        --temperature) TEMPERATURE="$2"; shift 2;;
        --codegen_n) CODEGEN_N="$2"; shift 2;;
        --n) N="$2"; shift 2;;
        --release_version) RELEASE_VERSION="$2"; shift 2;;
        --local_data) LOCAL_DATA_PATH="$2"; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

if [ -z "${MODEL}" ]; then
    echo "❌ 请指定模型: --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LCB_DIR="${SCRIPT_DIR}/LiveCodeBench"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -n "${MODEL_PATH}" ] && [[ "${MODEL_PATH}" != /* ]]; then
    MODEL_PATH="$(cd "$(dirname "${MODEL_PATH}")" && pwd)/$(basename "${MODEL_PATH}")"
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

echo "=========================================="
echo "  LiveCodeBench 标准评测"
echo "  模型: ${MODEL}"
echo "  GPU: ${GPU_IDS}"
echo "=========================================="

LCB_ARGS=(
    --model "${MODEL}"
    --scenario "${SCENARIO}"
    --max_tokens "${MAX_TOKENS}"
    --release_version "${RELEASE_VERSION}"
    --evaluate
    --codegen_n "${CODEGEN_N}"
    --n "${N}"
    --temperature "${TEMPERATURE}"
    --stop "${STOP_WORDS}"
)

if [ -n "${MODEL_PATH}" ]; then
    LCB_ARGS+=(--local_model_path "${MODEL_PATH}")
fi

if [ -n "${LOCAL_DATA_PATH}" ]; then
    LCB_ARGS+=(--local_dataset_path "${LOCAL_DATA_PATH}")
elif [ -d "${PROJECT_ROOT}/datasets/livecodebench/code_generation_lite" ]; then
    LCB_ARGS+=(--local_dataset_path "${PROJECT_ROOT}/datasets/livecodebench/code_generation_lite")
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${LCB_DIR}:${PYTHONPATH}"
uv run python -m lcb_runner.runner.main "${LCB_ARGS[@]}"

MODEL_NAME=$(basename "${MODEL}")
OUTPUT_FILE="${LCB_DIR}/output/${MODEL_NAME}/Scenario.${SCENARIO}_${CODEGEN_N}_${TEMPERATURE}.json"

if [ -f "${OUTPUT_FILE}" ]; then
    echo "  ✅ 评测完成！结果: ${OUTPUT_FILE}"
    uv run python -m lcb_runner.utils.get_length_lcb \
        --model_name "${MODEL}" \
        --file_path "${OUTPUT_FILE}" \
        2>/dev/null || echo "  ⚠️  token 统计失败"
else
    echo "  ⚠️  未找到输出文件: ${OUTPUT_FILE}"
fi
