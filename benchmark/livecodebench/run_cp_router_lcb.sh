#!/bin/bash
set -e
eval "$(conda shell.bash hook)"
conda activate othink-r1

LLM_PATH=""
LRM_PATH=""
GPU_IDS="0"
MAX_SAMPLES=""
PROBE_TOKENS=50
LLM_MAX_TOKENS=2048
LRM_MAX_TOKENS=4096
LOCAL_DATA=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --llm) LLM_PATH="$2"; shift 2;;
        --lrm) LRM_PATH="$2"; shift 2;;
        --gpu_ids) GPU_IDS="$2"; shift 2;;
        --max_samples) MAX_SAMPLES="$2"; shift 2;;
        --probe_tokens) PROBE_TOKENS="$2"; shift 2;;
        --llm_max_tokens) LLM_MAX_TOKENS="$2"; shift 2;;
        --lrm_max_tokens) LRM_MAX_TOKENS="$2"; shift 2;;
        --local_data) LOCAL_DATA="$2"; shift 2;;
        *) echo "Unknown: $1"; exit 1;;
    esac
done

[ -z "${LLM_PATH}" ] && { echo "❌ --llm required"; exit 1; }
[[ "${LLM_PATH}" != /* ]] && LLM_PATH="$(cd "$(dirname "${LLM_PATH}")" && pwd)/$(basename "${LLM_PATH}")"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ARGS=(--llm_path "${LLM_PATH}" --gpu_ids "${GPU_IDS}"
      --probe_tokens "${PROBE_TOKENS}"
      --llm_max_tokens "${LLM_MAX_TOKENS}" --lrm_max_tokens "${LRM_MAX_TOKENS}")
[ -n "${LRM_PATH}" ] && ARGS+=(--lrm_path "${LRM_PATH}")
[ -n "${MAX_SAMPLES}" ] && ARGS+=(--max_samples "${MAX_SAMPLES}")

if [ -n "${LOCAL_DATA}" ]; then
    ARGS+=(--dataset_path "${LOCAL_DATA}")
elif [ -d "${PROJECT_ROOT}/datasets/livecodebench/code_generation_lite" ]; then
    ARGS+=(--dataset_path "${PROJECT_ROOT}/datasets/livecodebench/code_generation_lite")
fi

cd "${PROJECT_ROOT}"
uv run python "${SCRIPT_DIR}/cp_router_lcb.py" "${ARGS[@]}"
