#!/bin/bash
set -e

MODEL_PATH=""
GPU_IDS="0"
THRESHOLD=0.95
MAX_LEN=16384
DATASETS="math aime"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL_PATH="$2"; shift 2;;
        --gpu_ids) GPU_IDS="$2"; shift 2;;
        --threshold) THRESHOLD="$2"; shift 2;;
        --max_len) MAX_LEN="$2"; shift 2;;
        --datasets) DATASETS="$2"; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

if [ -z "${MODEL_PATH}" ]; then
    echo "❌ 请指定模型路径: --model /path/to/model"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "  DEER Full Baseline Evaluation"
echo "=========================================="

bash "${SCRIPT_DIR}/prepare_data.sh"

for ds in ${DATASETS}; do
    echo ""
    echo "  ====== 运行 DEER on ${ds} ======"
    bash "${SCRIPT_DIR}/run_deer.sh" \
        --model "${MODEL_PATH}" \
        --dataset "${ds}" \
        --threshold "${THRESHOLD}" \
        --max_len "${MAX_LEN}" \
        --gpu_ids "${GPU_IDS}"
done

echo ""
echo "  ✅ 全部评测完成！"
