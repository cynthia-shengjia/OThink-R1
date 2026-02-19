#!/bin/bash
# 完整评测: 在所有数据集上运行 CP-Router
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CP_ROUTER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${CP_ROUTER_DIR}/../.." && pwd)"

# ==================== 配置 ====================
LLM_PATH="${1:-${PROJECT_ROOT}/models/Qwen2.5-0.5B-Instruct}"
LRM_PATH="${2:-${PROJECT_ROOT}/models/DeepSeek-R1-Distill-Qwen-1.5B}"
GPU_ID="${3:-0}"
DATASETS="mmlu_elementary_math mmlu_high_school_math mmlu_college_math logiqa gpqa stem_mcqa"
# ================================================

cd "${CP_ROUTER_DIR}"

echo "=========================================="
echo "  CP-Router 完整评测"
echo "=========================================="
echo "  LLM: ${LLM_PATH}"
echo "  LRM: ${LRM_PATH}"
echo "  GPU: ${GPU_ID}"
echo "=========================================="

for dataset in ${DATASETS}; do
    echo ""
    echo "  ====== ${dataset} ======"
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} uv run python run_cp_router.py \
        --llm_path "${LLM_PATH}" \
        --lrm_path "${LRM_PATH}" \
        --dataset "${dataset}" \
        --cal_ratio 0.3 \
        --beta 3.0 \
        --tau 1 \
        --max_tokens 4096 \
        --gpu_id "${GPU_ID}" \
        --output_dir ./results \
        --run_baselines \
        2>&1 || echo "  ⚠️  ${dataset} 失败"
done

echo ""
echo "  ✅ 完整评测完成"
echo "  结果保存在: ${CP_ROUTER_DIR}/results/"
