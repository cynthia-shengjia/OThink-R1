#!/bin/bash
# 论文 Table 1: Qwen pairing 评测
# Qwen-2.5-14B (LLM) + DeepSeek-R1-Distill-Qwen-14B (LRM)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CP_ROUTER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${CP_ROUTER_DIR}/../.." && pwd)"

LLM_PATH="${1:-/path/to/Qwen2.5-14B-Instruct}"
LRM_PATH="${2:-/path/to/DeepSeek-R1-Distill-Qwen-14B}"
GPU_IDS="${3:-0,1,2,3}"

cd "${CP_ROUTER_DIR}"

echo "=========================================="
echo "  Qwen Pairing Evaluation (Table 1)"
echo "=========================================="

DATASETS="mmlu_elementary_math mmlu_high_school_math mmlu_college_math logiqa gpqa stem_mcqa"

for dataset in ${DATASETS}; do
    echo ""
    echo "  ====== ${dataset} ======"
    
    CUDA_VISIBLE_DEVICES=${GPU_IDS} uv run python run_cp_router.py \
        --llm_path "${LLM_PATH}" \
        --lrm_path "${LRM_PATH}" \
        --dataset "${dataset}" \
        --cal_ratio 0.3 \
        --beta 3.0 \
        --tau 1 \
        --max_tokens 16384 \
        --gpu_id "${GPU_IDS}" \
        --output_dir ./results/qwen_pairing \
        --run_baselines
done

echo ""
echo "  ✅ Qwen Pairing 评测完成"
