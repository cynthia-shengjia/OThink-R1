#!/bin/bash
# =============================================================
# CP-Router baseline: DS-R1-7B-fixed × 6数据集 + LiveCodeBench(PPL版)
# =============================================================
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

BASE_MODEL="${ROOT}/models/DeepSeek-R1-Distill-Qwen-7B-fixed"
CP_DIR="baseline/cp-router"

echo "=============================================="
echo "  CP-Router Baseline (DS-R1-7B-fixed)"
echo "=============================================="

# --- 6 个 MCQA 数据集 ---
DATASETS=("math" "aime" "asdiv" "gsm8k" "commonsenseqa" "openbookqa")

for DS in "${DATASETS[@]}"; do
    echo ""
    echo "========== CP-Router: ${DS} =========="
    python ${CP_DIR}/test_cp_router.py \
        --model_path "${BASE_MODEL}" \
        --datasets_dir "datasets" \
        --dataset "${DS}" \
        --max_samples 500 \
        --cal_ratio 0.3 \
        --tau 1 \
        --beta 3.0 \
        --batch_size 8 \
        --lrm_max_tokens 4096
    echo "  ✅ CP-Router ${DS}"
done

# --- LiveCodeBench (Perplexity-Based Routing) ---
echo ""
echo "========== CP-Router: LiveCodeBench (PPL-Based) =========="
export PYTHONPATH="benchmark/livecodebench/LiveCodeBench:${PYTHONPATH:-}"
python benchmark/livecodebench/cp_router_lcb_codegen.py \
    --model_path "${BASE_MODEL}" \
    --max_model_len 16384 \
    --llm_max_tokens 4096 \
    --lrm_max_tokens 16384 \
    --cal_ratio 0.3 \
    --ppl_quantile 0.7 \
    --batch_size 8
echo "  ✅ CP-Router LiveCodeBench"

echo ""
echo "✅ CP-Router all done!"
