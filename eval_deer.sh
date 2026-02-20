#!/bin/bash
# =============================================================
# Deer baseline: DS-R1-7B-fixed × 6数据集 + LiveCodeBench
# =============================================================
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

BASE_MODEL="${ROOT}/models/DeepSeek-R1-Distill-Qwen-7B-fixed"
DEER_DIR="baseline/deer"

echo "=============================================="
echo "  Deer Baseline (DS-R1-7B-fixed)"
echo "=============================================="

# --- 6 个标准数据集 ---
DATASETS=("math" "aime" "gsm8k" "asdiv" "commonsenseqa" "openbookqa")

for DS in "${DATASETS[@]}"; do
    echo ""
    echo "========== Deer: ${DS} =========="
    python ${DEER_DIR}/vllm-deer.py \
        --model_name_or_path "${BASE_MODEL}" \
        --dataset_dir "${DEER_DIR}/data" \
        --dataset "${DS}" \
        --output_path "${DEER_DIR}/outputs" \
        --max_len 16384 \
        --threshold 0.95 \
        --think_ratio 0.9 \
        --temperature 0.0 \
        --top_p 1.0 \
        --batch_size 2000 \
        --max_judge_steps 10 \
        --policy avg1 \
        --points 1 \
        --prob_check_max_tokens 20
    echo "  ✅ Deer ${DS}"
done

# --- LiveCodeBench ---
echo ""
echo "========== Deer: LiveCodeBench =========="
bash benchmark/livecodebench/run_deer.sh \
    --model_path "${BASE_MODEL}" \
    --gpu_ids 0,1,2,3 \
    --max_model_len 16384 \
    --max_tokens 16384 \
    --threshold 0.95 \
    --max_rounds 5

echo ""
echo "✅ Deer all done!"
