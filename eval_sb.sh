#!/bin/bash
# =============================================================
# SB-DS7B-alpha-2: 标准测评 (6数据集) + LiveCodeBench
# =============================================================
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_PATH="${ROOT}/models/SB_DS7B_alpha_2"

echo "=============================================="
echo "  SB-DS7B-alpha-2 Evaluation"
echo "=============================================="

# --- 标准测评 ---
echo ""
echo "========== [1/2] Standard Eval × 6 datasets =========="
cd "${ROOT}/OThinkR1Training"
python eval_all.py --model SB-DS7B-alpha-2 --dataset all --tp 4 --gpu_util 0.95

# --- LiveCodeBench ---
echo ""
echo "========== [2/2] LiveCodeBench =========="
cd "$ROOT"
bash benchmark/livecodebench/run_standard.sh \
    --model_path "${MODEL_PATH}" \
    --gpu_ids 0,1,2,3 \
    --max_model_len 16384 \
    --max_tokens 16384

echo ""
echo "✅ SB-DS7B-alpha-2 all done!"
