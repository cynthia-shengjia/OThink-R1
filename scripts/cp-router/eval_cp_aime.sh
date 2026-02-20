#!/bin/bash
# CP-Router: aime | Model: DeepSeek-R1-Distill-Qwen-7B-fixed
set -e
cd /home/notebook/code/personal/S9059888/OThink-R1-main
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "========== CP-Router: aime =========="
python baseline/cp-router/test_cp_router.py \
    --model_path "/home/notebook/code/personal/S9059888/OThink-R1-main/models/DeepSeek-R1-Distill-Qwen-7B-fixed" \
    --datasets_dir "datasets" \
    --dataset "aime" \
    --max_samples 500 \
    --cal_ratio 0.3 \
    --tau 1 \
    --beta 3.0 \
    --batch_size 8 \
    --lrm_max_tokens 4096
echo "✅ CP-Router aime done!"
