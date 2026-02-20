#!/bin/bash
# Deer: gsm8k | Model: DeepSeek-R1-Distill-Qwen-7B-fixed
set -e
cd /home/notebook/code/personal/S9059888/OThink-R1-main
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "========== Deer: gsm8k =========="
python baseline/deer/vllm-deer.py \
    --model_name_or_path "/home/notebook/code/personal/S9059888/OThink-R1-main/models/DeepSeek-R1-Distill-Qwen-7B-fixed" \
    --dataset_dir "baseline/deer/data" \
    --dataset "gsm8k" \
    --output_path "baseline/deer/outputs" \
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
echo "✅ Deer gsm8k done!"
