#!/bin/bash
# ============================================================
# 正式评测脚本: MATH + AIME
# 在 A100 上使用，替换 model.path 为实际训练好的模型路径
# ============================================================

cd ../..

modelsize=7B
GPUNUM=4

# 需要替换的变量
MODEL_CONFIG="DeepSeek-R1-Distill-Qwen-${modelsize}-Fix"
MODE="YOUR-MODE"
beta1=0.0001
beta2=0.0001

for model_path in ./save_models/YOUR-SAVE-PREFIX/SFT_R1_lr_*/*; do

    # ---- MATH 评测 ----
    python eval.py \
        model=${MODEL_CONFIG} \
        model.path=${model_path} \
        +model.mode="${MODE}-beta1-${beta1}-beta2-${beta2}" \
        data=MATHBench \
        data.datasets.MATHBench.splits.test.slice=\"[:100%]\" \
        model.inference.tensor_parallel_size=${GPUNUM} \
        model.inference.gpu_memory_utilization=0.9 \
        +model.inference.repetition_penalty=1.0 \
        model.inference.temperature=0.9 \
        model.inference.top_p=0.95 \
        model.inference.max_tokens=16384

    # ---- AIME 评测 ----
    python eval.py \
        model=${MODEL_CONFIG} \
        model.path=${model_path} \
        +model.mode="${MODE}-beta1-${beta1}-beta2-${beta2}" \
        data=AIME \
        data.datasets.AIME.splits.train.slice=\"[:100%]\" \
        model.inference.tensor_parallel_size=${GPUNUM} \
        model.inference.gpu_memory_utilization=0.9 \
        +model.inference.repetition_penalty=1.0 \
        model.inference.temperature=0.9 \
        model.inference.top_p=0.95 \
        model.inference.max_tokens=16384

done
