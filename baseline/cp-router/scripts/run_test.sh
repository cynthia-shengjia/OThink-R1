#!/bin/bash
# 快速测试: 用小模型在少量样本上验证 CP-Router 是否能跑通
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CP_ROUTER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${CP_ROUTER_DIR}/../.." && pwd)"

# 默认模型路径 (可修改)
LLM_PATH="${PROJECT_ROOT}/models/Qwen2.5-0.5B-Instruct"
GPU_ID="${1:-0}"

cd "${CP_ROUTER_DIR}"

echo "=========================================="
echo "  CP-Router 快速测试"
echo "=========================================="
echo "  LLM: ${LLM_PATH}"
echo "  GPU: ${GPU_ID}"
echo "=========================================="

# 仅测试 logits 提取 + 路由决策
CUDA_VISIBLE_DEVICES=${GPU_ID} uv run python run_logits_only.py \
    --llm_path "${LLM_PATH}" \
    --dataset mmlu_elementary_math \
    --max_samples 50 \
    --beta 3.0 \
    --tau 1 \
    --batch_size 4 \
    --gpu_id "${GPU_ID}"

echo ""
echo "  ✅ CP-Router 快速测试完成"
