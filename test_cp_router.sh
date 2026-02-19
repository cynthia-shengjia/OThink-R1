#!/bin/bash
set -e
eval "$(conda shell.bash hook)"
conda activate othink-r1

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
CP_ROUTER_DIR="${PROJECT_ROOT}/baseline/cp-router"
MODEL_PATH="${PROJECT_ROOT}/models/Qwen2.5-0.5B-Instruct"
DATASETS_DIR="${PROJECT_ROOT}/datasets"

export PYTHONUNBUFFERED=1

echo "=========================================="
echo "  CP-Router 测试 (Qwen2.5-0.5B-Instruct)"
echo "=========================================="

# ---- 测试1: 仅路由决策 ----
echo ""
echo "[1/3] 测试1: 仅路由决策 (skip_lrm)..."
cd "${CP_ROUTER_DIR}"

uv run python -u test_cp_router.py \
    --model_path "${MODEL_PATH}" \
    --datasets_dir "${DATASETS_DIR}" \
    --dataset asdiv \
    --max_samples 30 \
    --batch_size 4 \
    --tau 1 \
    --beta 3.0 \
    --skip_lrm

echo ""
echo "  ✅ 测试1完成"

# ---- 测试2: 端到端 ----
echo ""
echo "[2/3] 测试2: 端到端 (含 LRM 推理)..."

uv run python -u test_cp_router.py \
    --model_path "${MODEL_PATH}" \
    --datasets_dir "${DATASETS_DIR}" \
    --dataset asdiv \
    --max_samples 20 \
    --batch_size 4 \
    --tau 1 \
    --beta 3.0 \
    --lrm_max_tokens 256

echo ""
echo "  ✅ 测试2完成"

# ---- 测试3: 多数据集 ----
echo ""
echo "[3/3] 测试3: 多数据集..."

for ds in math aime asdiv; do
    echo ""
    echo "  ------ ${ds} ------"
    uv run python -u test_cp_router.py \
        --model_path "${MODEL_PATH}" \
        --datasets_dir "${DATASETS_DIR}" \
        --dataset "${ds}" \
        --max_samples 20 \
        --batch_size 4 \
        --skip_lrm || echo "  ⚠️ ${ds} 失败"
done

echo ""
echo "=========================================="
echo "  ✅ CP-Router 全部测试完成!"
echo "=========================================="
