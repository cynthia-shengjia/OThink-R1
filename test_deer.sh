#!/bin/bash
# test_deer.sh
# 在项目根目录运行: bash test_deer.sh
# 用 Qwen2.5-0.5B-Instruct 在 math 前10条上快速测试 DEER 是否能跑通

set -e
eval "$(conda shell.bash hook)"
conda activate othink-r1

cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"
DEER_DIR="${PROJECT_ROOT}/baseline/deer"
MODEL_PATH="${PROJECT_ROOT}/models/Qwen2.5-0.5B-Instruct"

echo "=========================================="
echo "  DEER Baseline 快速测试"
echo "=========================================="
echo "  模型: ${MODEL_PATH}"
echo "  数据: ${DEER_DIR}/data/math/test.jsonl"
echo "=========================================="

# Step 1: 检查文件完整性
echo ""
echo "[1/4] 检查文件..."

MISSING=0
for f in vllm-deer.py check.py check_fixed.py utils/parser.py utils/grader.py utils/math_normalization.py utils/data_loader.py utils/utils.py utils/examples.py; do
    if [ ! -f "${DEER_DIR}/${f}" ]; then
        echo "  ❌ 缺少: ${DEER_DIR}/${f}"
        MISSING=1
    fi
done

if [ ! -f "${DEER_DIR}/data/math/test.jsonl" ]; then
    echo "  ❌ 缺少数据: ${DEER_DIR}/data/math/test.jsonl"
    MISSING=1
fi

if [ ! -d "${MODEL_PATH}" ]; then
    echo "  ❌ 模型不存在: ${MODEL_PATH}"
    MISSING=1
fi

if [ "${MISSING}" -eq 1 ]; then
    echo "  请先运行 bash copy_deer_files.sh"
    exit 1
fi
echo "  ✅ 文件完整"

# Step 2: 测试 import
echo ""
echo "[2/4] 测试 Python import..."

cd "${DEER_DIR}"
uv run python -c "from utils.parser import extract_answer; print('  ✅ parser OK')"
uv run python -c "from utils.data_loader import load_data; print('  ✅ data_loader OK')"
uv run python -c "
import json
with open('data/math/test.jsonl') as f:
    lines = [json.loads(l) for l in f]
print(f'  ✅ math 数据: {len(lines)} 条')
"

# Step 3: 截取前10条数据做测试
echo ""
echo "[3/4] 准备测试数据（前10条）..."

mkdir -p "${DEER_DIR}/data/math_test10"
head -10 "${DEER_DIR}/data/math/test.jsonl" > "${DEER_DIR}/data/math_test10/test.jsonl"
echo "  ✅ 已截取10条 → data/math_test10/test.jsonl"

# Step 4: 运行 DEER 推理
echo ""
echo "[4/4] 运行 DEER 推理（10条, threshold=0.95）..."
echo ""

export CUDA_VISIBLE_DEVICES=1

cd "${DEER_DIR}"
uv run python vllm-deer.py \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_dir "${DEER_DIR}/data" \
    --dataset "math_test10" \
    --threshold 0.95 \
    --max-len 4096 \
    --think_ratio 0.9 \
    --temperature 0.0 \
    --top_p 1.0 \
    --policy "avg1" \
    --batch_size 100 \
    --output_path "${DEER_DIR}/outputs" \
    --no_thinking 0 \
    --rep 0 \
    --points 1 \
    --af 0 \
    --max_judge_steps 10 \
    --prob_check_max_tokens 20 \
    --run_time 1

echo ""
echo "=========================================="
echo "  ✅ DEER 推理完成！"
echo "=========================================="

# 查找输出文件
OUTPUT_FILE=$(find "${DEER_DIR}/outputs" -name "*.jsonl" -path "*math_test10*" 2>/dev/null | sort | tail -1)

if [ -n "${OUTPUT_FILE}" ] && [ -f "${OUTPUT_FILE}" ]; then
    echo "  输出文件: ${OUTPUT_FILE}"
    echo ""
    echo "  前2条结果预览:"
    head -2 "${OUTPUT_FILE}" | uv run python -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line)
    q = d['question'][:80]
    a = d['gold_answer']
    r = d['generated_responses'][0][:100]
    print(f'  Q: {q}...')
    print(f'  Gold: {a}')
    print(f'  Resp: {r}...')
    print()
"

    # 用 OThink-R1 verifier 评估
    echo "=========================================="
    echo "  OThink-R1 Verifier 评估"
    echo "=========================================="
    cd "${PROJECT_ROOT}"
    uv run python "${DEER_DIR}/scripts/eval_with_othink.py" \
        --generation_path "${OUTPUT_FILE}" \
        --dataset math \
        2>&1 || echo "  ⚠️  OThink-R1 评估失败（不影响 DEER 本身）"
else
    echo "  ⚠️  未找到输出文件，请检查上面的报错信息"
fi

# 清理测试数据
rm -rf "${DEER_DIR}/data/math_test10"

echo ""
echo "=========================================="
echo "  测试结束"
echo "=========================================="