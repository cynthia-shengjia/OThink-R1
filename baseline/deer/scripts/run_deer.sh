#!/bin/bash
# ============================================================
# 运行 DEER baseline 推理 + 评估
#
# 用法（在项目任意位置运行均可）:
#   bash baseline/deer/scripts/run_deer.sh \
#       --model ./models/Qwen2.5-0.5B-Instruct \
#       --dataset math \
#       --threshold 0.95 \
#       --gpu_ids 0
# ============================================================
set -e
eval "$(conda shell.bash hook)"
conda activate othink-r1

# 默认参数
MODEL_PATH=""
DATASET="math"
THRESHOLD=0.95
MAX_LEN=16384
THINK_RATIO=0.9
GPU_IDS="0"
TEMPERATURE=0.0
TOP_P=1.0
POLICY="avg1"
BATCH_SIZE=2000
NO_THINKING=0
REP=0
POINTS=1
AF=0
MAX_JUDGE_STEPS=10
PROB_CHECK_MAX_TOKENS=20
RUN_TIME=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL_PATH="$2"; shift 2;;
        --dataset) DATASET="$2"; shift 2;;
        --threshold) THRESHOLD="$2"; shift 2;;
        --max_len) MAX_LEN="$2"; shift 2;;
        --think_ratio) THINK_RATIO="$2"; shift 2;;
        --gpu_ids) GPU_IDS="$2"; shift 2;;
        --temperature) TEMPERATURE="$2"; shift 2;;
        --top_p) TOP_P="$2"; shift 2;;
        --policy) POLICY="$2"; shift 2;;
        --batch_size) BATCH_SIZE="$2"; shift 2;;
        --no_thinking) NO_THINKING="$2"; shift 2;;
        --rep) REP="$2"; shift 2;;
        --points) POINTS="$2"; shift 2;;
        --af) AF="$2"; shift 2;;
        --max_judge_steps) MAX_JUDGE_STEPS="$2"; shift 2;;
        --run_time) RUN_TIME="$2"; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

if [ -z "${MODEL_PATH}" ]; then
    echo "❌ 请指定模型路径: --model /path/to/model"
    exit 1
fi

# 定位目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${DEER_DIR}/../.." && pwd)"

# 如果 MODEL_PATH 是相对路径，基于调用时的 pwd 解析
if [[ "${MODEL_PATH}" != /* ]]; then
    MODEL_PATH="$(cd "$(dirname "${MODEL_PATH}")" && pwd)/$(basename "${MODEL_PATH}")"
fi

cd "${DEER_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

echo "=========================================="
echo "  DEER Baseline 推理"
echo "=========================================="
echo "  模型: ${MODEL_PATH}"
echo "  数据集: ${DATASET}"
echo "  阈值: ${THRESHOLD}"
echo "  最大长度: ${MAX_LEN}"
echo "  思考比例: ${THINK_RATIO}"
echo "  GPU: ${GPU_IDS}"
echo "  策略: ${POLICY}"
echo "  工作目录: ${DEER_DIR}"
echo "=========================================="

# 检查数据
if [ ! -f "${DEER_DIR}/data/${DATASET}/test.jsonl" ]; then
    echo "  ⚠️  数据文件不存在: ${DEER_DIR}/data/${DATASET}/test.jsonl"
    echo "  正在自动转换数据..."
    cd "${PROJECT_ROOT}"
    uv run python "${DEER_DIR}/scripts/convert_data.py" \
        --dataset "${DATASET}" \
        --output_dir "${DEER_DIR}/data"
    cd "${DEER_DIR}"
fi

# 检查 vllm-deer.py
if [ ! -f "${DEER_DIR}/vllm-deer.py" ]; then
    echo "  ❌ vllm-deer.py 不存在！请先复制 DEER 源文件到 ${DEER_DIR}/"
    exit 1
fi

START_TIME=$(date +%s)

# 运行 DEER
cd "${PROJECT_ROOT}"
uv run python "${DEER_DIR}/vllm-deer.py" \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_dir "${DEER_DIR}/data" \
    --dataset "${DATASET}" \
    --threshold "${THRESHOLD}" \
    --max-len "${MAX_LEN}" \
    --think_ratio "${THINK_RATIO}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --policy "${POLICY}" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${DEER_DIR}/outputs" \
    --no_thinking "${NO_THINKING}" \
    --rep "${REP}" \
    --points "${POINTS}" \
    --af "${AF}" \
    --max_judge_steps "${MAX_JUDGE_STEPS}" \
    --prob_check_max_tokens "${PROB_CHECK_MAX_TOKENS}" \
    --run_time "${RUN_TIME}"

echo ""
echo "  ✅ DEER 推理完成"

# 查找输出文件
OUTPUT_FILE=$(find "${DEER_DIR}/outputs" -name "*.jsonl" -newer /proc/$$ 2>/dev/null | head -1)
if [ -z "${OUTPUT_FILE}" ]; then
    OUTPUT_FILE=$(find "${DEER_DIR}/outputs" -name "greedy_p${THRESHOLD}*.jsonl" -path "*${DATASET}*" 2>/dev/null | sort | tail -1)
fi

if [ -n "${OUTPUT_FILE}" ] && [ -f "${OUTPUT_FILE}" ]; then
    echo "  输出文件: ${OUTPUT_FILE}"

    echo ""
    echo "=========================================="
    echo "  [评估1] DEER 自带评估"
    echo "=========================================="
    cd "${DEER_DIR}"
    uv run python check_fixed.py \
        --model_name_or_path "${MODEL_PATH}" \
        --data_name "${DATASET}" \
        --data_dir "${DEER_DIR}/data" \
        --generation_path "${OUTPUT_FILE}" \
        2>&1 || echo "  ⚠️  DEER 自带评估失败"

    echo ""
    echo "=========================================="
    echo "  [评估2] OThink-R1 Verifier 评估"
    echo "=========================================="
    cd "${PROJECT_ROOT}"
    uv run python "${DEER_DIR}/scripts/eval_with_othink.py" \
        --generation_path "${OUTPUT_FILE}" \
        --dataset "${DATASET}" \
        2>&1 || echo "  ⚠️  OThink-R1 评估失败"
else
    echo "  ⚠️  未找到输出文件，请检查 ${DEER_DIR}/outputs/"
fi
