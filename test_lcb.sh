#!/bin/bash
# ============================================================
# test_lcb_smoke.sh
# LiveCodeBench 冒烟测试脚本
#
# 用一个小模型 (Qwen2.5-0.5B-Instruct) 跑 2 道题，
# 验证整个 pipeline 能否正常走通。
#
# 用法:
#   cd ~/ACL-ARR-Jan-Rebuttal/OThink-R1
#   bash test_lcb_smoke.sh
# ============================================================

set -e

eval "$(conda shell.bash hook)"
conda activate othink-r1
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

MODEL_PATH="${PROJECT_ROOT}/models/Qwen2.5-0.5B-Instruct"
DATASET_DIR="${PROJECT_ROOT}/datasets/livecodebench"
LCB_SRC_DIR="${PROJECT_ROOT}/benchmark/livecodebench/LiveCodeBench"
OUTPUT_DIR="${PROJECT_ROOT}/benchmark/livecodebench/outputs/smoke_test"

echo "=========================================="
echo "  LiveCodeBench 冒烟测试"
echo "=========================================="
echo "  项目根目录: ${PROJECT_ROOT}"
echo "  模型路径:   ${MODEL_PATH}"
echo "  数据集路径: ${DATASET_DIR}"
echo "=========================================="

# ============================================================
# 前置检查
# ============================================================

echo ""
echo "[1/5] 前置检查..."

PASS=true

# 检查模型
if [ -d "${MODEL_PATH}" ] && [ -f "${MODEL_PATH}/config.json" ]; then
    echo "  ✅ 模型存在: ${MODEL_PATH}"
else
    echo "  ❌ 模型不存在或不完整: ${MODEL_PATH}"
    echo "     需要包含 config.json 文件"
    PASS=false
fi

# 检查数据集 - 遍历可能的子目录结构
DATASET_FOUND=false
DATASET_ACTUAL_PATH=""

if [ -d "${DATASET_DIR}" ]; then
    # 情况1: datasets/livecodebench/ 下直接有 parquet/json 数据文件
    if ls "${DATASET_DIR}"/*.parquet >/dev/null 2>&1 || \
       ls "${DATASET_DIR}"/*.json >/dev/null 2>&1 || \
       ls "${DATASET_DIR}"/*.jsonl >/dev/null 2>&1 || \
       ls "${DATASET_DIR}"/*.arrow >/dev/null 2>&1; then
        DATASET_FOUND=true
        DATASET_ACTUAL_PATH="${DATASET_DIR}"
    fi

    # 情况2: datasets/livecodebench/code_generation_lite/ 子目录
    if [ -d "${DATASET_DIR}/code_generation_lite" ]; then
        if ls "${DATASET_DIR}/code_generation_lite"/*.parquet >/dev/null 2>&1 || \
           ls "${DATASET_DIR}/code_generation_lite"/*.json >/dev/null 2>&1 || \
           ls "${DATASET_DIR}/code_generation_lite"/*.jsonl >/dev/null 2>&1 || \
           ls "${DATASET_DIR}/code_generation_lite"/*.arrow >/dev/null 2>&1; then
            DATASET_FOUND=true
            DATASET_ACTUAL_PATH="${DATASET_DIR}/code_generation_lite"
        fi
    fi

    # 情况3: datasets/livecodebench/livecodebench/ 子目录
    if [ -d "${DATASET_DIR}/livecodebench" ]; then
        DATASET_FOUND=true
        DATASET_ACTUAL_PATH="${DATASET_DIR}/livecodebench"
    fi

    # 情况4: 目录存在但可能是 HuggingFace datasets cache 格式
    if [ "${DATASET_FOUND}" = false ]; then
        FILE_COUNT=$(find "${DATASET_DIR}" -type f | head -20 | wc -l)
        if [ "${FILE_COUNT}" -gt 0 ]; then
            DATASET_FOUND=true
            DATASET_ACTUAL_PATH="${DATASET_DIR}"
        fi
    fi
fi

if [ "${DATASET_FOUND}" = true ]; then
    echo "  ✅ 数据集存在: ${DATASET_ACTUAL_PATH}"
    echo "     文件列表 (前10个):"
    find "${DATASET_ACTUAL_PATH}" -type f | head -10 | while read f; do
        echo "       $(basename "$f")"
    done
else
    echo "  ❌ 数据集不存在或为空: ${DATASET_DIR}"
    echo "     请先运行: bash benchmark/livecodebench/download_data.sh"
    PASS=false
fi

# 检查 LiveCodeBench 源码
if [ -d "${LCB_SRC_DIR}" ] && [ -d "${LCB_SRC_DIR}/lcb_runner" ]; then
    echo "  ✅ LiveCodeBench 源码存在: ${LCB_SRC_DIR}"
else
    echo "  ❌ LiveCodeBench 源码不存在: ${LCB_SRC_DIR}"
    echo "     请先运行: bash setup_livecodebench.sh"
    PASS=false
fi

# 检查 Python 依赖
echo ""
echo "[2/5] 检查 Python 依赖..."

MISSING_DEPS=()

uv run python -c "import torch" 2>/dev/null && echo "  ✅ torch" || { echo "  ❌ torch"; MISSING_DEPS+=("torch"); }
uv run python -c "import transformers" 2>/dev/null && echo "  ✅ transformers" || { echo "  ❌ transformers"; MISSING_DEPS+=("transformers"); }
uv run python -c "import vllm" 2>/dev/null && echo "  ✅ vllm" || { echo "  ❌ vllm"; MISSING_DEPS+=("vllm"); }
uv run python -c "import datasets" 2>/dev/null && echo "  ✅ datasets" || { echo "  ❌ datasets"; MISSING_DEPS+=("datasets"); }
uv run python -c "import tqdm" 2>/dev/null && echo "  ✅ tqdm" || { echo "  ❌ tqdm"; MISSING_DEPS+=("tqdm"); }
uv run python -c "import numpy" 2>/dev/null && echo "  ✅ numpy" || { echo "  ❌ numpy"; MISSING_DEPS+=("numpy"); }

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo ""
    echo "  ❌ 缺少依赖: ${MISSING_DEPS[*]}"
    echo "     请安装: pip install ${MISSING_DEPS[*]}"
    PASS=false
fi


# 汇总检查结果
if [ "${PASS}" = false ]; then
    echo ""
    echo "=========================================="
    echo "  ❌ 前置检查未通过，请修复上述问题后重试"
    echo "=========================================="
    exit 1
fi

echo ""
echo "  ✅ 所有前置检查通过！"

# ============================================================
# 运行冒烟测试
# ============================================================

echo ""
echo "[4/5] 运行冒烟测试 (2 道题)..."
echo ""

mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH="${LCB_SRC_DIR}:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0

uv run python << 'PYEOF'
import os
import sys
import json
import time
import math
import re

# ---------------------------------------------------------------------------
# 安全常量
# ---------------------------------------------------------------------------
BACKTICK3 = chr(96) * 3
BACKTICK3_PY = chr(96) * 3 + "python"
THINK_CLOSE_TAG = "<" + "/think" + ">"

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Qwen2.5-0.5B-Instruct")
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", "livecodebench")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "benchmark", "livecodebench", "outputs", "smoke_test")

print("=" * 60)
print("  冒烟测试: 加载模型 + 数据 + 生成 + 提取代码")
print("=" * 60)

# ---- Step A: 加载数据集 ----
print("\n[A] 加载数据集...")
from datasets import load_dataset

dataset = None
# 尝试多种加载方式
for try_path in [
    DATASET_DIR,
    os.path.join(DATASET_DIR, "code_generation_lite"),
    os.path.join(DATASET_DIR, "livecodebench"),
]:
    if os.path.isdir(try_path):
        try:
            dataset = load_dataset(try_path, split="test")
            print(f"  ✅ 从 {try_path} 加载成功, 共 {len(dataset)} 道题")
            break
        except Exception as e:
            print(f"  ⚠️  尝试 {try_path} 失败: {e}")
            continue

if dataset is None:
    # 最后尝试从 HuggingFace 加载
    try:
        dataset = load_dataset("livecodebench/code_generation_lite", split="test")
        print(f"  ✅ 从 HuggingFace 加载成功, 共 {len(dataset)} 道题")
    except Exception as e:
        print(f"  ❌ 所有加载方式均失败: {e}")
        sys.exit(1)

# 只取前 2 道题
problems = [dict(dataset[i]) for i in range(min(2, len(dataset)))]
print(f"  取前 {len(problems)} 道题进行测试")

for i, p in enumerate(problems):
    qid = p.get("question_id", f"unknown_{i}")
    title = p.get("question_title", p.get("title", "N/A"))
    content_preview = p.get("question_content", p.get("problem", ""))[:80]
    print(f"  题目 {i}: [{qid}] {title}")
    print(f"          {content_preview}...")

# ---- Step B: 加载模型 ----
print("\n[B] 加载模型 (vLLM)...")
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"  ✅ Tokenizer 加载成功: {tokenizer.__class__.__name__}")

llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=1,
    dtype="bfloat16",
    max_model_len=4096,
    gpu_memory_utilization=0.8,
    trust_remote_code=True,
)
print(f"  ✅ vLLM 引擎加载成功")

# ---- Step C: 生成 ----
print("\n[C] 生成代码...")

sys_prompt = (
    "You are an expert Python programmer. "
    "Solve the given competitive programming problem. "
    "Provide your solution as a complete Python program."
)

results = []
for i, prob in enumerate(problems):
    question_content = prob.get("question_content", prob.get("problem", ""))
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question_content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    print(f"\n  --- 题目 {i} ---")
    start = time.time()

    outputs = llm.generate(
        [prompt],
        SamplingParams(
            max_tokens=1024,
            temperature=0.0,
            top_p=1.0,
        ),
        use_tqdm=False,
    )

    gen_text = outputs[0].outputs[0].text
    gen_tokens = len(outputs[0].outputs[0].token_ids)
    elapsed = time.time() - start

    print(f"  生成 tokens: {gen_tokens}, 耗时: {elapsed:.1f}s")
    print(f"  输出前 200 字符:")
    print(f"  {gen_text[:200]}...")

    # 提取代码
    bt3 = BACKTICK3
    pattern_py = bt3 + r'python\s*\n(.*?)' + bt3
    pattern_any = bt3 + r'\s*\n(.*?)' + bt3

    code = None
    matches = re.findall(pattern_py, gen_text, re.DOTALL)
    if matches:
        code = matches[-1].strip()
        print(f"  ✅ 提取到 python 代码块 ({len(code)} 字符)")
    else:
        matches = re.findall(pattern_any, gen_text, re.DOTALL)
        if matches:
            code = matches[-1].strip()
            print(f"  ✅ 提取到普通代码块 ({len(code)} 字符)")
        else:
            code = gen_text.strip()
            print(f"  ⚠️  未找到代码块，使用原始输出 ({len(code)} 字符)")

    results.append({
        "question_id": prob.get("question_id", i),
        "output": gen_text,
        "code": code,
        "tokens": gen_tokens,
        "time": elapsed,
    })

# ---- Step D: 保存结果 ----
print("\n[D] 保存结果...")

os.makedirs(OUTPUT_DIR, exist_ok=True)
out_file = os.path.join(OUTPUT_DIR, "smoke_test_results.json")
with open(out_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"  ✅ 结果保存到: {out_file}")

# ---- Step E: 汇总 ----
print("\n" + "=" * 60)
print("  冒烟测试汇总")
print("=" * 60)
total_tokens = sum(r['tokens'] for r in results)
total_time = sum(r['time'] for r in results)
print(f"  测试题数:   {len(results)}")
print(f"  总 tokens:  {total_tokens}")
print(f"  总耗时:     {total_time:.1f}s")
print(f"  平均 tok/s: {total_tokens/max(total_time,0.1):.1f}")
for i, r in enumerate(results):
    has_code = BACKTICK3 in r['output']
    print(f"  题目 {i}: {r['tokens']} tokens, {r['time']:.1f}s, 代码块: {'✅' if has_code else '⚠️'}")
print("=" * 60)
print("  ✅ 冒烟测试完成！Pipeline 正常工作。")
print("=" * 60)
PYEOF

echo ""
echo "[5/5] 测试完成！"
echo ""
echo "=========================================="
echo "  ✅ LiveCodeBench 冒烟测试通过"
echo "=========================================="
echo ""
echo "  下一步可以运行完整评测:"
echo "    bash benchmark/livecodebench/run_lcb.sh \\"
echo "        --model Qwen2.5-0.5B-Instruct \\"
echo "        --model_path ${MODEL_PATH} \\"
echo "        --gpu_ids 0"
echo ""
echo "  或 DEER 评测:"
echo "    bash benchmark/livecodebench/run_deer_lcb.sh \\"
echo "        --model ${MODEL_PATH} \\"
echo "        --gpu_ids 0 \\"
echo "        --threshold 0.95 \\"
echo "        --max_samples 10"
echo ""

