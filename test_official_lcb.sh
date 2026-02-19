#!/bin/bash
# ============================================================
# run_lcb_full.sh
# 用 Qwen2.5-0.5B-Instruct 跑完整 LiveCodeBench 评测 (1055 题)
#
# 用法:
#   cd ~/ACL-ARR-Jan-Rebuttal/OThink-R1
#   bash run_lcb_full.sh
# ============================================================

set -e

eval "$(conda shell.bash hook)"
conda activate othink-r1

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

MODEL_PATH="${PROJECT_ROOT}/models/Qwen2.5-0.5B-Instruct"
MODEL_NAME="Qwen2.5-0.5B-Instruct"
DATASET_DIR="${PROJECT_ROOT}/datasets/livecodebench/code_generation_lite"
LCB_SRC_DIR="${PROJECT_ROOT}/benchmark/livecodebench/LiveCodeBench"
OUTPUT_DIR="${PROJECT_ROOT}/benchmark/livecodebench/outputs/${MODEL_NAME}"

export PYTHONPATH="${LCB_SRC_DIR}:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export HF_DATASETS_TRUST_REMOTE_CODE=1

echo "=========================================="
echo "  LiveCodeBench 完整评测"
echo "=========================================="
echo "  模型:   ${MODEL_NAME}"
echo "  数据集: ${DATASET_DIR}"
echo "  输出:   ${OUTPUT_DIR}"
echo "=========================================="

# 前置检查
if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "❌ 模型不存在: ${MODEL_PATH}"
    exit 1
fi

if [ ! -d "${DATASET_DIR}" ]; then
    echo "❌ 数据集不存在: ${DATASET_DIR}"
    exit 1
fi

if [ ! -d "${LCB_SRC_DIR}/lcb_runner" ]; then
    echo "❌ LiveCodeBench 源码不存在: ${LCB_SRC_DIR}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo ""
echo "开始评测..."
echo ""

uv run python << 'PYEOF'
import os
import sys
import json
import time
import re
import math
import numpy as np

# ---------------------------------------------------------------------------
# 安全常量
# ---------------------------------------------------------------------------
BACKTICK3 = chr(96) * 3
BACKTICK3_PY = chr(96) * 3 + "python"
THINK_CLOSE_TAG = "<" + "/think" + ">"

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Qwen2.5-0.5B-Instruct")
MODEL_NAME = "Qwen2.5-0.5B-Instruct"
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", "livecodebench", "code_generation_lite")
LCB_SRC_DIR = os.path.join(PROJECT_ROOT, "benchmark", "livecodebench", "LiveCodeBench")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "benchmark", "livecodebench", "outputs", MODEL_NAME)

# =========================================================================
# Step 1: 加载数据集
# =========================================================================
print("=" * 60)
print("  LiveCodeBench 完整评测")
print("=" * 60)

print("\n[1/4] 加载数据集...")
from datasets import load_dataset

dataset = load_dataset(DATASET_DIR, split="test", trust_remote_code=True)
problems = [dict(dataset[i]) for i in range(len(dataset))]
print(f"  ✅ 加载 {len(problems)} 道题")

# =========================================================================
# Step 2: 加载模型
# =========================================================================
print("\n[2/4] 加载模型...")
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=1,
    dtype="bfloat16",
    max_model_len=16384,
    gpu_memory_utilization=0.9,
    trust_remote_code=True,
)
print(f"  ✅ 模型加载完成")

# =========================================================================
# Step 3: 批量生成
# =========================================================================
print("\n[3/4] 批量生成...")

sys_prompt = (
    "You are an expert Python programmer. "
    "Solve the given competitive programming problem by reading from standard input "
    "and writing to standard output. "
    "Think step by step, then provide your final solution as a complete Python program."
)

# 构建所有 prompt
all_prompts = []
for prob in problems:
    question_content = prob.get("question_content", prob.get("problem", ""))
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question_content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    all_prompts.append(prompt)

print(f"  构建了 {len(all_prompts)} 个 prompt")
print(f"  开始批量推理 (max_tokens=2048, temperature=0.0)...")

start_time = time.time()

# vLLM 批量推理 - 一次性传入所有 prompt
sampling_params = SamplingParams(
    max_tokens=2048,
    temperature=0.0,
    top_p=1.0,
)

outputs = llm.generate(all_prompts, sampling_params)

elapsed = time.time() - start_time
print(f"  ✅ 推理完成, 耗时 {elapsed:.1f}s ({elapsed/60:.1f}min)")

# =========================================================================
# Step 4: 提取代码 + 保存 + 评测
# =========================================================================
print("\n[4/4] 提取代码并评测...")

def extract_code(response):
    """从模型输出中提取代码"""
    bt3 = BACKTICK3
    pattern_py = bt3 + r'python\s*\n(.*?)' + bt3
    pattern_any = bt3 + r'\s*\n(.*?)' + bt3

    # 如果有 think 标签，取之后的部分
    if THINK_CLOSE_TAG in response:
        parts = response.split(THINK_CLOSE_TAG, 1)
        answer_part = parts[1].strip()
    else:
        answer_part = response.strip()

    # python 代码块
    matches = re.findall(pattern_py, answer_part, re.DOTALL)
    if matches:
        return matches[-1].strip()
    matches = re.findall(pattern_py, response, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # 普通代码块
    matches = re.findall(pattern_any, answer_part, re.DOTALL)
    if matches:
        return matches[-1].strip()
    matches = re.findall(pattern_any, response, re.DOTALL)
    if matches:
        return matches[-1].strip()

    if answer_part:
        return answer_part
    return response.strip()

# 整理结果
all_results = []
total_tokens = 0
code_found = 0

for i, (prob, output) in enumerate(zip(problems, outputs)):
    gen_text = output.outputs[0].text
    gen_tokens = len(output.outputs[0].token_ids)
    total_tokens += gen_tokens

    code = extract_code(gen_text)

    bt3 = BACKTICK3
    if bt3 in gen_text:
        code_found += 1

    all_results.append({
        "question_id": prob.get("question_id", i),
        "question_title": prob.get("question_title", ""),
        "output_list": [gen_text],
        "code_list": [code],
        "tokens": gen_tokens,
    })

print(f"  总 tokens: {total_tokens}")
print(f"  平均 tokens/题: {total_tokens / len(all_results):.0f}")
print(f"  代码块提取率: {code_found}/{len(all_results)} ({100*code_found/len(all_results):.1f}%)")
print(f"  吞吐量: {total_tokens / elapsed:.1f} tok/s")

# 保存生成结果
os.makedirs(OUTPUT_DIR, exist_ok=True)
gen_file = os.path.join(OUTPUT_DIR, "generations.json")
with open(gen_file, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print(f"  ✅ 生成结果保存到: {gen_file}")

# ---- 评测 ----
print("\n" + "=" * 60)
print("  开始 LiveCodeBench 评测 (代码执行 + 判分)...")
print("=" * 60)

try:
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem

    # 重建 CodeGenerationProblem 对象
    lcb_problems = []
    for prob in problems:
        try:
            lcb_problems.append(CodeGenerationProblem(**prob))
        except Exception:
            lcb_problems.append(prob)

    # 构建 (output_list, code_list) 格式
    combined = [
        (r['output_list'], r['code_list'])
        for r in all_results
    ]

    print(f"  评测 {len(combined)} 道题 (每题 timeout=6s, 12 进程并行)...")
    eval_start = time.time()

    metrics = codegen_metrics(
        lcb_problems,
        combined,
        num_process_evaluate=12,
        timeout=6,
    )

    eval_elapsed = time.time() - eval_start
    print(f"  评测耗时: {eval_elapsed:.1f}s ({eval_elapsed/60:.1f}min)")

    # 打印结果
    print("\n" + "=" * 60)
    print("  评测结果")
    print("=" * 60)

    if isinstance(metrics, tuple):
        for i, m in enumerate(metrics):
            print(f"\n  --- metrics[{i}] ---")
            if isinstance(m, dict):
                for k, v in m.items():
                    print(f"    {k}: {v}")
            else:
                print(f"    {m}")
    elif isinstance(metrics, dict):
        for k, v in metrics.items():
            print(f"    {k}: {v}")
    else:
        print(f"    {metrics}")

    # 保存评测结果
    eval_file = os.path.join(OUTPUT_DIR, "eval_results.json")
    with open(eval_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n  ✅ 评测结果保存到: {eval_file}")

except ImportError as e:
    print(f"\n  ⚠️  无法导入评测模块: {e}")
    print(f"  生成结果已保存，可以稍后手动评测:")
    print(f"    {gen_file}")
except Exception as e:
    print(f"\n  ⚠️  评测出错: {e}")
    import traceback
    traceback.print_exc()
    print(f"\n  生成结果已保存: {gen_file}")

# 最终汇总
print("\n" + "=" * 60)
print("  完成汇总")
print("=" * 60)
print(f"  模型:         {MODEL_NAME}")
print(f"  题目数:       {len(all_results)}")
print(f"  推理耗时:     {elapsed:.1f}s ({elapsed/60:.1f}min)")
print(f"  总 tokens:    {total_tokens}")
print(f"  平均 tok/题:  {total_tokens / len(all_results):.0f}")
print(f"  吞吐量:       {total_tokens / elapsed:.1f} tok/s")
print(f"  代码提取率:   {100*code_found/len(all_results):.1f}%")
print(f"  输出目录:     {OUTPUT_DIR}")
print("=" * 60)
PYEOF

echo ""
echo "=========================================="
echo "  ✅ LiveCodeBench 完整评测完成"
echo "=========================================="
echo "  结果目录: ${OUTPUT_DIR}"
echo ""