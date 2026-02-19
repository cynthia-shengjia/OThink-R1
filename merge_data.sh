#!/bin/bash
# ============================================================
# setup_livecodebench.sh
# 将 LiveCodeBench 整合到 OThink-R1 项目的 benchmark/ 目录下
#
# 使用方法:
#   cd ~/ACL-ARR-Jan-Rebuttal/OThink-R1
#   bash setup_livecodebench.sh [/path/to/local/LiveCodeBench]
#
# 参数:
#   $1 (可选): 本地已 clone 的 LiveCodeBench 路径
# ============================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
BENCHMARK_DIR="${PROJECT_ROOT}/benchmark/livecodebench"
LCB_SRC_DIR="${BENCHMARK_DIR}/LiveCodeBench"

echo "=========================================="
echo "  LiveCodeBench → OThink-R1 整合"
echo "  项目根目录: ${PROJECT_ROOT}"
echo "=========================================="

# ============================================================
# Step 1: 创建 benchmark 目录结构 + 拷贝/clone LiveCodeBench
# ============================================================

echo ""
echo "[1/6] 创建 benchmark/livecodebench/ 目录并导入 LiveCodeBench 源码..."

mkdir -p "${BENCHMARK_DIR}"

if [ -n "$1" ] && [ -d "$1" ]; then
    echo "  使用本地 LiveCodeBench: $1"
    if [ -d "${LCB_SRC_DIR}" ]; then
        echo "  ⚠️  ${LCB_SRC_DIR} 已存在，跳过拷贝"
    else
        cp -r "$1" "${LCB_SRC_DIR}"
        echo "  ✅ 已拷贝到 ${LCB_SRC_DIR}"
    fi
else
    if [ -d "${LCB_SRC_DIR}" ]; then
        echo "  ⚠️  ${LCB_SRC_DIR} 已存在，跳过 clone"
    else
        echo "  从 GitHub clone LiveCodeBench..."
        git clone https://github.com/cynthia-shengjia/LiveCodeBench.git "${LCB_SRC_DIR}"
        echo "  ✅ clone 完成"
    fi
fi

# ============================================================
# Step 2: 更新 pyproject.toml 添加 LiveCodeBench 依赖
# ============================================================

echo ""
echo "[2/6] 检查并更新 pyproject.toml 依赖..."

PYPROJECT="${PROJECT_ROOT}/pyproject.toml"
DEPS_TO_ADD=("pebble>=5.1.0" "annotated-types>=0.7.0")

for dep in "${DEPS_TO_ADD[@]}"; do
    dep_name=$(echo "$dep" | sed 's/[>=<].*//')
    if grep -qi "${dep_name}" "${PYPROJECT}"; then
        echo "  ⚠️  ${dep_name} 已在 pyproject.toml 中"
    else
        if grep -q "# DEER Baseline Dependencies" "${PYPROJECT}"; then
            sed -i "/# DEER Baseline Dependencies/i\\    \"${dep}\"," "${PYPROJECT}"
        else
            sed -i "/^]/i\\    \"${dep}\"," "${PYPROJECT}"
        fi
        echo "  ✅ 添加 ${dep}"
    fi
done

echo "  ✅ 依赖更新完成"

# ============================================================
# Step 3: 创建标准评测入口脚本 run_lcb.sh
# ============================================================

echo ""
echo "[3/6] 创建标准评测脚本 run_lcb.sh..."

cat > "${BENCHMARK_DIR}/run_lcb.sh" << 'RUNEOF'
#!/bin/bash
# ============================================================
# LiveCodeBench 标准评测入口
# ============================================================

set -e
eval "$(conda shell.bash hook)"
conda activate othink-r1

MODEL=""
MODEL_PATH=""
GPU_IDS="0"
MAX_TOKENS=16289
TEMPERATURE=0.9
CODEGEN_N=1
N=1
RELEASE_VERSION="release_v5"
SCENARIO="codegeneration"
STOP_WORDS="None"
LOCAL_DATA_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2;;
        --model_path) MODEL_PATH="$2"; shift 2;;
        --gpu_ids) GPU_IDS="$2"; shift 2;;
        --max_tokens) MAX_TOKENS="$2"; shift 2;;
        --temperature) TEMPERATURE="$2"; shift 2;;
        --codegen_n) CODEGEN_N="$2"; shift 2;;
        --n) N="$2"; shift 2;;
        --release_version) RELEASE_VERSION="$2"; shift 2;;
        --local_data) LOCAL_DATA_PATH="$2"; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

if [ -z "${MODEL}" ]; then
    echo "❌ 请指定模型: --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LCB_DIR="${SCRIPT_DIR}/LiveCodeBench"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -n "${MODEL_PATH}" ] && [[ "${MODEL_PATH}" != /* ]]; then
    MODEL_PATH="$(cd "$(dirname "${MODEL_PATH}")" && pwd)/$(basename "${MODEL_PATH}")"
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

echo "=========================================="
echo "  LiveCodeBench 标准评测"
echo "  模型: ${MODEL}"
echo "  GPU: ${GPU_IDS}"
echo "=========================================="

LCB_ARGS=(
    --model "${MODEL}"
    --scenario "${SCENARIO}"
    --max_tokens "${MAX_TOKENS}"
    --release_version "${RELEASE_VERSION}"
    --evaluate
    --codegen_n "${CODEGEN_N}"
    --n "${N}"
    --temperature "${TEMPERATURE}"
    --stop "${STOP_WORDS}"
)

if [ -n "${MODEL_PATH}" ]; then
    LCB_ARGS+=(--local_model_path "${MODEL_PATH}")
fi

if [ -n "${LOCAL_DATA_PATH}" ]; then
    LCB_ARGS+=(--local_dataset_path "${LOCAL_DATA_PATH}")
elif [ -d "${PROJECT_ROOT}/datasets/livecodebench/code_generation_lite" ]; then
    LCB_ARGS+=(--local_dataset_path "${PROJECT_ROOT}/datasets/livecodebench/code_generation_lite")
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${LCB_DIR}:${PYTHONPATH}"
uv run python -m lcb_runner.runner.main "${LCB_ARGS[@]}"

MODEL_NAME=$(basename "${MODEL}")
OUTPUT_FILE="${LCB_DIR}/output/${MODEL_NAME}/Scenario.${SCENARIO}_${CODEGEN_N}_${TEMPERATURE}.json"

if [ -f "${OUTPUT_FILE}" ]; then
    echo "  ✅ 评测完成！结果: ${OUTPUT_FILE}"
    uv run python -m lcb_runner.utils.get_length_lcb \
        --model_name "${MODEL}" \
        --file_path "${OUTPUT_FILE}" \
        2>/dev/null || echo "  ⚠️  token 统计失败"
else
    echo "  ⚠️  未找到输出文件: ${OUTPUT_FILE}"
fi
RUNEOF
chmod +x "${BENCHMARK_DIR}/run_lcb.sh"
echo "  ✅ run_lcb.sh 创建完成"

# ============================================================
# Step 4: 创建 DEER 适配脚本 deer_lcb.py
# ============================================================

echo ""
echo "[4/6] 创建 DEER 适配脚本 deer_lcb.py..."

cat > "${BENCHMARK_DIR}/deer_lcb.py" << 'DEEREOF'
"""
DEER (Dynamic Early Exit for Reasoning) 适配 LiveCodeBench 代码生成任务

核心适配点:
- 代码生成任务使用代码块标记替代 boxed 作为 answer inducing prompt
- 在代码块结束标记处检测 confidence
- confidence 计算复用 DEER 的 avg logprob 策略
- 生成完成后提取代码块，调用 LiveCodeBench 评测管线评分

用法:
    cd OThink-R1 项目根目录
    python benchmark/livecodebench/deer_lcb.py \
        --model_name_or_path ./models/DeepSeek-R1-Distill-Qwen-7B \
        --threshold 0.95 \
        --max_len 16384 \
        --dataset_path ./datasets/livecodebench/code_generation_lite \
        --release_version release_v5 \
        --gpu_ids 0
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import time
import argparse
import re
import math
import numpy as np
import random

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# 安全常量: 避免在模板/heredoc 中直接写三反引号和 HTML 标签
# ---------------------------------------------------------------------------
BACKTICK3 = chr(96) * 3
BACKTICK3_PY = chr(96) * 3 + "python"
THINK_CLOSE_TAG = "<" + "/think" + ">"

# ---------------------------------------------------------------------------
# 将 LiveCodeBench 加入 Python path
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LCB_DIR = os.path.join(SCRIPT_DIR, "LiveCodeBench")
if LCB_DIR not in sys.path:
    sys.path.insert(0, LCB_DIR)

# ---------------------------------------------------------------------------
# DEER 核心: answer confidence 计算
# ---------------------------------------------------------------------------
def calculate_avg_prob_from_logprobs(logprobs_list, policy='avg1') -> float:
    """从 vLLM logprobs 计算平均 token 概率"""
    num_tokens = len(logprobs_list)
    if num_tokens < 2:
        return 0.0

    total_prob = 0.0
    log_prob_sum = 0.0
    count = 0
    min_prob = 1.0

    for i in range(1, num_tokens):
        if i < len(logprobs_list) and logprobs_list[i]:
            try:
                logprob_obj = list(logprobs_list[i].values())[0]
                if hasattr(logprob_obj, 'logprob'):
                    prob = math.exp(logprob_obj.logprob)
                    min_prob = min(min_prob, prob)
                    total_prob += prob
                    log_prob_sum += math.log(max(prob, 1e-10))
                    count += 1
            except (IndexError, KeyError, AttributeError):
                pass

    if count == 0:
        return 0.0

    if policy == 'min':
        return min_prob
    elif policy == 'avg1':
        return total_prob / count
    elif policy == 'avg2':
        return math.exp(log_prob_sum / count)
    return 0.0

# ---------------------------------------------------------------------------
# 代码提取
# ---------------------------------------------------------------------------
def extract_code_from_response(response: str) -> str:
    """
    从模型回复中提取 Python 代码块。

    优先级:
      1. think结束标签之后的 python 代码块
      2. 全文中最后一个 python 代码块
      3. 全文中最后一个普通代码块
      4. think结束标签之后的全部文本
      5. 返回原始文本
    """
    bt3 = BACKTICK3
    pattern_py = bt3 + r'python\s*\n(.*?)' + bt3
    pattern_any = bt3 + r'\s*\n(.*?)' + bt3

    if THINK_CLOSE_TAG in response:
        parts = response.split(THINK_CLOSE_TAG, 1)
        answer_part = parts[1].strip()
    else:
        answer_part = response.strip()

    # 优先级 1: answer 部分中的 python 代码块
    matches = re.findall(pattern_py, answer_part, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # 优先级 2: 全文中的 python 代码块
    matches = re.findall(pattern_py, response, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # 优先级 3: answer 部分中的普通代码块
    matches = re.findall(pattern_any, answer_part, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # 优先级 4: 全文中的普通代码块
    matches = re.findall(pattern_any, response, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # 优先级 5
    if answer_part:
        return answer_part
    return response.strip()

# ---------------------------------------------------------------------------
# 加载 LiveCodeBench 数据集
# ---------------------------------------------------------------------------
def load_lcb_dataset(dataset_path, release_version="release_v5"):
    """加载 LiveCodeBench 代码生成数据集"""
    from datasets import load_dataset as hf_load_dataset

    if dataset_path and os.path.isdir(dataset_path):
        print(f"  从本地加载: {dataset_path}")
        dataset = hf_load_dataset(dataset_path, split="test")
    else:
        print(f"  从 HuggingFace 加载: livecodebench/code_generation_lite")
        dataset = hf_load_dataset("livecodebench/code_generation_lite", split="test")

    problems = []
    for item in dataset:
        problems.append(dict(item))

    print(f"  加载了 {len(problems)} 道题")
    return problems

# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="DEER on LiveCodeBench")
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, default=None,
                        help="本地 LiveCodeBench 数据集路径")
    parser.add_argument('--release_version', type=str, default='release_v5')
    parser.add_argument('--threshold', type=float, default=0.95,
                        help="DEER early exit confidence 阈值")
    parser.add_argument('--max_len', type=int, default=16384,
                        help="最大生成 token 数")
    parser.add_argument('--think_ratio', type=float, default=0.87,
                        help="思考阶段占总 token 预算的比例")
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--policy', type=str, default='avg1',
                        choices=['min', 'avg1', 'avg2'])
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--max_judge_steps', type=int, default=10)
    parser.add_argument('--prob_check_max_tokens', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--max_samples', type=int, default=None,
                        help="最大样本数（调试用）")
    parser.add_argument('--no_evaluate', action='store_true',
                        help="只生成不评测")
    args = parser.parse_args()
    return args

# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    available_gpus = args.gpu_ids.split(',')

    print("=" * 60)
    print("  DEER x LiveCodeBench (Code Generation)")
    print("=" * 60)
    print(f"  模型: {args.model_name_or_path}")
    print(f"  阈值: {args.threshold}")
    print(f"  最大长度: {args.max_len}")
    print(f"  思考比例: {args.think_ratio}")
    print(f"  策略: {args.policy}")
    print("=" * 60)

    # ---- 初始化 vLLM ----
    model_context_len = args.max_len + 8000
    llm_engine = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=len(available_gpus),
        dtype="bfloat16",
        max_model_len=model_context_len,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 加载数据集 ----
    print("\n加载 LiveCodeBench 数据集...")
    problems = load_lcb_dataset(args.dataset_path, args.release_version)
    if args.max_samples:
        problems = problems[:args.max_samples]
    print(f"  实际评测: {len(problems)} 道题")

    # ---- DEER 适配: 代码生成的 prompt 和 stop tokens ----
    code_answer_prompt = "\n\nHere is my solution:\n\n" + BACKTICK3_PY + "\n"

    continue_str = "Wait"
    last_token_strs = [THINK_CLOSE_TAG]
    generation_stop_tokens = [continue_str] + last_token_strs + [tokenizer.eos_token]

    prob_check_stop_tokens = [BACKTICK3 + "\n", BACKTICK3, "\n" + BACKTICK3]

    answer_stop_tokens = [tokenizer.eos_token]

    think_limit = int(args.max_len * args.think_ratio)

    # ---- 构建 prompts ----
    sys_prompt = (
        "You are an expert Python programmer. "
        "Solve the given competitive programming problem by reading from standard input and writing to standard output. "
        "Think step by step, then provide your final solution as a complete Python program."
    )

    formatted_prompts = []
    for prob in problems:
        question_content = prob.get("question_content", prob.get("problem", ""))
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question_content},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted_prompts.append(prompt)

    # ---- DEER 主循环 ----
    print("\n开始 DEER 推理...")
    start_time = time.time()

    all_results = []

    for idx, (prob, prompt) in enumerate(tqdm(
        zip(problems, formatted_prompts), total=len(problems), desc="DEER"
    )):
        thinking_history = ""
        current_seq = prompt
        thinking_steps = 0
        early_exit = False
        too_long = False
        regular_end = False

        # ---- Phase 1: 迭代思考 + confidence 检查 ----
        while True:
            think_tokens_used = len(tokenizer.encode(
                thinking_history, add_special_tokens=False
            ))
            remaining = think_limit - think_tokens_used
            if remaining <= 50:
                too_long = True
                break

            if thinking_steps < args.max_judge_steps:
                stop = generation_stop_tokens
            else:
                stop = last_token_strs + [tokenizer.eos_token]

            outputs = llm_engine.generate(
                [current_seq],
                SamplingParams(
                    max_tokens=min(remaining, 4096),
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stop=stop,
                ),
                use_tqdm=False,
            )

            gen_text = outputs[0].outputs[0].text
            gen_ids = outputs[0].outputs[0].token_ids

            thinking_history += gen_text
            current_seq = prompt + thinking_history

            stripped = thinking_history.rstrip()
            if stripped.endswith(THINK_CLOSE_TAG) or stripped.endswith(THINK_CLOSE_TAG + "\n"):
                regular_end = True
                break

            if gen_ids and gen_ids[-1] == tokenizer.eos_token_id:
                regular_end = True
                break

            if not gen_text.strip():
                break

            # ---- Phase 2: Confidence 检查 ----
            thinking_steps += 1
            prob_check_prompt = current_seq + code_answer_prompt

            prob_outputs = llm_engine.generate(
                [prob_check_prompt],
                SamplingParams(
                    max_tokens=args.prob_check_max_tokens,
                    stop=prob_check_stop_tokens,
                    logprobs=1,
                ),
                use_tqdm=False,
            )

            pred_prob = 0.0
            if (prob_outputs[0].outputs[0].logprobs and
                    len(prob_outputs[0].outputs[0].logprobs) > 1):
                pred_prob = calculate_avg_prob_from_logprobs(
                    prob_outputs[0].outputs[0].logprobs, args.policy
                )

            if pred_prob > args.threshold:
                early_exit = True
                print(f"\n  Q{idx}: Early exit (conf={pred_prob:.4f} > {args.threshold})")
                break

            if not stripped.endswith(continue_str):
                current_seq += continue_str
                thinking_history += continue_str

        # ---- Phase 3: 生成最终代码 ----
        stripped_think = thinking_history.rstrip()
        if not stripped_think.endswith(THINK_CLOSE_TAG):
            thinking_history = thinking_history.rstrip() + "\n" + THINK_CLOSE_TAG + "\n\n"
        else:
            thinking_history = thinking_history.rstrip() + "\n\n"

        final_prompt = prompt + thinking_history

        answer_budget = args.max_len - len(
            tokenizer.encode(thinking_history, add_special_tokens=False)
        )
        answer_budget = max(answer_budget, 512)
        answer_budget = min(answer_budget, 4096)

        final_outputs = llm_engine.generate(
            [final_prompt],
            SamplingParams(
                max_tokens=answer_budget,
                temperature=args.temperature,
                top_p=args.top_p,
                stop=answer_stop_tokens,
            ),
            use_tqdm=False,
        )

        final_text = final_outputs[0].outputs[0].text
        full_response = thinking_history + final_text

        code = extract_code_from_response(full_response)

        total_tokens = len(tokenizer.encode(full_response, add_special_tokens=False))

        all_results.append({
            "question_id": prob.get("question_id", idx),
            "question_content": prob.get("question_content", prob.get("problem", "")),
            "output_list": [full_response],
            "code_list": [code],
            "thinking_steps": thinking_steps,
            "early_exit": early_exit,
            "too_long": too_long,
            "regular_end": regular_end,
            "total_tokens": total_tokens,
        })

    elapsed = time.time() - start_time
    print(f"\n推理完成! 共 {len(all_results)} 道题, 耗时 {elapsed:.1f}s")

    # ---- 保存结果 ----
    model_name = os.path.basename(args.model_name_or_path.rstrip('/'))
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(SCRIPT_DIR, "outputs", "deer", model_name)
    os.makedirs(out_dir, exist_ok=True)

    out_file = os.path.join(
        out_dir,
        f"deer_p{args.threshold}_ratio{args.think_ratio}_len{args.max_len}.json"
    )
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"结果保存到: {out_file}")

    # ---- 统计 ----
    early_exits = sum(1 for r in all_results if r['early_exit'])
    too_longs = sum(1 for r in all_results if r['too_long'])
    regular_ends = sum(1 for r in all_results if r['regular_end'])
    avg_tokens = np.mean([r['total_tokens'] for r in all_results])
    avg_steps = np.mean([r['thinking_steps'] for r in all_results])

    print(f"\n============= DEER 统计 =============")
    print(f"  总题数:       {len(all_results)}")
    print(f"  Early Exit:   {early_exits} ({100*early_exits/max(len(all_results),1):.1f}%)")
    print(f"  Regular End:  {regular_ends} ({100*regular_ends/max(len(all_results),1):.1f}%)")
    print(f"  Too Long:     {too_longs} ({100*too_longs/max(len(all_results),1):.1f}%)")
    print(f"  平均 tokens:  {avg_tokens:.0f}")
    print(f"  平均思考步数: {avg_steps:.1f}")

    # ---- 评测 (可选) ----
    if not args.no_evaluate:
        print(f"\n开始 LiveCodeBench 评测...")
        try:
            from lcb_runner.evaluation.compute_code_generation_metrics import (
                codegen_metrics,
            )
            from lcb_runner.benchmarks.code_generation import CodeGenerationProblem

            lcb_problems = []
            for r in all_results:
                orig = None
                for p in problems:
                    pid = p.get("question_id", None)
                    if pid == r["question_id"]:
                        orig = p
                        break
                if orig is not None:
                    try:
                        lcb_problems.append(CodeGenerationProblem(**orig))
                    except Exception:
                        lcb_problems.append(orig)
                else:
                    lcb_problems.append(r)

            combined = [
                ([r['output_list'][0]], [r['code_list'][0]])
                for r in all_results
            ]

            metrics = codegen_metrics(
                lcb_problems,
                combined,
                num_process_evaluate=12,
                timeout=6,
            )

            print(f"\n============= LiveCodeBench 评测结果 =============")
            if isinstance(metrics, tuple) and len(metrics) >= 1:
                summary = metrics[0] if isinstance(metrics[0], dict) else metrics
                print(json.dumps(summary, indent=2, default=str))
            elif isinstance(metrics, dict):
                print(json.dumps(metrics, indent=2, default=str))
            else:
                print(metrics)

            eval_file = out_file.replace('.json', '_eval.json')
            with open(eval_file, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            print(f"评测结果保存到: {eval_file}")

        except ImportError as e:
            print(f"  ⚠️  无法导入 LiveCodeBench 评测模块: {e}")
            print(f"  请确保 LiveCodeBench 已正确安装")
        except Exception as e:
            print(f"  ⚠️  评测失败: {e}")
            import traceback
            traceback.print_exc()
            print(f"  结果已保存，可以稍后手动评测")

if __name__ == "__main__":
    main()
DEEREOF
echo "  ✅ deer_lcb.py 创建完成"

# ============================================================
# Step 5: 创建 CP-Router stub
# ============================================================

echo ""
echo "[5/6] 创建 CP-Router stub..."

cat > "${BENCHMARK_DIR}/cp_router_lcb_stub.py" << 'CPEOF'
"""
CP-Router x LiveCodeBench Stub

CP-Router 不适用于代码生成任务:
1. 代码生成是开放式生成任务，没有固定选项集合
2. 无法计算选项级别的 nonconformity scores
3. 无法构建有意义的预测集
"""
import sys

def main():
    print("=" * 60)
    print("  CP-Router x LiveCodeBench")
    print("=" * 60)
    print()
    print("  CP-Router 不适用于代码生成任务")
    print()
    sys.exit(0)

if __name__ == "__main__":
    main()
CPEOF
echo "  ✅ cp_router_lcb_stub.py 创建完成"

# ============================================================
# Step 6: 创建运行脚本 + 数据下载脚本 + README
# ============================================================

echo ""
echo "[6/6] 创建运行脚本和 README..."

cat > "${BENCHMARK_DIR}/run_deer_lcb.sh" << 'DEERSHEOF'
#!/bin/bash
# ============================================================
# DEER x LiveCodeBench 运行脚本
# ============================================================

set -e
eval "$(conda shell.bash hook)"
conda activate othink-r1

MODEL_PATH=""
GPU_IDS="0"
THRESHOLD=0.95
MAX_LEN=16384
THINK_RATIO=0.87
POLICY="avg1"
TEMPERATURE=0.0
MAX_SAMPLES=""
LOCAL_DATA=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL_PATH="$2"; shift 2;;
        --gpu_ids) GPU_IDS="$2"; shift 2;;
        --threshold) THRESHOLD="$2"; shift 2;;
        --max_len) MAX_LEN="$2"; shift 2;;
        --think_ratio) THINK_RATIO="$2"; shift 2;;
        --policy) POLICY="$2"; shift 2;;
        --temperature) TEMPERATURE="$2"; shift 2;;
        --max_samples) MAX_SAMPLES="$2"; shift 2;;
        --local_data) LOCAL_DATA="$2"; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

if [ -z "${MODEL_PATH}" ]; then
    echo "请指定模型路径: --model /path/to/model"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ "${MODEL_PATH}" != /* ]]; then
    MODEL_PATH="$(cd "$(dirname "${MODEL_PATH}")" && pwd)/$(basename "${MODEL_PATH}")"
fi

DEER_ARGS=(
    --model_name_or_path "${MODEL_PATH}"
    --threshold "${THRESHOLD}"
    --max_len "${MAX_LEN}"
    --think_ratio "${THINK_RATIO}"
    --policy "${POLICY}"
    --temperature "${TEMPERATURE}"
    --gpu_ids "${GPU_IDS}"
)

if [ -n "${LOCAL_DATA}" ]; then
    DEER_ARGS+=(--dataset_path "${LOCAL_DATA}")
elif [ -d "${PROJECT_ROOT}/datasets/livecodebench/code_generation_lite" ]; then
    DEER_ARGS+=(--dataset_path "${PROJECT_ROOT}/datasets/livecodebench/code_generation_lite")
fi

if [ -n "${MAX_SAMPLES}" ]; then
    DEER_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${SCRIPT_DIR}/LiveCodeBench:${PYTHONPATH}"
uv run python "${SCRIPT_DIR}/deer_lcb.py" "${DEER_ARGS[@]}"
DEERSHEOF
chmod +x "${BENCHMARK_DIR}/run_deer_lcb.sh"

cat > "${BENCHMARK_DIR}/download_data.sh" << 'DLEOF'
#!/bin/bash
# 下载 LiveCodeBench 数据集
set -e
eval "$(conda shell.bash hook)"
conda activate othink-r1

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/datasets/livecodebench/code_generation_lite"

export HF_ENDPOINT=https://hf-mirror.com

if [ -d "${DATA_DIR}" ] && [ "$(ls -A ${DATA_DIR} 2>/dev/null)" ]; then
    echo "  数据集已存在: ${DATA_DIR}"
else
    echo "  下载 LiveCodeBench 数据集..."
    mkdir -p "${DATA_DIR}"
    uv run huggingface-cli download \
        --repo-type dataset \
        livecodebench/code_generation_lite \
        --local-dir "${DATA_DIR}" \
        --local-dir-use-symlinks False \
        --resume-download
    echo "  下载完成: ${DATA_DIR}"
fi
DLEOF
chmod +x "${BENCHMARK_DIR}/download_data.sh"

cat > "${BENCHMARK_DIR}/README.md" << 'READMEEOF'
# LiveCodeBench Benchmark for OThink-R1

支持标准评测和 DEER early-exit 适配。

## 快速开始

1. 下载数据集: bash benchmark/livecodebench/download_data.sh
2. 标准评测: bash benchmark/livecodebench/run_lcb.sh --model NAME --model_path PATH
3. DEER 评测: bash benchmark/livecodebench/run_deer_lcb.sh --model PATH --threshold 0.95
READMEEOF

echo "  ✅ 运行脚本和 README 创建完成"

echo ""
echo "=========================================="
echo "  ✅ 整合完成!"
echo "=========================================="
echo ""
echo "  后续步骤:"
echo "    1. bash benchmark/livecodebench/download_data.sh"
echo "    2. bash benchmark/livecodebench/run_lcb.sh --model <model_name>"
echo "    3. bash benchmark/livecodebench/run_deer_lcb.sh --model <model_path>"
echo ""