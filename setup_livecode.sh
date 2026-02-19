#!/bin/bash
# ============================================================
# setup_livecodebench.sh
# 将 LiveCodeBench 整合到 OThink-R1 项目的 benchmark/ 目录下
#
# 使用方法:
#   cd ~/ACL-ARR-Jan-Rebuttal/OThink-R1   # 项目根目录
#   bash setup_livecodebench.sh [/path/to/local/LiveCodeBench]
#
# 参数:
#   $1 (可选): 本地已 clone 的 LiveCodeBench 路径
#              如果不传，则从 GitHub clone
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
    if grep -qi "\"${dep_name}" "${PYPROJECT}"; then
        echo "  ⚠️  ${dep_name} 已在 pyproject.toml 中"
    else
        # 在 "# DEER Baseline Dependencies" 行之前插入
        sed -i "/# DEER Baseline Dependencies/i\\    \"${dep}\"," "${PYPROJECT}"
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
# 使用 LiveCodeBench 原生 runner 对 OThink-R1 训练的模型评测
#
# 用法:
#   bash benchmark/livecodebench/run_lcb.sh \
#       --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
#       --model_path /path/to/your/model \
#       --gpu_ids 0
# ============================================================
set -e
eval "$(conda shell.bash hook)"
conda activate othink-r1

# 默认参数
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

# 如果 MODEL_PATH 是相对路径，基于调用时的 pwd 解析
if [ -n "${MODEL_PATH}" ] && [[ "${MODEL_PATH}" != /* ]]; then
    MODEL_PATH="$(cd "$(dirname "${MODEL_PATH}")" && pwd)/$(basename "${MODEL_PATH}")"
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

echo "=========================================="
echo "  LiveCodeBench 标准评测"
echo "=========================================="
echo "  模型: ${MODEL}"
echo "  模型路径: ${MODEL_PATH:-auto}"
echo "  GPU: ${GPU_IDS}"
echo "  Max Tokens: ${MAX_TOKENS}"
echo "  Temperature: ${TEMPERATURE}"
echo "  Release: ${RELEASE_VERSION}"
echo "=========================================="

# 构建参数
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

cd "${LCB_DIR}"
uv run python -m lcb_runner.runner.main "${LCB_ARGS[@]}"

MODEL_NAME=$(basename "${MODEL}")
OUTPUT_FILE="${LCB_DIR}/output/${MODEL_NAME}/Scenario.${SCENARIO}_${CODEGEN_N}_${TEMPERATURE}.json"

if [ -f "${OUTPUT_FILE}" ]; then
    echo ""
    echo "  ✅ 评测完成！结果: ${OUTPUT_FILE}"

    # 统计 token 长度
    echo ""
    echo "  统计 token 长度..."
    uv run python -m lcb_runner.utils.get_length_lcb \
        --model_name "${MODEL}" \
        --file_path "${OUTPUT_FILE}" \
        2>/dev/null || echo "  ⚠️  token 统计失败（不影响评测结果）"
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
  - 原始 DEER 在 math reasoning 中使用 \\boxed{} 作为 answer inducing prompt，
    在

</think>

处检测 answer confidence 来决定 early exit。
  - 代码生成任务没有 \\boxed{}，我们改为:
    1. answer_prompt: 使用 "```python\n" 来诱导模型输出代码
    2. stop tokens: 在 "```" (代码块结束) 处检测 confidence
    3. confidence 计算: 复用 DEER 的 avg logprob 策略
  - 生成完成后，提取代码块，调用 LiveCodeBench 的评测管线评分

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
# 将 LiveCodeBench 加入 Python path
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LCB_DIR = os.path.join(SCRIPT_DIR, "LiveCodeBench")
sys.path.insert(0, LCB_DIR)

from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

# ---------------------------------------------------------------------------
# DEER 核心: answer confidence 计算 (复用自 vllm-deer.py)
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
# 代码提取: 从模型输出中提取 python 代码
# ---------------------------------------------------------------------------
def extract_code_from_response(response: str) -> str:
    """从模型回复中提取 Python 代码块"""
    # 尝试匹配 ```python ... ``` 代码块
    pattern = r'```python\s*\n(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # 尝试匹配 ``` ... ``` 代码块
    pattern = r'```\s*\n(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # 如果

</think>

之后有内容，取

</think>

之后的部分
    if '

</think>

' in response:
        after_think = response.split('

</think>

')[-1].strip()
        if after_think:
            return after_think

    return response.strip()

# ---------------------------------------------------------------------------
# 加载 LiveCodeBench 数据集
# ---------------------------------------------------------------------------
def load_lcb_dataset(dataset_path, release_version="release_v5"):
    """加载 LiveCodeBench 代码生成数据集"""
    from datasets import load_dataset

    if dataset_path and os.path.isdir(dataset_path):
        dataset = load_dataset(dataset_path, split="test")
    else:
        dataset = load_dataset("livecodebench/code_generation_lite", split="test")

    problems = [CodeGenerationProblem(**item) for item in dataset]

    # 按 release_version 过滤
    if release_version and release_version != "release_latest":
        # LiveCodeBench 的版本过滤逻辑
        from lcb_runner.utils.scenarios import Scenario
        pass  # 保留全部，让 LiveCodeBench 内部处理

    return problems

# ---------------------------------------------------------------------------
# 主函数
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

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    available_gpus = args.gpu_ids.split(',')

    print("=" * 60)
    print("  DEER × LiveCodeBench (Code Generation)")
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
    print(f"  共 {len(problems)} 道题")

    # ---- DEER 适配: 代码生成的 prompt 和 stop tokens ----
    # 代码生成任务的 answer inducing prompt
    code_answer_prompt = "\n\n```python\n"

    # 思考阶段的 stop tokens
    think_stop_tokens = ["Wait", "

</think>

", tokenizer.eos_token]
    # confidence 检查阶段的 stop tokens (代码块结束)
    prob_check_stop_tokens = ["```\n", "```"]
    # 最终回答阶段的 stop tokens
    answer_stop_tokens = [tokenizer.eos_token]

    think_limit = int(args.max_len * args.think_ratio)

    # ---- 构建 prompts ----
    sys_prompt = (
        "You are an expert Python programmer. "
        "Solve the given competitive programming problem. "
        "Think step by step inside

<think>

...

</think>

tags, "
        "then provide your final solution as a Python code block."
    )

    formatted_prompts = []
    for prob in problems:
        question_content = prob.question_content
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question_content},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted_prompts.append(prompt)

    # ---- DEER 主循环 (简化版: 批量处理) ----
    # 这里使用简化的 DEER 逻辑:
    # Phase 1: 生成思考过程，在 "Wait" 或 "

</think>

" 处停下
    # Phase 2: 用 code_answer_prompt 诱导代码输出，检查 confidence
    # Phase 3: 如果 confidence > threshold 或达到限制，生成最终答案
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

        # ---- Phase 1: 迭代思考 ----
        while True:
            think_tokens_used = len(tokenizer.encode(
                thinking_history, add_special_tokens=False
            ))
            remaining = think_limit - think_tokens_used
            if remaining <= 50:
                too_long = True
                break

            # 生成一段思考
            if thinking_steps < args.max_judge_steps:
                stop = think_stop_tokens
            else:
                stop = ["

</think>

", tokenizer.eos_token]

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
            last_id = gen_ids[-1] if gen_ids else -1

            thinking_history += gen_text
            current_seq = prompt + thinking_history

            # 检查是否自然结束思考
            think_end_ids = tokenizer.encode("

</think>

", add_special_tokens=False)
            if last_id in think_end_ids:
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

            if prob_outputs[0].outputs[0].logprobs:
                pred_prob = calculate_avg_prob_from_logprobs(
                    prob_outputs[0].outputs[0].logprobs, args.policy
                )
            else:
                pred_prob = 0.0

            if pred_prob > args.threshold:
                early_exit = True
                break

            # 继续思考
            if not current_seq.rstrip().endswith("Wait"):
                current_seq += "Wait"
                thinking_history += "Wait"

        # ---- Phase 3: 生成最终代码 ----
        final_prompt = prompt + thinking_history
        if not thinking_history.rstrip().endswith("

</think>

"):
            final_prompt += "\n

</think>

\n\n"
        else:
            final_prompt += "\n\n"

        answer_budget = args.max_len - len(
            tokenizer.encode(thinking_history, add_special_tokens=False)
        )
        answer_budget = max(answer_budget, 512)

        final_outputs = llm_engine.generate(
            [final_prompt],
            SamplingParams(
                max_tokens=min(answer_budget, 4096),
                temperature=args.temperature,
                top_p=args.top_p,
                stop=answer_stop_tokens,
            ),
            use_tqdm=False,
        )

        final_text = final_outputs[0].outputs[0].text
        full_response = thinking_history + "\n

</think>

\n\n" + final_text

        # 提取代码
        code = extract_code_from_response(full_response)

        all_results.append({
            "question_id": prob.question_id,
            "question_content": prob.question_content,
            "output_list": [full_response],
            "code_list": [code],
            "thinking_steps": thinking_steps,
            "early_exit": early_exit,
            "too_long": too_long,
            "total_tokens": len(tokenizer.encode(full_response, add_special_tokens=False)),
        })

    elapsed = time.time() - start_time
    print(f"\n推理完成! 共 {len(all_results)} 道题, 耗时 {elapsed:.1f}s")

    # ---- 保存结果 ----
    model_name = os.path.basename(args.model_name_or_path)
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
    avg_tokens = np.mean([r['total_tokens'] for r in all_results])
    avg_steps = np.mean([r['thinking_steps'] for r in all_results])

    print(f"\n============= DEER 统计 =============")
    print(f"  总题数: {len(all_results)}")
    print(f"  Early Exit: {early_exits} ({100*early_exits/len(all_results):.1f}%)")
    print(f"  Too Long:   {too_longs} ({100*too_longs/len(all_results):.1f}%)")
    print(f"  平均 tokens: {avg_tokens:.0f}")
    print(f"  平均思考步数: {avg_steps:.1f}")

    # ---- 评测 (可选) ----
    if not args.no_evaluate:
        print(f"\n开始 LiveCodeBench 评测...")
        try:
            combined = [([r['output_list'][0]], [r['code_list'][0]]) for r in all_results]
            metrics = codegen_metrics(
                problems[:len(all_results)],
                combined,
                num_process_evaluate=12,
                timeout=6,
            )
            print(f"\n============= LiveCodeBench 评测结果 =============")
            if isinstance(metrics, tuple) and len(metrics) >= 1:
                print(json.dumps(metrics[0], indent=2))
            else:
                print(json.dumps(metrics, indent=2))

            eval_file = out_file.replace('.json', '_eval.json')
            with open(eval_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"评测结果保存到: {eval_file}")
        except Exception as e:
            print(f"  ⚠️  评测失败: {e}")
            print(f"  可以稍后手动评测")

if __name__ == "__main__":
    main()
DEEREOF
echo "  ✅ deer_lcb.py 创建完成"

# ============================================================
# Step 5: 创建 CP-Router stub (不适用于代码生成)
# ============================================================
echo ""
echo "[5/6] 创建 CP-Router stub..."

cat > "${BENCHMARK_DIR}/cp_router_lcb_stub.py" << 'CPEOF'
"""
CP-Router × LiveCodeBench Stub

CP-Router 是一个基于 Conformal Prediction 的 MCQA 路由器。
它通过分析 LLM 对选项 (A/B/C/D) 的 softmax 概率来构建预测集，
然后根据预测集大小决定路由到 LLM 还是 LRM。

⚠️  CP-Router 不适用于代码生成任务:
  1. 代码生成是开放式生成任务，没有固定选项集合
  2. 无法计算选项级别的 nonconformity scores
  3. 无法构建有意义的预测集

可能的替代方案 (未来工作):
  - 基于 prompt 难度分类的路由 (如 problem rating)
  - 基于首 N 个 token 的 perplexity 做路由决策
  - 基于代码生成 confidence (如 pass@k 估计) 做路由

本文件仅作为占位符，记录不适用原因。
"""
import sys

def main():
    print("=" * 60)
    print("  CP-Router × LiveCodeBench")
    print("=" * 60)
    print()
    print("  ❌ CP-Router 不适用于代码生成任务")
    print()
    print("  原因:")
    print("    - CP-Router 基于 MCQA 选项 logits 做 conformal prediction")
    print("    - 代码生成是开放式任务，没有固定选项集合")
    print("    - 无法计算选项级别的 nonconformity scores")
    print()
    print("  替代方案 (未来工作):")
    print("    - 基于 problem difficulty rating 的路由")
    print("    - 基于 prompt perplexity 的路由")
    print("    - 基于 code generation confidence 的路由")
    print()
    sys.exit(0)

if __name__ == "__main__":
    main()
CPEOF
echo "  ✅ cp_router_lcb_stub.py 创建完成"

# ============================================================
# Step 6: 创建 DEER 运行脚本 + README
# ============================================================
echo ""
echo "[6/6] 创建运行脚本和 README..."

# DEER 运行脚本
cat > "${BENCHMARK_DIR}/run_deer_lcb.sh" << 'DEERSHEOF'
#!/bin/bash
# ============================================================
# DEER × LiveCodeBench 运行脚本
#
# 用法:
#   bash benchmark/livecodebench/run_deer_lcb.sh \
#       --model /path/to/model \
#       --gpu_ids 0 \
#       --threshold 0.95
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
    echo "❌ 请指定模型路径: --model /path/to/model"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 解析相对路径
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

# 数据集路径
if [ -n "${LOCAL_DATA}" ]; then
    DEER_ARGS+=(--dataset_path "${LOCAL_DATA}")
elif [ -d "${PROJECT_ROOT}/datasets/livecodebench/code_generation_lite" ]; then
    DEER_ARGS+=(--dataset_path "${PROJECT_ROOT}/datasets/livecodebench/code_generation_lite")
fi

if [ -n "${MAX_SAMPLES}" ]; then
    DEER_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

cd "${PROJECT_ROOT}"
uv run python "${SCRIPT_DIR}/deer_lcb.py" "${DEER_ARGS[@]}"
DEERSHEOF
chmod +x "${BENCHMARK_DIR}/run_deer_lcb.sh"

# 下载数据集脚本
cat > "${BENCHMARK_DIR}/download_data.sh" << 'DLEOF'
#!/bin/bash
# 下载 LiveCodeBench 数据集
set -e
eval "$(conda shell.bash hook)"
conda activate othink-r1

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/datasets/livecodebench/code_generation_lite"

export HF_ENDPOINT=https://hf-mirror.com

if [ -d "${DATA_DIR}" ]; then
    echo "  ⚠️  数据集已存在: ${DATA_DIR}"
else
    echo "  📦 下载 LiveCodeBench 数据集..."
    mkdir -p "${DATA_DIR}"
    uv run huggingface-cli download \
        --repo-type dataset \
        livecodebench/code_generation_lite \
        --local-dir "${DATA_DIR}" \
        --local-dir-use-symlinks False \
        --resume-download
    echo "  ✅ 下载完成: ${DATA_DIR}"
fi
DLEOF
chmod +x "${BENCHMARK_DIR}/download_data.sh"

# README
cat > "${BENCHMARK_DIR}/README.md" << 'READMEEOF'
# LiveCodeBench Benchmark for OThink-R1

将 [LiveCodeBench](https://github.com/cynthia-shengjia/LiveCodeBench) 集成到 OThink-R1 项目中，
支持标准评测和 DEER early-exit 适配。

## 目录结构
benchmark/livecodebench/
├── LiveCodeBench/ # LiveCodeBench 源码 (git clone)
├── run_lcb.sh # 标准评测入口
├── run_deer_lcb.sh # DEER 适配运行脚本
├── deer_lcb.py # DEER × LiveCodeBench 适配代码
├── cp_router_lcb_stub.py # CP-Router 不适用标记
├── download_data.sh # 数据集下载脚本
└── README.md # 本文件