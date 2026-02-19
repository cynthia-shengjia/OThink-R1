#!/usr/bin/env python3
"""
LiveCodeBench 标准评测 (lcb_eval.py)
使用 vLLM 批量推理 + LCB 原生评测链路

用法:
    cd ~/ACL-ARR-Jan-Rebuttal/OThink-R1
    export PYTHONPATH="benchmark/livecodebench/LiveCodeBench:$PYTHONPATH"
    conda run -n othink-r1 --no-banner uv run python benchmark/livecodebench/lcb_eval.py \
        --model_path ./models/Qwen2.5-0.5B-Instruct \
        --max_problems 0 \
        --max_model_len 4096
"""

import os
import sys
import json
import time
import argparse
import re

# ── 安全常量 (heredoc / markdown 不会破坏) ──
BACKTICK3 = chr(96) * 3
THINK_OPEN_TAG = "<" + "think" + ">"
THINK_CLOSE_TAG = "<" + "/" + "think" + ">"

# ── 确保 LCB 源码在 path 中 ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LCB_ROOT = os.path.join(SCRIPT_DIR, "LiveCodeBench")
if LCB_ROOT not in sys.path:
    sys.path.insert(0, LCB_ROOT)


# ═══════════════════════════════════════════════════════
#  1. 数据集加载 (使用 LCB 原生 CodeGenerationProblem)
# ═══════════════════════════════════════════════════════

def load_problems(dataset_path, max_problems=0):
    """加载数据集并转为 CodeGenerationProblem 对象列表"""
    from lcb_runner.benchmarks.code_generation import (
        CodeGenerationProblem,
        load_code_generation_dataset,
    )

    print(f"[INFO] 加载数据集: {dataset_path}")
    problems = load_code_generation_dataset(local_path=dataset_path)
    problems = sorted(problems, key=lambda x: x.question_id)

    if max_problems > 0:
        problems = problems[:max_problems]
        print(f"[INFO] 限制为前 {max_problems} 题")

    print(f"[INFO] 共 {len(problems)} 题")
    return problems


# ═══════════════════════════════════════════════════════
#  2. Prompt 构建 (DeepSeek-R1 风格)
# ═══════════════════════════════════════════════════════

SYSTEM_MESSAGE = (
    "You are an expert Python programmer. "
    "You will be given a question (problem specification) and will generate "
    "a correct Python program that matches the specification and passes all tests."
)

FORMATTING_WITH_STARTER = (
    "You will use the following starter code to write the solution "
    "to the problem and enclose your code within delimiters."
)

FORMATTING_WITHOUT_STARTER = (
    "Read the inputs from stdin solve the problem and write the answer to stdout "
    "(do not directly test on the sample inputs). "
    "Enclose your code within delimiters as follows. "
    "Ensure that when the python program runs, it reads the inputs, "
    "runs the algorithm and writes output to STDOUT."
)


def build_prompt(problem):
    """构建单个问题的 prompt (chat messages 格式)"""
    user_content = f"### Question:\n{problem.question_content}\n\n"

    if problem.starter_code:
        user_content += f"### Format: {FORMATTING_WITH_STARTER}\n"
        user_content += f"{BACKTICK3}python\n{problem.starter_code}\n{BACKTICK3}\n\n"
    else:
        user_content += f"### Format: {FORMATTING_WITHOUT_STARTER}\n"
        user_content += f"{BACKTICK3}python\n# YOUR CODE HERE\n{BACKTICK3}\n\n"

    user_content += "### Answer: (use the provided format with backticks)\n\n"

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]
    return messages


# ═══════════════════════════════════════════════════════
#  3. 代码提取 (从模型输出中提取 python 代码块)
# ═══════════════════════════════════════════════════════

def extract_code(model_output):
    """
    从模型输出中提取代码。
    优先提取 </think> 之后的内容，然后找最后一对 ``` 之间的代码。
    """
    text = model_output

    # 如果有 </think> 标签，只看标签之后的内容
    think_close = "<" + "/" + "think" + ">"
    if think_close in text:
        text = text.split(think_close, 1)[1]

    lines = text.split("\n")
    # 找所有 ``` 行的索引
    fence_indices = [i for i, line in enumerate(lines) if "```" in line]

    if len(fence_indices) < 2:
        # 没有代码块，返回空字符串
        return ""

    # 取最后一对 ``` 之间的内容 (与官方 extract_code 一致)
    return "\n".join(lines[fence_indices[-2] + 1 : fence_indices[-1]])


# ═══════════════════════════════════════════════════════
#  4. vLLM 推理
# ═══════════════════════════════════════════════════════

def run_inference(problems, model_path, max_model_len=4096, n=1,
                  temperature=0.0, max_tokens=4096):
    """使用 vLLM 批量推理"""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"\n[INFO] 加载模型: {model_path}")
    print(f"[INFO] max_model_len={max_model_len}, n={n}, temperature={temperature}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        max_model_len=max_model_len,
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=0.90,
    )

    sampling_params = SamplingParams(
        n=n,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95 if temperature > 0 else 1.0,
    )

    # 构建 prompts
    print("[INFO] 构建 prompts...")
    prompts = []
    for prob in problems:
        messages = build_prompt(prob)
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)

    # 推理
    print(f"[INFO] 开始推理 {len(prompts)} 个问题...")
    t0 = time.time()
    vllm_outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0

    total_tokens = sum(
        len(out.outputs[0].token_ids) for out in vllm_outputs
    )
    print(f"[INFO] 推理完成: {elapsed:.1f}s, {total_tokens} tokens, "
          f"{total_tokens/elapsed:.1f} tok/s")

    # 整理结果
    all_results = []
    for prob, vllm_out in zip(problems, vllm_outputs):
        output_list = [o.text for o in vllm_out.outputs]
        code_list = [extract_code(o) for o in output_list]
        all_results.append({
            "question_id": prob.question_id,
            "question_title": prob.question_title,
            "output_list": output_list,
            "code_list": code_list,
        })

    return all_results


# ═══════════════════════════════════════════════════════
#  5. 评测 (使用 LCB 原生 codegen_metrics)
# ═══════════════════════════════════════════════════════

def run_evaluation(problems, all_results, num_process=12, timeout=6):
    """
    使用 LCB 原生评测:
    1. CodeGenerationProblem.get_evaluation_sample() 构建 eval_samples
    2. codegen_metrics(eval_samples, generations) 执行判分
    """
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

    print("\n" + "=" * 60)
    print("  LiveCodeBench 评测 (代码执行 + 判分)")
    print("=" * 60)

    # 构建 eval_samples (每个是 {"input_output": json_str})
    eval_samples = []
    for prob in problems:
        eval_samples.append(prob.get_evaluation_sample())

    # 构建 generations (每个是 list[str] 的代码列表)
    generations = [r["code_list"] for r in all_results]

    print(f"[INFO] 评测 {len(eval_samples)} 题, "
          f"每题 {len(generations[0]) if generations else 0} 个生成...")

    t0 = time.time()
    metrics_result = codegen_metrics(
        eval_samples,
        generations,
        num_process_evaluate=num_process,
        timeout=timeout,
    )
    elapsed = time.time() - t0

    # metrics_result = [metrics_dict, results_dict, metadata_list]
    metrics = metrics_result[0]
    results = metrics_result[1]

    print(f"\n[INFO] 评测完成: {elapsed:.1f}s")
    print("=" * 60)
    print("  评测结果")
    print("=" * 60)

    # 打印 pass@k
    for key, value in sorted(metrics.items()):
        if key == "detail":
            continue
        print(f"  {key}: {value:.4f}")

    # 统计通过/失败
    n_pass = 0
    n_total = len(results)
    for idx in results:
        # results[idx] 是 list of list, 每个内层 list 是各 test case 的结果
        # 如果所有 test case 都是 True (即 1), 则通过
        for gen_result in results[idx]:
            if all(r == True or r == 1 for r in gen_result):
                n_pass += 1
                break  # 至少一个 generation 通过即可

    print(f"\n  通过: {n_pass}/{n_total}")

    return metrics_result


# ═══════════════════════════════════════════════════════
#  6. 保存结果
# ═══════════════════════════════════════════════════════

def save_results(problems, all_results, metrics_result, output_dir):
    """保存推理结果和评测指标"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存推理结果
    output_path = os.path.join(output_dir, "generation_results.json")
    save_data = []
    for prob, res in zip(problems, all_results):
        save_data.append({
            "question_id": prob.question_id,
            "question_title": prob.question_title,
            "platform": prob.platform.value,
            "difficulty": prob.difficulty.value,
            "output_list": res["output_list"],
            "code_list": res["code_list"],
        })
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] 推理结果已保存: {output_path}")

    # 保存评测指标
    if metrics_result is not None:
        metrics_path = os.path.join(output_dir, "metrics.json")
        # metrics_result[0] 是 metrics dict, metrics_result[1] 是 results
        with open(metrics_path, "w") as f:
            json.dump(metrics_result[0], f, indent=2)
        print(f"[INFO] 评测指标已保存: {metrics_path}")


# ═══════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="LiveCodeBench 标准评测")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径")
    parser.add_argument("--dataset_path", type=str,
                        default="datasets/livecodebench/code_generation_lite",
                        help="数据集路径")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录 (默认: results/lcb/<model_name>)")
    parser.add_argument("--max_problems", type=int, default=0,
                        help="最大题数 (0=全部)")
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="vLLM max_model_len")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="最大生成 token 数")
    parser.add_argument("--n", type=int, default=1,
                        help="每题生成数")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="采样温度")
    parser.add_argument("--num_process_evaluate", type=int, default=12,
                        help="评测并行进程数")
    parser.add_argument("--timeout", type=int, default=6,
                        help="单个测试用例超时(秒)")
    parser.add_argument("--no_eval", action="store_true",
                        help="跳过评测，只做推理")
    parser.add_argument("--gpu_ids", type=str, default="0",
                        help="GPU ID")
    args = parser.parse_args()


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    available_gpus = args.gpu_ids.split(",")
    print(f"  GPU: {args.gpu_ids}")

    # 输出目录
    if args.output_dir is None:
        model_name = os.path.basename(args.model_path.rstrip("/"))
        args.output_dir = os.path.join("results", "lcb", model_name, "standard")

    print("=" * 60)
    print("  LiveCodeBench 标准评测")
    print("=" * 60)
    print(f"  模型: {args.model_path}")
    print(f"  数据集: {args.dataset_path}")
    print(f"  输出: {args.output_dir}")
    print(f"  max_model_len: {args.max_model_len}")
    print(f"  max_tokens: {args.max_tokens}")
    print(f"  n: {args.n}, temperature: {args.temperature}")
    print("=" * 60)

    # Step 1: 加载数据集
    problems = load_problems(args.dataset_path, args.max_problems)

    # Step 2: 推理
    all_results = run_inference(
        problems,
        args.model_path,
        max_model_len=args.max_model_len,
        n=args.n,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Step 3: 评测
    metrics_result = None
    if not args.no_eval:
        metrics_result = run_evaluation(
            problems,
            all_results,
            num_process=args.num_process_evaluate,
            timeout=args.timeout,
        )

    # Step 4: 保存
    save_results(problems, all_results, metrics_result, args.output_dir)

    print("\n[DONE] 评测完成!")


if __name__ == "__main__":
    main()
