#!/usr/bin/env python3
"""
LiveCodeBench 标准评测 (lcb_eval.py)
使用 vLLM 批量推理 + LCB 原生评测链路
"""

import os
import sys
import json
import time
import argparse

BACKTICK3 = chr(96) * 3
THINK_CLOSE_TAG = "<" + "/" + "think" + ">"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LCB_ROOT = os.path.join(SCRIPT_DIR, "LiveCodeBench")
if LCB_ROOT not in sys.path:
    sys.path.insert(0, LCB_ROOT)


def load_problems(dataset_path, max_problems=0):
    from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
    print(f"[INFO] 加载数据集: {dataset_path}")
    problems = load_code_generation_dataset(local_path=dataset_path)
    problems = sorted(problems, key=lambda x: x.question_id)
    if max_problems > 0:
        problems = problems[:max_problems]
        print(f"[INFO] 限制为前 {max_problems} 题")
    print(f"[INFO] 共 {len(problems)} 题")
    return problems


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
    user_content = f"### Question:\n{problem.question_content}\n\n"
    if problem.starter_code:
        user_content += f"### Format: {FORMATTING_WITH_STARTER}\n"
        user_content += f"{BACKTICK3}python\n{problem.starter_code}\n{BACKTICK3}\n\n"
    else:
        user_content += f"### Format: {FORMATTING_WITHOUT_STARTER}\n"
        user_content += f"{BACKTICK3}python\n# YOUR CODE HERE\n{BACKTICK3}\n\n"
    user_content += "### Answer: (use the provided format with backticks)\n\n"
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]


def extract_code(model_output):
    text = model_output
    think_close = "<" + "/" + "think" + ">"
    if think_close in text:
        text = text.split(think_close, 1)[1]
    lines = text.split("\n")
    fence_indices = [i for i, line in enumerate(lines) if "```" in line]
    if len(fence_indices) < 2:
        return ""
    return "\n".join(lines[fence_indices[-2] + 1 : fence_indices[-1]])


def run_inference(problems, model_path, max_model_len=4096, n=1,
                  temperature=0.0, max_tokens=4096):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"\n[INFO] 加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path, tokenizer=model_path,
        max_model_len=max_model_len, trust_remote_code=True,
        enforce_eager=True, gpu_memory_utilization=0.90,
    )
    sampling_params = SamplingParams(
        n=n, max_tokens=max_tokens, temperature=temperature,
        top_p=0.95 if temperature > 0 else 1.0,
    )

    print("[INFO] 构建 prompts...")
    prompts = []
    for prob in problems:
        messages = build_prompt(prob)
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompts.append(prompt_text)

    print(f"[INFO] 开始推理 {len(prompts)} 个问题...")
    t0 = time.time()
    vllm_outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    total_tokens = sum(len(out.outputs[0].token_ids) for out in vllm_outputs)
    print(f"[INFO] 推理完成: {elapsed:.1f}s, {total_tokens} tokens, "
          f"{total_tokens/elapsed:.1f} tok/s")

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


def run_evaluation(problems, all_results, num_process=12, timeout=6):
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

    print("\n" + "=" * 60)
    print("  LiveCodeBench 评测 (代码执行 + 判分)")
    print("=" * 60)

    eval_samples = [prob.get_evaluation_sample() for prob in problems]
    generations = [r["code_list"] for r in all_results]

    print(f"[INFO] 评测 {len(eval_samples)} 题, "
          f"每题 {len(generations[0]) if generations else 0} 个生成...")

    t0 = time.time()
    metrics_result = codegen_metrics(
        eval_samples, generations,
        num_process_evaluate=num_process, timeout=timeout,
    )
    elapsed = time.time() - t0

    metrics = metrics_result[0]
    results = metrics_result[1]

    print(f"\n[INFO] 评测完成: {elapsed:.1f}s")
    print("=" * 60)
    for key, value in sorted(metrics.items()):
        if key == "detail":
            continue
        print(f"  {key}: {value:.4f}")

    n_pass = 0
    for idx in results:
        for gen_result in results[idx]:
            if all(r == True or r == 1 for r in gen_result):
                n_pass += 1
                break
    print(f"\n  通过: {n_pass}/{len(results)}")
    return metrics_result


def save_results(problems, all_results, metrics_result, output_dir):
    os.makedirs(output_dir, exist_ok=True)
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

    if metrics_result is not None:
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_result[0], f, indent=2)
        print(f"[INFO] 评测指标已保存: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="LiveCodeBench 标准评测")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str,
                        default="datasets/livecodebench/code_generation_lite")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_problems", type=int, default=0)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_process_evaluate", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=6)
    parser.add_argument("--no_eval", action="store_true")
    args = parser.parse_args()

    if args.output_dir is None:
        model_name = os.path.basename(args.model_path.rstrip("/"))
        args.output_dir = os.path.join("results", "lcb", model_name, "standard")

    print("=" * 60)
    print("  LiveCodeBench 标准评测")
    print("=" * 60)
    print(f"  模型: {args.model_path}")
    print(f"  数据集: {args.dataset_path}")
    print(f"  输出: {args.output_dir}")
    print("=" * 60)

    problems = load_problems(args.dataset_path, args.max_problems)
    all_results = run_inference(
        problems, args.model_path,
        max_model_len=args.max_model_len, n=args.n,
        temperature=args.temperature, max_tokens=args.max_tokens,
    )

    metrics_result = None
    if not args.no_eval:
        metrics_result = run_evaluation(
            problems, all_results,
            num_process=args.num_process_evaluate, timeout=args.timeout,
        )

    save_results(problems, all_results, metrics_result, args.output_dir)
    print("\n[DONE] 评测完成!")


if __name__ == "__main__":
    main()
