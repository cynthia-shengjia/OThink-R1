#!/usr/bin/env python3
"""
LiveCodeBench DEER Early-Exit 评测 (deer_lcb.py)
"""

import os
import sys
import json
import time
import argparse
import math
import numpy as np

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


def compute_confidence(logprobs_list):
    if not logprobs_list:
        return 0.0
    probs = [math.exp(lp) for lp in logprobs_list if lp is not None]
    if not probs:
        return 0.0
    return float(np.mean(probs))


def has_complete_code(text):
    think_close = "<" + "/" + "think" + ">"
    if think_close in text:
        after_think = text.split(think_close, 1)[1]
    else:
        after_think = text
    lines = after_think.split("\n")
    fence_indices = [i for i, line in enumerate(lines) if "```" in line]
    return len(fence_indices) >= 2


def deer_generate_single(llm, tokenizer, prompt_text, sampling_params,
                         threshold=0.95, max_rounds=5, extra_tokens=512):
    from vllm import SamplingParams

    current_text = prompt_text
    accumulated_output = ""
    confidence_history = []

    for round_idx in range(max_rounds):
        outputs = llm.generate([current_text], sampling_params)
        vllm_out = outputs[0].outputs[0]
        new_text = vllm_out.text
        accumulated_output += new_text

        logprobs = []
        if vllm_out.logprobs:
            for lp_dict in vllm_out.logprobs:
                if lp_dict:
                    top_token = max(lp_dict.values(), key=lambda x: x.logprob)
                    logprobs.append(top_token.logprob)

        confidence = compute_confidence(logprobs)
        confidence_history.append(confidence)

        print(f"    Round {round_idx+1}: conf={confidence:.4f}, "
              f"tokens={len(vllm_out.token_ids)}, "
              f"has_code={has_complete_code(accumulated_output)}")

        if confidence >= threshold and has_complete_code(accumulated_output):
            print(f"    -> Early exit (conf >= {threshold})")
            break
        if has_complete_code(accumulated_output) and round_idx >= 1:
            print(f"    -> Exit (complete code found)")
            break
        if vllm_out.finish_reason == "stop":
            print(f"    -> Natural stop")
            break

        current_text = prompt_text + accumulated_output
        sampling_params = SamplingParams(
            n=1, max_tokens=extra_tokens,
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p, logprobs=1,
        )

    return accumulated_output, round_idx + 1, confidence_history


def run_deer_inference(problems, model_path, threshold=0.95,
                       max_rounds=5, max_model_len=4096, max_tokens=4096):
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
        n=1, max_tokens=max_tokens, temperature=0.0, top_p=1.0, logprobs=1,
    )

    print("[INFO] 构建 prompts...")
    prompts = []
    for prob in problems:
        messages = build_prompt(prob)
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompts.append(prompt_text)

    print(f"\n[INFO] DEER 推理 {len(prompts)} 个问题...")
    t0 = time.time()
    all_results = []
    total_rounds = 0

    for i, (prob, prompt_text) in enumerate(zip(problems, prompts)):
        print(f"\n[{i+1}/{len(problems)}] {prob.question_title}")
        output, rounds, conf_hist = deer_generate_single(
            llm, tokenizer, prompt_text, sampling_params,
            threshold=threshold, max_rounds=max_rounds,
        )
        code = extract_code(output)
        total_rounds += rounds
        all_results.append({
            "question_id": prob.question_id,
            "question_title": prob.question_title,
            "output_list": [output],
            "code_list": [code],
            "deer_rounds": rounds,
            "deer_confidence_history": conf_hist,
        })

    elapsed = time.time() - t0
    avg_rounds = total_rounds / len(problems) if problems else 0
    print(f"\n[INFO] DEER 完成: {elapsed:.1f}s, avg_rounds={avg_rounds:.2f}")
    return all_results


def run_evaluation(problems, all_results, num_process=12, timeout=6):
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

    print("\n" + "=" * 60)
    print("  LiveCodeBench DEER 评测")
    print("=" * 60)

    eval_samples = [prob.get_evaluation_sample() for prob in problems]
    generations = [r["code_list"] for r in all_results]

    t0 = time.time()
    metrics_result = codegen_metrics(
        eval_samples, generations,
        num_process_evaluate=num_process, timeout=timeout,
    )
    elapsed = time.time() - t0
    metrics = metrics_result[0]

    print(f"\n[INFO] 评测完成: {elapsed:.1f}s")
    for key, value in sorted(metrics.items()):
        if key == "detail":
            continue
        print(f"  {key}: {value:.4f}")

    rounds_list = [r["deer_rounds"] for r in all_results]
    print(f"\n  DEER 统计: avg_rounds={np.mean(rounds_list):.2f}, "
          f"1轮退出={rounds_list.count(1)}/{len(rounds_list)}")
    return metrics_result


def save_results(problems, all_results, metrics_result, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "deer_results.json")
    save_data = []
    for prob, res in zip(problems, all_results):
        save_data.append({
            "question_id": prob.question_id,
            "question_title": prob.question_title,
            "platform": prob.platform.value,
            "difficulty": prob.difficulty.value,
            "output_list": res["output_list"],
            "code_list": res["code_list"],
            "deer_rounds": res["deer_rounds"],
            "deer_confidence_history": res["deer_confidence_history"],
        })
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] 结果已保存: {output_path}")
    if metrics_result is not None:
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics_result[0], f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="LiveCodeBench DEER 评测")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str,
                        default="datasets/livecodebench/code_generation_lite")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_problems", type=int, default=0)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--max_rounds", type=int, default=5)
    parser.add_argument("--num_process_evaluate", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=6)
    parser.add_argument("--no_eval", action="store_true")
    args = parser.parse_args()

    if args.output_dir is None:
        model_name = os.path.basename(args.model_path.rstrip("/"))
        args.output_dir = os.path.join("results", "lcb", model_name,
                                       f"deer_t{args.threshold}")

    print("=" * 60)
    print("  LiveCodeBench DEER 评测")
    print("=" * 60)
    print(f"  模型: {args.model_path}")
    print(f"  DEER: threshold={args.threshold}, max_rounds={args.max_rounds}")
    print("=" * 60)

    problems = load_problems(args.dataset_path, args.max_problems)
    all_results = run_deer_inference(
        problems, args.model_path,
        threshold=args.threshold, max_rounds=args.max_rounds,
        max_model_len=args.max_model_len, max_tokens=args.max_tokens,
    )

    metrics_result = None
    if not args.no_eval:
        metrics_result = run_evaluation(
            problems, all_results,
            num_process=args.num_process_evaluate, timeout=args.timeout,
        )

    save_results(problems, all_results, metrics_result, args.output_dir)
    print("\n[DONE] DEER 评测完成!")


if __name__ == "__main__":
    main()
