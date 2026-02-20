#!/usr/bin/env python3
"""
CP-Router x LiveCodeBench — Perplexity-Based Routing for Code Generation

原版 CP-Router 基于 MCQA 选项 logits，不适用于代码生成。
本版本改用 prompt perplexity 作为不确定性度量:
  - 计算 LLM 对每个 problem prompt 的 perplexity
  - 低 perplexity (简单题) → LLM 直接生成 (短 budget)
  - 高 perplexity (难题)  → LRM 深度推理 (长 budget)

校准阶段用 calibration set 确定 perplexity 阈值。
"""

import os
import sys
import json
import time
import math
import argparse
import numpy as np
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LCB_ROOT = os.path.join(SCRIPT_DIR, "LiveCodeBench")
if LCB_ROOT not in sys.path:
    sys.path.insert(0, LCB_ROOT)

BACKTICK3 = chr(96) * 3

SYSTEM_MESSAGE = (
    "You are an expert Python programmer. "
    "You will be given a question (problem specification) and will generate "
    "a correct Python program that matches the specification and passes all tests."
)

FORMATTING_WITHOUT_STARTER = (
    "Read the inputs from stdin solve the problem and write the answer to stdout "
    "(do not directly test on the sample inputs). "
    "Enclose your code within delimiters as follows. "
    "Ensure that when the python program runs, it reads the inputs, "
    "runs the algorithm and writes output to STDOUT."
)

FORMATTING_WITH_STARTER = (
    "You will use the following starter code to write the solution "
    "to the problem and enclose your code within delimiters."
)


def load_problems(dataset_path, max_problems=0):
    from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
    problems = load_code_generation_dataset(local_path=dataset_path)
    problems = sorted(problems, key=lambda x: x.question_id)
    if max_problems > 0:
        problems = problems[:max_problems]
    print(f"[INFO] 共 {len(problems)} 题")
    return problems


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


def compute_prompt_perplexity(llm, tokenizer, prompts, batch_size=8):
    """
    计算每个 prompt 的 perplexity (用 LLM 的 prompt_logprobs)。
    perplexity = exp(-mean(log_probs))
    """
    from vllm import SamplingParams

    # 只生成 1 个 token，但开启 prompt_logprobs 来获取 prompt 的 log-likelihood
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)

    perplexities = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        outputs = llm.generate(batch, sampling_params)

        for out in outputs:
            if out.prompt_logprobs is not None:
                log_probs = []
                for lp_dict in out.prompt_logprobs:
                    if lp_dict is not None:
                        # 每个位置取 top token 的 logprob
                        top_lp = max(lp_dict.values(), key=lambda x: x.logprob)
                        log_probs.append(top_lp.logprob)
                if log_probs:
                    avg_neg_logprob = -np.mean(log_probs)
                    ppl = math.exp(avg_neg_logprob)
                else:
                    ppl = float('inf')
            else:
                ppl = float('inf')
            perplexities.append(ppl)

    return np.array(perplexities)


def calibrate_threshold(perplexities, cal_ratio=0.3, quantile=0.7, seed=42):
    """
    用 calibration set 确定 perplexity 路由阈值。
    低于阈值 → LLM (简单), 高于阈值 → LRM (难)
    """
    np.random.seed(seed)
    n = len(perplexities)
    n_cal = int(n * cal_ratio)
    indices = np.random.permutation(n)
    cal_indices = indices[:n_cal]
    test_indices = indices[n_cal:]

    cal_ppls = perplexities[cal_indices]
    threshold = np.quantile(cal_ppls, quantile)

    return threshold, cal_indices, test_indices


def main():
    parser = argparse.ArgumentParser(description="CP-Router LiveCodeBench (Perplexity-Based)")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径 (同时作 LLM 和 LRM)")
    parser.add_argument("--dataset_path", type=str,
                        default="datasets/livecodebench/code_generation_lite")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_problems", type=int, default=0)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--llm_max_tokens", type=int, default=4096,
                        help="LLM 路由: 短 budget")
    parser.add_argument("--lrm_max_tokens", type=int, default=16384,
                        help="LRM 路由: 长 budget (深度推理)")
    parser.add_argument("--cal_ratio", type=float, default=0.3)
    parser.add_argument("--ppl_quantile", type=float, default=0.7,
                        help="perplexity 分位数阈值 (越高越多题路由到 LRM)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_process_evaluate", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=6)
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output_dir is None:
        model_name = os.path.basename(args.model_path.rstrip("/"))
        args.output_dir = os.path.join("results", "lcb", model_name, "cp_router_ppl")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print("=" * 60)
    print("  CP-Router x LiveCodeBench (Perplexity-Based Routing)")
    print("=" * 60)
    print(f"  模型: {args.model_path}")
    print(f"  LLM budget: {args.llm_max_tokens} tokens")
    print(f"  LRM budget: {args.lrm_max_tokens} tokens")
    print(f"  PPL quantile: {args.ppl_quantile}")
    print("=" * 60)

    # Step 1: 加载
    problems = load_problems(args.dataset_path, args.max_problems)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path, tokenizer=args.model_path,
        max_model_len=args.max_model_len, trust_remote_code=True,
        enforce_eager=True, gpu_memory_utilization=0.90,
    )

    # Step 2: 构建 prompts
    prompts = []
    for prob in problems:
        messages = build_prompt(prob)
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompts.append(prompt_text)

    # Step 3: 计算 prompt perplexity
    print("\n[Step 1] 计算 prompt perplexity...")
    perplexities = compute_prompt_perplexity(llm, tokenizer, prompts, args.batch_size)
    print(f"  PPL 统计: min={perplexities.min():.2f}, max={perplexities.max():.2f}, "
          f"mean={perplexities.mean():.2f}, median={np.median(perplexities):.2f}")

    # Step 4: 校准阈值
    print("\n[Step 2] 校准路由阈值...")
    threshold, cal_idx, test_idx = calibrate_threshold(
        perplexities, args.cal_ratio, args.ppl_quantile, args.seed
    )
    print(f"  阈值 = {threshold:.4f}")

    # Step 5: 路由决策 (对所有题目)
    llm_indices = [i for i in range(len(problems)) if perplexities[i] <= threshold]
    lrm_indices = [i for i in range(len(problems)) if perplexities[i] > threshold]
    print(f"\n[Step 3] 路由决策:")
    print(f"  → LLM (简单, 短budget): {len(llm_indices)} 题 ({100*len(llm_indices)/len(problems):.1f}%)")
    print(f"  → LRM (难, 长budget):   {len(lrm_indices)} 题 ({100*len(lrm_indices)/len(problems):.1f}%)")

    # Step 6: 生成
    print(f"\n[Step 4] LLM 生成 ({len(llm_indices)} 题, max_tokens={args.llm_max_tokens})...")
    all_results = [None] * len(problems)

    if llm_indices:
        llm_prompts = [prompts[i] for i in llm_indices]
        llm_params = SamplingParams(n=1, max_tokens=args.llm_max_tokens, temperature=0.0, top_p=1.0)
        llm_outputs = llm.generate(llm_prompts, llm_params)
        for idx, out in zip(llm_indices, llm_outputs):
            text = out.outputs[0].text
            code = extract_code(text)
            all_results[idx] = {
                "question_id": problems[idx].question_id,
                "question_title": problems[idx].question_title,
                "output_list": [text],
                "code_list": [code],
                "route": "LLM",
                "perplexity": float(perplexities[idx]),
                "tokens": len(out.outputs[0].token_ids),
            }

    print(f"\n[Step 5] LRM 生成 ({len(lrm_indices)} 题, max_tokens={args.lrm_max_tokens})...")
    if lrm_indices:
        lrm_prompts = [prompts[i] for i in lrm_indices]
        lrm_params = SamplingParams(n=1, max_tokens=args.lrm_max_tokens, temperature=0.0, top_p=1.0)
        lrm_outputs = llm.generate(lrm_prompts, lrm_params)
        for idx, out in zip(lrm_indices, lrm_outputs):
            text = out.outputs[0].text
            code = extract_code(text)
            all_results[idx] = {
                "question_id": problems[idx].question_id,
                "question_title": problems[idx].question_title,
                "output_list": [text],
                "code_list": [code],
                "route": "LRM",
                "perplexity": float(perplexities[idx]),
                "tokens": len(out.outputs[0].token_ids),
            }

    # Step 7: 评测
    metrics_result = None
    if not args.no_eval:
        from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
        print(f"\n[Step 6] 评测...")
        eval_samples = [prob.get_evaluation_sample() for prob in problems]
        generations = [r["code_list"] for r in all_results]
        metrics_result = codegen_metrics(
            eval_samples, generations,
            num_process_evaluate=args.num_process_evaluate, timeout=args.timeout,
        )
        metrics = metrics_result[0]
        results_detail = metrics_result[1]

        print(f"\n{'='*60}")
        print(f"  CP-Router (PPL) LiveCodeBench Results")
        print(f"{'='*60}")
        for k, v in sorted(metrics.items()):
            if k == "detail": continue
            print(f"  {k}: {v:.4f}")

        # 分路由统计
        llm_pass = lrm_pass = 0
        for idx in results_detail:
            for gen_result in results_detail[idx]:
                if all(r == True or r == 1 for r in gen_result):
                    if all_results[idx]["route"] == "LLM":
                        llm_pass += 1
                    else:
                        lrm_pass += 1
                    break

        total_llm_tokens = sum(r["tokens"] for r in all_results if r["route"] == "LLM")
        total_lrm_tokens = sum(r["tokens"] for r in all_results if r["route"] == "LRM")
        all_lrm_tokens = sum(r["tokens"] for r in all_results)  # 假设全部用长 budget

        print(f"\n  路由统计:")
        print(f"    LLM: {len(llm_indices)} 题, pass={llm_pass}, tokens={total_llm_tokens}")
        print(f"    LRM: {len(lrm_indices)} 题, pass={lrm_pass}, tokens={total_lrm_tokens}")
        print(f"    Token 节省: {100*(1 - (total_llm_tokens+total_lrm_tokens)/max(all_lrm_tokens,1)):.1f}%")
        print(f"{'='*60}")

    # 保存
    os.makedirs(args.output_dir, exist_ok=True)
    save_data = []
    for prob, res in zip(problems, all_results):
        save_data.append({
            "question_id": prob.question_id,
            "question_title": prob.question_title,
            "platform": prob.platform.value,
            "difficulty": prob.difficulty.value,
            "output_list": res["output_list"],
            "code_list": res["code_list"],
            "route": res["route"],
            "perplexity": res["perplexity"],
            "tokens": res["tokens"],
        })
    with open(os.path.join(args.output_dir, "cp_router_results.json"), "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    summary = {
        "model": args.model_path,
        "ppl_threshold": float(threshold),
        "ppl_quantile": args.ppl_quantile,
        "llm_count": len(llm_indices),
        "lrm_count": len(lrm_indices),
        "llm_max_tokens": args.llm_max_tokens,
        "lrm_max_tokens": args.lrm_max_tokens,
        "timestamp": datetime.now().isoformat(),
    }
    if metrics_result:
        summary.update(metrics_result[0])
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[DONE] 结果: {args.output_dir}/")


if __name__ == "__main__":
    main()
