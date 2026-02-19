"""
CP-Router 适配 LiveCodeBench 代码生成任务

=== 适配思路 ===

原始 CP-Router (MCQA):
  softmax(A,B,C,D) → nonconformity score = 1 - p(correct)
  → conformal prediction 选 α → 预测集大小 ≤ τ 则用 LLM

适配 Code Generation:
  对每道题, 先让模型生成少量 token (probe phase, 如 50 tokens),
  收集这些 token 的 avg logprob 作为 "generation confidence".
  - confidence 高 → 模型对这道题有把握 → 用 LLM (短 budget)
  - confidence 低 → 模型没把握 → 路由到 LRM (长 budget)

  用 conformal prediction 在校准集上确定阈值, 保证覆盖率.

  nonconformity score = 1 - normalized_avg_logprob
  (归一化到 [0,1] 区间)

=== 用法 ===
  python benchmark/livecodebench/cp_router_lcb.py \
      --llm_path ./models/Qwen2.5-0.5B-Instruct \
      --dataset_path ./datasets/livecodebench/code_generation_lite \
      --cal_ratio 0.3 \
      --max_samples 30 \
      --gpu_ids 0
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import math
import time
import argparse
import random
import numpy as np
from datetime import datetime

import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams

# LiveCodeBench path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LCB_DIR = os.path.join(SCRIPT_DIR, "LiveCodeBench")
sys.path.insert(0, LCB_DIR)

from lcb_runner.benchmarks.code_generation import CodeGenerationProblem

# =====================================================================
# Conformal Prediction 核心 (从 cp-router/core/ 移植并适配)
# =====================================================================
def compute_nonconformity_scores(confidences: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    nonconformity score = 1 - confidence
    labels: 1 = 模型在该题上生成了正确代码, 0 = 错误
    这里我们用 confidence 本身 (不依赖 label), 因为校准阶段
    我们用 "confidence 能否预测成功" 来校准阈值
    """
    return 1.0 - confidences

def compute_quantile_threshold(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    quantile_level = math.ceil((n + 1) * (1 - alpha)) / n
    quantile_level = min(quantile_level, 1.0)
    return float(np.quantile(scores, quantile_level))

def select_alpha_by_fbe(
    cal_confidences: np.ndarray,
    test_confidences: np.ndarray,
    tau: float,
    beta: float = 3.0,
    alpha_candidates: np.ndarray = None,
) -> tuple:
    """
    FBE 自适应选择 α
    对代码生成: 预测集 = {LLM} 如果 confidence > threshold, 否则 {LLM, LRM}
    set_size = 1 (路由到 LLM) 或 2 (路由到 LRM)
    """
    if alpha_candidates is None:
        alpha_candidates = np.arange(0.01, 1.0, 0.01)

    cal_scores = 1.0 - cal_confidences
    best_alpha = 0.5
    best_fbe = -float('inf')

    for alpha in alpha_candidates:
        q_hat = compute_quantile_threshold(cal_scores, alpha)
        # 对测试集: confidence > (1 - q_hat) → LLM, 否则 LRM
        set_sizes = np.where(test_confidences >= (1.0 - q_hat), 1, 2)

        # FBE 计算
        total = len(set_sizes)
        if total == 0:
            continue

        # H_binary
        p1 = np.sum(set_sizes == 1) / total
        p2 = 1.0 - p1
        h_binary = 0.0
        if p1 > 0:
            h_binary -= p1 * np.log(p1 + 1e-10)
        if p2 > 0:
            h_binary -= p2 * np.log(p2 + 1e-10)

        # H_full (只有 size 1 和 2)
        h_full = h_binary  # 对二元情况 H_full = H_binary

        fbe = beta * h_full + h_binary
        if fbe > best_fbe:
            best_fbe = fbe
            best_alpha = alpha

    return float(best_alpha), float(best_fbe)

# =====================================================================
# 从模型生成中提取 confidence
# =====================================================================
def extract_generation_confidence(
    llm_engine, tokenizer, prompts, probe_tokens=50
) -> np.ndarray:
    """
    对每个 prompt 生成 probe_tokens 个 token, 收集 avg logprob 作为 confidence.
    返回归一化到 [0, 1] 的 confidence 数组.
    """
    outputs = llm_engine.generate(
        prompts,
        SamplingParams(
            max_tokens=probe_tokens,
            temperature=0.0,
            logprobs=1,
        ),
        use_tqdm=True,
    )

    confidences = []
    for output in outputs:
        logprobs_list = output.outputs[0].logprobs
        if not logprobs_list or len(logprobs_list) < 2:
            confidences.append(0.0)
            continue

        total_logprob = 0.0
        count = 0
        for i in range(1, len(logprobs_list)):
            if logprobs_list[i]:
                try:
                    lp = list(logprobs_list[i].values())[0].logprob
                    total_logprob += lp
                    count += 1
                except (IndexError, KeyError, AttributeError):
                    pass

        if count > 0:
            avg_logprob = total_logprob / count
            # 将 avg_logprob (负值, 通常在 [-5, 0] 范围) 映射到 [0, 1]
            # 用 sigmoid-like 变换: confidence = exp(avg_logprob)
            confidence = math.exp(avg_logprob)
        else:
            confidence = 0.0

        confidences.append(confidence)

    return np.array(confidences)

# =====================================================================
# 代码提取
# =====================================================================
def extract_code(response: str) -> str:
    import re
    patterns = [
        r'```python\s*\n(.*?)```',
        r'```\s*\n(.*?)```',
    ]
    for pat in patterns:
        matches = re.findall(pat, response, re.DOTALL)
        if matches:
            return matches[-1].strip()
    if '</think>' in response:
        after = response.split('</think>')[-1].strip()
        if after:
            return after
    return response.strip()

# =====================================================================
# 加载数据集
# =====================================================================
def load_lcb_problems(dataset_path):
    from datasets import load_dataset
    if dataset_path and os.path.isdir(dataset_path):
        dataset = load_dataset(dataset_path, split="test")
    else:
        dataset = load_dataset("livecodebench/code_generation_lite", split="test")
    return [CodeGenerationProblem(**item) for item in dataset]

# =====================================================================
# 主函数
# =====================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="CP-Router × LiveCodeBench")
    parser.add_argument('--llm_path', type=str, required=True,
                        help="LLM 模型路径 (轻量模型)")
    parser.add_argument('--lrm_path', type=str, default=None,
                        help="LRM 模型路径 (重量模型, 不传则 LLM=LRM)")
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--cal_ratio', type=float, default=0.3)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--tau', type=float, default=0.5,
                        help="路由阈值: confidence > threshold → LLM")
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--probe_tokens', type=int, default=50,
                        help="探测阶段生成的 token 数")
    parser.add_argument('--llm_max_tokens', type=int, default=2048,
                        help="LLM 生成 budget")
    parser.add_argument('--lrm_max_tokens', type=int, default=4096,
                        help="LRM 生成 budget")
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    np.random.seed(args.seed)
    random.seed(args.seed)

    print("=" * 60)
    print("  CP-Router × LiveCodeBench (Code Generation)")
    print("=" * 60)
    print(f"  LLM: {args.llm_path}")
    print(f"  LRM: {args.lrm_path or '(same as LLM)'}")
    print(f"  Probe tokens: {args.probe_tokens}")
    print(f"  β={args.beta}")
    print("=" * 60)

    # ---- 加载数据 ----
    print("\n[1/5] 加载数据集...")
    problems = load_lcb_problems(args.dataset_path)
    if args.max_samples and args.max_samples < len(problems):
        problems = problems[:args.max_samples]
    print(f"  共 {len(problems)} 道题")

    # ---- 划分校准集/测试集 ----
    n = len(problems)
    indices = list(range(n))
    random.shuffle(indices)
    cal_size = max(int(n * args.cal_ratio), 1)
    cal_indices = indices[:cal_size]
    test_indices = indices[cal_size:]
    print(f"  校准集: {len(cal_indices)}, 测试集: {len(test_indices)}")

    # ---- 初始化 LLM ----
    print("\n[2/5] 初始化 LLM...")
    available_gpus = args.gpu_ids.split(',')
    llm_engine = LLM(
        model=args.llm_path,
        tensor_parallel_size=len(available_gpus),
        dtype="bfloat16",
        max_model_len=args.lrm_max_tokens + 2048,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 构建 prompts ----
    sys_prompt = (
        "You are an expert Python programmer. "
        "Solve the given competitive programming problem by writing Python code."
    )

    all_prompts = []
    for prob in problems:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prob.question_content},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        all_prompts.append(prompt)

    # ---- Probe: 提取所有题目的 confidence ----
    print("\n[3/5] Probe phase: 提取 generation confidence...")
    all_confidences = extract_generation_confidence(
        llm_engine, tokenizer, all_prompts, args.probe_tokens
    )

    cal_confidences = all_confidences[cal_indices]
    test_confidences = all_confidences[test_indices]

    print(f"  校准集 confidence: mean={cal_confidences.mean():.4f}, "
          f"std={cal_confidences.std():.4f}")
    print(f"  测试集 confidence: mean={test_confidences.mean():.4f}, "
          f"std={test_confidences.std():.4f}")

    # ---- Conformal Prediction 校准 ----
    print("\n[4/5] Conformal Prediction 校准...")
    # 对校准集: 用 confidence 作为 score (无需 ground truth label)
    # 选择 α 使得 FBE 最大化
    best_alpha, best_fbe = select_alpha_by_fbe(
        cal_confidences, test_confidences,
        tau=args.tau, beta=args.beta
    )
    print(f"  α* = {best_alpha:.4f}, FBE = {best_fbe:.4f}")

    # 计算阈值
    cal_scores = 1.0 - cal_confidences
    q_hat = compute_quantile_threshold(cal_scores, best_alpha)
    confidence_threshold = 1.0 - q_hat
    print(f"  q_hat = {q_hat:.4f}")
    print(f"  confidence threshold = {confidence_threshold:.4f}")

    # ---- 路由决策 + 生成 ----
    print("\n[5/5] 路由 + 生成...")
    llm_indices_local = []  # 在 test_indices 中的局部索引
    lrm_indices_local = []

    for i, tidx in enumerate(test_indices):
        if all_confidences[tidx] >= confidence_threshold:
            llm_indices_local.append(i)
        else:
            lrm_indices_local.append(i)

    print(f"  → LLM (短 budget): {len(llm_indices_local)} 题")
    print(f"  → LRM (长 budget): {len(lrm_indices_local)} 题")

    # 生成: LLM 路由的题目用短 budget
    test_prompts = [all_prompts[test_indices[i]] for i in range(len(test_indices))]
    results = [None] * len(test_indices)

    if llm_indices_local:
        llm_prompts = [test_prompts[i] for i in llm_indices_local]
        llm_outputs = llm_engine.generate(
            llm_prompts,
            SamplingParams(
                max_tokens=args.llm_max_tokens,
                temperature=args.temperature,
                stop=[tokenizer.eos_token],
            ),
            use_tqdm=True,
        )
        for local_i, out in zip(llm_indices_local, llm_outputs):
            text = out.outputs[0].text
            tokens = len(out.outputs[0].token_ids)
            results[local_i] = {
                "routed_to": "LLM",
                "response": text,
                "code": extract_code(text),
                "tokens": tokens,
            }

    # 生成: LRM 路由的题目用长 budget
    if lrm_indices_local:
        lrm_prompts = [test_prompts[i] for i in lrm_indices_local]
        # 如果有单独的 LRM, 这里应该切换模型
        # 测试阶段 LLM=LRM, 只是 budget 不同
        lrm_outputs = llm_engine.generate(
            lrm_prompts,
            SamplingParams(
                max_tokens=args.lrm_max_tokens,
                temperature=args.temperature,
                stop=[tokenizer.eos_token],
            ),
            use_tqdm=True,
        )
        for local_i, out in zip(lrm_indices_local, lrm_outputs):
            text = out.outputs[0].text
            tokens = len(out.outputs[0].token_ids)
            results[local_i] = {
                "routed_to": "LRM",
                "response": text,
                "code": extract_code(text),
                "tokens": tokens,
            }

    # ---- 统计 ----
    llm_tokens = sum(r['tokens'] for r in results if r and r['routed_to'] == 'LLM')
    lrm_tokens = sum(r['tokens'] for r in results if r and r['routed_to'] == 'LRM')
    total_tokens = llm_tokens + lrm_tokens
    all_lrm_tokens = sum(r['tokens'] for r in results if r) if not lrm_indices_local else \
        len(test_indices) * (lrm_tokens / max(len(lrm_indices_local), 1))

    trr = 1.0 - (total_tokens / max(all_lrm_tokens, 1)) if all_lrm_tokens > 0 else 0

    # ---- 保存结果 ----
    model_name = os.path.basename(args.llm_path)
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(SCRIPT_DIR, "outputs", "cp_router", model_name)
    os.makedirs(out_dir, exist_ok=True)

    save_results = []
    for i, tidx in enumerate(test_indices):
        r = results[i]
        if r is None:
            continue
        save_results.append({
            "question_id": problems[tidx].question_id,
            "routed_to": r["routed_to"],
            "confidence": float(all_confidences[tidx]),
            "tokens": r["tokens"],
            "code": r["code"],
            "output_list": [r["response"]],
        })

    out_file = os.path.join(out_dir, f"cp_router_alpha{best_alpha:.2f}.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  CP-Router × LiveCodeBench 结果")
    print(f"{'='*60}")
    print(f"  α* = {best_alpha:.4f}")
    print(f"  Confidence threshold = {confidence_threshold:.4f}")
    print(f"  → LLM: {len(llm_indices_local)} 题 ({100*len(llm_indices_local)/len(test_indices):.1f}%)")
    print(f"  → LRM: {len(lrm_indices_local)} 题 ({100*len(lrm_indices_local)/len(test_indices):.1f}%)")
    print(f"  LLM tokens: {llm_tokens}")
    print(f"  LRM tokens: {lrm_tokens}")
    print(f"  TRR (token reduction): {trr:.4f}")
    print(f"  结果保存到: {out_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
