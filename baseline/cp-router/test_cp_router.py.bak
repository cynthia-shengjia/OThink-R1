"""
CP-Router 测试脚本

测试阶段: LLM 和 LRM 都用 Qwen2.5-0.5B-Instruct
目的: 验证 CP-Router 的路由逻辑是否正确

流程:
  1. 加载本地数据集 (MATH/AIME/ASDIV → MCQA 格式)
  2. 用 LLM 提取选项 logits
  3. Conformal Prediction 校准
  4. FBE 自适应选择 α
  5. 路由决策
  6. LRM 推理 (对路由到 LRM 的样本)
  7. 汇总结果
"""
import os
import sys
import json
import argparse
import numpy as np
import time
import torch
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.local_data_loader import load_local_dataset
from core.data_loader import split_calibration_test, format_mcqa_prompt
from core.logit_extractor import LogitExtractor
from core.conformal_prediction import (
    compute_nonconformity_scores,
    compute_quantile_threshold,
    construct_prediction_set,
    compute_prediction_set_sizes,
    compute_apss,
    compute_coverage
)
from core.fbe import select_optimal_alpha, compute_fbe
from core.router import CPRouter


def _infer_model_size(model_name: str) -> str:
    """从模型名推断参数量: Qwen2.5-0.5B-Instruct → 0.5B"""
    import re
    m = re.search(r'(\d+\.?\d*B)', model_name, re.IGNORECASE)
    return m.group(1) if m else "unknown"

def parse_args():
    parser = argparse.ArgumentParser(description="CP-Router Test")
    parser.add_argument('--model_path', type=str, required=True,
                        help="模型路径 (同时用作 LLM 和 LRM)")
    parser.add_argument('--datasets_dir', type=str, required=True,
                        help="数据集根目录 (包含 MATH/, AIME/, ASDIV/)")
    parser.add_argument('--dataset', type=str, default='math',
                        choices=['math', 'aime', 'asdiv'],
                        help="测试数据集")
    parser.add_argument('--max_samples', type=int, default=30,
                        help="最大样本数 (测试阶段用少量)")
    parser.add_argument('--cal_ratio', type=float, default=0.3,
                        help="校准集比例")
    parser.add_argument('--tau', type=int, default=1,
                        help="路由阈值")
    parser.add_argument('--beta', type=float, default=3.0,
                        help="FBE 权重")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="批处理大小")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip_lrm', action='store_true',
                        help="跳过 LRM 推理 (仅测试路由决策)")
    parser.add_argument('--lrm_max_tokens', type=int, default=512,
                        help="LRM 最大生成 token 数 (测试阶段用小值)")
    args = parser.parse_args()
    return args


def run_lrm_inference(model_path, prompts, max_tokens=512):
    """
    用 vLLM 对路由到 LRM 的样本进行推理
    测试阶段: LRM 也是 Qwen2.5-0.5B-Instruct
    """
    import re
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    
    print(f"  Loading LRM (vLLM): {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    llm_engine = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        max_model_len=max_tokens + 1024
    )
    
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        skip_special_tokens=True
    )
    
    # 构建 chat 格式
    formatted = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": "Please answer the multiple choice question. Output only the letter (A, B, C, or D)."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted.append(text)
    
    outputs = llm_engine.generate(formatted, sampling_params, use_tqdm=True)
    
    results = []
    option_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    
    for output in outputs:
        text = output.outputs[0].text
        tokens = len(output.outputs[0].token_ids)
        
        # 提取答案
        answer_idx = -1
        # 尝试多种模式
        for pattern in [
            r'\b([A-D])\b',
            r'([A-D])\.',
            r'answer\s*(?:is|:)\s*([A-D])',
        ]:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                answer_idx = option_map.get(matches[-1].upper(), -1)
                break
        
        results.append({
            "text": text[:200],  # 截断用于显示
            "tokens": tokens,
            "answer_idx": answer_idx
        })
    
    # 清理
    del llm_engine
    torch.cuda.empty_cache()
    
    return results


def main():
    args = parse_args()
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("  CP-Router 测试")
    print("  (LLM = LRM = Qwen2.5-0.5B-Instruct)")
    print("=" * 60)
    print(f"  模型: {args.model_path}")
    print(f"  数据集: {args.dataset}")
    print(f"  样本数: {args.max_samples}")
    print(f"  τ={args.tau}, β={args.beta}")
    print("=" * 60)
    
    # ========== Step 1: 加载数据 ==========
    print("\n[Step 1/6] 加载数据集...")
    data = load_local_dataset(
        args.dataset,
        args.datasets_dir,
        max_samples=args.max_samples,
        seed=args.seed
    )
    
    # 划分校准集和测试集
    cal_data, test_data = split_calibration_test(data, args.cal_ratio, args.seed)
    n_cal = len(cal_data["questions"])
    n_test = len(test_data["questions"])
    print(f"  校准集: {n_cal} 条")
    print(f"  测试集: {n_test} 条")
    
    # 显示样例
    print(f"\n  样例问题: {test_data['questions'][0][:80]}...")
    print(f"  选项: {test_data['options'][0]}")
    print(f"  正确答案索引: {test_data['labels'][0]}")
    
    # ========== Step 2: 提取 LLM logits ==========
    print("\n[Step 2/6] 提取 LLM logits...")
    extractor = LogitExtractor(args.model_path)
    
    # 格式化提示
    cal_prompts = [
        format_mcqa_prompt(q, opts)
        for q, opts in zip(cal_data["questions"], cal_data["options"])
    ]
    test_prompts = [
        format_mcqa_prompt(q, opts)
        for q, opts in zip(test_data["questions"], test_data["options"])
    ]
    
    print(f"  提取校准集 logits ({n_cal} 条)...")
    cal_logits, cal_probs = extractor.extract_logits(cal_prompts, args.batch_size)
    
    print(f"  提取测试集 logits ({n_test} 条)...")
    test_logits, test_probs = extractor.extract_logits(test_prompts, args.batch_size)
    
    # LLM 准确率
    llm_preds = np.argmax(test_probs, axis=1)
    llm_acc = np.mean(llm_preds == test_data["labels"])
    print(f"\n  ✅ LLM 准确率: {llm_acc:.4f} ({int(llm_acc * n_test)}/{n_test})")
    
    # 显示概率分布样例
    print(f"\n  样例概率分布 (前3条):")
    for i in range(min(3, n_test)):
        probs_str = ", ".join(f"{p:.3f}" for p in test_probs[i])
        pred = chr(65 + llm_preds[i])
        gold = chr(65 + test_data["labels"][i])
        correct = "✓" if llm_preds[i] == test_data["labels"][i] else "✗"
        print(f"    [{i}] P=[{probs_str}] pred={pred} gold={gold} {correct}")
    
    # 释放模型内存
    del extractor
    torch.cuda.empty_cache()
    
    # ========== Step 3: FBE 自适应校准 ==========
    print("\n[Step 3/6] FBE 自适应选择 α...")
    
    best_alpha, best_fbe = select_optimal_alpha(
        cal_probs=cal_probs,
        cal_labels=cal_data["labels"],
        test_probs=test_probs,
        beta=args.beta,
        num_choices=data["num_choices"]
    )
    print(f"  ✅ 最优 α* = {best_alpha:.4f}")
    print(f"  ✅ FBE score = {best_fbe:.4f}")
    
    # 对比不同 α 的效果
    print(f"\n  α 敏感性分析:")
    cal_scores = compute_nonconformity_scores(cal_probs, cal_data["labels"])
    for alpha in [0.05, 0.1, 0.2, 0.3, 0.5, best_alpha]:
        q_hat = compute_quantile_threshold(cal_scores, alpha)
        ps = construct_prediction_set(test_probs, q_hat, data["num_choices"])
        sizes = compute_prediction_set_sizes(ps)
        llm_n = int(np.sum(sizes <= args.tau))
        lrm_n = int(np.sum(sizes > args.tau))
        apss = float(np.mean(sizes))
        coverage = compute_coverage(ps, test_data["labels"])
        marker = " ← α*" if abs(alpha - best_alpha) < 0.001 else ""
        print(f"    α={alpha:.3f}: APSS={apss:.2f}, Coverage={coverage:.3f}, "
              f"LLM={llm_n}, LRM={lrm_n}{marker}")
    
    # ========== Step 4: CP-Router 路由决策 ==========
    print("\n[Step 4/6] CP-Router 路由决策...")
    
    router = CPRouter(
        tau=args.tau,
        beta=args.beta,
        num_choices=data["num_choices"],
        fixed_alpha=best_alpha  # 使用 FBE 选出的 α
    )
    router.calibrate(cal_probs, cal_data["labels"], test_probs)
    
    results, llm_indices, lrm_indices = router.route(test_probs, test_data["labels"])
    
    print(f"\n  路由结果:")
    print(f"    → LLM: {len(llm_indices)} 条 ({100*len(llm_indices)/n_test:.1f}%)")
    print(f"    → LRM: {len(lrm_indices)} 条 ({100*len(lrm_indices)/n_test:.1f}%)")
    
    # 预测集大小分布
    set_sizes = np.array([r.prediction_set_size for r in results])
    print(f"\n  预测集大小分布:")
    for s in range(1, data["num_choices"] + 1):
        count = int(np.sum(set_sizes == s))
        if count > 0:
            print(f"    Size {s}: {count} 条 ({100*count/n_test:.1f}%)")
    
    # LLM 路由部分的准确率
    if llm_indices:
        llm_correct = sum(1 for i in llm_indices if llm_preds[i] == test_data["labels"][i])
        llm_route_acc = llm_correct / len(llm_indices)
        print(f"\n  LLM 路由部分准确率: {llm_route_acc:.4f} ({llm_correct}/{len(llm_indices)})")
    
    # ========== Step 5: LRM 推理 ==========
    if not args.skip_lrm and len(lrm_indices) > 0:
        print(f"\n[Step 5/6] LRM 推理 ({len(lrm_indices)} 条)...")
        
        lrm_prompts = [
            format_mcqa_prompt(
                test_data["questions"][i],
                test_data["options"][i]
            )
            for i in lrm_indices
        ]
        
        lrm_results = run_lrm_inference(
            args.model_path,  # 测试阶段: LRM 也用同一个模型
            lrm_prompts,
            max_tokens=args.lrm_max_tokens
        )
        
        # 填充结果
        lrm_correct = 0
        for idx, lrm_idx in enumerate(lrm_indices):
            answer_idx = lrm_results[idx]["answer_idx"]
            results[lrm_idx].lrm_answer = answer_idx
            results[lrm_idx].final_answer = answer_idx
            results[lrm_idx].correct = (answer_idx == test_data["labels"][lrm_idx])
            results[lrm_idx].lrm_tokens = lrm_results[idx]["tokens"]
            if results[lrm_idx].correct:
                lrm_correct += 1
        
        print(f"  LRM 路由部分准确率: {lrm_correct}/{len(lrm_indices)}")
    else:
        print(f"\n[Step 5/6] 跳过 LRM 推理 (--skip_lrm)")
        for i in lrm_indices:
            results[i].final_answer = results[i].llm_answer
            results[i].correct = (results[i].llm_answer == test_data["labels"][i])
    
    # ========== Step 6: 汇总结果 ==========
    print(f"\n[Step 6/6] 汇总结果...")
    
    total_correct = sum(1 for r in results if r.correct)
    router_acc = total_correct / n_test
    
    # Token 统计
    total_lrm_tokens = sum(r.lrm_tokens for r in results if r.lrm_tokens > 0)
    avg_lrm_tokens = total_lrm_tokens / len(lrm_indices) if lrm_indices else 100
    all_lrm_tokens = n_test * avg_lrm_tokens  # 全部用 LRM 的估算 token
    
    # 路由器实际 token: LLM 部分几乎不消耗 (只提取 logits), LRM 部分消耗推理 token
    router_tokens = total_lrm_tokens  # LLM 路由部分 token ≈ 0 (只取 logits)
    trr = 1.0 - (router_tokens / all_lrm_tokens) if all_lrm_tokens > 0 else 0
    
    # U_token
    u_token = (router_acc - llm_acc) / (1.0 - trr) if trr < 1.0 else 0
    
    print(f"\n{'='*60}")
    print(f"  CP-Router 测试结果 ({args.dataset.upper()})")
    print(f"{'='*60}")
    print(f"  α*           = {best_alpha:.4f}")
    print(f"  APSS         = {float(np.mean(set_sizes)):.3f}")
    print(f"  Coverage     = {compute_coverage(construct_prediction_set(test_probs, router.q_hat, data['num_choices']), test_data['labels']):.4f}")
    print(f"")
    print(f"  LLM-only Acc = {llm_acc:.4f}")
    print(f"  Router Acc   = {router_acc:.4f}")
    print(f"  TRR          = {trr:.4f}")
    print(f"  U_token      = {u_token:.4f}")
    print(f"")
    print(f"  路由分配:")
    print(f"    → LLM: {len(llm_indices)} ({100*len(llm_indices)/n_test:.1f}%)")
    print(f"    → LRM: {len(lrm_indices)} ({100*len(lrm_indices)/n_test:.1f}%)")
    print(f"{'='*60}")
    
    # 保存结果
    model_basename = os.path.basename(os.path.normpath(args.model_path))
    model_size = _infer_model_size(model_basename)
    result_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", model_size, args.dataset,
    )
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(
        result_dir,
        f"test_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    
    result_dict = {
        "dataset": args.dataset,
        "model": args.model_path,
        "n_cal": n_cal,
        "n_test": n_test,
        "alpha_star": best_alpha,
        "fbe_score": best_fbe,
        "apss": float(np.mean(set_sizes)),
        "llm_acc": float(llm_acc),
        "router_acc": float(router_acc),
        "trr": float(trr),
        "u_token": float(u_token),
        "llm_count": len(llm_indices),
        "lrm_count": len(lrm_indices),
        "tau": args.tau,
        "beta": args.beta,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(result_file, 'w') as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    print(f"\n  结果保存到: {result_file}")


if __name__ == "__main__":
    main()
