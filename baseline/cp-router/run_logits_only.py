"""
轻量版 CP-Router: 仅提取 LLM logits 并做路由决策
不需要加载 LRM, 适合快速验证和调试

用法:
  python run_logits_only.py \
      --llm_path ../models/Qwen2.5-0.5B-Instruct \
      --dataset mmlu_elementary_math \
      --max_samples 50
"""
import os
import sys
import json
import argparse
import numpy as np

from core.data_loader import load_dataset_by_name, split_calibration_test, format_mcqa_prompt
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='mmlu_elementary_math')
    parser.add_argument('--cal_ratio', type=float, default=0.3)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--tau', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    np.random.seed(args.seed)
    
    print("=" * 50)
    print("  CP-Router (Logits Only Mode)")
    print("=" * 50)
    
    # 加载数据
    data = load_dataset_by_name(args.dataset)
    if args.max_samples:
        indices = list(range(min(args.max_samples, len(data["questions"]))))
        data = {
            "questions": [data["questions"][i] for i in indices],
            "options": [data["options"][i] for i in indices],
            "labels": data["labels"][indices],
            "name": data["name"],
            "num_choices": data["num_choices"]
        }
    
    cal_data, test_data = split_calibration_test(data, args.cal_ratio, args.seed)
    print(f"  Cal: {len(cal_data['questions'])}, Test: {len(test_data['questions'])}")
    
    # 提取 logits
    extractor = LogitExtractor(args.llm_path)
    
    cal_prompts = [format_mcqa_prompt(q, o) for q, o in zip(cal_data["questions"], cal_data["options"])]
    test_prompts = [format_mcqa_prompt(q, o) for q, o in zip(test_data["questions"], test_data["options"])]
    
    _, cal_probs = extractor.extract_logits(cal_prompts, args.batch_size)
    _, test_probs = extractor.extract_logits(test_prompts, args.batch_size)
    
    # LLM 准确率
    llm_preds = np.argmax(test_probs, axis=1)
    llm_acc = np.mean(llm_preds == test_data["labels"])
    print(f"\n  LLM Accuracy: {llm_acc:.4f}")
    
    # FBE 选择最优 α
    best_alpha, best_fbe = select_optimal_alpha(
        cal_probs, cal_data["labels"], test_probs,
        beta=args.beta, num_choices=data["num_choices"]
    )
    print(f"  Best α = {best_alpha:.4f} (FBE = {best_fbe:.4f})")
    
    # 构建预测集
    cal_scores = compute_nonconformity_scores(cal_probs, cal_data["labels"])
    q_hat = compute_quantile_threshold(cal_scores, best_alpha)
    prediction_sets = construct_prediction_set(test_probs, q_hat, data["num_choices"])
    set_sizes = compute_prediction_set_sizes(prediction_sets)
    
    # 路由统计
    llm_count = np.sum(set_sizes <= args.tau)
    lrm_count = np.sum(set_sizes > args.tau)
    
    print(f"\n  APSS: {compute_apss(prediction_sets):.3f}")
    print(f"  Coverage: {compute_coverage(prediction_sets, test_data['labels']):.4f}")
    print(f"  Route to LLM: {llm_count} ({100*llm_count/len(set_sizes):.1f}%)")
    print(f"  Route to LRM: {lrm_count} ({100*lrm_count/len(set_sizes):.1f}%)")
    
    # 预测集大小分布
    print(f"\n  Prediction set size distribution:")
    for s in range(1, data["num_choices"] + 1):
        count = np.sum(set_sizes == s)
        print(f"    Size {s}: {count} ({100*count/len(set_sizes):.1f}%)")
    
    # 不同 α 的效果
    print(f"\n  α sensitivity analysis:")
    for alpha in [0.05, 0.1, 0.2, 0.3, 0.5]:
        q = compute_quantile_threshold(cal_scores, alpha)
        ps = construct_prediction_set(test_probs, q, data["num_choices"])
        ss = compute_prediction_set_sizes(ps)
        llm_n = np.sum(ss <= args.tau)
        print(f"    α={alpha:.2f}: APSS={np.mean(ss):.3f}, LLM={llm_n}, LRM={len(ss)-llm_n}")


if __name__ == "__main__":
    main()
