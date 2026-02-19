"""
CP-Router 主运行脚本

用法:
  python run_cp_router.py \
      --llm_path ../models/Qwen2.5-0.5B-Instruct \
      --lrm_path ../models/DeepSeek-R1-Distill-Qwen-1.5B \
      --dataset mmlu_elementary_math \
      --cal_ratio 0.3 \
      --beta 3.0 \
      --tau 1
"""
import os
import sys
import json
import argparse
import numpy as np
import time
from datetime import datetime

from core.data_loader import load_dataset_by_name, split_calibration_test, format_mcqa_prompt
from core.logit_extractor import LogitExtractor
from core.router import CPRouter
from core.lrm_inference import LRMInference
from core.baselines import random_routing, top1_probability_routing, entropy_routing


def parse_args():
    parser = argparse.ArgumentParser(description="CP-Router: LLM/LRM Routing")
    
    # 模型
    parser.add_argument('--llm_path', type=str, required=True, help="LLM 模型路径")
    parser.add_argument('--lrm_path', type=str, required=True, help="LRM 模型路径")
    
    # 数据
    parser.add_argument('--dataset', type=str, default='mmlu_elementary_math',
                        help="数据集名称")
    parser.add_argument('--cal_ratio', type=float, default=0.3,
                        help="校准集比例")
    parser.add_argument('--max_samples', type=int, default=None,
                        help="最大样本数 (用于快速测试)")
    
    # CP-Router 参数
    parser.add_argument('--tau', type=int, default=1,
                        help="路由阈值 (预测集大小 ≤ τ 则用 LLM)")
    parser.add_argument('--beta', type=float, default=3.0,
                        help="FBE 中 H_full 的权重")
    parser.add_argument('--fixed_alpha', type=float, default=None,
                        help="固定 α (不使用 FBE)")
    
    # 推理参数
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--gpu_id', type=str, default='0')
    
    # 输出
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=42)
    
    # 模式
    parser.add_argument('--skip_lrm', action='store_true',
                        help="跳过 LRM 推理 (仅评估路由决策)")
    parser.add_argument('--run_baselines', action='store_true',
                        help="同时运行 baseline 方法")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("  CP-Router: Uncertainty-Aware LLM/LRM Routing")
    print("=" * 60)
    print(f"  LLM: {args.llm_path}")
    print(f"  LRM: {args.lrm_path}")
    print(f"  Dataset: {args.dataset}")
    print(f"  τ={args.tau}, β={args.beta}")
    print("=" * 60)
    
    # ==================== Step 0: 加载数据 ====================
    print("\n[Step 0] Loading dataset...")
    data = load_dataset_by_name(args.dataset)
    
    if args.max_samples and args.max_samples < len(data["questions"]):
        indices = list(range(len(data["questions"])))
        np.random.shuffle(indices)
        indices = indices[:args.max_samples]
        data = {
            "questions": [data["questions"][i] for i in indices],
            "options": [data["options"][i] for i in indices],
            "labels": data["labels"][indices],
            "name": data["name"],
            "num_choices": data["num_choices"]
        }
    
    print(f"  Total samples: {len(data['questions'])}")
    
    # 划分校准集和测试集
    cal_data, test_data = split_calibration_test(data, args.cal_ratio, args.seed)
    print(f"  Calibration set: {len(cal_data['questions'])}")
    print(f"  Test set: {len(test_data['questions'])}")
    
    # ==================== Step 1: LLM Logit 提取 ====================
    print("\n[Step 1] Extracting LLM logits...")
    extractor = LogitExtractor(args.llm_path)
    
    # 格式化提示
    cal_prompts = [
        format_mcqa_prompt(q, opts)
        for q, opts in zip(cal_data["questions"], cal_data["options"])
    ]
    test_prompts = [
        format_mcqa_prompt(q, opts)
        for q, opts in zip(test_data["questions"], test_data["options"])
    ]
    
    # 提取 logits
    print("  Extracting calibration set logits...")
    cal_logits, cal_probs = extractor.extract_logits(cal_prompts, args.batch_size)
    
    print("  Extracting test set logits...")
    test_logits, test_probs = extractor.extract_logits(test_prompts, args.batch_size)
    
    # LLM 准确率
    llm_predictions = np.argmax(test_probs, axis=1)
    llm_acc = np.mean(llm_predictions == test_data["labels"])
    print(f"  LLM accuracy: {llm_acc:.4f}")
    
    # 释放 LLM 模型内存
    del extractor
    import torch
    torch.cuda.empty_cache()
    
    # ==================== Step 2: CP-Router 校准 ====================
    print("\n[Step 2] CP-Router calibration...")
    router = CPRouter(
        tau=args.tau,
        beta=args.beta,
        num_choices=data["num_choices"],
        fixed_alpha=args.fixed_alpha
    )
    
    alpha_star = router.calibrate(cal_probs, cal_data["labels"], test_probs)
    print(f"  Selected α* = {alpha_star:.4f}")
    
    # ==================== Step 3: 路由决策 ====================
    print("\n[Step 3] Routing decisions...")
    results, llm_indices, lrm_indices = router.route(test_probs, test_data["labels"])
    
    print(f"  Routed to LLM: {len(llm_indices)} ({100*len(llm_indices)/len(results):.1f}%)")
    print(f"  Routed to LRM: {len(lrm_indices)} ({100*len(lrm_indices)/len(results):.1f}%)")
    
    # ==================== Step 4: LRM 推理 ====================
    if not args.skip_lrm and len(lrm_indices) > 0:
        print(f"\n[Step 4] LRM inference on {len(lrm_indices)} samples...")
        
        lrm = LRMInference(
            model_path=args.lrm_path,
            max_tokens=args.max_tokens,
            gpu_memory_utilization=0.9
        )
        
        # 准备 LRM 提示
        lrm_prompts = [
            format_mcqa_prompt(
                test_data["questions"][i],
                test_data["options"][i]
            )
            for i in lrm_indices
        ]
        
        lrm_results = lrm.generate(lrm_prompts)
        
        # 填充结果
        option_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        for idx, lrm_idx in enumerate(lrm_indices):
            answer_str = lrm_results[idx]["answer"]
            answer_idx = option_map.get(answer_str, -1) if answer_str else -1
            
            results[lrm_idx].lrm_answer = answer_idx
            results[lrm_idx].final_answer = answer_idx
            results[lrm_idx].correct = (answer_idx == test_data["labels"][lrm_idx])
            results[lrm_idx].lrm_tokens = lrm_results[idx]["tokens"]
        
        del lrm
        torch.cuda.empty_cache()
    else:
        print("\n[Step 4] Skipping LRM inference")
        # 用 LLM 预测填充 LRM 部分 (仅用于评估路由决策)
        for i in lrm_indices:
            results[i].final_answer = results[i].llm_answer
            results[i].correct = (results[i].llm_answer == test_data["labels"][i])
    
    # ==================== Step 5: 评估 ====================
    print("\n[Step 5] Evaluation...")
    
    # 计算 LRM-only 的 token 数 (估算)
    avg_lrm_tokens = 500  # 默认估算
    if any(r.lrm_tokens > 0 for r in results):
        lrm_token_list = [r.lrm_tokens for r in results if r.lrm_tokens > 0]
        avg_lrm_tokens = int(np.mean(lrm_token_list))
    
    lrm_total_tokens = len(results) * avg_lrm_tokens
    
    metrics = router.evaluate(results, llm_acc, lrm_total_tokens)
    
    print(f"\n{'='*50}")
    print(f"  CP-Router Results on {args.dataset}")
    print(f"{'='*50}")
    print(f"  α* = {metrics.alpha_star:.4f}")
    print(f"  APSS = {metrics.apss:.3f}")
    print(f"  Accuracy = {metrics.accuracy:.4f}")
    print(f"  TRR = {metrics.trr:.4f}")
    print(f"  U_token = {metrics.u_token:.4f}")
    print(f"  LLM count = {metrics.llm_count}")
    print(f"  LRM count = {metrics.lrm_count}")
    print(f"  LLM-only Acc = {llm_acc:.4f}")
    print(f"{'='*50}")
    
    # ==================== 保存结果 ====================
    os.makedirs(args.output_dir, exist_ok=True)
    
    result_dict = {
        "dataset": args.dataset,
        "llm_path": args.llm_path,
        "lrm_path": args.lrm_path,
        "tau": args.tau,
        "beta": args.beta,
        "alpha_star": metrics.alpha_star,
        "apss": metrics.apss,
        "accuracy": metrics.accuracy,
        "trr": metrics.trr,
        "u_token": metrics.u_token,
        "llm_acc": float(llm_acc),
        "llm_count": metrics.llm_count,
        "lrm_count": metrics.lrm_count,
        "timestamp": datetime.now().isoformat()
    }
    
    output_file = os.path.join(
        args.output_dir,
        f"cp_router_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_file, 'w') as f:
        json.dump(result_dict, f, indent=2)
    print(f"\n  Results saved to: {output_file}")
    
    # ==================== Baselines ====================
    if args.run_baselines:
        print("\n" + "=" * 50)
        print("  Running Baselines")
        print("=" * 50)
        
        # Random routing
        for threshold in [0.2, 0.3, 0.4, 0.5]:
            llm_idx, lrm_idx = random_routing(len(test_data["labels"]), threshold)
            correct = sum(1 for i in llm_idx if llm_predictions[i] == test_data["labels"][i])
            # LRM 部分暂用 LLM 预测
            correct += sum(1 for i in lrm_idx if llm_predictions[i] == test_data["labels"][i])
            acc = correct / len(test_data["labels"])
            trr = len(llm_idx) / len(test_data["labels"])
            print(f"  Random (t={threshold}): Acc={acc:.4f}, TRR={trr:.4f}")
        
        # Top-1 probability routing
        for threshold in [0.6, 0.7, 0.8]:
            llm_idx, lrm_idx = top1_probability_routing(test_probs, threshold)
            print(f"  Top-1 (t={threshold}): LLM={len(llm_idx)}, LRM={len(lrm_idx)}")
        
        # Entropy routing
        for threshold in [1.0, 1.2, 1.4]:
            llm_idx, lrm_idx = entropy_routing(test_probs, threshold)
            print(f"  Entropy (t={threshold}): LLM={len(llm_idx)}, LRM={len(lrm_idx)}")


if __name__ == "__main__":
    main()
