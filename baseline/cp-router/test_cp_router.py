"""
CP-Router 测试脚本

测试阶段: LLM 和 LRM 都用 Qwen2.5-0.5B-Instruct
目的: 验证 CP-Router 的路由逻辑是否正确
"""
import os
import sys
import json
import argparse
import numpy as np
import time
import torch
import logging
from datetime import datetime

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

OPTION_LETTERS = ['A', 'B', 'C', 'D']

def create_eval_logger(log_filename):
    """创建日志器，同时输出到文件和终端"""
    logger = logging.getLogger("cp_router_eval")
    logger.setLevel(logging.INFO)
    # 清除旧 handler，避免重复
    logger.handlers.clear()

    formatter = logging.Formatter('%(message)s')

    log_dir = os.path.dirname(log_filename)
    os.makedirs(log_dir, exist_ok=True)

    fh = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def parse_args():
    parser = argparse.ArgumentParser(description="CP-Router Test")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--datasets_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='math',
                        choices=['math', 'aime', 'asdiv'])
    parser.add_argument('--max_samples', type=int, default=30)
    parser.add_argument('--cal_ratio', type=float, default=0.3)
    parser.add_argument('--tau', type=int, default=1)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip_lrm', action='store_true')
    parser.add_argument('--lrm_max_tokens', type=int, default=512)
    return parser.parse_args()

def run_lrm_inference(model_path, prompts, max_tokens=512):
    """用 vLLM 对路由到 LRM 的样本进行推理"""
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
        temperature=0.0, top_p=1.0,
        max_tokens=max_tokens, skip_special_tokens=True
    )

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
        answer_idx = -1
        for pattern in [r'\b([A-D])\b', r'([A-D])\.', r'answer\s*(?:is|:)\s*([A-D])']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                answer_idx = option_map.get(matches[-1].upper(), -1)
                break
        results.append({"text": text, "tokens": tokens, "answer_idx": answer_idx})

    del llm_engine
    torch.cuda.empty_cache()
    return results

def main():
    args = parse_args()
    np.random.seed(args.seed)

    # ========== 构建日志路径 ==========
    model_name = os.path.basename(args.model_path).replace('-', '')
    log_filename = (
        f"log/{args.dataset}/CP-Router/"
        f"{model_name}-tau{args.tau}-beta{args.beta}.log"
    )
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_filename)

    # ========== Step 1: 加载数据 ==========
    print(f"\n[Step 1/6] 加载数据集 {args.dataset}...")
    data = load_local_dataset(args.dataset, args.datasets_dir,
                              max_samples=args.max_samples, seed=args.seed)
    cal_data, test_data = split_calibration_test(data, args.cal_ratio, args.seed)
    n_cal = len(cal_data["questions"])
    n_test = len(test_data["questions"])
    print(f"  校准集: {n_cal}, 测试集: {n_test}")

    # ========== Step 2: 提取 LLM logits ==========
    print(f"\n[Step 2/6] 提取 LLM logits...")
    extractor = LogitExtractor(args.model_path)

    cal_prompts = [format_mcqa_prompt(q, o) for q, o in zip(cal_data["questions"], cal_data["options"])]
    test_prompts = [format_mcqa_prompt(q, o) for q, o in zip(test_data["questions"], test_data["options"])]

    cal_logits, cal_probs = extractor.extract_logits(cal_prompts, args.batch_size)
    test_logits, test_probs = extractor.extract_logits(test_prompts, args.batch_size)

    llm_preds = np.argmax(test_probs, axis=1)
    llm_acc = np.mean(llm_preds == test_data["labels"])
    print(f"  LLM 准确率: {llm_acc:.4f} ({int(llm_acc * n_test)}/{n_test})")

    del extractor
    torch.cuda.empty_cache()

    # ========== Step 3: FBE 校准 ==========
    print(f"\n[Step 3/6] FBE 自适应选择 α...")
    best_alpha, best_fbe = select_optimal_alpha(
        cal_probs, cal_data["labels"], test_probs,
        beta=args.beta, num_choices=data["num_choices"]
    )
    print(f"  α* = {best_alpha:.4f}, FBE = {best_fbe:.4f}")

    # ========== Step 4: 路由决策 ==========
    print(f"\n[Step 4/6] CP-Router 路由决策...")
    router = CPRouter(tau=args.tau, beta=args.beta,
                      num_choices=data["num_choices"], fixed_alpha=best_alpha)
    router.calibrate(cal_probs, cal_data["labels"], test_probs)
    results, llm_indices, lrm_indices = router.route(test_probs, test_data["labels"])
    print(f"  → LLM: {len(llm_indices)}, → LRM: {len(lrm_indices)}")

    # ========== Step 5: LRM 推理 ==========
    lrm_texts = {}  # idx -> generated text
    if not args.skip_lrm and len(lrm_indices) > 0:
        print(f"\n[Step 5/6] LRM 推理 ({len(lrm_indices)} 条)...")
        lrm_prompts = [format_mcqa_prompt(test_data["questions"][i], test_data["options"][i]) for i in lrm_indices]
        lrm_results = run_lrm_inference(args.model_path, lrm_prompts, max_tokens=args.lrm_max_tokens)
        for idx, lrm_idx in enumerate(lrm_indices):
            answer_idx = lrm_results[idx]["answer_idx"]
            results[lrm_idx].lrm_answer = answer_idx
            results[lrm_idx].final_answer = answer_idx
            results[lrm_idx].correct = (answer_idx == test_data["labels"][lrm_idx])
            results[lrm_idx].lrm_tokens = lrm_results[idx]["tokens"]
            lrm_texts[lrm_idx] = lrm_results[idx]["text"]
    else:
        print(f"\n[Step 5/6] 跳过 LRM 推理")
        for i in lrm_indices:
            results[i].final_answer = results[i].llm_answer
            results[i].correct = (results[i].llm_answer == test_data["labels"][i])

    # ========== Step 6: 日志输出 (参考 OThink-R1 eval_utils 格式) ==========
    print(f"\n[Step 6/6] 写入日志...")
    logger = create_eval_logger(log_path)

    logger.info(f"\n======== CP-Router Evaluation Start ========")
    logger.info(f"Model:       {args.model_path}")
    logger.info(f"Dataset:     {args.dataset}")
    logger.info(f"Samples:     {n_test} (test) + {n_cal} (cal)")
    logger.info(f"Tau:         {args.tau}")
    logger.info(f"Beta:        {args.beta}")
    logger.info(f"Alpha*:      {best_alpha:.4f}")
    logger.info(f"FBE score:   {best_fbe:.4f}")
    logger.info(f"APSS:        {compute_apss(construct_prediction_set(test_probs, router.q_hat, data['num_choices'])):.3f}")
    logger.info(f"Coverage:    {compute_coverage(construct_prediction_set(test_probs, router.q_hat, data['num_choices']), test_data['labels']):.4f}")
    logger.info(f"q_hat:       {router.q_hat:.4f}")
    logger.info(f"Skip LRM:    {args.skip_lrm}")

    num_generated_tokens_list = []
    verify_result_list = []
    failed_parsed = 0

    for idx in range(n_test):
        r = results[idx]
        question = test_data["questions"][idx]
        options = test_data["options"][idx]
        gold_idx = test_data["labels"][idx]
        gold_answer = OPTION_LETTERS[gold_idx]

        # 模型回答
        if r.final_answer is not None and 0 <= r.final_answer < len(OPTION_LETTERS):
            model_answer = OPTION_LETTERS[r.final_answer]
        else:
            model_answer = "FAILED"
            failed_parsed += 1

        verify_result = r.correct if r.correct is not None else False
        verify_result_list.append(verify_result)

        # 概率分布
        probs_str = ", ".join(f"{OPTION_LETTERS[j]}:{test_probs[idx][j]:.3f}" for j in range(len(OPTION_LETTERS)))

        # 预测集
        pred_set_str = "{" + ", ".join(OPTION_LETTERS[j] for j in r.prediction_set) + "}"

        # token 数
        tokens = r.lrm_tokens if r.lrm_tokens > 0 else 0
        num_generated_tokens_list.append(tokens)

        # LRM 生成文本
        generated_text = lrm_texts.get(idx, "(LLM direct answer, no generation)")

        logger.info("------------------------------------------")
        logger.info(f"Index: {idx}")
        logger.info(f"Question: {question[:120]}...")
        logger.info(f"Options: {options}")
        logger.info(f"Probabilities: [{probs_str}]")
        logger.info(f"Prediction set: {pred_set_str} (size={r.prediction_set_size})")
        logger.info(f"Routed to: {r.routed_to}")
        logger.info(f"Gold answer: {gold_answer} ({options[gold_idx]})")
        logger.info(f"Model answer: {model_answer}")
        logger.info(f"Verify result: {verify_result}")
        logger.info(f"Generated tokens: {tokens}")
        if r.routed_to == "LRM" and idx in lrm_texts:
            logger.info(f"Generated text:\n{generated_text}")

    # ========== Summary ==========
    total_correct = sum(verify_result_list)
    router_acc = total_correct / n_test if n_test > 0 else 0

    total_lrm_tokens = sum(num_generated_tokens_list)
    avg_lrm_tokens = total_lrm_tokens / len(lrm_indices) if lrm_indices else 100
    all_lrm_tokens = n_test * avg_lrm_tokens
    router_tokens = total_lrm_tokens
    trr = 1.0 - (router_tokens / all_lrm_tokens) if all_lrm_tokens > 0 else 0
    u_token = (router_acc - llm_acc) / (1.0 - trr) if trr < 1.0 else 0

    llm_route_correct = sum(1 for i in llm_indices if llm_preds[i] == test_data["labels"][i])
    lrm_route_correct = total_correct - llm_route_correct

    logger.info(f"\n============= Summary =============")
    logger.info(f"Total cases:       {n_test}")
    logger.info(f"Correct:           {total_correct}")
    logger.info(f"Failed parsed:     {failed_parsed}")
    logger.info(f"")
    logger.info(f"LLM-only Acc:      {llm_acc:.4f}")
    logger.info(f"Router Acc:        {router_acc:.4f}")
    logger.info(f"TRR:               {trr:.4f}")
    logger.info(f"U_token:           {u_token:.4f}")
    logger.info(f"")
    logger.info(f"Alpha*:            {best_alpha:.4f}")
    logger.info(f"APSS:              {compute_apss(construct_prediction_set(test_probs, router.q_hat, data['num_choices'])):.3f}")
    logger.info(f"")
    logger.info(f"Routing:")
    logger.info(f"  → LLM:           {len(llm_indices)} ({100*len(llm_indices)/n_test:.1f}%)  correct: {llm_route_correct}/{len(llm_indices) if llm_indices else 0}")
    logger.info(f"  → LRM:           {len(lrm_indices)} ({100*len(lrm_indices)/n_test:.1f}%)  correct: {lrm_route_correct}/{len(lrm_indices) if lrm_indices else 0}")
    logger.info(f"")
    if num_generated_tokens_list and any(t > 0 for t in num_generated_tokens_list):
        nonzero_tokens = [t for t in num_generated_tokens_list if t > 0]
        logger.info(f"Average LRM tokens: {sum(nonzero_tokens)/len(nonzero_tokens):.1f}")
    logger.info(f"Total LRM tokens:  {total_lrm_tokens}")
    logger.info(f"")
    logger.info(f"Log saved to: {log_path}")

    # ========== 保存 JSON ==========
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"), exist_ok=True)
    result_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results",
        f"test_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    result_dict = {
        "dataset": args.dataset,
        "model": args.model_path,
        "n_cal": n_cal, "n_test": n_test,
        "alpha_star": best_alpha, "fbe_score": best_fbe,
        "apss": float(compute_apss(construct_prediction_set(test_probs, router.q_hat, data['num_choices']))),
        "llm_acc": float(llm_acc), "router_acc": float(router_acc),
        "trr": float(trr), "u_token": float(u_token),
        "llm_count": len(llm_indices), "lrm_count": len(lrm_indices),
        "tau": args.tau, "beta": args.beta,
        "timestamp": datetime.now().isoformat()
    }
    with open(result_file, 'w') as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON 保存到: {result_file}")
    print(f"  Log 保存到:  {log_path}")

if __name__ == "__main__":
    main()
