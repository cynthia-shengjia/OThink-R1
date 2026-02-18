"""
用 OThink-R1 的 AnswerVerifier 评估 DEER 的输出结果。

用法:
  cd baseline/deer
  uv run python scripts/eval_with_othink.py \
      --generation_path ./outputs/.../greedy_p0.95_....jsonl \
      --dataset math
"""
import os
import sys
import json
import argparse
from tqdm import tqdm

# baseline/deer/scripts/ → ../../.. → PROJECT_ROOT → OThinkR1Training
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
OTHINK_TRAINING_DIR = os.path.join(PROJECT_ROOT, 'OThinkR1Training')
sys.path.insert(0, OTHINK_TRAINING_DIR)

from core.verifier import AnswerVerifier

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def format_gold_answer(answer_str):
    answer_str = str(answer_str).strip()
    if not answer_str.startswith('$'):
        answer_str = f"${answer_str}$"
    return answer_str

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generation_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='math')
    parser.add_argument('--verify_type', type=str, default='math_verify')
    args = parser.parse_args()

    print(f"========================================")
    print(f"  DEER → OThink-R1 Verifier 评估")
    print(f"  输入: {args.generation_path}")
    print(f"  数据集: {args.dataset}")
    print(f"  验证方式: {args.verify_type}")
    print(f"========================================")

    file_outputs = read_jsonl(args.generation_path)
    print(f"  加载了 {len(file_outputs)} 条结果")

    correct_cnt = 0
    results = []

    for idx, item in enumerate(tqdm(file_outputs, desc="Evaluating")):
        generated_text = item['generated_responses'][0]
        gold_answer = format_gold_answer(item['gold_answer'])

        verify_result, answers = AnswerVerifier.answer_verify_and_parse(
            content=generated_text,
            solution=gold_answer,
            verify_type=args.verify_type,
        )

        if verify_result:
            correct_cnt += 1

        results.append({
            'idx': idx,
            'correct': verify_result,
            'model_answer': str(answers[0]) if answers[0] else None,
            'gold_answer': str(answers[1]) if answers[1] else None,
        })

    total = len(file_outputs)
    acc = correct_cnt / total if total > 0 else 0

    print(f"\n============= Results (OThink-R1 Verifier) =============")
    print(f"  Total: {total}")
    print(f"  Correct: {correct_cnt}")
    print(f"  Accuracy: {acc:.4f}")

    result_path = args.generation_path.replace('.jsonl', '_othink_eval.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump({
            'accuracy': acc, 'correct': correct_cnt, 'total': total,
            'dataset': args.dataset, 'verify_type': args.verify_type,
            'details': results
        }, f, indent=2, ensure_ascii=False)
    print(f"  详细结果保存到: {result_path}")

if __name__ == "__main__":
    main()
