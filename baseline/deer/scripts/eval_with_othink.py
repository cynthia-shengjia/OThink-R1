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


def load_tokenizer(model_path):
    """尝试加载 tokenizer，失败则返回 None"""
    if not model_path:
        return None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"  ✅ Tokenizer 已加载: {model_path}")
        return tokenizer
    except Exception as e:
        print(f"  ⚠️  Tokenizer 加载失败: {e}")
        print(f"     将使用 word count 近似估算 token 数")
        return None


def count_tokens(text, tokenizer=None):
    """
    计算文本的 token 数。
    有 tokenizer 时精确计算，否则用 word count 近似。
    """
    if tokenizer is not None:
        return len(tokenizer(text)['input_ids'])
    else:
        # 近似: 英文约 0.75 word/token, 中文约 1.5 char/token
        # 简单用 word count * 1.3 近似
        return int(len(text.split()) * 1.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generation_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='math')
    parser.add_argument('--verify_type', type=str, default='math_verify')
    parser.add_argument('--model_path', type=str, default=None,
                        help="模型路径，用于精确计算 token 数 (可选)")
    args = parser.parse_args()

    # 如果没指定 model_path，尝试从 generation_path 推断
    if args.model_path is None:
        # generation_path 格式: .../outputs/<model_name>/<dataset>/xxx.jsonl
        parts = args.generation_path.split(os.sep)
        for i, p in enumerate(parts):
            if p == 'outputs' and i + 1 < len(parts):
                candidate = os.path.join(PROJECT_ROOT, 'models', parts[i + 1])
                if os.path.isdir(candidate):
                    args.model_path = candidate
                    print(f"  ℹ️  自动检测模型路径: {candidate}")
                break

    print(f"========================================")
    print(f"  DEER → OThink-R1 Verifier 评估")
    print(f"  输入: {args.generation_path}")
    print(f"  数据集: {args.dataset}")
    print(f"  验证方式: {args.verify_type}")
    print(f"  模型路径: {args.model_path or '(未指定, 使用近似估算)'}")
    print(f"========================================")

    file_outputs = read_jsonl(args.generation_path)
    print(f"  加载了 {len(file_outputs)} 条结果")

    # 加载 tokenizer
    tokenizer = load_tokenizer(args.model_path)

    correct_cnt = 0
    token_counts = []
    results = []

    for idx, item in enumerate(tqdm(file_outputs, desc="Evaluating")):
        generated_text = item['generated_responses'][0]
        gold_answer = format_gold_answer(item['gold_answer'])

        # 验证正确性
        verify_result, answers = AnswerVerifier.answer_verify_and_parse(
            content=generated_text,
            solution=gold_answer,
            verify_type=args.verify_type,
        )

        if verify_result:
            correct_cnt += 1

        # 统计 token 数
        num_tokens = count_tokens(generated_text, tokenizer)
        token_counts.append(num_tokens)

        results.append({
            'idx': idx,
            'correct': verify_result,
            'model_answer': str(answers[0]) if answers[0] else None,
            'gold_answer': str(answers[1]) if answers[1] else None,
            'num_tokens': num_tokens,
        })

    total = len(file_outputs)
    acc = correct_cnt / total if total > 0 else 0
    avg_tokens = sum(token_counts) / total if total > 0 else 0
    max_tokens = max(token_counts) if token_counts else 0
    min_tokens = min(token_counts) if token_counts else 0

    # 统计 DEER 特有指标 (如果 jsonl 中有这些字段)
    too_long_cnt = sum(1 for item in file_outputs if item.get('too_long'))
    high_prob_cnt = sum(1 for item in file_outputs if item.get('high_prob'))
    regular_end_cnt = sum(1 for item in file_outputs if item.get('regular_end'))
    thinking_steps_list = [item.get('thinking_steps', 0) for item in file_outputs]
    avg_thinking_steps = sum(thinking_steps_list) / total if total > 0 else 0

    print(f"\n============= Results (OThink-R1 Verifier) =============")
    print(f"  Total:            {total}")
    print(f"  Correct:          {correct_cnt}")
    print(f"  Accuracy:         {acc:.4f}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Avg tokens:       {avg_tokens:.1f}")
    print(f"  Min tokens:       {min_tokens}")
    print(f"  Max tokens:       {max_tokens}")
    print(f"  Token count mode: {'tokenizer (精确)' if tokenizer else 'word count (近似)'}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Too long exits:   {too_long_cnt}")
    print(f"  High prob exits:  {high_prob_cnt}")
    print(f"  Regular ends:     {regular_end_cnt}")
    print(f"  Avg think steps:  {avg_thinking_steps:.2f}")
    print(f"=========================================================")

    # 保存结果
    result_path = args.generation_path.replace('.jsonl', '_othink_eval.json')
    eval_result = {
        'accuracy': acc,
        'correct': correct_cnt,
        'total': total,
        'avg_tokens': avg_tokens,
        'min_tokens': min_tokens,
        'max_tokens': max_tokens,
        'token_count_mode': 'tokenizer' if tokenizer else 'approximate',
        'too_long_exits': too_long_cnt,
        'high_prob_exits': high_prob_cnt,
        'regular_ends': regular_end_cnt,
        'avg_thinking_steps': avg_thinking_steps,
        'dataset': args.dataset,
        'verify_type': args.verify_type,
        'model_path': args.model_path,
        'details': results,
    }

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(eval_result, f, indent=2, ensure_ascii=False)
    print(f"  详细结果保存到: {result_path}")


if __name__ == "__main__":
    main()
