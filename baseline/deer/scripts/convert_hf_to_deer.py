"""
将 datasets/ 下的 HuggingFace parquet 数据集转换为 DEER jsonl 格式（仅 test split）。
每行: {"problem": "...", "answer": "..."}
用法:
  python convert_hf_to_deer.py --hf_dir ../../datasets --output_dir ./data --dataset all
  python convert_hf_to_deer.py --dataset custom --custom_jsonl /path/to/data.jsonl --custom_name xxx
"""
import os
import sys
import json
import argparse
import re
from datasets import load_dataset
def convert_math(hf_dir, output_dir):
    """MATH-500 (ricdomolm/MATH-500) - 只有 test split"""
    math_dir = os.path.join(hf_dir, "MATH")
    data_sub = os.path.join(math_dir, "data")
    src = data_sub if os.path.isdir(data_sub) else math_dir
    out_dir = os.path.join(output_dir, "math_hf")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test.jsonl")
    dataset = load_dataset(src, split="test")
    count = 0
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            # MATH-500 直接提供 answer 字段，无需从 solution 中提取
            answer = item.get('answer', '')
            if not answer and 'solution' in item:
                solution = item['solution']
                match = re.search(r'\boxed\{(.*)\}', solution, re.DOTALL)
                answer = match.group(1) if match else solution
            f.write(json.dumps({
                "problem": item['problem'],
                "answer": str(item['answer']).strip()
            }, ensure_ascii=False) + '\n')
            count += 1
    print(f"  ✅ MATH-500 (test): {count} 条 → {out_path}")

def convert_aime(hf_dir, output_dir):
    """
    AIME (AI-MO/aimo-validation-aime)
    这个数据集只有 train split，全部都是测试题，所以直接用 train
    """
    aime_dir = os.path.join(hf_dir, "AIME")
    data_sub = os.path.join(aime_dir, "data")
    src = data_sub if os.path.isdir(data_sub) else aime_dir
    out_dir = os.path.join(output_dir, "aime_hf")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test.jsonl")
    # AIME 只有 train split，本身就是评测数据
    dataset = load_dataset(src, split="train")
    count = 0
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps({
                "problem": item['problem'],
                "answer": str(item['answer']).strip()
            }, ensure_ascii=False) + '\n')
            count += 1
    print(f"  ✅ AIME (全部作为test): {count} 条 → {out_path}")
def convert_asdiv(hf_dir, output_dir):
    """
    ASDIV (EleutherAI/asdiv)
    只有 validation split，全部都是测试题
    """
    asdiv_dir = os.path.join(hf_dir, "ASDIV")
    out_dir = os.path.join(output_dir, "asdiv_hf")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test.jsonl")
    dataset = load_dataset(asdiv_dir, "asdiv", split="validation")
    count = 0
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            problem = f"{item['body']} {item['question']}"
            raw = item['answer']
            match = re.match(r'([\d.]+)', raw)
            answer = match.group(1) if match else raw
            f.write(json.dumps({
                "problem": problem,
                "answer": answer
            }, ensure_ascii=False) + '\n')
            count += 1
    print(f"  ✅ ASDIV (validation作为test): {count} 条 → {out_path}")


def convert_gsm8k(hf_dir, output_dir):
    """GSM8K (openai/gsm8k) - test split"""
    gsm8k_dir = os.path.join(hf_dir, "GSM8K")
    out_dir = os.path.join(output_dir, "gsm8k_hf")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test.jsonl")

    dataset = load_dataset(gsm8k_dir, "main", split="test")
    count = 0
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            problem = item['question']
            # answer 格式: "...\n#### 123"
            raw_answer = item['answer']
            match = re.search(r'####\s*(.*)', raw_answer)
            answer = match.group(1).strip() if match else raw_answer
            f.write(json.dumps({
                "problem": problem,
                "answer": answer
            }, ensure_ascii=False) + '\n')
            count += 1
    print(f"  ✅ GSM8K (test): {count} 条 → {out_path}")

def convert_commonsenseqa(hf_dir, output_dir):
    """CommonsenseQA (tau/commonsense_qa) - 多选题, validation 作为 test"""
    csqa_dir = os.path.join(hf_dir, "CommonsenseQA")
    out_dir = os.path.join(output_dir, "commonsenseqa_hf")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test.jsonl")

    # CommonsenseQA 的 test split 没有标签, 用 validation
    dataset = load_dataset(csqa_dir, split="validation")
    count = 0
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            choices = item['choices']
            labels = choices['label']
            texts = choices['text']
            options_str = "\n".join([f"{l}. {t}" for l, t in zip(labels, texts)])
            problem = f"{item['question']}\n{options_str}"
            answer = item['answerKey']
            f.write(json.dumps({
                "problem": problem,
                "answer": answer
            }, ensure_ascii=False) + '\n')
            count += 1
    print(f"  ✅ CommonsenseQA (validation→test): {count} 条 → {out_path}")


def convert_openbookqa(hf_dir, output_dir):
    """OpenBookQA (allenai/openbookqa) - 多选题, test split"""
    obqa_dir = os.path.join(hf_dir, "OpenBookQA")
    out_dir = os.path.join(output_dir, "openbookqa_hf")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test.jsonl")

    dataset = load_dataset(obqa_dir, "main", split="test")
    count = 0
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            choices = item['choices']
            labels = choices['label']
            texts = choices['text']
            options_str = "\n".join([f"{l}. {t}" for l, t in zip(labels, texts)])
            problem = f"{item['question_stem']}\n{options_str}"
            answer = item['answerKey']
            f.write(json.dumps({
                "problem": problem,
                "answer": answer
            }, ensure_ascii=False) + '\n')
            count += 1
    print(f"  ✅ OpenBookQA (test): {count} 条 → {out_path}")

def convert_custom(jsonl_path, output_dir, name):
    """
    通用转换：任意 jsonl 文件。
    自动适配 problem/question/input 和 answer/solution/output 字段。
    """
    out_dir = os.path.join(output_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test.jsonl")
    count = 0
    with open(jsonl_path, 'r', encoding='utf-8') as fin, \
         open(out_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            item = json.loads(line.strip())
            problem = item.get('problem') or item.get('question') or item.get('input', '')
            answer = item.get('answer') or item.get('solution') or item.get('output', '')
            match = re.search(r'\\boxed\{(.*)\}', str(answer), re.DOTALL)
            if match:
                answer = match.group(1)
            fout.write(json.dumps({
                "problem": problem,
                "answer": str(answer).strip()
            }, ensure_ascii=False) + '\n')
            count += 1
    print(f"  ✅ {name}: {count} 条 → {out_path}")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_dir', type=str, default='../../datasets')
    parser.add_argument('--output_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['math', 'aime', 'asdiv', 'gsm8k', 'commonsenseqa', 'openbookqa', 'all', 'custom'])
    parser.add_argument('--custom_jsonl', type=str, default=None)
    parser.add_argument('--custom_name', type=str, default='custom')
    args = parser.parse_args()
    hf_dir = os.path.abspath(args.hf_dir)
    print(f"  HF 数据集目录: {hf_dir}")
    print(f"  输出目录: {os.path.abspath(args.output_dir)}")
    print()
    converters = {
        'math': lambda: convert_math(hf_dir, args.output_dir),
        'aime': lambda: convert_aime(hf_dir, args.output_dir),
        'asdiv': lambda: convert_asdiv(hf_dir, args.output_dir),
        'gsm8k': lambda: convert_gsm8k(hf_dir, args.output_dir),
        'commonsenseqa': lambda: convert_commonsenseqa(hf_dir, args.output_dir),
        'openbookqa': lambda: convert_openbookqa(hf_dir, args.output_dir),
    }
    if args.dataset == 'custom':
        if not args.custom_jsonl:
            print("  ❌ --custom_jsonl 必填")
            sys.exit(1)
        convert_custom(args.custom_jsonl, args.output_dir, args.custom_name)
    elif args.dataset == 'all':
        for name, fn in converters.items():
            try:
                fn()
            except Exception as e:
                print(f"  ⚠️  {name} 失败: {e}")
    else:
        converters[args.dataset]()
if __name__ == "__main__":
    main()
