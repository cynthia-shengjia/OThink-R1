"""
将 OThink-R1 的 HuggingFace datasets 转换为 DEER 需要的 jsonl 格式。

用法:
  cd baseline/deer
  uv run python scripts/convert_data.py --dataset math
  uv run python scripts/convert_data.py --dataset all
"""
import os
import sys
import json
import argparse
import re

def convert_math(input_dir, output_dir):
    from datasets import load_dataset
    os.makedirs(f"{output_dir}/math", exist_ok=True)
    # MATH-lighteval 本地目录加载，需要指定 data_dir 或直接加载
    # 数据在 input_dir/data/ 下有 parquet 文件
    dataset = load_dataset(input_dir, split="test")
    output_path = f"{output_dir}/math/test.jsonl"
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            problem = item['problem']
            solution = item['solution']
            match = re.search(r'\\boxed\{(.*)\}', solution, re.DOTALL)
            if match:
                answer = match.group(1)
            else:
                answer = solution
            entry = {"problem": problem, "answer": answer}
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            count += 1
    print(f"  ✅ MATH: {count} 条 → {output_path}")

def convert_aime(input_dir, output_dir):
    from datasets import load_dataset
    os.makedirs(f"{output_dir}/aime", exist_ok=True)
    dataset = load_dataset(input_dir, split="train")
    output_path = f"{output_dir}/aime/test.jsonl"
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            entry = {"problem": item['problem'], "answer": str(item['answer']).strip()}
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            count += 1
    print(f"  ✅ AIME: {count} 条 → {output_path}")

def convert_asdiv(input_dir, output_dir):
    from datasets import load_dataset
    os.makedirs(f"{output_dir}/asdiv", exist_ok=True)
    dataset = load_dataset(input_dir, "asdiv", split="validation")
    output_path = f"{output_dir}/asdiv/test.jsonl"
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            problem = f"{item['body']} {item['question']}"
            raw_answer = item['answer']
            match = re.match(r'([\d.]+)', raw_answer)
            answer = match.group(1) if match else raw_answer
            entry = {"problem": problem, "answer": answer}
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            count += 1
    print(f"  ✅ ASDIV: {count} 条 → {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['math', 'aime', 'asdiv', 'all'])
    parser.add_argument('--output_dir', type=str, default='./data')
    parser.add_argument('--othink_dataset_dir', type=str, default=None)
    args = parser.parse_args()

    # 关键：正确定位数据集目录
    # 脚本在 baseline/deer/scripts/，项目根在 ../../..
    # 数据集在 PROJECT_ROOT/datasets/
    if args.othink_dataset_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
        args.othink_dataset_dir = os.path.join(project_root, 'datasets')

    print(f"  数据集源目录: {args.othink_dataset_dir}")

    if not os.path.exists(args.othink_dataset_dir):
        print(f"  ❌ 数据集目录不存在: {args.othink_dataset_dir}")
        sys.exit(1)

    converters = {
        'math': lambda: convert_math(
            os.path.join(args.othink_dataset_dir, 'MATH'),
            args.output_dir),
        'aime': lambda: convert_aime(
            os.path.join(args.othink_dataset_dir, 'AIME'),
            args.output_dir),
        'asdiv': lambda: convert_asdiv(
            os.path.join(args.othink_dataset_dir, 'ASDIV'),
            args.output_dir),
    }

    if args.dataset == 'all':
        for name, converter in converters.items():
            try:
                converter()
            except Exception as e:
                print(f"  ⚠️  {name} 转换失败: {e}")
    else:
        converters[args.dataset]()

if __name__ == "__main__":
    main()
