#!/bin/bash
# setup_deer_datasets.sh
# 将 datasets/ 下的 HuggingFace 数据集转换为 DEER jsonl 格式
# 并提供一键运行 DEER 的入口
#
# 使用:
#   cd ~/ACL-ARR-Jan-Rebuttal/OThink-R1
#   bash setup_deer_datasets.sh

set -e
export CUDA_VISIBLE_DEVICES=1
eval "$(conda shell.bash hook)"
conda activate othink-r1

cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"
DEER_DIR="${PROJECT_ROOT}/baseline/deer"
HF_DATASETS="${PROJECT_ROOT}/datasets"

echo "=========================================="
echo "  转换 HuggingFace 数据集 → DEER 格式"
echo "=========================================="

# 写转换脚本（覆盖旧版）
CONVERT_PY="$(pwd)/baseline/deer/scripts/convert_hf_to_deer.py"
cat > "${CONVERT_PY}" << 'PYEOF'
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
    """MATH (DigitalLearningGmbH/MATH-lighteval) - 只取 test"""
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
            solution = item['solution']
            match = re.search(r'\\boxed\{(.*)\}', solution, re.DOTALL)
            answer = match.group(1) if match else solution
            f.write(json.dumps({
                "problem": item['problem'],
                "answer": answer
            }, ensure_ascii=False) + '\n')
            count += 1
    print(f"  ✅ MATH (test): {count} 条 → {out_path}")
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
                        choices=['math', 'aime', 'asdiv', 'all', 'custom'])
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
PYEOF
echo "✅ convert_hf_to_deer.py 已更新（只转换 test split）"


# 执行转换
echo ""
cd "${DEER_DIR}"
uv run python scripts/convert_hf_to_deer.py \
    --hf_dir "${HF_DATASETS}" \
    --output_dir "${DEER_DIR}/data" \
    --dataset all

# 显示结果
echo ""
echo "=========================================="
echo "  DEER data/ 目录当前内容"
echo "=========================================="
for d in "${DEER_DIR}"/data/*/; do
    name=$(basename "$d")
    if [ -f "${d}/test.jsonl" ]; then
        count=$(wc -l < "${d}/test.jsonl")
        echo "  ${name}: ${count} 条"
    fi
done

echo ""
echo "=========================================="
echo "  ✅ 转换完成！"
echo "=========================================="
echo ""
echo "  现在可以用以下命令运行 DEER:"
echo ""
echo "  # DEER 原始数据集（已有）"
echo "  bash baseline/deer/scripts/run_deer.sh --model ./models/XXX --dataset math"
echo "  bash baseline/deer/scripts/run_deer.sh --model ./models/XXX --dataset aime"
echo "  bash baseline/deer/scripts/run_deer.sh --model ./models/XXX --dataset gsm8k"
echo ""
echo "  # 从 HuggingFace 转换的数据集（新增）"
echo "  bash baseline/deer/scripts/run_deer.sh --model ./models/XXX --dataset math_hf"
echo "  bash baseline/deer/scripts/run_deer.sh --model ./models/XXX --dataset aime_hf"
echo "  bash baseline/deer/scripts/run_deer.sh --model ./models/XXX --dataset asdiv_hf"
echo ""
echo "  # 自定义数据集"
echo "  cd baseline/deer"
echo "  uv run python scripts/convert_hf_to_deer.py \\"
echo "      --dataset custom \\"
echo "      --custom_jsonl /path/to/your/data.jsonl \\"
echo "      --custom_name my_dataset"
echo "  # 然后:"
echo "  bash scripts/run_deer.sh --model /path/to/model --dataset my_dataset"