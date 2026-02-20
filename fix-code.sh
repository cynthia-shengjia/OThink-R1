#!/usr/bin/env bash
# ============================================================================
# fix_gsm8k.sh — 补全 GSM8K 的 DEER 数据转换
# 用法: cd ~/ACL-ARR-Jan-Rebuttal/OThink-R1 && bash fix_gsm8k.sh
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
echo "📁 项目根目录: $PROJECT_ROOT"

# ============================================================================
# 1. 在 convert_hf_to_deer.py 中添加 GSM8K 转换函数
# ============================================================================
echo ""
echo "🔧 [1/2] 修改 convert_hf_to_deer.py 添加 GSM8K ..."

cd "$PROJECT_ROOT"

python3 << 'PYEOF'
import re

with open("baseline/deer/scripts/convert_hf_to_deer.py", "r") as f:
    content = f.read()

# 检查是否已经添加过
if "convert_gsm8k" in content:
    print("  ⏭️  GSM8K 转换函数已存在，跳过")
else:
    # 在 convert_commonsenseqa 之前插入 convert_gsm8k
    gsm8k_func = '''
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
            # answer 格式: "...\\n#### 123"
            raw_answer = item['answer']
            match = re.search(r'####\\s*(.*)', raw_answer)
            answer = match.group(1).strip() if match else raw_answer
            f.write(json.dumps({
                "problem": problem,
                "answer": answer
            }, ensure_ascii=False) + '\\n')
            count += 1
    print(f"  ✅ GSM8K (test): {count} 条 → {out_path}")

'''

    # 在 convert_commonsenseqa 之前插入
    content = content.replace(
        "def convert_commonsenseqa(",
        gsm8k_func + "def convert_commonsenseqa("
    )

    # 在 converters 字典中添加 gsm8k
    content = content.replace(
        "'commonsenseqa': lambda: convert_commonsenseqa(hf_dir, args.output_dir),",
        "'gsm8k': lambda: convert_gsm8k(hf_dir, args.output_dir),\n"
        "        'commonsenseqa': lambda: convert_commonsenseqa(hf_dir, args.output_dir),"
    )

    # 扩展 choices 列表
    content = content.replace(
        "choices=['math', 'aime', 'asdiv', 'commonsenseqa', 'openbookqa', 'all', 'custom']",
        "choices=['math', 'aime', 'asdiv', 'gsm8k', 'commonsenseqa', 'openbookqa', 'all', 'custom']"
    )

    with open("baseline/deer/scripts/convert_hf_to_deer.py", "w") as f:
        f.write(content)
    print("  ✅ convert_hf_to_deer.py 已添加 GSM8K")

# 修改 othink_cli.py: gsm8k 的 deer_name 改为 gsm8k_hf, 加入 DEER_CONVERTIBLE
with open("othink_cli.py", "r") as f:
    content = f.read()

# gsm8k deer_name: "gsm8k" → "gsm8k_hf" (使用新转换的统一格式)
content = re.sub(
    r'("gsm8k"\s*:\s*\{[^}]*?"deer_name"\s*:\s*)"gsm8k"',
    r'\1"gsm8k_hf"',
    content
)

# 扩展 DEER_CONVERTIBLE (如果还没有 gsm8k)
if '"gsm8k"' not in content.split('DEER_CONVERTIBLE')[1].split('\n')[0] if 'DEER_CONVERTIBLE' in content else True:
    content = content.replace(
        'DEER_CONVERTIBLE = {"math", "aime", "asdiv", "commonsenseqa", "openbookqa"}',
        'DEER_CONVERTIBLE = {"math", "aime", "asdiv", "gsm8k", "commonsenseqa", "openbookqa"}'
    )

with open("othink_cli.py", "w") as f:
    f.write(content)
print("  ✅ othink_cli.py 已更新 GSM8K deer_name")

PYEOF

# ============================================================================
# 2. 运行全量数据转换
# ============================================================================
echo ""
echo "🔧 [2/2] 运行全量数据转换 ..."

cd "$PROJECT_ROOT"
uv run python baseline/deer/scripts/convert_hf_to_deer.py \
    --hf_dir datasets \
    --output_dir baseline/deer/data \
    --dataset all

# ============================================================================
# 3. 验证全部文件
# ============================================================================
echo ""
echo "=========================================="
echo "  ✅ 全部 DEER 数据验证"
echo "=========================================="

for ds in math_hf aime_hf asdiv_hf gsm8k_hf commonsenseqa_hf openbookqa_hf; do
    f="baseline/deer/data/$ds/test.jsonl"
    if [ -f "$f" ]; then
        cnt=$(wc -l < "$f")
        echo "  ✅ $ds: ${cnt} 条"
    else
        echo "  ❌ $ds: 文件不存在!"
    fi
done

echo ""
echo "=========================================="
echo "  📊 数据集总览"
echo "=========================================="
echo ""
echo "  数据集           DEER格式    标准评测    eval_split   条数"
echo "  ─────────────────────────────────────────────────────────"
echo "  math             math_hf     MATHBench   test         5000"
echo "  aime             aime_hf     AIME        train(全部)  90"
echo "  asdiv            asdiv_hf    ASDIV       validation   2305"
echo "  gsm8k            gsm8k_hf    GSM8K       test         1319"
echo "  commonsenseqa    csqa_hf     CommonsenseQA validation  1221"
echo "  openbookqa       obqa_hf     OpenBookQA  test         500"
echo ""
echo "🎉 全部完成！"