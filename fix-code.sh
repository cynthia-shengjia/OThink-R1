#!/bin/bash
# fix_cp_router_cal_ratio.sh
set -e

echo "🔧 为 othink_cli.py 的 eval-cp-router 添加 --cal_ratio 参数 ..."

python3 << 'PYEOF'
with open("othink_cli.py", "r") as f:
    content = f.read()

# 1. 在 eval-cp-router 的 argparse 定义中添加 --cal_ratio
#    插在 --batch_size 那行之后
content = content.replace(
    'ec.add_argument("--batch_size", type=int, default=8)',
    'ec.add_argument("--batch_size", type=int, default=8)\n'
    '    ec.add_argument("--cal_ratio", type=float, default=0.3, help="校准集比例 (默认 0.3)")'
)

# 2. 在 cmd_eval_cp_router 构建 cmd 时透传 --cal_ratio
#    插在 "--beta", str(args.beta), 之后
content = content.replace(
    '"--beta", str(args.beta),\n        ]',
    '"--beta", str(args.beta),\n'
    '            "--cal_ratio", str(args.cal_ratio),\n'
    '        ]'
)

with open("othink_cli.py", "w") as f:
    f.write(content)

print("  ✅ 已添加 --cal_ratio 参数")
PYEOF

# 验证
echo ""
echo "🔍 验证:"
grep -n "cal_ratio" othink_cli.py
echo ""
echo "✅ 完成! 现在可以运行:"
echo '  python othink_cli.py eval-cp-router --llm_model Qwen2.5-0.5B-Instruct --datasets aime --gpu_ids 1 --skip_lrm --cal_ratio 0.1'