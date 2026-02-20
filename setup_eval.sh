#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo ">>> 创建 4 个测评脚本 ..."

# ================================================================
# 脚本1: eval_arm.sh
# ================================================================
cat > eval_arm.sh << 'EOF'
#!/bin/bash
set -e

echo "========== ARM-7B: Standard =========="

echo ">>> ARM-7B: AIME"
python othink_cli.py eval --model ARM-7B --datasets aime --gpu_ids 0

echo ">>> ARM-7B: MATHBench"
python othink_cli.py eval --model ARM-7B --datasets math --gpu_ids 0

echo ">>> ARM-7B: GSM8K"
python othink_cli.py eval --model ARM-7B --datasets gsm8k --gpu_ids 0

echo ">>> ARM-7B: ASDIV"
python othink_cli.py eval --model ARM-7B --datasets asdiv --gpu_ids 0

echo ">>> ARM-7B: CommonsenseQA"
python othink_cli.py eval --model ARM-7B --datasets commonsenseqa --gpu_ids 0

echo ">>> ARM-7B: OpenBookQA"
python othink_cli.py eval --model ARM-7B --datasets openbookqa --gpu_ids 0

echo "========== ARM-7B: LiveCodeBench =========="

echo ">>> ARM-7B: LCB Standard"
python othink_cli.py eval-lcb --model ARM-7B --mode standard --gpu_ids 0

echo ">>> ARM-7B: LCB DEER"
python othink_cli.py eval-lcb --model ARM-7B --mode deer --gpu_ids 0

echo "✅ ARM-7B 全部完成"
EOF
chmod +x eval_arm.sh
echo "  ✅ eval_arm.sh"

# ================================================================
# 脚本2: eval_sb.sh
# ================================================================
cat > eval_sb.sh << 'EOF'
#!/bin/bash
set -e

echo "========== SB-DS7B-alpha-2: Standard =========="

echo ">>> SB: AIME"
python othink_cli.py eval --model SB_DS7B_alpha_2 --datasets aime --gpu_ids 0

echo ">>> SB: MATHBench"
python othink_cli.py eval --model SB_DS7B_alpha_2 --datasets math --gpu_ids 0

echo ">>> SB: GSM8K"
python othink_cli.py eval --model SB_DS7B_alpha_2 --datasets gsm8k --gpu_ids 0

echo ">>> SB: ASDIV"
python othink_cli.py eval --model SB_DS7B_alpha_2 --datasets asdiv --gpu_ids 0

echo ">>> SB: CommonsenseQA"
python othink_cli.py eval --model SB_DS7B_alpha_2 --datasets commonsenseqa --gpu_ids 0

echo ">>> SB: OpenBookQA"
python othink_cli.py eval --model SB_DS7B_alpha_2 --datasets openbookqa --gpu_ids 0

echo "========== SB: LiveCodeBench =========="

echo ">>> SB: LCB Standard"
python othink_cli.py eval-lcb --model SB_DS7B_alpha_2 --mode standard --gpu_ids 0

echo ">>> SB: LCB DEER"
python othink_cli.py eval-lcb --model SB_DS7B_alpha_2 --mode deer --gpu_ids 0

echo "✅ SB 全部完成"
EOF
chmod +x eval_sb.sh
echo "  ✅ eval_sb.sh"

# ================================================================
# 脚本3: eval_deer.sh
# ================================================================
cat > eval_deer.sh << 'EOF'
#!/bin/bash
set -e

BASE=DeepSeek-R1-Distill-Qwen-7B-fixed

echo "========== Deer: ${BASE} =========="

echo ">>> Deer: math"
python othink_cli.py eval-deer --model $BASE --datasets math --gpu_ids 0

echo ">>> Deer: aime"
python othink_cli.py eval-deer --model $BASE --datasets aime --gpu_ids 0

echo ">>> Deer: gsm8k"
python othink_cli.py eval-deer --model $BASE --datasets gsm8k --gpu_ids 0

echo ">>> Deer: asdiv"
python othink_cli.py eval-deer --model $BASE --datasets asdiv --gpu_ids 0

echo ">>> Deer: commonsenseqa"
python othink_cli.py eval-deer --model $BASE --datasets commonsenseqa --gpu_ids 0

echo ">>> Deer: openbookqa"
python othink_cli.py eval-deer --model $BASE --datasets openbookqa --gpu_ids 0

echo "========== Deer: LiveCodeBench =========="

echo ">>> Deer: LCB"
python othink_cli.py eval-lcb --model $BASE --mode deer --gpu_ids 0

echo "✅ Deer 全部完成"
EOF
chmod +x eval_deer.sh
echo "  ✅ eval_deer.sh"

# ================================================================
# 脚本4: eval_cp_router.sh
# ================================================================
cat > eval_cp_router.sh << 'EOF'
#!/bin/bash
set -e

BASE=DeepSeek-R1-Distill-Qwen-7B-fixed

echo "========== CP-Router: ${BASE} =========="

echo ">>> CP-Router: math"
python othink_cli.py eval-cp-router --llm_model $BASE --datasets math --gpu_ids 0 --skip_lrm

echo ">>> CP-Router: aime"
python othink_cli.py eval-cp-router --llm_model $BASE --datasets aime --gpu_ids 0 --skip_lrm

echo ">>> CP-Router: gsm8k"
python othink_cli.py eval-cp-router --llm_model $BASE --datasets gsm8k --gpu_ids 0 --skip_lrm

echo ">>> CP-Router: asdiv"
python othink_cli.py eval-cp-router --llm_model $BASE --datasets asdiv --gpu_ids 0 --skip_lrm

echo ">>> CP-Router: commonsenseqa"
python othink_cli.py eval-cp-router --llm_model $BASE --datasets commonsenseqa --gpu_ids 0 --skip_lrm

echo ">>> CP-Router: openbookqa"
python othink_cli.py eval-cp-router --llm_model $BASE --datasets openbookqa --gpu_ids 0 --skip_lrm

echo "✅ CP-Router 全部完成"
EOF
chmod +x eval_cp_router.sh
echo "  ✅ eval_cp_router.sh"

# ================================================================
echo ""
echo "✅ 完成! 在项目根目录下运行:"
echo "  bash eval_arm.sh"
echo "  bash eval_sb.sh"
echo "  bash eval_deer.sh"
echo "  bash eval_cp_router.sh"