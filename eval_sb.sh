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
