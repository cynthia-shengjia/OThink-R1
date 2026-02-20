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
