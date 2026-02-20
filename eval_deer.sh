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
