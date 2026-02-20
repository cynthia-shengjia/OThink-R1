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
