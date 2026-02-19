#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH="" GPU_IDS="0" MAX_PROBLEMS=0 MAX_MODEL_LEN=4096 MAX_TOKENS=4096
THRESHOLD=0.95 MAX_ROUNDS=5 NUM_PROCESS=12 TIMEOUT=6 NO_EVAL="" OUTPUT_DIR=""
DATASET_PATH="datasets/livecodebench/code_generation_lite"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)    MODEL_PATH="$2";      shift 2 ;;
        --gpu_ids)       GPU_IDS="$2";         shift 2 ;;
        --max_problems)  MAX_PROBLEMS="$2";    shift 2 ;;
        --max_model_len) MAX_MODEL_LEN="$2";   shift 2 ;;
        --max_tokens)    MAX_TOKENS="$2";      shift 2 ;;
        --threshold)     THRESHOLD="$2";       shift 2 ;;
        --max_rounds)    MAX_ROUNDS="$2";      shift 2 ;;
        --num_process)   NUM_PROCESS="$2";     shift 2 ;;
        --timeout)       TIMEOUT="$2";         shift 2 ;;
        --dataset_path)  DATASET_PATH="$2";    shift 2 ;;
        --output_dir)    OUTPUT_DIR="$2";      shift 2 ;;
        --no_eval)       NO_EVAL="--no_eval";  shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

[[ -z "$MODEL_PATH" ]] && { echo "用法: bash run_deer.sh --model_path <path> [--threshold 0.95]"; exit 1; }

PROJ_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJ_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONPATH="benchmark/livecodebench/LiveCodeBench:${PYTHONPATH:-}"

echo "=============================="
echo " DEER LiveCodeBench 评测"
echo " 模型: $MODEL_PATH"
echo " 阈值: $THRESHOLD"
echo " GPU: $GPU_IDS"
echo "=============================="

CMD="uv run python benchmark/livecodebench/deer_lcb.py"
CMD="$CMD --model_path $MODEL_PATH --dataset_path $DATASET_PATH"
CMD="$CMD --max_problems $MAX_PROBLEMS --max_model_len $MAX_MODEL_LEN --max_tokens $MAX_TOKENS"
CMD="$CMD --threshold $THRESHOLD --max_rounds $MAX_ROUNDS"
CMD="$CMD --num_process_evaluate $NUM_PROCESS --timeout $TIMEOUT"
[[ -n "$OUTPUT_DIR" ]] && CMD="$CMD --output_dir $OUTPUT_DIR"
[[ -n "$NO_EVAL" ]] && CMD="$CMD --no_eval"
echo "[CMD] $CMD"
eval $CMD
