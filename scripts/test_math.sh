#!/bin/bash
set -e
cd ..
cd ./OThinkR1Training

echo "=========================================="
echo "  测试 MATH 数据集推理 (前10条)"
echo "=========================================="

uv run python eval.py \
    model=Qwen2.5-0.5B-Instruct \
    model.inference.tensor_parallel_size=1 \
    model.inference.gpu_memory_utilization=0.9 \
    +model.inference.repetition_penalty=1.0 \
    model.inference.temperature=0.9 \
    model.inference.top_p=0.95 \
    model.inference.max_tokens=2048 \
    +model.mode="test" \
    data=MATHBench \
    'data.datasets.MATHBench.splits.test.slice="[:10]"'

echo ""
echo "  ✅ MATH 测试完成"
