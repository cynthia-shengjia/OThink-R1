#!/bin/bash
set -e

cd ~/ACL-ARR-Jan-Rebuttal/OThink-R1/OThinkR1Training

uv run python eval.py \
    model=Qwen2.5-0.5B-Instruct \
    model.inference.tensor_parallel_size=1 \
    model.inference.gpu_memory_utilization=0.9 \
    +model.inference.repetition_penalty=1.0 \
    model.inference.temperature=0.9 \
    model.inference.top_p=0.95 \
    model.inference.max_tokens=2048 \
    +model.mode="test" \
    data=ASDIV \
    'data.datasets.ASDIV.splits.validation.slice="[:10]"'
