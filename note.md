

### Deer 

Transfer data to target dataset
```
uv run python scripts/convert_hf_to_deer.py \
    --hf_dir ../../datasets \
    --output_dir ./data \
    --dataset all
```

### CP-Router


```
python othink_cli.py eval-cp-router \
    --llm_model Qwen2.5-0.5B-Instruct \
    --datasets aime \
    --gpu_ids 1 \
    --skip_lrm
```

```
python othink_cli.py eval-cp-router \
    --llm_model Qwen2.5-0.5B-Instruct \
    --lrm_model Qwen2.5-0.5B-Instruct \
    --datasets math \
    --gpu_ids 1 \
    --tau 1 --beta 3.0 --batch_size 128
```