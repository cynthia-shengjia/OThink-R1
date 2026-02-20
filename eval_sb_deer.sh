uv run python baseline/deer/scripts/convert_hf_to_deer.py \
    --hf_dir datasets \
    --output_dir baseline/deer/data \
    --dataset commonsenseqa
# 转换 OpenBookQA
uv run python baseline/deer/scripts/convert_hf_to_deer.py \
    --hf_dir datasets \
    --output_dir baseline/deer/data \
    --dataset openbookqa

python othink_cli.py eval-deer \
    --model Qwen2.5-0.5B-Instruct \
    --datasets commonsenseqa \
    --gpu_ids 1 \
    --max_len 2048


 python othink_cli.py eval-deer \
    --model Qwen2.5-0.5B-Instruct \
    --datasets openbookqa \
    --gpu_ids 1 \
    --max_len 2048

