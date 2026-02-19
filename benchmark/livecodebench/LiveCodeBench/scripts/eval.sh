# LiveCodeBench
export CUDA_VISIBLE_DEVICES=0  # 仅使用第一个 GPU  
export VLLM_GPU_MEMORY_UTILIZATION=0.2 # 使用 80% 的 GPU 内存  

cd ..
# MODEL="/home/zsj/huggingface-models/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_PATH="/home/zsj/huggingface-models/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_NAME=$(basename $MODEL) 
SCENARIO=codegeneration
MAX_TOKENS=16289
RELEASE_VERSION=release_v5
codegen_n=1
temperature=0.9
n=1
STOP_WORDS="None"
LOCAL_CODE_DATA="/home/zsj/huggingface-models/datasets/livecodebench/code_generation_lite"


source .venv/bin/activate

python -m lcb_runner.runner.main --model $MODEL --local_model_path $MODEL_PATH --local_dataset_path $LOCAL_CODE_DATA --scenario $SCENARIO --max_tokens $MAX_TOKENS --release_version $RELEASE_VERSION --evaluate --codegen_n $codegen_n --n $n --temperature $temperature --stop $STOP_WORDS
# python -m lcb_runner.utils.get_length_lcb --model_name $MODEL --file_path ./output/$MODEL_NAME/Scenario.${SCENARIO}_${codegen_n}_${temperature}.json