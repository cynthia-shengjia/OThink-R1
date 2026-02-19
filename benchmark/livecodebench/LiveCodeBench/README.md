How to install
```
conda create -n lcb_env python=3.11  
uv venv --python 3.11
source .venv/bin/activate

uv pip install -e .
```


How to download Data
```
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --repo-type dataset livecodebench/code_generation_lite --local-dir ./datasets/livecodebench/code_generation_lite --local-dir-use-symlinks False --resume-download
```

How to evalaute

# LiveCodeBench
```
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MODEL_NAME=$(basename $MODEL) 
SCENARIO=codegeneration
MAX_TOKENS=16289
RELEASE_VERSION=release_v5
codegen_n=1
temperature=0.9
n=1
STOP_WORDS="None"

cd ./code_eval/LiveCodeBench
source .venv/bin/activate

python -m lcb_runner.runner.main --model $MODEL --scenario $SCENARIO --max_tokens $MAX_TOKENS --release_version $RELEASE_VERSION --evaluate --codegen_n $codegen_n --n $n --temperature $temperature --stop $STOP_WORDS
python -m lcb_runner.utils.get_length_lcb --model_name $MODEL --file_path ./output/$MODEL_NAME/Scenario.${SCENARIO}_${codegen_n}_${temperature}.json
```