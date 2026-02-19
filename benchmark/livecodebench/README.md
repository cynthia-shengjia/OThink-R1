# LiveCodeBench Benchmark for OThink-R1

支持标准评测和 DEER early-exit 适配。

## 快速开始

1. 下载数据集: bash benchmark/livecodebench/download_data.sh
2. 标准评测: bash benchmark/livecodebench/run_lcb.sh --model NAME --model_path PATH
3. DEER 评测: bash benchmark/livecodebench/run_deer_lcb.sh --model PATH --threshold 0.95
