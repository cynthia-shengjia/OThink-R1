```markdown
# OThink-R1 评测指南 (BENCHMARK_README.md)

> 本文档说明如何使用 `othink_cli.py` 在 A100 集群上进行一键部署和评测。

---

## 1. 快速开始 (4 步跑通)

```bash
# ① 激活环境
conda activate othink-r1

# ② 下载数据集 + 模型
python othink_cli.py download-data  --datasets math aime asdiv
python othink_cli.py download-model --model Qwen/Qwen2.5-0.5B-Instruct

# ③ 运行 DEER 评测 (单卡)
python othink_cli.py eval-deer --model Qwen2.5-0.5B-Instruct --datasets math --gpu_ids 0

# ④ 一键全量评测 (8卡并行)
python othink_cli.py eval-all --model Qwen2.5-0.5B-Instruct --gpu_ids 0,1,2,3,4,5,6,7
```

---

## 2. 环境准备

### 2.1 创建环境
```bash
conda env create -n othink-r1 python=3.11
conda activate othink-r1
```

### 2.2 安装依赖 (uv)
```bash
pip install uv
uv sync
```

### 2.3 验证
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
python -c "import vllm; print('vLLM OK')"
```

---

## 3. 数据集下载

### 3.1 下载全部
```bash
python othink_cli.py download-data --datasets all
```

### 3.2 下载指定数据集
```bash
python othink_cli.py download-data --datasets math aime asdiv livecodebench
```

### 3.3 使用自定义镜像
```bash
python othink_cli.py download-data --datasets all --hf_mirror https://hf-mirror.com
```

### 3.4 数据集速查表

| CLI 名称 | HuggingFace 仓库 | 本地路径 | 支持评测 |
|:---------:|:----------------:|:--------:|:--------:|
| `math` | `DigitalLearningGmbH/MATH-lighteval` | `datasets/MATH` | Standard, DEER, CP-Router |
| `aime` | `AI-MO/aimo-validation-aime` | `datasets/AIME` | Standard, DEER, CP-Router |
| `asdiv` | `EleutherAI/asdiv` | `datasets/ASDIV` | Standard, DEER, CP-Router |
| `gsm8k` | `openai/gsm8k` | `datasets/GSM8K` | DEER |
| `gpqa` | `Idavidrein/gpqa` | `datasets/GPQA` | DEER |
| `livecodebench` | `livecodebench/code_generation_lite` | `datasets/livecodebench/...` | LCB |

> 💡 下载完成后会自动调用 `baseline/deer/scripts/convert_hf_to_deer.py` 转换 DEER 格式。

---

## 4. 模型下载

### 4.1 从 HuggingFace 下载
```bash
python othink_cli.py download-model --model Qwen/Qwen2.5-0.5B-Instruct
python othink_cli.py download-model --model Qwen/Qwen2.5-7B-Instruct
python othink_cli.py download-model --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

### 4.2 自定义保存名称
```bash
python othink_cli.py download-model --model Qwen/Qwen2.5-7B-Instruct --name Qwen7B
```

### 4.3 链接本地已有模型
```bash
python othink_cli.py download-model --model Qwen2.5-7B-Instruct --local_path /data/shared/models/Qwen2.5-7B-Instruct
```

### 4.4 幂等性
所有下载操作都是幂等的：如果 `models/<name>` 已存在且非空，会自动跳过。

---

## 5. 评测方式详解

### 5.1 标准评测 (Standard)

调用 `OThinkR1Training/eval.py` + Hydra 配置。

```bash
# 单数据集
python othink_cli.py eval \
    --model Qwen2.5-0.5B-Instruct \
    --datasets aime \
    --gpu_ids 1

# 多数据集并行 (每个数据集分配一张卡)
python othink_cli.py eval \
    --model Qwen2.5-0.5B-Instruct \
    --datasets math aime asdiv \
    --gpu_ids 0,1,2

# 自定义参数
python othink_cli.py eval \
    --model Qwen2.5-0.5B-Instruct \
    --datasets math \
    --gpu_ids 0 \
    --temperature 0.6 \
    --top_p 0.9 \
    --max_tokens 8192
```

### 5.2 DEER 评测 (Dynamic Early Exit)

调用 `baseline/deer/vllm-deer.py`，支持动态提前退出。

```bash
# 基本用法
python othink_cli.py eval-deer \
    --model Qwen2.5-0.5B-Instruct \
    --datasets aime \
    --gpu_ids 1

# 自定义阈值和长度
python othink_cli.py eval-deer \
    --model Qwen2.5-0.5B-Instruct \
    --datasets math \
    --gpu_ids 0 \
    --threshold 0.90 \
    --max_len 8192

# 扫描多个阈值
for t in 0.80 0.85 0.90 0.95 0.99; do
    python othink_cli.py eval-deer \
        --model Qwen2.5-0.5B-Instruct \
        --datasets math \
        --gpu_ids 0 \
        --threshold $t
done
```

**DEER 参数说明:**

| 参数 | 默认值 | 说明 |
|:----:|:------:|:----:|
| `--threshold` | 0.95 | 退出置信度阈值 |
| `--max_len` | 16384 | 最大生成长度 |

### 5.3 CP-Router 评测

调用 `baseline/cp-router/test_cp_router.py`。

```bash
# 仅路由决策 (skip LRM)
python othink_cli.py eval-cp-router \
    --llm_model Qwen2.5-0.5B-Instruct \
    --datasets math aime asdiv \
    --gpu_ids 0 \
    --skip_lrm

# 端到端 (含 LRM 推理)
python othink_cli.py eval-cp-router \
    --llm_model Qwen2.5-0.5B-Instruct \
    --lrm_model Qwen2.5-0.5B-Instruct \
    --datasets aime \
    --gpu_ids 1

# 自定义参数
python othink_cli.py eval-cp-router \
    --llm_model Qwen2.5-14B-Instruct \
    --lrm_model DeepSeek-R1-Distill-Qwen-14B \
    --datasets math \
    --gpu_ids 0 \
    --tau 1 --beta 3.0 --batch_size 8
```

### 5.4 LiveCodeBench 评测

```bash
# 标准模式
python othink_cli.py eval-lcb \
    --model Qwen2.5-0.5B-Instruct \
    --mode standard \
    --gpu_ids 0

# DEER 模式
python othink_cli.py eval-lcb \
    --model Qwen2.5-0.5B-Instruct \
    --mode deer \
    --gpu_ids 0 \
    --threshold 0.95

# 限制题目数 (快速测试)
python othink_cli.py eval-lcb \
    --model Qwen2.5-0.5B-Instruct \
    --mode standard \
    --gpu_ids 0 \
    --max_problems 5
```

---

## 6. 多 GPU 并行评测

### 核心机制

`othink_cli.py` 内置 `GPUScheduler`，维护空闲 GPU 池：

1. 用户指定 `--gpu_ids 0,1,2,3`
2. 脚本将 (method, dataset) 组合生成任务队列
3. 有空闲 GPU 时自动取任务，设置 `CUDA_VISIBLE_DEVICES` 启动子进程
4. 所有任务完成后汇总结果

### 示例: 4 卡并行 DEER

```bash
# 3 个数据集分配到 4 张卡上并行
python othink_cli.py eval-deer \
    --model Qwen2.5-0.5B-Instruct \
    --datasets math aime asdiv \
    --gpu_ids 0,1,2,3
```

输出:
```
📋 共 3 个任务, 可用 GPU: [0, 1, 2, 3]
🖥️  [START] deer-math  →  GPU [0]
🖥️  [START] deer-aime  →  GPU [1]
🖥️  [START] deer-asdiv →  GPU [2]
⏱️  [DONE] ✅ deer-asdiv  耗时 120.3s  rc=0
⏱️  [DONE] ✅ deer-aime   耗时 245.1s  rc=0
⏱️  [DONE] ✅ deer-math   耗时 890.2s  rc=0

========================================================================
🎉 评测结果汇总
------------------------------------------------------------------------
  任务名                                   GPU        耗时     状态
------------------------------------------------------------------------
  deer-aime                                1          245.1s   ✅
  deer-asdiv                               2          120.3s   ✅
  deer-math                                0          890.2s   ✅
========================================================================
🎉 所有任务均已成功完成!
```

---

## 7. 一键全量评测

### 7.1 全方法 + 全数据集

```bash
python othink_cli.py eval-all \
    --model Qwen2.5-0.5B-Instruct \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --methods standard,deer,cp-router,lcb-standard,lcb-deer \
    --datasets math aime asdiv
```

### 7.2 仅 Standard + DEER

```bash
python othink_cli.py eval-all \
    --model Qwen2.5-0.5B-Instruct \
    --gpu_ids 0,1,2,3 \
    --methods standard,deer \
    --datasets math aime asdiv
```

### 7.3 含 CP-Router (需指定 LRM)

```bash
python othink_cli.py eval-all \
    --model Qwen2.5-14B-Instruct \
    --lrm_model DeepSeek-R1-Distill-Qwen-14B \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --methods standard,deer,cp-router \
    --datasets math aime asdiv
```

### 7.4 支持的方法列表

| 方法名 | 说明 | 调用脚本 |
|:------:|:----:|:--------:|
| `standard` | 标准评测 | `OThinkR1Training/eval.py` |
| `deer` | DEER 早退 | `baseline/deer/vllm-deer.py` |
| `cp-router` | CP-Router 路由 | `baseline/cp-router/test_cp_router.py` |
| `lcb-standard` | LCB 标准 | `benchmark/livecodebench/lcb_eval.py` |
| `lcb-deer` | LCB DEER | `benchmark/livecodebench/deer_lcb.py` |

---

## 8. 完整工作流示例

### A100 8卡集群完整评测流程

```bash
# 1. 环境
conda activate othink-r1

# 2. 下载所有数据
python othink_cli.py download-data --datasets all

# 3. 下载模型
python othink_cli.py download-model --model Qwen/Qwen2.5-0.5B-Instruct
python othink_cli.py download-model --model Qwen/Qwen2.5-7B-Instruct
python othink_cli.py download-model --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

# 4. 全量评测 (8卡并行)
python othink_cli.py eval-all \
    --model Qwen2.5-7B-Instruct \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --methods standard,deer \
    --datasets math aime asdiv

# 5. LiveCodeBench 单独跑 (需要独占 GPU)
python othink_cli.py eval-lcb \
    --model Qwen2.5-7B-Instruct \
    --mode standard \
    --gpu_ids 0

python othink_cli.py eval-lcb \
    --model Qwen2.5-7B-Instruct \
    --mode deer \
    --gpu_ids 1
```

---

## 9. 常见问题 (FAQ)

### Q1: 下载报 ConnectionError
确认镜像设置: `--hf_mirror https://hf-mirror.com`

### Q2: CUDA out of memory
- 减小 `--max_tokens` 或 `--max_len`
- 使用更小的模型先测试
- 14B 模型建议 2 卡 tensor parallel

### Q3: DEER 数据不存在
先运行 `python othink_cli.py download-data --datasets <name>`，会自动转换 DEER 格式。

### Q4: 如何只跑前 N 条数据
标准评测: 在 Hydra 配置中设置 slice
DEER: 修改 `baseline/deer/data/<name>/test.jsonl` 截取前 N 行
LCB: 使用 `--max_problems N`

### Q5: 结果保存在哪里
- 标准评测: `OThinkR1Training/save_configs/` 和 `OThinkR1Training/log/`
- DEER: `baseline/deer/outputs/<model_name>/<dataset>/`
- CP-Router: `baseline/cp-router/results/`
- LCB: `results/lcb/<model_name>/`

### Q6: 如何添加新数据集
1. 在 `othink_config.yaml` 的 `datasets` 下添加条目
2. 在 `othink_cli.py` 的 `DATASET_REGISTRY` 中添加对应映射
3. 如需 DEER 支持，在 `convert_hf_to_deer.py` 中添加转换逻辑

### Q7: 如何添加新模型
```bash
# 方法1: 从 HuggingFace 下载
python othink_cli.py download-model --model your-org/your-model

# 方法2: 链接本地模型
python othink_cli.py download-model --model YourModel --local_path /path/to/model
```
如需标准评测支持，还需在 `OThinkR1Training/config/model/` 下创建 Hydra 配置。
```