# CP-Router Baseline

复现论文: **CP-Router: An Uncertainty-Aware Router Between LLM and LRM** (arXiv:2505.19970)

## 核心思想

CP-Router 使用 Conformal Prediction (CP) 估计 LLM 对每个输入的预测不确定性，
然后根据预测集大小决定是使用 LLM 还是路由到 LRM：

- **预测集小** (size ≤ τ) → LLM 有信心，直接使用 LLM 的回答
- **预测集大** (size > τ) → LLM 不确定，路由到 LRM 进行深度推理

## 快速开始

```bash
# 1. 快速测试 (仅需 LLM, 不需要 LRM)
bash scripts/run_test.sh

# 2. 完整评测
bash scripts/run_full_eval.sh /path/to/llm /path/to/lrm 0

# 3. 论文 Table 1 复现
bash scripts/run_qwen_pairing.sh /path/to/Qwen2.5-14B /path/to/R1-Distill-Qwen-14B 0,1,2,3
bash scripts/run_llama_pairing.sh /path/to/Llama-3.1-8B /path/to/R1-Distill-Llama-8B 0,1
