"""
CP-Router × LiveCodeBench Stub

CP-Router 是一个基于 Conformal Prediction 的 MCQA 路由器。
它通过分析 LLM 对选项 (A/B/C/D) 的 softmax 概率来构建预测集，
然后根据预测集大小决定路由到 LLM 还是 LRM。

⚠️  CP-Router 不适用于代码生成任务:
  1. 代码生成是开放式生成任务，没有固定选项集合
  2. 无法计算选项级别的 nonconformity scores
  3. 无法构建有意义的预测集

可能的替代方案 (未来工作):
  - 基于 prompt 难度分类的路由 (如 problem rating)
  - 基于首 N 个 token 的 perplexity 做路由决策
  - 基于代码生成 confidence (如 pass@k 估计) 做路由

本文件仅作为占位符，记录不适用原因。
"""
import sys

def main():
    print("=" * 60)
    print("  CP-Router × LiveCodeBench")
    print("=" * 60)
    print()
    print("  ❌ CP-Router 不适用于代码生成任务")
    print()
    print("  原因:")
    print("    - CP-Router 基于 MCQA 选项 logits 做 conformal prediction")
    print("    - 代码生成是开放式任务，没有固定选项集合")
    print("    - 无法计算选项级别的 nonconformity scores")
    print()
    print("  替代方案 (未来工作):")
    print("    - 基于 problem difficulty rating 的路由")
    print("    - 基于 prompt perplexity 的路由")
    print("    - 基于 code generation confidence 的路由")
    print()
    sys.exit(0)

if __name__ == "__main__":
    main()
