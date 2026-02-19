"""
CP-Router x LiveCodeBench Stub

CP-Router 不适用于代码生成任务:
  - CP-Router 基于 MCQA 选项 (A/B/C/D) 的 softmax 概率做 conformal prediction
  - 代码生成是开放式任务，没有固定选项集合
  - 无法计算选项级别的 nonconformity scores

替代方案 (未来工作):
  1. 基于 problem difficulty rating 的路由
  2. 基于 prompt perplexity 的路由
  3. 基于 execution-based 验证的路由 (生成->测试->重路由)
  4. 基于 embedding 相似度的路由

用法:
    uv run python benchmark/livecodebench/cp_router_lcb.py --model_path /path/to/model
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="CP-Router LiveCodeBench Stub")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径 (不会实际使用)")
    parser.add_argument("--gpu_ids", type=str, default="0")
    args = parser.parse_args()

    print("=" * 60)
    print("  CP-Router x LiveCodeBench")
    print("=" * 60)
    print()
    print("  CP-Router 不适用于代码生成任务")
    print()
    print("  原因:")
    print("    - CP-Router 基于 MCQA 选项 logits 做 conformal prediction")
    print("    - 代码生成是开放式任务，没有固定选项集合")
    print("    - 无法计算选项级别的 nonconformity scores")
    print()
    print("  替代方案 (未来工作):")
    print("    1. 基于 problem difficulty rating 的路由")
    print("    2. 基于 prompt perplexity 的路由")
    print("    3. 基于 execution-based 验证的路由")
    print("    4. 基于 embedding 相似度的路由")
    print()
    if args.model_path:
        print(f"  收到 --model_path: {args.model_path}")
        print("  但不会加载或使用该模型。")
        print()
    print("  正常退出 (exit code 0)")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
