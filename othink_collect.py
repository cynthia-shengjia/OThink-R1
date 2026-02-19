#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OThink-R1 结果收集器 (othink_collect.py)
=========================================
从各评测脚本的原始输出中提取 accuracy 和 avg_tokens，
统一写入:
  log/<model_size>/<model_name>/log.txt        — 人类可读的汇总日志
  log/<model_size>/<model_name>/results.json   — 结构化 JSON (每个 dataset/method 的指标)

放置于项目根目录: OThink-R1/othink_collect.py

用法:
  # 收集指定模型的所有结果
  python othink_collect.py --model Qwen2.5-0.5B-Instruct --model_size 0.5B

  # 收集并指定方法
  python othink_collect.py --model Qwen2.5-0.5B-Instruct --model_size 0.5B --methods standard deer

  # 收集多个模型
  python othink_collect.py --model Qwen2.5-0.5B-Instruct --model_size 0.5B
  python othink_collect.py --model Qwen2.5-7B-Instruct --model_size 7B
  python othink_collect.py --model DeepSeek-R1-Distill-Qwen-7B --model_size 7B
"""

import argparse
import json
import os
import re
import sys
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

PROJECT_ROOT = Path(__file__).resolve().parent


# ──────────────────────────────────────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────────────────────────────────────
class EvalResult:
    """单次评测的结果"""
    def __init__(self, method: str, dataset: str):
        self.method = method          # standard / deer / cp-router / lcb-standard / lcb-deer
        self.dataset = dataset        # math / aime / asdiv / gsm8k / gpqa / livecodebench
        self.accuracy: Optional[float] = None
        self.avg_tokens: Optional[float] = None
        self.total_samples: Optional[int] = None
        self.correct_count: Optional[int] = None
        self.extra: Dict[str, Any] = {}  # 方法特有的指标 (如 DEER 的 threshold, CP-Router 的 trr)
        self.source_file: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            "method": self.method,
            "dataset": self.dataset,
            "accuracy": self.accuracy,
            "avg_tokens": self.avg_tokens,
            "total_samples": self.total_samples,
            "correct_count": self.correct_count,
        }
        d.update(self.extra)
        if self.source_file:
            d["source_file"] = self.source_file
        return d


# ──────────────────────────────────────────────────────────────────────────────
# 解析器: Standard (OThinkR1Training log)
# ──────────────────────────────────────────────────────────────────────────────
def parse_standard_logs(model_name: str) -> List[EvalResult]:
    """
    解析 OThinkR1Training/log/<dataset>/<size>/test/*.log
    
    日志格式 (来自 eval_utils.py write_responses):
      ============= Summary =============
      Total cases: 500
      Average tokens: 1234.567
      Correct rate: 0.680
    """
    results = []
    log_base = PROJECT_ROOT / "OThinkR1Training" / "log"
    
    if not log_base.exists():
        return results

    # 遍历所有数据集目录
    for dataset_dir in log_base.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name  # e.g. "AIME", "MATHBench", "ASDIV"

        # 遍历所有 size 目录
        for size_dir in dataset_dir.iterdir():
            if not size_dir.is_dir():
                continue

            # 遍历 test 目录下的 log 文件
            test_dir = size_dir / "test"
            if not test_dir.exists():
                continue

            for log_file in test_dir.glob("*.log"):
                # 检查是否属于目标模型
                # 文件名格式: models_Qwen2.50.5BInstruct-parallel-1-tmp-0.9-topp-0.95.log
                model_clean = model_name.replace("-", "").replace(".", "")
                if model_clean.lower() not in log_file.name.replace("-", "").replace(".", "").lower():
                    continue

                content = log_file.read_text(encoding="utf-8", errors="ignore")

                r = EvalResult(method="standard", dataset=_normalize_dataset_name(dataset_name))
                r.source_file = str(log_file.relative_to(PROJECT_ROOT))

                # 提取 Total cases
                m = re.search(r"Total cases:\s*(\d+)", content)
                if m:
                    r.total_samples = int(m.group(1))

                # 提取 Average tokens
                m = re.search(r"Average tokens:\s*([\d.]+)", content)
                if m:
                    r.avg_tokens = float(m.group(1))

                # 提取 Correct rate
                m = re.search(r"Correct rate:\s*([\d.]+)", content)
                if m:
                    r.accuracy = float(m.group(1))

                if r.total_samples and r.accuracy is not None:
                    r.correct_count = int(round(r.accuracy * r.total_samples))

                results.append(r)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 解析器: DEER
# ──────────────────────────────────────────────────────────────────────────────
def parse_deer_results(model_name: str) -> List[EvalResult]:
    """
    解析 DEER 输出:
    1. baseline/deer/outputs/<model_name>/<dataset>/*.jsonl  — 原始推理结果
    2. baseline/deer/outputs/<model_name>/<dataset>/*_othink_eval.json — OThink verifier 评估
    
    jsonl 每行: {"question": ..., "generated_responses": [...], "gold_answer": ..., 
                  "too_long": 0/1, "thinking_steps": N, "high_prob": 0/1, ...}
    
    _othink_eval.json: {"accuracy": 0.68, "correct": 34, "total": 50, ...}
    """
    results = []
    deer_outputs = PROJECT_ROOT / "baseline" / "deer" / "outputs"

    if not deer_outputs.exists():
        return results

    # 查找模型目录
    model_dir = deer_outputs / model_name
    if not model_dir.exists():
        # 尝试模糊匹配
        for d in deer_outputs.iterdir():
            if d.is_dir() and model_name.replace("-", "") in d.name.replace("-", ""):
                model_dir = d
                break
        if not model_dir.exists():
            return results

    for dataset_dir in model_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name

        # 优先使用 _othink_eval.json
        eval_jsons = list(dataset_dir.glob("*_othink_eval.json"))
        jsonl_files = list(dataset_dir.glob("*.jsonl"))

        if eval_jsons:
            # 取最新的
            eval_json = sorted(eval_jsons, key=lambda f: f.stat().st_mtime)[-1]
            data = json.loads(eval_json.read_text())

            r = EvalResult(method="deer", dataset=_normalize_dataset_name(dataset_name))
            r.accuracy = data.get("accuracy")
            r.total_samples = data.get("total")
            r.correct_count = data.get("correct")
            r.source_file = str(eval_json.relative_to(PROJECT_ROOT))

            # 从对应的 jsonl 提取 avg_tokens
            if jsonl_files:
                jsonl_file = sorted(jsonl_files, key=lambda f: f.stat().st_mtime)[-1]
                r.avg_tokens = _calc_avg_tokens_from_deer_jsonl(jsonl_file)
                r.extra["deer_stats"] = _calc_deer_stats(jsonl_file)

            results.append(r)

        elif jsonl_files:
            # 没有 eval json，直接从 jsonl 解析
            jsonl_file = sorted(jsonl_files, key=lambda f: f.stat().st_mtime)[-1]
            r = EvalResult(method="deer", dataset=_normalize_dataset_name(dataset_name))
            r.source_file = str(jsonl_file.relative_to(PROJECT_ROOT))
            r.avg_tokens = _calc_avg_tokens_from_deer_jsonl(jsonl_file)
            stats = _calc_deer_stats(jsonl_file)
            r.extra["deer_stats"] = stats
            # 如果 jsonl 中有 is_correct 字段
            if "accuracy" in stats:
                r.accuracy = stats["accuracy"]
                r.total_samples = stats["total"]
                r.correct_count = stats["correct"]
            results.append(r)

    return results


def _calc_avg_tokens_from_deer_jsonl(jsonl_path: Path) -> Optional[float]:
    """从 DEER jsonl 计算平均 token 数 (用 response 字符数近似, 或用 thinking_tokens 字段)"""
    try:
        total_tokens = 0
        count = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                # 优先用 total_tokens 字段
                if "total_tokens" in item:
                    total_tokens += item["total_tokens"]
                elif "generated_responses" in item and item["generated_responses"]:
                    # 用 response 长度的 token 近似 (中文~1.5字/token, 英文~0.75词/token)
                    resp = item["generated_responses"][0]
                    # 粗略估计: 按空格分词数 * 1.3
                    total_tokens += len(resp.split()) * 1.3
                count += 1
        return total_tokens / count if count > 0 else None
    except Exception:
        return None


def _calc_deer_stats(jsonl_path: Path) -> dict:
    """从 DEER jsonl 计算统计信息"""
    stats = {"total": 0, "correct": 0, "too_long": 0, "high_prob": 0, "regular_end": 0}
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                stats["total"] += 1
                if item.get("is_correct"):
                    stats["correct"] += 1
                if item.get("too_long"):
                    stats["too_long"] += 1
                if item.get("high_prob"):
                    stats["high_prob"] += 1
                if item.get("regular_end"):
                    stats["regular_end"] += 1
        if stats["total"] > 0:
            stats["accuracy"] = stats["correct"] / stats["total"]
    except Exception:
        pass
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# 解析器: CP-Router
# ──────────────────────────────────────────────────────────────────────────────
def parse_cp_router_results(model_name: str) -> List[EvalResult]:
    """
    解析 baseline/cp-router/results/test_<dataset>_*.json
    
    JSON 格式:
    {
      "dataset": "asdiv",
      "model": "...",
      "alpha_star": 0.15,
      "apss": 1.23,
      "llm_acc": 0.65,
      "router_acc": 0.70,
      "trr": 0.45,
      "u_token": 0.09,
      "llm_count": 14,
      "lrm_count": 6,
      ...
    }
    """
    results = []
    cp_results_dir = PROJECT_ROOT / "baseline" / "cp-router" / "results"

    if not cp_results_dir.exists():
        return results

    for json_file in cp_results_dir.glob("test_*.json"):
        try:
            data = json.loads(json_file.read_text())
        except Exception:
            continue

        # 检查是否属于目标模型
        model_field = data.get("model", "")
        if model_name not in model_field and model_name.replace("-", "") not in model_field.replace("-", ""):
            continue

        dataset = data.get("dataset", "unknown")
        r = EvalResult(method="cp-router", dataset=_normalize_dataset_name(dataset))
        r.accuracy = data.get("router_acc")
        r.total_samples = data.get("n_test")
        r.source_file = str(json_file.relative_to(PROJECT_ROOT))

        # CP-Router 特有指标
        r.extra["llm_acc"] = data.get("llm_acc")
        r.extra["trr"] = data.get("trr")
        r.extra["u_token"] = data.get("u_token")
        r.extra["apss"] = data.get("apss")
        r.extra["alpha_star"] = data.get("alpha_star")
        r.extra["llm_count"] = data.get("llm_count")
        r.extra["lrm_count"] = data.get("lrm_count")

        results.append(r)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 解析器: LiveCodeBench
# ──────────────────────────────────────────────────────────────────────────────
def parse_lcb_results(model_name: str) -> List[EvalResult]:
    """
    解析 results/lcb/<model_name>/standard/metrics.json
    和    results/lcb/<model_name>/deer_t*/metrics.json
    
    metrics.json 格式 (来自 lcb_eval.py):
    {
      "pass@1": 0.123,
      "easy_pass@1": 0.456,
      "medium_pass@1": 0.234,
      "hard_pass@1": 0.012,
      ...
    }
    """
    results = []
    lcb_base = PROJECT_ROOT / "results" / "lcb"

    if not lcb_base.exists():
        return results

    model_dir = lcb_base / model_name
    if not model_dir.exists():
        # 模糊匹配
        for d in lcb_base.iterdir():
            if d.is_dir() and model_name.replace("-", "") in d.name.replace("-", ""):
                model_dir = d
                break
        if not model_dir.exists():
            return results

    for sub_dir in model_dir.iterdir():
        if not sub_dir.is_dir():
            continue

        metrics_file = sub_dir / "metrics.json"
        if not metrics_file.exists():
            continue

        try:
            data = json.loads(metrics_file.read_text())
        except Exception:
            continue

        # 判断是 standard 还是 deer
        if "standard" in sub_dir.name:
            method = "lcb-standard"
        elif "deer" in sub_dir.name:
            method = "lcb-deer"
        else:
            method = f"lcb-{sub_dir.name}"

        r = EvalResult(method=method, dataset="livecodebench")
        r.accuracy = data.get("pass@1")
        r.source_file = str(metrics_file.relative_to(PROJECT_ROOT))

        # LCB 特有指标
        for key in ["easy_pass@1", "medium_pass@1", "hard_pass@1"]:
            if key in data:
                r.extra[key] = data[key]

        # 从 generation_results.json 或 deer_results.json 提取 token 信息
        gen_file = sub_dir / "generation_results.json"
        deer_file = sub_dir / "deer_results.json"
        result_file = gen_file if gen_file.exists() else (deer_file if deer_file.exists() else None)

        if result_file:
            try:
                gen_data = json.loads(result_file.read_text())
                total_tokens = 0
                count = 0
                for item in gen_data:
                    if "output_list" in item and item["output_list"]:
                        total_tokens += len(item["output_list"][0].split()) * 1.3
                        count += 1
                    if "deer_rounds" in item:
                        r.extra.setdefault("deer_stats", {})
                        r.extra["deer_stats"]["avg_rounds"] = r.extra["deer_stats"].get("avg_rounds", 0) + item["deer_rounds"]
                if count > 0:
                    r.avg_tokens = total_tokens / count
                    if "deer_stats" in r.extra:
                        r.extra["deer_stats"]["avg_rounds"] /= count
                r.total_samples = count
            except Exception:
                pass

        results.append(r)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────────────────────
def _normalize_dataset_name(name: str) -> str:
    """统一数据集名称"""
    mapping = {
        "MATHBench": "math",
        "MATH": "math",
        "math_hf": "math",
        "AIME": "aime",
        "aime_hf": "aime",
        "ASDIV": "asdiv",
        "asdiv_hf": "asdiv",
        "GSM8K": "gsm8k",
        "gsm8k": "gsm8k",
        "GPQA": "gpqa",
        "gpqa": "gpqa",
        "math_test10": "math(test10)",
    }
    return mapping.get(name, name.lower())


# ──────────────────────────────────────────────────────────────────────────────
# 主收集逻辑
# ──────────────────────────────────────────────────────────────────────────────
def collect_all(model_name: str, model_size: str,
                methods: Optional[List[str]] = None) -> List[EvalResult]:
    """
    收集指定模型的所有评测结果。
    
    Args:
        model_name: 模型名称 (如 Qwen2.5-0.5B-Instruct)
        model_size: 模型大小 (如 0.5B, 7B, 14B)
        methods: 要收集的方法列表, None=全部
    
    Returns:
        所有评测结果列表
    """
    all_results = []
    
    collectors = {
        "standard": parse_standard_logs,
        "deer": parse_deer_results,
        "cp-router": parse_cp_router_results,
        "lcb-standard": parse_lcb_results,
        "lcb-deer": parse_lcb_results,
    }

    target_methods = methods or list(collectors.keys())

    for method in target_methods:
        if method in ("lcb-standard", "lcb-deer"):
            # LCB 解析器统一处理
            if method == "lcb-deer" and "lcb-standard" in target_methods:
                continue  # 避免重复调用
            parser = collectors[method]
        else:
            parser = collectors.get(method)

        if parser is None:
            print(f"⚠️  未知方法: {method}, 跳过")
            continue

        try:
            results = parser(model_name)
            if methods:
                results = [r for r in results if r.method in target_methods]
            all_results.extend(results)
        except Exception as e:
            print(f"⚠️  解析 {method} 结果失败: {e}")

    return all_results


def write_outputs(model_name: str, model_size: str, results: List[EvalResult]):
    """
    将结果写入:
      log/<model_size>/<model_name>/log.txt
      log/<model_size>/<model_name>/results.json
    """
    output_dir = PROJECT_ROOT / "log" / model_size / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "log.txt"
    json_path = output_dir / "results.json"

    # ── 写 log.txt ──
    lines = []
    lines.append("=" * 70)
    lines.append(f"  OThink-R1 评测结果汇总")
    lines.append(f"  模型: {model_name}")
    lines.append(f"  大小: {model_size}")
    lines.append(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")

    # 按方法分组
    by_method: Dict[str, List[EvalResult]] = {}
    for r in results:
        by_method.setdefault(r.method, []).append(r)

    for method, method_results in sorted(by_method.items()):
        lines.append(f"── {method.upper()} {'─' * (60 - len(method))}")
        lines.append(f"  {'数据集':<15} {'Accuracy':>10} {'Avg Tokens':>12} {'Samples':>10}")
        lines.append(f"  {'-'*15} {'-'*10} {'-'*12} {'-'*10}")

        for r in sorted(method_results, key=lambda x: x.dataset):
            acc_str = f"{r.accuracy:.4f}" if r.accuracy is not None else "N/A"
            tok_str = f"{r.avg_tokens:.1f}" if r.avg_tokens is not None else "N/A"
            sam_str = str(r.total_samples) if r.total_samples is not None else "N/A"
            lines.append(f"  {r.dataset:<15} {acc_str:>10} {tok_str:>12} {sam_str:>10}")

            # 额外指标
            if r.extra:
                for k, v in r.extra.items():
                    if k == "deer_stats":
                        continue
                    if isinstance(v, float):
                        lines.append(f"    {k}: {v:.4f}")
                    elif v is not None:
                        lines.append(f"    {k}: {v}")

        lines.append("")

    # 总结
    lines.append("=" * 70)
    lines.append(f"  共 {len(results)} 条评测记录")
    lines.append("=" * 70)

    log_content = "\n".join(lines)
    log_path.write_text(log_content, encoding="utf-8")
    print(f"✅ 日志已写入: {log_path}")

    # ── 写 results.json ──
    json_data = {
        "model_name": model_name,
        "model_size": model_size,
        "collected_at": datetime.now().isoformat(),
        "results": [r.to_dict() for r in results],
    }
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ JSON 已写入: {json_path}")

    # 打印到终端
    print()
    print(log_content)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="🔍 OThink-R1 结果收集器 — 从各评测输出中提取 accuracy + tokens",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python othink_collect.py --model Qwen2.5-0.5B-Instruct --model_size 0.5B
  python othink_collect.py --model Qwen2.5-7B-Instruct --model_size 7B --methods standard deer
  python othink_collect.py --model DeepSeek-R1-Distill-Qwen-7B --model_size 7B --methods deer cp-router
        """,
    )
    parser.add_argument("--model", required=True, help="模型名称 (如 Qwen2.5-0.5B-Instruct)")
    parser.add_argument("--model_size", required=True, help="模型大小 (如 0.5B, 1.5B, 7B, 14B)")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="要收集的方法: standard deer cp-router lcb-standard lcb-deer (默认全部)")

    args = parser.parse_args()

    print(f"🔍 收集模型 [{args.model}] ({args.model_size}) 的评测结果...")
    print()

    results = collect_all(args.model, args.model_size, args.methods)

    if not results:
        print("⚠️  未找到任何评测结果!")
        print("   请确认:")
        print(f"   - 标准评测日志: OThinkR1Training/log/*/")
        print(f"   - DEER 输出: baseline/deer/outputs/{args.model}/")
        print(f"   - CP-Router 结果: baseline/cp-router/results/")
        print(f"   - LCB 结果: results/lcb/{args.model}/")
        sys.exit(1)

    write_outputs(args.model, args.model_size, results)


if __name__ == "__main__":
    main()