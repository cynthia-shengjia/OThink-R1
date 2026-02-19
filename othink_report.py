#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OThink-R1 报表生成器 (othink_report.py)
=========================================
读取 log/<model_size>/<model_name>/results.json，
汇总多个模型的结果，生成:
  1. 终端彩色表格
  2. Markdown 表格 (可直接粘贴到论文/README)
  3. CSV 文件 (可用 Excel 打开)
  4. LaTeX 表格 (可直接用于论文)

放置于项目根目录: OThink-R1/othink_report.py

用法:
  # 自动扫描 log/ 下所有模型，生成表格
  python othink_report.py

  # 指定模型
  python othink_report.py --models Qwen2.5-0.5B-Instruct Qwen2.5-7B-Instruct

  # 指定方法
  python othink_report.py --method deer

  # 输出到文件
  python othink_report.py --output report.md --format markdown
  python othink_report.py --output report.csv --format csv
  python othink_report.py --output report.tex --format latex
"""

import argparse
import csv
import io
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent


# ──────────────────────────────────────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────────────────────────────────────
def load_all_results(model_names: Optional[List[str]] = None) -> Dict[str, dict]:
    """
    扫描 log/ 目录，加载所有 results.json。
    
    Returns:
        { "0.5B/Qwen2.5-0.5B-Instruct": { json_data }, ... }
    """
    log_dir = PROJECT_ROOT / "log"
    if not log_dir.exists():
        print("❌ log/ 目录不存在，请先运行 othink_collect.py")
        sys.exit(1)

    all_data = {}

    for size_dir in sorted(log_dir.iterdir()):
        if not size_dir.is_dir():
            continue
        model_size = size_dir.name

        for model_dir in sorted(size_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name

            # 过滤
            if model_names and model_name not in model_names:
                continue

            json_path = model_dir / "results.json"
            if not json_path.exists():
                continue

            try:
                data = json.loads(json_path.read_text())
                key = f"{model_size}/{model_name}"
                all_data[key] = data
            except Exception as e:
                print(f"⚠️  读取 {json_path} 失败: {e}")

    return all_data


def build_table_data(all_data: Dict[str, dict],
                     method_filter: Optional[str] = None,
                     metric: str = "accuracy") -> Tuple[List[str], List[str], List[List[str]]]:
    """
    构建表格数据。
    
    Args:
        all_data: load_all_results 的返回值
        method_filter: 只看某个方法 (如 "deer"), None=所有方法合并
        metric: "accuracy" 或 "avg_tokens"
    
    Returns:
        (models, columns, rows)
        models: 模型名列表 (行标签)
        columns: 列标签 (dataset 或 method-dataset)
        rows: 二维字符串列表
    """
    # 收集所有 (method, dataset) 组合和所有模型
    all_columns = set()
    models_info = []  # [(model_size, model_name, key)]

    for key, data in all_data.items():
        model_size = data.get("model_size", "?")
        model_name = data.get("model_name", key)
        models_info.append((model_size, model_name, key))

        for r in data.get("results", []):
            m = r.get("method", "?")
            d = r.get("dataset", "?")
            if method_filter and m != method_filter:
                continue
            if method_filter:
                col = d  # 只看一个方法时，列名只用 dataset
            else:
                col = f"{m}/{d}"
            all_columns.add(col)

    # 排序
    columns = sorted(all_columns)
    models_info.sort(key=lambda x: (x[0], x[1]))

    # 构建行
    rows = []
    model_labels = []
    for model_size, model_name, key in models_info:
        data = all_data[key]
        label = f"{model_name} ({model_size})"
        model_labels.append(label)

        # 建立 (method, dataset) → result 的映射
        result_map = {}
        for r in data.get("results", []):
            m = r.get("method", "?")
            d = r.get("dataset", "?")
            if method_filter:
                col_key = d
            else:
                col_key = f"{m}/{d}"
            result_map[col_key] = r

        row = []
        for col in columns:
            r = result_map.get(col)
            if r is None:
                row.append("-")
            else:
                val = r.get(metric)
                if val is None:
                    row.append("-")
                elif metric == "accuracy":
                    row.append(f"{val:.4f}")
                elif metric == "avg_tokens":
                    row.append(f"{val:.1f}")
                else:
                    row.append(str(val))
        rows.append(row)

    return model_labels, columns, rows


# ──────────────────────────────────────────────────────────────────────────────
# 输出格式
# ──────────────────────────────────────────────────────────────────────────────
def format_terminal(model_labels: List[str], columns: List[str],
                    rows: List[List[str]], title: str = "") -> str:
    """终端表格 (带颜色)"""
    lines = []

    # 计算列宽
    col_widths = [max(len(label) for label in model_labels) + 2]  # 模型列
    for i, col in enumerate(columns):
        w = max(len(col), max(len(rows[j][i]) for j in range(len(rows)))) + 2
        col_widths.append(w)

    total_width = sum(col_widths) + len(col_widths) + 1

    if title:
        lines.append("")
        lines.append("=" * total_width)
        lines.append(f"  📊 {title}")
        lines.append("=" * total_width)

    # 表头
    header = "│" + f"{'模型':<{col_widths[0]}}" + "│"
    for i, col in enumerate(columns):
        header += f"{col:>{col_widths[i+1]}}" + "│"
    lines.append("┌" + "┬".join("─" * w for w in col_widths) + "┐")
    lines.append(header)
    lines.append("├" + "┼".join("─" * w for w in col_widths) + "┤")

    # 数据行
    for label, row in zip(model_labels, rows):
        line = "│" + f"{label:<{col_widths[0]}}" + "│"
        for i, val in enumerate(row):
            line += f"{val:>{col_widths[i+1]}}" + "│"
        lines.append(line)

    lines.append("└" + "┴".join("─" * w for w in col_widths) + "┘")
    lines.append("")

    return "\n".join(lines)


def format_markdown(model_labels: List[str], columns: List[str],
                    rows: List[List[str]], title: str = "") -> str:
    """Markdown 表格"""
    lines = []
    if title:
        lines.append(f"## {title}")
        lines.append("")

    # 表头
    header = "| 模型 | " + " | ".join(columns) + " |"
    separator = "|:---:|" + "|:---:" * len(columns) + "|"
    lines.append(header)
    lines.append(separator)

    # 数据行
    for label, row in zip(model_labels, rows):
        line = f"| {label} | " + " | ".join(row) + " |"
        lines.append(line)

    lines.append("")
    lines.append(f"*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    return "\n".join(lines)


def format_csv(model_labels: List[str], columns: List[str],
               rows: List[List[str]]) -> str:
    """CSV 格式"""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["模型"] + columns)
    for label, row in zip(model_labels, rows):
        writer.writerow([label] + row)
    return output.getvalue()


def format_latex(model_labels: List[str], columns: List[str],
                 rows: List[List[str]], title: str = "") -> str:
    """LaTeX 表格"""
    n_cols = len(columns) + 1
    col_spec = "l" + "c" * len(columns)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    if title:
        lines.append(f"\\caption{{{title}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # 表头
    header = "Model & " + " & ".join(columns) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # 数据行
    for label, row in zip(model_labels, rows):
        # 转义 LaTeX 特殊字符
        safe_label = label.replace("_", r"\_").replace("%", r"\%")
        safe_row = [v.replace("_", r"\_") for v in row]
        line = f"{safe_label} & " + " & ".join(safe_row) + r" \\"
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 综合报告 (同时输出 accuracy 和 tokens 两张表)
# ──────────────────────────────────────────────────────────────────────────────
def generate_full_report(all_data: Dict[str, dict],
                         method_filter: Optional[str] = None,
                         fmt: str = "terminal") -> str:
    """生成完整报告 (accuracy 表 + tokens 表)"""
    parts = []

    # Accuracy 表
    title_acc = f"Accuracy ({method_filter or 'All Methods'})"
    labels, cols, rows = build_table_data(all_data, method_filter, "accuracy")
    if fmt == "terminal":
        parts.append(format_terminal(labels, cols, rows, title_acc))
    elif fmt == "markdown":
        parts.append(format_markdown(labels, cols, rows, title_acc))
    elif fmt == "csv":
        parts.append(f"# {title_acc}\n")
        parts.append(format_csv(labels, cols, rows))
    elif fmt == "latex":
        parts.append(format_latex(labels, cols, rows, title_acc))

    # Avg Tokens 表
    title_tok = f"Average Tokens ({method_filter or 'All Methods'})"
    labels, cols, rows = build_table_data(all_data, method_filter, "avg_tokens")
    if fmt == "terminal":
        parts.append(format_terminal(labels, cols, rows, title_tok))
    elif fmt == "markdown":
        parts.append(format_markdown(labels, cols, rows, title_tok))
    elif fmt == "csv":
        parts.append(f"\n# {title_tok}\n")
        parts.append(format_csv(labels, cols, rows))
    elif fmt == "latex":
        parts.append(format_latex(labels, cols, rows, title_tok))

    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="📊 OThink-R1 报表生成器 — 将评测结果汇总为表格",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 终端表格 (自动扫描所有模型)
  python othink_report.py

  # 只看 DEER 方法
  python othink_report.py --method deer

  # 指定模型
  python othink_report.py --models Qwen2.5-0.5B-Instruct Qwen2.5-7B-Instruct

  # 输出 Markdown
  python othink_report.py --format markdown --output report.md

  # 输出 CSV (可用 Excel 打开)
  python othink_report.py --format csv --output report.csv

  # 输出 LaTeX
  python othink_report.py --format latex --output report.tex
        """,
    )
    parser.add_argument("--models", nargs="+", default=None,
                        help="指定模型名称 (默认扫描 log/ 下所有)")
    parser.add_argument("--method", default=None,
                        help="只看某个方法: standard / deer / cp-router / lcb-standard / lcb-deer")
    parser.add_argument("--format", choices=["terminal", "markdown", "csv", "latex"],
                        default="terminal", help="输出格式 (默认: terminal)")
    parser.add_argument("--output", default=None,
                        help="输出到文件 (默认打印到终端)")
    parser.add_argument("--metric", choices=["accuracy", "avg_tokens", "both"],
                        default="both", help="显示哪个指标 (默认: both)")

    args = parser.parse_args()

    # 加载数据
    all_data = load_all_results(args.models)

    if not all_data:
        print("❌ 未找到任何结果数据!")
        print("   请先运行: python othink_collect.py --model <name> --model_size <size>")
        sys.exit(1)

    print(f"📊 已加载 {len(all_data)} 个模型的结果")

    # 生成报告
    if args.metric == "both":
        report = generate_full_report(all_data, args.method, args.format)
    else:
        title = f"{args.metric.replace('_', ' ').title()} ({args.method or 'All Methods'})"
        labels, cols, rows = build_table_data(all_data, args.method, args.metric)

        if args.format == "terminal":
            report = format_terminal(labels, cols, rows, title)
        elif args.format == "markdown":
            report = format_markdown(labels, cols, rows, title)
        elif args.format == "csv":
            report = format_csv(labels, cols, rows)
        elif args.format == "latex":
            report = format_latex(labels, cols, rows, title)

    # 输出
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"✅ 报告已保存: {output_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
