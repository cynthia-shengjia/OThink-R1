#!/bin/bash
set -e
eval "$(conda shell.bash hook)"
conda activate othink-r1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${DEER_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

echo "=========================================="
echo "  转换数据集为 DEER 格式"
echo "  DEER 目录: ${DEER_DIR}"
echo "  项目根目录: ${PROJECT_ROOT}"
echo "=========================================="

for dataset in math aime asdiv; do
    echo ""
    echo "  转换 ${dataset}..."
    uv run python "${DEER_DIR}/scripts/convert_data.py" \
        --dataset "${dataset}" \
        --output_dir "${DEER_DIR}/data" \
        2>&1 || echo "  ⚠️  ${dataset} 转换失败（可能数据集未下载）"
done

echo ""
echo "  ✅ 数据转换完成"
echo "  生成的文件:"
find "${DEER_DIR}/data" -name "*.jsonl" 2>/dev/null | while read f; do
    count=$(wc -l < "$f")
    echo "    ${f} (${count} 条)"
done
