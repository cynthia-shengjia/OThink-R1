#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${PROJECT_ROOT}/models"
DATA_DIR="${PROJECT_ROOT}/datasets"

echo "=========================================="
echo "  下载模型和数据集 (via ModelScope)"
echo "=========================================="

# Step 1: 创建目录
echo "[1/5] 创建目录结构"
mkdir -p "${MODEL_DIR}"
mkdir -p "${DATA_DIR}"
echo "  ✅ 目录创建完成"

# Step 2: 安装 modelscope
echo ""
echo "[2/5] 检查/安装 modelscope"
if uv run python -c "import modelscope" 2>/dev/null; then
    echo "  ✅ modelscope 已安装"
else
    uv pip install modelscope
    echo "  ✅ modelscope 安装完成"
fi

# Step 3: 下载 Qwen3-0.6B
echo ""
echo "[3/5] 下载 Qwen3-0.6B 模型"
if [ -d "${MODEL_DIR}/Qwen3-0.6B" ] && [ -f "${MODEL_DIR}/Qwen3-0.6B/config.json" ]; then
    echo "  ⚠️  模型已存在，跳过下载"
else
    uv run modelscope download \
        --model Qwen/Qwen3-0.6B \
        --local_dir "${MODEL_DIR}/Qwen3-0.6B"
    echo "  ✅ Qwen3-0.6B 下载完成"
fi

# Step 4: 下载 ASDIV
echo ""
echo "[4/5] 下载 ASDIV 数据集"
if [ -d "${DATA_DIR}/ASDIV" ]; then
    echo "  ⚠️  数据集已存在，跳过下载"
else
    uv run modelscope download \
        --dataset EleutherAI/asdiv \
        --local_dir "${DATA_DIR}/ASDIV"
    echo "  ✅ ASDIV 下载完成"
fi

# Step 5: 创建 Hydra 配置文件
echo ""
echo "[5/5] 创建 Hydra 配置文件"

cat > "${PROJECT_ROOT}/OThinkR1Construct/config/model/Qwen3-0.6B.yaml" << EOF
# @package _global_
model:
  name: Qwen3-0.6B
  model_size: 0.6B
  path: ${MODEL_DIR}/Qwen3-0.6B
  inference:
    tensor_parallel_size: 1
    enable_prefix_caching: True
    gpu_memory_utilization: 0.95
    temperature: 0.9
    top_p: 0.95
    max_tokens: 16384
    skip_special_tokens: True
EOF

cat > "${PROJECT_ROOT}/OThinkR1Training/config/model/Qwen3-0.6B.yaml" << EOF
# @package _global_
model:
  name: Qwen3-0.6B
  model_size: 0.6B
  path: ${MODEL_DIR}/Qwen3-0.6B
  inference:
    tensor_parallel_size: 1
    enable_prefix_caching: True
    gpu_memory_utilization: 0.95
    temperature: 0.9
    top_p: 0.95
    max_tokens: 16384
    skip_special_tokens: True
EOF

cat > "${PROJECT_ROOT}/OThinkR1Construct/config/data/ASDIV.yaml" << EOF
# @package _global_
data:
  datasets:
    ASDIV:
      _target_: core.dataset_processor.ASDIVProcessor
      path: ${DATA_DIR}/ASDIV
      splits:
        train:
          name: train
          slice: "[:100%]"
          columns_to_remove: []
        test:
          name: test
          slice: "[:100%]"
          columns_to_remove: []
      verify: "math_verify"
      subset: 
      eval_split: "test"
EOF

cat > "${PROJECT_ROOT}/OThinkR1Training/config/data/ASDIV.yaml" << EOF
# @package _global_
data:
  datasets:
    ASDIV:
      _target_: core.dataset_processor.ASDIVProcessor
      path: ${DATA_DIR}/ASDIV
      splits:
        train:
          name: train
          slice: "[:100%]"
          columns_to_remove: []
        test:
          name: test
          slice: "[:100%]"
          columns_to_remove: []
      verify: "math_verify"
      subset: 
      eval_split: "test"
EOF

echo "  ✅ 配置文件创建完成"

echo ""
echo "=========================================="
echo "  ✅ 全部完成！"
echo "=========================================="
echo ""
echo "  OThink-R1/"
echo "  ├── models/Qwen3-0.6B/"
echo "  ├── datasets/ASDIV/"
echo "  ├── OThinkR1Construct/config/model/Qwen3-0.6B.yaml"
echo "  ├── OThinkR1Construct/config/data/ASDIV.yaml"
echo "  ├── OThinkR1Training/config/model/Qwen3-0.6B.yaml"
echo "  └── OThinkR1Training/config/data/ASDIV.yaml"
