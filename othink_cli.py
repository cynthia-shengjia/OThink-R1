#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OThink-R1 CLI — 一站式 LLM 评测入口
=====================================
放置于项目根目录: OThink-R1/othink_cli.py

用法:
  python othink_cli.py download-data   --datasets all
  python othink_cli.py download-model  --model Qwen/Qwen2.5-7B-Instruct
  python othink_cli.py eval            --model Qwen2.5-0.5B-Instruct --datasets math aime --gpu_ids 0,1,2,3
  python othink_cli.py eval-deer       --model Qwen2.5-0.5B-Instruct --datasets math aime --gpu_ids 0,1
  python othink_cli.py eval-cp-router  --llm_model Qwen2.5-0.5B-Instruct --datasets math aime --gpu_ids 0
  python othink_cli.py eval-lcb        --model Qwen2.5-0.5B-Instruct --mode standard --gpu_ids 0
  python othink_cli.py eval-all        --model Qwen2.5-0.5B-Instruct --gpu_ids 0,1,2,3,4,5,6,7
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
import threading
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict

# ──────────────────────────────────────────────────────────────────────────────
# 全局常量
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# 数据集注册表: key → { repo, local_dir, deer_name, standard_hydra_name }
DATASET_REGISTRY = {
    "math": {
        "repo": "DigitalLearningGmbH/MATH-lighteval",
        "local": "datasets/MATH",
        "deer_name": "math_hf",           # DEER 格式数据集名 (在 baseline/deer/data/ 下)
        "standard_name": "MATHBench",      # OThinkR1Training Hydra 配置名
    },
    "aime": {
        "repo": "AI-MO/aimo-validation-aime",
        "local": "datasets/AIME",
        "deer_name": "aime_hf",
        "standard_name": "AIME",
    },
    "asdiv": {
        "repo": "EleutherAI/asdiv",
        "local": "datasets/ASDIV",
        "deer_name": "asdiv_hf",
        "standard_name": "ASDIV",
    },
    "livecodebench": {
        "repo": "livecodebench/code_generation_lite",
        "local": "datasets/livecodebench/code_generation_lite",
        "deer_name": None,
        "standard_name": None,
    },
    "gsm8k": {
        "repo": "openai/gsm8k",
        "local": "datasets/GSM8K",
        "deer_name": "gsm8k",
        "standard_name": None,
    },
    "gpqa": {
        "repo": "Idavidrein/gpqa",
        "local": "datasets/GPQA",
        "deer_name": "gpqa",
        "standard_name": None,
    },
    
    "openbookqa": {
    "repo": "allenai/openbookqa",
    "local": "datasets/OpenBookQA",
    "deer_name": None,
    "standard_name": "OpenBookQA",
    },
    
    "commonsenseqa": {
        "repo": "tau/commonsense_qa",
        "local": "datasets/CommonsenseQA",
        "deer_name": None,
        "standard_name": "CommonsenseQA",
    }
}

DEER_CONVERTIBLE = {"math", "aime", "asdiv"}  # convert_hf_to_deer.py 支持的

DEFAULT_HF_MIRROR = "https://hf-mirror.com"


# ──────────────────────────────────────────────────────────────────────────────
# 日志辅助
# ──────────────────────────────────────────────────────────────────────────────
class Log:
    @staticmethod
    def info(msg):    print(f"ℹ️  {msg}", flush=True)
    @staticmethod
    def ok(msg):      print(f"✅ {msg}", flush=True)
    @staticmethod
    def warn(msg):    print(f"⚠️  {msg}", flush=True)
    @staticmethod
    def err(msg):     print(f"❌ {msg}", file=sys.stderr, flush=True)
    @staticmethod
    def run(msg):     print(f"🚀 {msg}", flush=True)
    @staticmethod
    def dl(msg):      print(f"📥 {msg}", flush=True)
    @staticmethod
    def gpu(msg):     print(f"🖥️  {msg}", flush=True)
    @staticmethod
    def time(msg):    print(f"⏱️  {msg}", flush=True)
    @staticmethod
    def done(msg):    print(f"🎉 {msg}", flush=True)
    @staticmethod
    def task(msg):    print(f"📋 {msg}", flush=True)
    @staticmethod
    def skip(msg):    print(f"⏭️  {msg}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# GPU 并行调度器
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TaskItem:
    name: str
    cmd: List[str]
    env_extra: Dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None
    gpu_count: int = 1


@dataclass
class TaskResult:
    name: str
    returncode: int
    elapsed: float
    gpu_ids: List[int]


class GPUScheduler:
    """维护空闲 GPU 池，自动将任务分配到空闲卡上并行运行。"""

    def __init__(self, gpu_ids: List[int]):
        self._all = list(gpu_ids)
        self._free: deque = deque(gpu_ids)
        self._lock = threading.Lock()
        self._results: List[TaskResult] = []
        self._rlock = threading.Lock()

    def run_all(self, tasks: List[TaskItem]) -> List[TaskResult]:
        queue: deque = deque(tasks)
        active: List[threading.Thread] = []
        Log.task(f"共 {len(tasks)} 个任务, 可用 GPU: {self._all}")

        while queue or active:
            active = [t for t in active if t.is_alive()]
            scheduled = True
            while scheduled and queue:
                scheduled = False
                task = queue[0]
                alloc = self._try_alloc(task.gpu_count)
                if alloc is not None:
                    queue.popleft()
                    t = threading.Thread(target=self._exec, args=(task, alloc), daemon=True)
                    t.start()
                    active.append(t)
                    scheduled = True
            time.sleep(0.5)
        return self._results

    def _try_alloc(self, n):
        with self._lock:
            if len(self._free) >= n:
                return [self._free.popleft() for _ in range(n)]
        return None

    def _release(self, ids):
        with self._lock:
            self._free.extend(ids)

    def _exec(self, task: TaskItem, gpu_ids: List[int]):
        cuda = ",".join(str(g) for g in gpu_ids)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cuda
        env.update(task.env_extra)

        Log.gpu(f"[START] {task.name}  →  GPU [{cuda}]")
        t0 = time.time()
        try:
            proc = subprocess.Popen(
                task.cmd, env=env,
                cwd=task.cwd or str(PROJECT_ROOT),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            prefix = f"[{task.name}|GPU {cuda}]"
            for line in proc.stdout:
                print(f"  {prefix} {line}", end="", flush=True)
            proc.wait()
            rc = proc.returncode
        except Exception as e:
            Log.err(f"任务 {task.name} 启动失败: {e}")
            rc = -1

        elapsed = time.time() - t0
        self._release(gpu_ids)
        r = TaskResult(name=task.name, returncode=rc, elapsed=elapsed, gpu_ids=gpu_ids)
        with self._rlock:
            self._results.append(r)
        icon = "✅" if rc == 0 else "❌"
        Log.time(f"[DONE] {icon} {task.name}  耗时 {elapsed:.1f}s  rc={rc}")


# ──────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────────────────────
def parse_gpu_ids(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def resolve_model(name: str) -> Path:
    """models/ 下的名称 或 绝对路径"""
    p = PROJECT_ROOT / "models" / name
    if p.exists():
        return p
    p2 = Path(name)
    if p2.exists():
        return p2.resolve()
    Log.err(f"模型路径不存在: {p} 或 {name}")
    sys.exit(1)


def run_cmd(cmd, env_extra=None, cwd=None):
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    Log.run(f"$ {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env, cwd=cwd or str(PROJECT_ROOT),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(f"  | {line}", end="", flush=True)
    proc.wait()
    return proc.returncode


def print_summary(results: List[TaskResult]):
    print("\n" + "=" * 72)
    Log.done("评测结果汇总")
    print("-" * 72)
    print(f"  {'任务名':<40} {'GPU':<10} {'耗时':>8} {'状态':>6}")
    print("-" * 72)
    for r in sorted(results, key=lambda x: x.name):
        st = "✅" if r.returncode == 0 else f"❌ rc={r.returncode}"
        gs = ",".join(str(g) for g in r.gpu_ids)
        print(f"  {r.name:<40} {gs:<10} {r.elapsed:>7.1f}s {st}")
    print("=" * 72)
    failed = [r for r in results if r.returncode != 0]
    if failed:
        Log.err(f"{len(failed)} 个任务失败!")
    else:
        Log.done("所有任务均已成功完成!")


# ──────────────────────────────────────────────────────────────────────────────
# download-data
# ──────────────────────────────────────────────────────────────────────────────
def cmd_download_data(args):
    if "all" in args.datasets:
        targets = list(DATASET_REGISTRY.keys())
    else:
        targets = [d.lower() for d in args.datasets]

    hf_mirror = args.hf_mirror
    env_hf = {"HF_ENDPOINT": hf_mirror} if hf_mirror else {}
    Log.info(f"HuggingFace 镜像: {hf_mirror or '官方源'}")

    downloaded = []
    for ds in targets:
        meta = DATASET_REGISTRY.get(ds)
        if not meta:
            Log.err(f"未知数据集: {ds}. 可选: {', '.join(DATASET_REGISTRY.keys())}, all")
            continue

        local = PROJECT_ROOT / meta["local"]
        if local.exists() and any(local.iterdir()):
            Log.skip(f"[{ds}] 已存在于 {local}，跳过")
            downloaded.append(ds)
            continue

        Log.dl(f"下载 [{ds}]: {meta['repo']} → {local}")
        local.mkdir(parents=True, exist_ok=True)

        rc = run_cmd([
            "uv", "run", "huggingface-cli", "download",
            "--repo-type", "dataset",
            meta["repo"],
            "--local-dir", str(local),
            "--local-dir-use-symlinks", "False",
            "--resume-download",
        ], env_extra=env_hf)

        if rc != 0:
            Log.err(f"[{ds}] 下载失败 (rc={rc})")
            continue
        Log.ok(f"[{ds}] 下载完成")
        downloaded.append(ds)

    # DEER 格式转换
    convert_py = PROJECT_ROOT / "baseline" / "deer" / "scripts" / "convert_hf_to_deer.py"
    if convert_py.exists():
        deer_targets = [d for d in downloaded if d in DEER_CONVERTIBLE]
        if deer_targets:
            Log.run("🦌 转换 DEER 格式...")
            rc = run_cmd([
                "uv", "run", "python", str(convert_py),
                "--hf_dir", str(PROJECT_ROOT / "datasets"),
                "--output_dir", str(PROJECT_ROOT / "baseline" / "deer" / "data"),
                "--dataset", "all",
            ])
            if rc == 0:
                Log.ok("DEER 格式转换完成")
            else:
                Log.warn(f"DEER 转换失败 (rc={rc}), 可手动运行")

    Log.done("数据集下载流程完毕!")


# ──────────────────────────────────────────────────────────────────────────────
# download-model
# ──────────────────────────────────────────────────────────────────────────────
def cmd_download_model(args):
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    save_name = args.name or args.model.rstrip("/").split("/")[-1]
    target = models_dir / save_name

    # 幂等
    if target.exists() and (target.is_symlink() or any(target.iterdir())):
        Log.skip(f"模型 [{save_name}] 已存在于 {target}，跳过")
        return

    # 本地路径 → 软链接
    if args.local_path:
        src = Path(args.local_path).resolve()
        if not src.exists():
            Log.err(f"本地路径不存在: {src}")
            sys.exit(1)
        Log.info(f"创建软链接: {target} → {src}")
        target.symlink_to(src)
        Log.ok(f"模型 [{save_name}] 已链接")
        return

    # HuggingFace 下载
    Log.dl(f"下载模型: {args.model} → {target}")
    target.mkdir(parents=True, exist_ok=True)
    env_hf = {"HF_ENDPOINT": args.hf_mirror} if args.hf_mirror else {}

    rc = run_cmd([
        "uv", "run", "huggingface-cli", "download",
        args.model,
        "--local-dir", str(target),
        "--local-dir-use-symlinks", "False",
        "--resume-download",
    ], env_extra=env_hf)

    if rc != 0:
        Log.err(f"模型下载失败 (rc={rc})")
        sys.exit(1)
    Log.ok(f"模型 [{save_name}] 下载完成 → {target}")


# ──────────────────────────────────────────────────────────────────────────────
# eval — 标准评测 (OThinkR1Training/eval.py + Hydra)
# ──────────────────────────────────────────────────────────────────────────────
def cmd_eval(args):
    model_path = resolve_model(args.model)
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    datasets = [d.lower() for d in args.datasets]

    eval_py = PROJECT_ROOT / "OThinkR1Training" / "eval.py"
    if not eval_py.exists():
        Log.err(f"评测脚本不存在: {eval_py}")
        sys.exit(1)

    tasks = []
    for ds in datasets:
        meta = DATASET_REGISTRY.get(ds)
        if not meta or not meta.get("standard_name"):
            Log.warn(f"[{ds}] 不支持标准评测 (无 Hydra 配置), 跳过")
            continue

        hydra_data = meta["standard_name"]
        # 推断 Hydra model config 名 (models/Qwen2.5-0.5B-Instruct → Qwen2.5-0.5B-Instruct)
        model_name = model_path.name

        cmd = [
            "uv", "run", "python", str(eval_py),
            f"model={model_name}",
            f"model.path={model_path}",
            f"model.inference.tensor_parallel_size=1",
            f"model.inference.gpu_memory_utilization=0.9",
            f"+model.inference.repetition_penalty=1.0",
            f"model.inference.temperature={args.temperature}",
            f"model.inference.top_p={args.top_p}",
            f"model.inference.max_tokens={args.max_tokens}",
            f'+model.mode="test"',
            f"data={hydra_data}",
        ]

        tasks.append(TaskItem(name=f"standard-{ds}", cmd=cmd,
                              cwd=str(PROJECT_ROOT / "OThinkR1Training")))

    if not tasks:
        Log.err("没有有效的评测任务")
        sys.exit(1)

    scheduler = GPUScheduler(gpu_ids)
    results = scheduler.run_all(tasks)
    print_summary(results)


# ──────────────────────────────────────────────────────────────────────────────
# eval-deer — DEER 早退评测
# ──────────────────────────────────────────────────────────────────────────────
def cmd_eval_deer(args):
    """DEER 早退评测 — 推理 + 评估 (生成 *_othink_eval.json)"""
    model_path = resolve_model(args.model)
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    datasets = [d.lower() for d in args.datasets]

    deer_py = PROJECT_ROOT / "baseline" / "deer" / "vllm-deer.py"
    eval_py = PROJECT_ROOT / "baseline" / "deer" / "scripts" / "eval_with_othink.py"
    check_py = PROJECT_ROOT / "baseline" / "deer" / "check_fixed.py"

    if not deer_py.exists():
        Log.err(f"DEER 脚本不存在: {deer_py}")
        sys.exit(1)

    tasks = []
    for ds in datasets:
        meta = DATASET_REGISTRY.get(ds)
        if not meta:
            Log.warn(f"未知数据集: {ds}, 跳过")
            continue

        deer_name = meta.get("deer_name")
        if not deer_name:
            Log.warn(f"[{ds}] 不支持 DEER 评测, 跳过")
            continue

        # 检查 DEER 数据是否存在
        deer_data_dir = PROJECT_ROOT / "baseline" / "deer" / "data"
        deer_data_file = deer_data_dir / deer_name / "test.jsonl"
        if not deer_data_file.exists():
            Log.warn(f"DEER 数据不存在: {deer_data_file}")
            Log.info(f"请先运行: python othink_cli.py download-data --datasets {ds}")
            continue

        output_dir = PROJECT_ROOT / "baseline" / "deer" / "outputs"
        model_basename = model_path.name  # e.g. Qwen2.5-0.5B-Instruct
        threshold = args.threshold
        max_len = args.max_len

        # 构建输出文件名 (与 vllm-deer.py 生成的一致)
        output_pattern = (
            f"greedy_p{threshold}_ratio0.9_len{max_len}_"
            f"temperature0.0_run_time1_no_thinking0_rep0_points1_policyavg1.jsonl"
        )
        expected_output = output_dir / model_basename / deer_name / output_pattern

        # 构建 bash -c 串联命令: 推理 → 查找输出 → 评估
        # 这样两步在同一个子进程中执行，共享 CUDA_VISIBLE_DEVICES
        bash_script = f'''
set -e

echo "=========================================="
echo "  DEER 推理: {deer_name}"
echo "=========================================="

cd "{PROJECT_ROOT}"

# Step 1: 推理
uv run python "{deer_py}" \\
    --model_name_or_path "{model_path}" \\
    --dataset_dir "{deer_data_dir}" \\
    --dataset "{deer_name}" \\
    --threshold {threshold} \\
    --max-len {max_len} \\
    --think_ratio 0.9 \\
    --temperature 0.0 \\
    --top_p 1.0 \\
    --policy "avg1" \\
    --batch_size 2000 \\
    --output_path "{output_dir}" \\
    --no_thinking 0 \\
    --rep 0 \\
    --points 1 \\
    --af 0 \\
    --max_judge_steps 10 \\
    --prob_check_max_tokens 20 \\
    --run_time 1

echo ""
echo "  ✅ DEER 推理完成"

# Step 2: 查找输出文件
OUTPUT_FILE=$(find "{output_dir}" -name "*.jsonl" -path "*{deer_name}*" 2>/dev/null | sort -t/ -k+1 | tail -1)

if [ -z "$OUTPUT_FILE" ]; then
    echo "  ⚠️  未找到输出文件"
    exit 1
fi

echo "  输出文件: $OUTPUT_FILE"

# Step 3: DEER 自带评估 (check_fixed.py)
echo ""
echo "=========================================="
echo "  DEER 自带评估"
echo "=========================================="
'''

        # check_fixed.py 需要原始数据集名 (不带 _hf 后缀的也行)
        if check_py.exists():
            bash_script += f'''
cd "{PROJECT_ROOT}/baseline/deer"
uv run python "{check_py}" \\
    --model_name_or_path "{model_path}" \\
    --data_name "{deer_name}" \\
    --data_dir "{deer_data_dir}" \\
    --generation_path "$OUTPUT_FILE" \\
    2>&1 || echo "  ⚠️  DEER 自带评估失败 (不影响后续)"
'''

        # eval_with_othink.py 生成 *_othink_eval.json
        if eval_py.exists():
            bash_script += f'''
# Step 4: OThink-R1 Verifier 评估 (生成 *_othink_eval.json)
echo ""
echo "=========================================="
echo "  OThink-R1 Verifier 评估"
echo "=========================================="
cd "{PROJECT_ROOT}"
uv run python "{eval_py}" \\
    --generation_path "$OUTPUT_FILE" \\
    --dataset "{ds}" \\
    2>&1 || echo "  ⚠️  OThink-R1 评估失败"

echo ""
echo "=========================================="
echo "  ✅ DEER 评测完成: {deer_name}"
echo "=========================================="
'''

        cmd = ["bash", "-c", bash_script]
        tasks.append(TaskItem(name=f"deer-{ds}", cmd=cmd))

    if not tasks:
        Log.err("没有有效的 DEER 评测任务")
        sys.exit(1)

    scheduler = GPUScheduler(gpu_ids)
    results = scheduler.run_all(tasks)
    print_summary(results)


# ──────────────────────────────────────────────────────────────────────────────
# eval-cp-router — CP-Router 评测
# ──────────────────────────────────────────────────────────────────────────────
def cmd_eval_cp_router(args):
    llm_model = resolve_model(args.llm_model)
    lrm_model = resolve_model(args.lrm_model) if args.lrm_model else llm_model
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    datasets = [d.lower() for d in args.datasets]

    cp_script = PROJECT_ROOT / "baseline" / "cp-router" / "test_cp_router.py"
    if not cp_script.exists():
        Log.err(f"CP-Router 脚本不存在: {cp_script}")
        sys.exit(1)

    tasks = []
    for ds in datasets:
        cmd = [
            "uv", "run", "python", str(cp_script),
            "--model_path", str(llm_model),
            "--datasets_dir", str(PROJECT_ROOT / "datasets"),
            "--dataset", ds,
            "--max_samples", str(args.max_samples),
            "--batch_size", str(args.batch_size),
            "--tau", str(args.tau),
            "--beta", str(args.beta),
        ]
        if args.skip_lrm:
            cmd.append("--skip_lrm")
        else:
            cmd.extend(["--lrm_max_tokens", str(args.lrm_max_tokens)])

        tasks.append(TaskItem(name=f"cp-router-{ds}", cmd=cmd,
                              cwd=str(PROJECT_ROOT / "baseline" / "cp-router")))

    if not tasks:
        Log.err("没有有效的 CP-Router 评测任务")
        sys.exit(1)

    scheduler = GPUScheduler(gpu_ids)
    results = scheduler.run_all(tasks)
    print_summary(results)


# ──────────────────────────────────────────────────────────────────────────────
# eval-lcb — LiveCodeBench 评测
# ──────────────────────────────────────────────────────────────────────────────
def cmd_eval_lcb(args):
    model_path = resolve_model(args.model)
    gpu_ids = parse_gpu_ids(args.gpu_ids)

    lcb_dir = PROJECT_ROOT / "benchmark" / "livecodebench"

    if args.mode == "standard":
        script = lcb_dir / "run_standard.sh"
        cmd = [
            "bash", str(script),
            "--model_path", str(model_path),
            "--gpu_ids", ",".join(str(g) for g in gpu_ids),
        ]
        if args.max_problems > 0:
            cmd.extend(["--max_problems", str(args.max_problems)])
    elif args.mode == "deer":
        script = lcb_dir / "run_deer.sh"
        cmd = [
            "bash", str(script),
            "--model_path", str(model_path),
            "--gpu_ids", ",".join(str(g) for g in gpu_ids),
            "--threshold", str(args.threshold),
        ]
        if args.max_problems > 0:
            cmd.extend(["--max_problems", str(args.max_problems)])
    else:
        Log.err(f"未知 LCB 模式: {args.mode}")
        sys.exit(1)

    if not script.exists():
        Log.err(f"LCB 脚本不存在: {script}")
        sys.exit(1)

    # LCB 直接运行，不走调度器 (因为 GPU 已在脚本内设置)
    Log.run(f"LiveCodeBench [{args.mode}] 评测")
    rc = run_cmd(cmd)
    if rc == 0:
        Log.ok("LiveCodeBench 评测完成")
    else:
        Log.err(f"LiveCodeBench 评测失败 (rc={rc})")


# ──────────────────────────────────────────────────────────────────────────────
# eval-all — 一键全量评测
# ──────────────────────────────────────────────────────────────────────────────
def cmd_eval_all(args):
    model_path = resolve_model(args.model)
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    methods = [m.strip().lower() for m in args.methods.split(",")]
    datasets = [d.strip().lower() for d in args.datasets]

    lrm_model = resolve_model(args.lrm_model) if args.lrm_model else model_path
    threshold = args.threshold
    max_len = args.max_len

    tasks = []

    for method in methods:
        if method == "standard":
            eval_py = PROJECT_ROOT / "OThinkR1Training" / "eval.py"
            if not eval_py.exists():
                Log.warn("标准评测脚本不存在, 跳过")
                continue
            for ds in datasets:
                meta = DATASET_REGISTRY.get(ds)
                if not meta or not meta.get("standard_name"):
                    continue
                model_name = model_path.name
                cmd = [
                    "uv", "run", "python", str(eval_py),
                    f"model={model_name}",
                    f"model.path={model_path}",
                    f"model.inference.tensor_parallel_size=1",
                    f"model.inference.gpu_memory_utilization=0.9",
                    f"+model.inference.repetition_penalty=1.0",
                    f"model.inference.temperature=0.9",
                    f"model.inference.top_p=0.95",
                    f"model.inference.max_tokens=4096",
                    f'+model.mode="test"',
                    f"data={meta['standard_name']}",
                ]
                tasks.append(TaskItem(name=f"standard-{ds}", cmd=cmd,
                                      cwd=str(PROJECT_ROOT / "OThinkR1Training")))

        elif method == "deer":
            deer_py = PROJECT_ROOT / "baseline" / "deer" / "vllm-deer.py"
            if not deer_py.exists():
                Log.warn("DEER 脚本不存在, 跳过")
                continue
            for ds in datasets:
                meta = DATASET_REGISTRY.get(ds)
                if not meta or not meta.get("deer_name"):
                    continue
                deer_name = meta["deer_name"]
                deer_data = PROJECT_ROOT / "baseline" / "deer" / "data" / deer_name / "test.jsonl"
                if not deer_data.exists():
                    Log.warn(f"DEER 数据 [{deer_name}] 不存在, 跳过")
                    continue
                cmd = [
                    "uv", "run", "python", str(deer_py),
                    "--model_name_or_path", str(model_path),
                    "--dataset_dir", str(PROJECT_ROOT / "baseline" / "deer" / "data"),
                    "--dataset", deer_name,
                    "--threshold", str(threshold),
                    "--max-len", str(max_len),
                    "--think_ratio", "0.9",
                    "--temperature", "0.0",
                    "--top_p", "1.0",
                    "--policy", "avg1",
                    "--batch_size", "2000",
                    "--output_path", str(PROJECT_ROOT / "baseline" / "deer" / "outputs"),
                    "--no_thinking", "0", "--rep", "0", "--points", "1",
                    "--af", "0", "--max_judge_steps", "10",
                    "--prob_check_max_tokens", "20", "--run_time", "1",
                ]
                tasks.append(TaskItem(name=f"deer-{ds}", cmd=cmd))

        elif method == "cp-router":
            cp_script = PROJECT_ROOT / "baseline" / "cp-router" / "test_cp_router.py"
            if not cp_script.exists():
                Log.warn("CP-Router 脚本不存在, 跳过")
                continue
            for ds in datasets:
                cmd = [
                    "uv", "run", "python", str(cp_script),
                    "--model_path", str(model_path),
                    "--datasets_dir", str(PROJECT_ROOT / "datasets"),
                    "--dataset", ds,
                    "--skip_lrm",
                ]
                tasks.append(TaskItem(name=f"cp-router-{ds}", cmd=cmd,
                                      cwd=str(PROJECT_ROOT / "baseline" / "cp-router")))

        elif method == "lcb-standard":
            script = PROJECT_ROOT / "benchmark" / "livecodebench" / "lcb_eval.py"
            if not script.exists():
                Log.warn("LCB 标准脚本不存在, 跳过")
                continue
            env = {"PYTHONPATH": str(PROJECT_ROOT / "benchmark" / "livecodebench" / "LiveCodeBench")}
            cmd = [
                "uv", "run", "python", str(script),
                "--model_path", str(model_path),
                "--dataset_path", str(PROJECT_ROOT / "datasets" / "livecodebench" / "code_generation_lite"),
            ]
            tasks.append(TaskItem(name="lcb-standard", cmd=cmd, env_extra=env))

        elif method == "lcb-deer":
            script = PROJECT_ROOT / "benchmark" / "livecodebench" / "deer_lcb.py"
            if not script.exists():
                Log.warn("LCB-DEER 脚本不存在, 跳过")
                continue
            env = {"PYTHONPATH": str(PROJECT_ROOT / "benchmark" / "livecodebench" / "LiveCodeBench")}
            cmd = [
                "uv", "run", "python", str(script),
                "--model_path", str(model_path),
                "--dataset_path", str(PROJECT_ROOT / "datasets" / "livecodebench" / "code_generation_lite"),
                "--threshold", str(threshold),
            ]
            tasks.append(TaskItem(name="lcb-deer", cmd=cmd, env_extra=env))

        else:
            Log.warn(f"未知方法: {method}, 跳过")

    if not tasks:
        Log.err("没有生成任何评测任务")
        sys.exit(1)

    Log.run(f"📊 共 {len(tasks)} 个任务, {len(gpu_ids)} 张 GPU")
    for i, t in enumerate(tasks, 1):
        Log.task(f"  {i:>3}. {t.name}")
    print()

    t0 = time.time()
    scheduler = GPUScheduler(gpu_ids)
    results = scheduler.run_all(tasks)
    total = time.time() - t0

    print_summary(results)
    Log.time(f"总耗时: {total:.1f}s ({total/60:.1f} min)")




# ──────────────────────────────────────────────────────────────────────────────
# 以下代码追加到 othink_cli.py 中
# 在 cmd_eval_all 函数之后、build_parser 函数之前插入
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# collect — 结果收集
# ──────────────────────────────────────────────────────────────────────────────
def cmd_collect(args):
    """收集指定模型的评测结果，写入 log/<model_size>/<model_name>/"""
    collect_script = PROJECT_ROOT / "othink_collect.py"
    if not collect_script.exists():
        Log.err(f"结果收集脚本不存在: {collect_script}")
        sys.exit(1)

    cmd = [
        "uv", "run", "python", str(collect_script),
        "--model", args.model,
        "--model_size", args.model_size,
    ]
    if args.methods:
        cmd.extend(["--methods"] + args.methods)

    rc = run_cmd(cmd)
    if rc == 0:
        Log.ok(f"结果已收集到 log/{args.model_size}/{args.model}/")
    else:
        Log.err(f"结果收集失败 (rc={rc})")


# ──────────────────────────────────────────────────────────────────────────────
# report — 生成报表
# ──────────────────────────────────────────────────────────────────────────────
def cmd_report(args):
    """生成评测报表 (终端/Markdown/CSV/LaTeX)"""
    report_script = PROJECT_ROOT / "othink_report.py"
    if not report_script.exists():
        Log.err(f"报表生成脚本不存在: {report_script}")
        sys.exit(1)

    cmd = [
        "uv", "run", "python", str(report_script),
        "--format", args.format,
    ]
    if args.models:
        cmd.extend(["--models"] + args.models)
    if args.method:
        cmd.extend(["--method", args.method])
    if args.output:
        cmd.extend(["--output", args.output])
    if args.metric:
        cmd.extend(["--metric", args.metric])

    rc = run_cmd(cmd)
    if rc != 0:
        Log.err(f"报表生成失败 (rc={rc})")


# ──────────────────────────────────────────────────────────────────────────────
# 在 build_parser() 函数中，eval-all 子命令之后添加以下两个子命令:
# ──────────────────────────────────────────────────────────────────────────────

"""
    # ── collect ──
    pc = sub.add_parser("collect", help="🔍 收集评测结果")
    pc.add_argument("--model", required=True, help="模型名称")
    pc.add_argument("--model_size", required=True, help="模型大小 (0.5B, 1.5B, 7B, 14B)")
    pc.add_argument("--methods", nargs="+", default=None,
                    help="方法列表: standard deer cp-router lcb-standard lcb-deer")
    pc.set_defaults(func=cmd_collect)

    # ── report ──
    pr = sub.add_parser("report", help="📊 生成评测报表")
    pr.add_argument("--models", nargs="+", default=None, help="指定模型 (默认全部)")
    pr.add_argument("--method", default=None, help="只看某个方法")
    pr.add_argument("--format", choices=["terminal", "markdown", "csv", "latex"],
                    default="terminal", help="输出格式")
    pr.add_argument("--output", default=None, help="输出到文件")
    pr.add_argument("--metric", choices=["accuracy", "avg_tokens", "both"],
                    default="both", help="显示指标")
    pr.set_defaults(func=cmd_report)
"""
 
def build_parser():
    p = argparse.ArgumentParser(
        prog="othink_cli",
        description="🧠 OThink-R1 CLI — 一站式 LLM 评测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    示例:
    python othink_cli.py download-data   --datasets all
    python othink_cli.py download-model  --model Qwen/Qwen2.5-7B-Instruct
    python othink_cli.py eval            --model Qwen2.5-0.5B-Instruct --datasets math aime --gpu_ids 0,1
    python othink_cli.py eval-deer       --model Qwen2.5-0.5B-Instruct --datasets math --gpu_ids 0,1
    python othink_cli.py eval-all        --model Qwen2.5-0.5B-Instruct --gpu_ids 0,1,2,3
    python othink_cli.py collect         --model Qwen2.5-0.5B-Instruct --model_size 0.5B
    python othink_cli.py report          --format markdown --output report.md
            """,
    )
    sub = p.add_subparsers(dest="command", help="子命令")

    # ── download-data ──
    dd = sub.add_parser("download-data", help="📥 下载数据集")
    dd.add_argument("--datasets", nargs="+", required=True,
                    help="数据集: all, math, aime, asdiv, livecodebench, gsm8k, gpqa")
    dd.add_argument("--hf_mirror", default=DEFAULT_HF_MIRROR, help="HF 镜像")
    dd.set_defaults(func=cmd_download_data)

    # ── download-model ──
    dm = sub.add_parser("download-model", help="📥 下载模型")
    dm.add_argument("--model", required=True, help="HF 模型 ID 如 Qwen/Qwen2.5-7B-Instruct")
    dm.add_argument("--name", default=None, help="保存名称 (默认从 ID 推断)")
    dm.add_argument("--local_path", default=None, help="本地路径 (创建软链接)")
    dm.add_argument("--hf_mirror", default=DEFAULT_HF_MIRROR, help="HF 镜像")
    dm.set_defaults(func=cmd_download_model)

    # ── eval ──
    ev = sub.add_parser("eval", help="📊 标准评测")
    ev.add_argument("--model", required=True, help="模型名称 (models/ 下)")
    ev.add_argument("--datasets", nargs="+", required=True, help="数据集列表")
    ev.add_argument("--gpu_ids", required=True, help="GPU 卡号, 如 0,1,2,3")
    ev.add_argument("--temperature", type=float, default=0.9)
    ev.add_argument("--top_p", type=float, default=0.95)
    ev.add_argument("--max_tokens", type=int, default=4096)
    ev.set_defaults(func=cmd_eval)

    # ── eval-deer ──
    ed = sub.add_parser("eval-deer", help="🦌 DEER 评测")
    ed.add_argument("--model", required=True, help="模型名称或路径")
    ed.add_argument("--datasets", nargs="+", required=True, help="数据集列表")
    ed.add_argument("--gpu_ids", required=True, help="GPU 卡号")
    ed.add_argument("--threshold", type=float, default=0.95, help="DEER 阈值")
    ed.add_argument("--max_len", type=int, default=16384, help="最大长度")
    ed.set_defaults(func=cmd_eval_deer)

    # ── eval-cp-router ──
    ec = sub.add_parser("eval-cp-router", help="🔀 CP-Router 评测")
    ec.add_argument("--llm_model", required=True, help="LLM 模型")
    ec.add_argument("--lrm_model", default=None, help="LRM 模型 (默认同 LLM)")
    ec.add_argument("--datasets", nargs="+", required=True, help="数据集列表")
    ec.add_argument("--gpu_ids", required=True, help="GPU 卡号")
    ec.add_argument("--tau", type=int, default=1)
    ec.add_argument("--beta", type=float, default=3.0)
    ec.add_argument("--max_samples", type=int, default=0, help="0=全部")
    ec.add_argument("--batch_size", type=int, default=8)
    ec.add_argument("--skip_lrm", action="store_true", help="跳过 LRM 推理")
    ec.add_argument("--lrm_max_tokens", type=int, default=512)
    ec.set_defaults(func=cmd_eval_cp_router)

    # ── eval-lcb ──
    el = sub.add_parser("eval-lcb", help="💻 LiveCodeBench 评测")
    el.add_argument("--model", required=True, help="模型名称或路径")
    el.add_argument("--mode", choices=["standard", "deer"], default="standard")
    el.add_argument("--gpu_ids", required=True, help="GPU 卡号")
    el.add_argument("--threshold", type=float, default=0.95)
    el.add_argument("--max_problems", type=int, default=0, help="0=全部")
    el.set_defaults(func=cmd_eval_lcb)

    # ── eval-all ──
    ea = sub.add_parser("eval-all", help="🚀 一键全量评测")
    ea.add_argument("--model", required=True, help="模型名称或路径")
    ea.add_argument("--gpu_ids", required=True, help="GPU 卡号")
    ea.add_argument("--methods", default="standard,deer,cp-router,lcb-standard,lcb-deer",
                    help="方法列表, 逗号分隔")
    ea.add_argument("--datasets", nargs="+",
                    default=["math", "aime", "asdiv"], help="数据集列表")
    ea.add_argument("--lrm_model", default=None, help="LRM 模型 (cp-router 用)")
    ea.add_argument("--threshold", type=float, default=0.95)
    ea.add_argument("--max_len", type=int, default=16384)
    ea.set_defaults(func=cmd_eval_all)

    # ── collect (新增) ──
    pc = sub.add_parser("collect", help="🔍 收集评测结果到 log/ 目录")
    pc.add_argument("--model", required=True, help="模型名称 (如 Qwen2.5-0.5B-Instruct)")
    pc.add_argument("--model_size", required=True, help="模型大小 (0.5B, 1.5B, 7B, 14B)")
    pc.add_argument("--methods", nargs="+", default=None,
                    help="方法: standard deer cp-router lcb-standard lcb-deer")
    pc.set_defaults(func=cmd_collect)

    # ── report (新增) ──
    pr = sub.add_parser("report", help="📊 生成评测报表")
    pr.add_argument("--models", nargs="+", default=None, help="指定模型 (默认全部)")
    pr.add_argument("--method", default=None,
                    help="只看某个方法: standard / deer / cp-router / lcb-standard / lcb-deer")
    pr.add_argument("--format", choices=["terminal", "markdown", "csv", "latex"],
                    default="terminal", help="输出格式 (默认: terminal)")
    pr.add_argument("--output", default=None, help="输出到文件")
    pr.add_argument("--metric", choices=["accuracy", "avg_tokens", "both"],
                    default="both", help="显示指标 (默认: both)")
    pr.set_defaults(func=cmd_report)

    return p




# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    print(r"""
    ╔═══════════════════════════════════════════════════════════╗
    ║        🧠  OThink-R1 CLI  —  LLM Evaluation Suite       ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    Log.info(f"项目根目录: {PROJECT_ROOT}")
    Log.run(f"命令: {args.command}")
    print()

    try:
        args.func(args)
    except KeyboardInterrupt:
        print()
        Log.warn("用户中断 (Ctrl+C)")
        sys.exit(130)
    except Exception as e:
        Log.err(f"异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()