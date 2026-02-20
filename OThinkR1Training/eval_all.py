#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, subprocess, argparse, itertools
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ALL_MODELS = ["ARM-7B", "SB-DS7B-alpha-2"]
ALL_DATASETS = ["AIME", "MATHBench", "GSM8K", "ASDIV", "CommonsenseQA", "OpenBookQA"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="all")
    p.add_argument("--dataset", type=str, default="all")
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--gpu_util", type=float, default=0.95)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=16384)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()

def resolve_list(v, a):
    return a if v.strip().lower() == "all" else [x.strip() for x in v.split(",")]

def run_eval(m, d, args):
    cmd = [sys.executable, os.path.join(PROJECT_ROOT, "eval.py"),
           f"model={m}", f"data={d}",
           f"model.inference.tensor_parallel_size={args.tp}",
           f"model.inference.gpu_memory_utilization={args.gpu_util}",
           f"model.inference.temperature={args.temperature}",
           f"model.inference.top_p={args.top_p}",
           f"model.inference.max_tokens={args.max_tokens}"]
    print(f"\n{'='*70}\n[{datetime.now().strftime('%H:%M:%S')}] {m} x {d}\n{'='*70}")
    if args.dry_run: print("  (dry run)"); return 0
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode

def main():
    args = parse_args()
    ms = resolve_list(args.model, ALL_MODELS)
    ds = resolve_list(args.dataset, ALL_DATASETS)
    t = len(ms)*len(ds)
    res = []
    for i,(m,d) in enumerate(itertools.product(ms,ds),1):
        print(f"\n>>> [{i}/{t}] {m} x {d}")
        rc = run_eval(m,d,args)
        res.append((m,d,"OK" if rc==0 else f"FAIL({rc})"))
    print(f"\n{'='*70}\n  Summary\n{'='*70}")
    for m,d,s in res: print(f"  {m:35s} | {d:20s} | {s}")
    f = sum(1 for _,_,s in res if s!="OK")
    if f: print(f"\n  ⚠️ {f}/{t} failed!"); sys.exit(1)
    else: print(f"\n  ✅ All {t} OK!")

if __name__ == "__main__": main()
