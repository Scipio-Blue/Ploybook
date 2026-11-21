#!/usr/bin/env python3
"""run_phase2_ext — concealed-ext 四臂收尾（疑点2：32→130 扩容）。

复用 run_phase2 全部逻辑，仅把三个路径指向 ext 产物：
  UNMASK 预计算 = results/ext/unmask_extractions_ext.jsonl（GPU 侧已完成）
  RAD 预计算   = results/ext/rad_baseline_ext.csv（GPU 侧已完成）
  输出         = results/ext/exp_ext.csv
用法（代理开启窗口）：python3 scripts/runs/run_phase2_ext.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "runs"))
sys.path.insert(0, str(ROOT / "src"))

import run_phase2 as rp

EXT_DIR = ROOT / "results" / "ext"
rp.UNMASK_PRE = EXT_DIR / "unmask_extractions_ext.jsonl"
rp.RAD_PRE = EXT_DIR / "rad_baseline_ext.csv"
rp.OUT = EXT_DIR / "exp_ext.csv"
rp.RESP = EXT_DIR / "exp_ext_responses.jsonl"

if __name__ == "__main__":
    sys.exit(rp.main())
