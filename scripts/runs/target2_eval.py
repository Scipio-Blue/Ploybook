#!/usr/bin/env python3
"""target2_eval — 多目标复制（实验A：第二目标模型 Qwen2.5-72B-Instruct）。

复用主实验四臂的防御判决（exp_full.csv 的 blocked），只对放行者调用第二目标
+ judge；none/SR 臂全量过目标。产出 results/ext/target2_eval.csv。
用法（代理侧）：https_proxy=http://127.0.0.1:12000 python3 scripts/runs/target2_eval.py
"""
from __future__ import annotations

import csv
import sys
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "runs"))

from run_phase2 import SR_PREFIX, run_target_judge  # noqa: E402
from rad_judge import or_target_chat, score_harmful  # noqa: E402

import os
TARGET2 = os.environ.get("TARGET2_MODEL", "deepseek/deepseek-v4.1-flash-1017")
EXT = ROOT / "results" / "ext"
OUT = EXT / os.environ.get("TARGET2_OUT", "target2_eval.csv")
FIELDS = ["sample_id", "family", "arm", "blocked", "attack_score", "refusal"]


def main() -> int:
    # 主实验：sample_id × arm → blocked + prompt/query/family
    main_rows = {}
    for r in csv.DictReader((ROOT / "results" / "full" / "exp_full.csv").open()):
        main_rows[(r["sample_id"], r["arm"])] = r
    tasks = {}
    for line in (ROOT / "results" / "full" / "unmask_extractions.jsonl").open():
        import json
        r = json.loads(line)
        tasks[r["sample_id"]] = {"sample_id": r["sample_id"], "family": r["family"],
                                 "prompt": r["prompt"], "query": r.get("query", "")}
    done = set()
    if OUT.exists():
        with OUT.open(newline="", encoding="utf-8") as f:
            done = {(r["sample_id"], r["arm"]) for r in csv.DictReader(f) if r["attack_score"] not in ("ERROR", "")}
    todo = [(t, arm) for arm in ("none", "sr", "rad", "unmask") for t in tasks.values()
            if (t["sample_id"], arm) not in done]
    print(f"目标2（{TARGET2}）：任务 {len(tasks)*4}，已完成 {len(done)}，待跑 {len(todo)}")

    lock = threading.Lock()

    def one(item):
        t, arm = item
        blocked = main_rows.get((t["sample_id"], arm), {}).get("blocked") == "1"
        if blocked:
            return {"sample_id": t["sample_id"], "family": t["family"], "arm": arm,
                    "blocked": 1, "attack_score": 0.0, "refusal": 1}
        prompt = (SR_PREFIX + t["prompt"]) if arm == "sr" else t["prompt"]
        resp = or_target_chat(prompt, model=TARGET2, max_tokens=3072)
        if not resp or not resp.strip():
            return {"sample_id": t["sample_id"], "family": t["family"], "arm": arm,
                    "blocked": 0, "attack_score": "ERROR", "refusal": ""}
        if t["family"] == "benign":
            from rad_judge import is_refusal
            return {"sample_id": t["sample_id"], "family": t["family"], "arm": arm,
                    "blocked": 0, "attack_score": 0.0,
                    "refusal": int(is_refusal(t["query"], resp))}
        s = score_harmful(t["query"], resp)
        return {"sample_id": t["sample_id"], "family": t["family"], "arm": arm,
                "blocked": 0, "attack_score": round(s.score, 4), "refusal": s.refusal}

    new = not OUT.exists()
    n = 0
    with ThreadPoolExecutor(max_workers=32) as pool, \
            OUT.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        fut_map = {pool.submit(one, it): it for it in todo}
        for fut in as_completed(fut_map):
            try:
                row = fut.result()
            except Exception as e:  # noqa: BLE001
                it = fut_map[fut]
                row = {"sample_id": it[0]["sample_id"], "family": it[0]["family"],
                       "arm": it[1], "blocked": 0, "attack_score": "ERROR",
                       "refusal": str(e)[:40]}
            with lock:
                w.writerow(row)
                f.flush()
            n += 1
            if n % 200 == 0:
                print(f"[{n}/{len(todo)}]")
    print(f"完成 → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
