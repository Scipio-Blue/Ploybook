#!/usr/bin/env python3
"""qwen38_target3_eval — 第三目标 Qwen3.8-27B（DashScope 官方 API, QWEN_KEY 直连免代理）。

判官照旧 OpenRouter GPT-4o-mini（需 https_proxy）。复用主实验四臂防御判决。
产出 results/ext/target3_eval.csv。
用法：https_proxy=http://127.0.0.1:12000 python3 scripts/runs/qwen38_target3_eval.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
import threading
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "runs"))

from run_phase2 import SR_PREFIX  # noqa: E402
from rad_judge import score_harmful, is_refusal  # noqa: E402

T3_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
T3_KEY = os.environ["QWEN_KEY"]
NOPROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))

EXT = ROOT / "results" / "ext"
OUT = EXT / "target3_eval.csv"
FIELDS = ["sample_id", "family", "arm", "blocked", "attack_score", "refusal"]


def t3_chat(prompt: str) -> str:
    for _ in range(2):
        body = json.dumps({"model": "qwen3.8-27b",
                           "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": 2048}).encode()
        req = urllib.request.Request(T3_URL, data=body, headers={
            "Authorization": f"Bearer {T3_KEY}", "Content-Type": "application/json"})
        with NOPROXY.open(req, timeout=240) as r:
            m = json.load(r)
        content = (m["choices"][0]["message"].get("content") or "").strip()
        if content:
            return content
    return ""


def main() -> int:
    main_rows = {}
    for r in csv.DictReader((ROOT / "results" / "full" / "exp_full.csv").open()):
        main_rows[(r["sample_id"], r["arm"])] = r
    tasks = {}
    for line in (ROOT / "results" / "full" / "unmask_extractions.jsonl").open():
        r = json.loads(line)
        tasks[r["sample_id"]] = {"sample_id": r["sample_id"], "family": r["family"],
                                 "prompt": r["prompt"], "query": r.get("query", "")}
    good = set()
    if OUT.exists():
        with OUT.open(newline="", encoding="utf-8") as f:
            good = {(r["sample_id"], r["arm"]) for r in csv.DictReader(f)
                    if r["attack_score"] not in ("ERROR", "")}
    todo = [(t, arm) for arm in ("none", "sr", "rad", "unmask") for t in tasks.values()
            if (t["sample_id"], arm) not in good]
    print(f"Qwen3.8-27B DashScope: 待跑 {len(todo)}", flush=True)

    lock = threading.Lock()
    JUDGE = threading.BoundedSemaphore(8)

    def one(item):
        t, arm = item
        if main_rows.get((t["sample_id"], arm), {}).get("blocked") == "1":
            return {"sample_id": t["sample_id"], "family": t["family"], "arm": arm,
                    "blocked": 1, "attack_score": 0.0, "refusal": 1}
        prompt = (SR_PREFIX + t["prompt"]) if arm == "sr" else t["prompt"]
        resp = t3_chat(prompt)
        if not resp:
            return {"sample_id": t["sample_id"], "family": t["family"], "arm": arm,
                    "blocked": 0, "attack_score": "ERROR", "refusal": "empty"}
        if t["family"] == "benign":
            with JUDGE:
                ref = int(is_refusal(t["query"], resp))
            return {"sample_id": t["sample_id"], "family": t["family"], "arm": arm,
                    "blocked": 0, "attack_score": 0.0, "refusal": ref}
        with JUDGE:
            s = score_harmful(t["query"], resp)
        return {"sample_id": t["sample_id"], "family": t["family"], "arm": arm,
                "blocked": 0, "attack_score": round(s.score, 4), "refusal": s.refusal}

    n = 0
    with ThreadPoolExecutor(max_workers=12) as pool, \
            OUT.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
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
            if n % 100 == 0:
                print(f"[{n}/{len(todo)}]", flush=True)
    print("完成", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
