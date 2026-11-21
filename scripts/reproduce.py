"""Recompute the conference manuscript's numeric summaries from frozen records.

Read-only: no model calls, experiments, source edits, or data cleaning.
Run from any directory with the project Python environment.
"""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "reproduced"
HERE.mkdir(exist_ok=True)
INPUTS = {}


def read(name):
    p = ROOT / name
    INPUTS[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    if p.suffix == ".csv":
        with p.open(newline="") as f:
            return list(csv.DictReader(f))
    if p.suffix == ".jsonl":
        return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
    if p.suffix == ".npy":
        return np.load(p)
    return json.loads(p.read_text())


def score_mean(rows):
    return statistics.mean(float(r["attack_score"]) * 100 for r in rows)


arms = ["none", "sr", "rad", "unmask"]
families = ["naive", "aim", "pair", "pap", "concealed"]
full = read("results/full/exp_full.csv")
assert len(full) == 5176
assert len({(r["sample_id"], r["arm"]) for r in full}) == 5176
expected = {
    "naive": [1.32, 0.28, 0.24, 0.24],
    "aim": [3.83, 0, 0, 0],
    "pair": [49.96, 32.87, 32.47, 19.69],
    "pap": [1.32, 0.28, 0.24, 0.24],
    "concealed": [27.73, 5.47, 18.36, 9.38],
    "harmful": [14.45, 8.28, 8.49, 5.15],
}
primary = {}
for family, wanted in expected.items():
    primary[family] = {}
    for arm, number in zip(arms, wanted):
        rows = [r for r in full if r["arm"] == arm and
                (r["family"] != "benign" if family == "harmful" else r["family"] == family)]
        n = 1284 if family == "harmful" else 32 if family == "concealed" else 313
        assert len(rows) == n
        mean = score_mean(rows)
        assert round(mean, 2) == number, (family, arm, mean, number)
        primary[family][arm] = {"n": n, "mean": mean,
            "blocked": sum(int(r["blocked"]) for r in rows),
            "judge_positive": sum(float(r["attack_score"]) > 0 for r in rows)}

benign = {}
for arm, wanted in zip(arms, [0, 3, 0, 0]):
    rows = [r for r in full if r["arm"] == arm and r["family"] == "benign"]
    assert len(rows) == 10
    benign[arm] = sum(float(r["refusal"]) for r in rows)
    assert benign[arm] == wanted
benign_ext = read("results/ext/benign_ext100_scores.json")
assert len(benign_ext) == 100 and not any(r["blocked"] for r in benign_ext)
pair_passers = [r for r in full if r["family"] == "pair" and r["arm"] == "unmask" and not int(r["blocked"])]
assert len(pair_passers) == 72
residual_bands = [sum(float(r["p_harmful"]) < .20 for r in pair_passers),
                  sum(.20 <= float(r["p_harmful"]) < .35 for r in pair_passers),
                  sum(.35 <= float(r["p_harmful"]) < .50 for r in pair_passers)]
assert residual_bands == [29, 19, 24]

concealed = [r for r in full if r["family"] == "concealed"] + read("results/ext/exp_ext.csv")
pairs = defaultdict(dict)
for r in concealed:
    assert r["arm"] not in pairs[r["sample_id"]]
    pairs[r["sample_id"]][r["arm"]] = float(r["attack_score"])
assert len(pairs) == 130 and all(set(p) == set(arms) for p in pairs.values())
diffs = [p["rad"] - p["unmask"] for p in pairs.values()]
assert sum(d > 0 for d in diffs) == 6 and sum(d < 0 for d in diffs) == 0
concealed_summary = {a: score_mean([r for r in concealed if r["arm"] == a]) for a in arms}
assert [round(concealed_summary[a], 2) for a in arms] == [22.40, 8.85, 18.56, 14.04]
assert [sum(int(r["blocked"]) for r in concealed if r["arm"] == a) for a in ["rad", "unmask"]] == [62, 82]
# All six nonzero signed ranks favor the same arm: two extreme allocations / 2**6.
paired_p = 2 / 2**6
assert paired_p == 0.03125

ablation = read("results/ext/ablation_no_extract.csv")
ab_harmful = [r for r in ablation if r["family"] != "benign"]
assert len(ab_harmful) == 1284
assert sum(int(r["blocked"]) for r in ab_harmful) == 1108
assert round(100 * statistics.mean(int(r["blocked"]) for r in ab_harmful if r["family"] == "pair"), 1) == 67.1

targets = {}
for name, filename, n, rad, ploybook in [
    ("DeepSeek V4.1-Flash", "target2_eval.csv", 5173, 6.76, 4.36),
    ("Qwen3.8-27B", "target3_eval.csv", 5135, 6.18, 4.17),
]:
    # Append-only retry records: retain the last successfully scored row per task.
    latest = {}
    for r in read("results/ext/" + filename):
        try:
            value = float(r["attack_score"])
        except ValueError:
            continue
        assert math.isfinite(value) and 0 <= value <= 1
        latest[r["sample_id"], r["arm"]] = r
    assert len(latest) == n
    result = {a: score_mean([r for r in latest.values() if r["arm"] == a and r["family"] != "benign"])
              for a in ["rad", "unmask"]}
    assert [round(result[a], 2) for a in ["rad", "unmask"]] == [rad, ploybook]
    targets[name] = {"completed_tasks": n, "means": result,
                    "harmful_n": dict(Counter(r["arm"] for r in latest.values() if r["family"] != "benign"))}

rounds = []
breached = set()
for i, wanted in enumerate([(100, 90, 10, 8), (95, 73, 22, 16), (100, 74, 26, 21)], 1):
    rows = read(f"results/ext/adaptive_verdicts_r{i}.jsonl")
    candidates = read(f"benchmarks/adaptive/adaptive_r{i}.csv")
    assert len(candidates) == len(rows)
    blocked = sum(bool(r["blocked"]) for r in rows)
    wins = [r for r in rows if not r["blocked"] and float(r["attack_score"] or 0) > 0]
    values = (len(rows), blocked, len(rows) - blocked, len(wins))
    assert values == wanted
    # Verdict goal_id is a candidate-index alias, not the experimental goal.
    # Restore the true goal from the original candidate CSV using sample_id.
    breached.update(candidates[int(r["sample_id"].rsplit("_", 1)[1])]["goal_id"] for r in wins)
    rounds.append(values)
assert len(breached) == 9
transfer = read("results/ext/adaptive_final_rad_tj.csv")
assert len(transfer) == 9 and sum(float(r["attack_score"]) > 0 for r in transfer) == 8
maintenance = read("results/ext/maint_verdicts_v2.json")
assert len(maintenance) == 30
assert [sum(r[k] for r in maintenance) for k in ["blocked_pre", "blocked_post"]] == [5, 9]

probes = read("results/ext/injection_probes.jsonl")
controls = {r["goal_id"]: r for r in probes if r["injection"] == "none"}
injected = [r for r in probes if r["injection"] != "none"]
assert len(controls) == 15 and len(injected) == 90
assert not any(controls[r["goal_id"]]["blocked"] and not r["blocked"] for r in injected)
shifts = [r["core"] - controls[r["goal_id"]]["core"] for r in injected]
assert round(min(shifts), 3) == -0.25
up_flips = [r for r in injected if r["blocked"] and not controls[r["goal_id"]]["blocked"]]
assert len(up_flips) == 3 and len({r["goal_id"] for r in up_flips}) == 1
assert all(round(r["core"] - controls[r["goal_id"]]["core"], 3) == 0.328 for r in up_flips)

navigation = read("results/m13_nav_eval.json")
vectors = read("benchmarks/RAD-wiki/page_emb.npy").astype(float)
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
cosine = vectors @ vectors.T
nav_rates = {}
assert len(navigation["rows"]) == 165
for key, strict, tolerant in [("choice_top3", 0.497, 0.721), ("emb_top3", 0.364, 0.521)]:
    hit = [any(int(p[1:]) == r["true_page"] for p in r[key]) for r in navigation["rows"]]
    near = [any(cosine[r["true_page"], int(p[1:])] > .85 for p in r[key]) for r in navigation["rows"]]
    assert [round(statistics.mean(v), 3) for v in [hit, near]] == [strict, tolerant]
    nav_rates[key] = {"strict": sum(hit), "tolerant": sum(near), "n": len(hit)}

timing = [float(r["def_seconds"]) for r in full if r["arm"] == "unmask" and r["family"] != "benign"]
assert statistics.median(timing) == 1.25 and round(statistics.mean(timing), 2) == 4.52
assert sum(v > 30 for v in timing) == 37 and max(timing) == 132.6
token_pairs = defaultdict(dict)
for r in read("results/exp2_pilot.csv"):
    if r["arm"] in ["rad", "wiki"]:
        token_pairs[r["sample_id"]][r["arm"]] = float(r["def_tokens"])
assert len(token_pairs) == 42
tokens = {a: statistics.mean(p[a] for p in token_pairs.values()) for a in ["rad", "wiki"]}
assert [round(tokens[a]) for a in ["rad", "wiki"]] == [12660, 6565]
assert sum(p["wiki"] < p["rad"] for p in token_pairs.values()) == 30
pages = [read(str(p.relative_to(ROOT))) for p in sorted((ROOT / "benchmarks/RAD-wiki/pages").glob("P*.json"))]
assert len(pages) == 55 and min(p["n_members"] for p in pages) == 48 and max(p["n_members"] for p in pages) == 447
lint = read("results/m14_lint_decisions.json")
assert Counter(p["decision"] for p in lint) == {"merge": 12, "link-variant": 2}

report = {"status": "PASS", "primary": primary, "benign_refusals_of_10": benign,
          "PAIR_residual_score_bands_below020_020to035_035to050": residual_bands,
          "benign_extension_filter_blocks_of_100": 0, "concealed_means": concealed_summary,
          "concealed_ties_favorable_unfavorable": [124, 6, 0], "paired_p": paired_p,
          "raw_text_ablation_blocks_of_1284": 1108, "target_replications": targets,
          "whitebox_rounds_candidates_blocked_passed_judge_positive": rounds,
          "breached_goals_of_20": len(breached), "maintenance_blocks_of_30": [5, 9],
          "navigation_hits": nav_rates, "decision_timing_mean_seconds": statistics.mean(timing),
          "decision_input_tokens": tokens, "input_sha256": INPUTS}
(HERE / "data-validation.json").write_text(json.dumps(report, indent=2))
print(f"PASS: primary tables and supporting summaries recomputed from {len(INPUTS)} frozen files.")
