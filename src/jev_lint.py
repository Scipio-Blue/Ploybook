"""jev_lint — Jev 语义级 wiki lint：近重复判定 + 矛盾检测（维护层的两工作整合）。

entity_alignment 三档模式：「错并比漏并贵」→ Score 三级（不同/变体/同策略）切点由措辞涌现；
矛盾检测独立 Noul（同策略但抽取指引冲突 → 必须人审）。
用途：①对 cosine 疑似近重复对产出合并决策 ②lint 常规项。
"""
from __future__ import annotations

import json
from pathlib import Path

from jev_client import system_one

LINT_QUESTIONS = {
    "relation": {
        "type": "score",
        "instructions": ("Do the two strategy pages below describe the same attack "
                         "strategy, a variant of one strategy, or different strategies?"),
        "criteria": ["Different strategies: distinct disguise techniques",
                     "Variant of one strategy: same core technique, different flavor or domain",
                     "Same strategy: a wiki should have only one page for both"],
    },
    "contradiction": {
        "type": "noul",
        "instructions": ("Do the two extraction guidance sections give conflicting "
                         "instructions for how to recover intent from the same kind of prompt?"),
    },
}

MERGE_SAME = 1.5      # ≥1.5 → 合并（same）
MERGE_VARIANT = 0.5   # 0.5-1.5 → 变体（链接+观察，不急并）


def lint_pair(page_a: dict, page_b: dict) -> dict:
    state = {"page_a": {"name": page_a["name"], "trigger": page_a["trigger"],
                        "strategy": page_a["strategy"][:400],
                        "guidance": page_a["guidance"][:400]},
             "page_b": {"name": page_b["name"], "trigger": page_b["trigger"],
                        "strategy": page_b["strategy"][:400],
                        "guidance": page_b["guidance"][:400]},
             "note": "Wiki maintenance analysis of two internal pages."}
    answers, meta = system_one(state, LINT_QUESTIONS)
    rel = float(answers["relation"]["score"])
    contra = float(answers["contradiction"]["noul"])
    decision = ("merge" if rel >= MERGE_SAME else
                "link-variant" if rel >= MERGE_VARIANT else "keep-separate")
    if contra >= 0.5:
        decision += "+HUMAN-REVIEW(contradiction)"
    return {"a": page_a["name"], "b": page_b["name"],
            "relation": round(rel, 2), "contradiction": round(contra, 2),
            "decision": decision}


def lint_duplicates(page_emb_path: Path, pages: list[dict],
                    sim_threshold: float = 0.92, top: int | None = None) -> list[dict]:
    """对 cosine 疑似近重复对（>阈值）逐一语义裁决。"""
    import numpy as np
    emb = np.load(page_emb_path)
    Pn = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    S = Pn @ Pn.T
    pairs = [(i, j) for i in range(len(pages)) for j in range(i + 1, len(pages))
             if S[i, j] > sim_threshold]
    if top:
        pairs = pairs[:top]
    return [lint_pair(pages[i], pages[j]) for i, j in pairs]
