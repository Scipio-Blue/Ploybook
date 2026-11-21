#!/usr/bin/env python3
"""build_wiki — 从 RAD 攻击数据库构建 LLM 维护的策略 wiki（知识层，工作1）。

流水线：
  1. 嵌入 10,991 条 analysis（Contriever 复用，缓存 npy）
  2. 球面 k-means 聚成 ~56 个策略簇（k=56, seed=42, 15 iter）
  3. 每簇选 top-8 成员（质心最近）+ 2 条代表 jailbreak 样本
  4. 本地 MLX Qwen 逐簇撰写页面（JSON 结构化：Name/Trigger/Strategy/Guidance/Examples）
  5. 汇总 index.md + meta.json（成员归属/质心/lint 标记）
  6. 页面文本再嵌入（页级检索用）+ 近重复 lint（cos>0.92 警告）

产物：benchmarks/RAD-wiki/{pages/,index.md,meta.json,page_emb.npy}
"""
from __future__ import annotations

import os
import json
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "RAD-code" / "src"))
sys.path.insert(0, str(ROOT / "src"))

WIKI_DIR = ROOT / "benchmarks" / "RAD-wiki"
PAGES_DIR = WIKI_DIR / "pages"
K = 56
SEED = 42

PAGE_PROMPT = """You are building a defensive wiki of jailbreak attack strategies for an LLM safety team. You are given member analyses (each describes how one attack prompt disguised a harmful intent) and two representative attack prompt snippets.

Synthesize ONE wiki page for this attack strategy family. Output ONLY a JSON object:
{{
  "name": "short strategy name (<=6 words, hyphenated style)",
  "trigger": "one sentence: when should a defender consult this page (surface cues in the input)",
  "strategy": "2-3 sentences: how this attack disguises or reframes the harmful intent",
  "guidance": "2-3 imperative sentences: how to strip the disguise and recover the user's true intent",
  "examples": ["one short cue from prompt A (<=25 words)", "one short cue from prompt B (<=25 words)"]
}}

Member analyses:
{analyses}

Representative attack prompt snippets:
A: {snip_a}
B: {snip_b}

JSON only:"""


def spherical_kmeans(X, k, iters=15, seed=SEED):
    import numpy as np
    rng = np.random.RandomState(seed)
    # L2 归一化后余弦 = 点积；初始化用随机样本，空簇用最远点重启
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    C = Xn[rng.choice(len(Xn), k, replace=False)].copy()
    assign = None
    for it in range(iters):
        sims = Xn @ C.T                      # (n, k)
        new_assign = sims.argmax(axis=1)
        if assign is not None and (new_assign == assign).all():
            print(f"  kmeans 第{it}轮收敛")
            break
        assign = new_assign
        for j in range(k):
            members = Xn[assign == j]
            if len(members) == 0:            # 空簇：用离质心最远的点重启
                farthest = sims.min(axis=1).argmax()
                C[j] = Xn[farthest]
            else:
                m = members.mean(axis=0)
                C[j] = m / np.linalg.norm(m)
    sizes = [(assign == j).sum() for j in range(k)]
    print(f"  簇大小: min={min(sizes)} max={max(sizes)} 中位={sorted(sizes)[k//2]}")
    return assign, C


def main() -> int:
    t0 = time.time()
    import numpy as np
    from rad_defense.data import RADDatabase, PAPER_DATABASE_DIGEST
    from rad_defense.models import ContrieverEncoder

    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    PAGES_DIR.mkdir(exist_ok=True)
    db = RADDatabase.from_csv(ROOT / "benchmarks" / "RAD-Database" / "full.csv",
                              expected_digest=PAPER_DATABASE_DIGEST)
    analyses = [e.analysis for e in db]
    jailbreaks = [e.jailbreak for e in db]
    print(f"1/6 数据库 {len(db)} 条")

    emb_path = WIKI_DIR / "analysis_emb.npy"
    if emb_path.exists():
        emb = np.load(emb_path)
    else:
        enc = ContrieverEncoder(str(ROOT / ".local" / "models" / "contriever-msmarco"),
                                "main", device="cpu")
        print("2/6 嵌入 analysis（~6 分钟）…")
        emb = enc.encode(analyses, batch_size=64)
        np.save(emb_path, emb)
    print(f"2/6 嵌入 {emb.shape}")

    print("3/6 球面 k-means…")
    assign, C = spherical_kmeans(emb, K)
    np.save(WIKI_DIR / "centroids.npy", C)

    # 4/6 LLM 写页
    from rad_hybrid import MLXQwenScorer
    from mlx_lm import generate as mlx_generate
    scorer = MLXQwenScorer()
    print("4/6 MLX 写页面…")
    pages = []
    for j in range(K):
        members = np.where(assign == j)[0]
        if len(members) < 10:
            continue
        sims = (emb[members] / np.linalg.norm(emb[members], axis=1, keepdims=True)) @ C[j]
        top = members[np.argsort(-sims)]
        ana_txt = "\n".join(f"- {db[i].analysis[:220]}" for i in top[:8])
        prompt = PAGE_PROMPT.format(analyses=ana_txt,
                                    snip_a=jailbreaks[top[0]][:300],
                                    snip_b=jailbreaks[top[min(1, len(top)-1)]][:300])
        out_path = PAGES_DIR / f"P{j:03d}.json"
        if out_path.exists():
            pages.append(json.loads(out_path.read_text(encoding="utf-8")))
            continue
        text = scorer.tokenizer.apply_chat_template(
            [{"role": "system", "content": "You are a careful safety analyst."},
             {"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)
        import mlx.core as mx
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=0.0)
        raw = mlx_generate(scorer.model, scorer.tokenizer, prompt=text,
                           max_tokens=700, sampler=sampler, verbose=False)
        try:
            s, e = raw.index("{"), raw.rindex("}") + 1
            page = json.loads(raw[s:e])
        except (ValueError, json.JSONDecodeError):
            page = {"name": f"strategy-{j}", "trigger": raw[:200],
                    "strategy": "", "guidance": "", "examples": []}
        page["cluster"] = j
        page["n_members"] = int(len(members))
        page["member_ids"] = [int(i) for i in sorted(members.tolist())]
        out_path.write_text(json.dumps(page, ensure_ascii=False, indent=1), encoding="utf-8")
        pages.append(page)
        if (len(pages)) % 8 == 0:
            print(f"  已写 {len(pages)}/{K} 页（{time.time()-t0:.0f}s）")

    # 5/6 index + 页面嵌入
    print("5/6 index.md + 页面嵌入…")
    enc = ContrieverEncoder(str(ROOT / ".local" / "models" / "contriever-msmarco"),
                            "main", device="cpu")
    page_texts, index_lines = [], []
    for p in pages:
        txt = f"{p['name']}. {p['trigger']} {p['strategy']}"
        page_texts.append(txt)
        index_lines.append(f"P{p['cluster']:03d} | {p['name']} (n={p['n_members']}) | {p['trigger']}")
    (WIKI_DIR / "index.md").write_text(
        "# RAD 攻击策略 wiki 索引\n\n" + "\n".join(index_lines) + "\n", encoding="utf-8")
    page_emb = enc.encode(page_texts, batch_size=32)
    np.save(WIKI_DIR / "page_emb.npy", page_emb)

    # 6/6 近重复 lint
    print("6/6 lint：近重复页检测…")
    Pn = page_emb / np.linalg.norm(page_emb, axis=1, keepdims=True)
    S = Pn @ Pn.T
    dup = [(pages[i]["name"], pages[j]["name"], float(S[i, j]))
           for i in range(len(pages)) for j in range(i + 1, len(pages)) if S[i, j] > 0.92]
    meta = {"k": K, "n_pages": len(pages), "n_examples": len(db),
            "dup_warnings": dup, "build_seconds": round(time.time() - t0)}
    (WIKI_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    print(f"完成：{len(pages)} 页，dup警告 {len(dup)}，总耗时 {time.time()-t0:.0f}s → {WIKI_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
