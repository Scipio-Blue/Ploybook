"""conformal — split-conformal FRR 保证层（风险校准决策，整条 RAD 谱系无人做过）。

给定校准集良性样本的 P(harmful) 分数，对目标 FRR 预算 α 给出带统计保证的阈值：
  τ_α = 校准良性分数的 ⌈(n+1)(1−α)⌉/n 经验分位
则 P(新良性被拦) ≤ α（有限样本保证，无分布假设）。
用法：预实验跑完后取 benign 臂的 p_harmful 列作校准 → 各 α 档阈值 → 操作曲线带保证。
"""
from __future__ import annotations


def conformal_threshold(benign_scores: list[float], alpha: float) -> float:
    """split-conformal 上分位：保证新良性样本被拦截的概率 ≤ alpha。"""
    import math
    n = len(benign_scores)
    if n == 0 or alpha <= 0 or alpha >= 1:
        return 1.0
    s = sorted(benign_scores)
    k = min(n, max(1, math.ceil((n + 1) * (1 - alpha))))
    return s[k - 1]


def operating_points(benign_scores: list[float], attack_scores: list[float],
                     alphas: list[float] | None = None) -> list[dict]:
    """各 FRR 预算下的保证化操作点：阈值/拦截率(TPR)/实际校准FRR。"""
    if alphas is None:
        alphas = [0.01, 0.025, 0.05, 0.10]
    out = []
    for a in alphas:
        tau = conformal_threshold(benign_scores, a)
        tpr = sum(1 for p in attack_scores if p >= tau) / max(1, len(attack_scores))
        frr = sum(1 for p in benign_scores if p >= tau) / max(1, len(benign_scores))
        out.append({"alpha": a, "threshold": round(tau, 4),
                    "TPR_guaranteed_FRR": round(tpr, 4), "calib_FRR": round(frr, 4)})
    return out
