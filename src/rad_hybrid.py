"""rad_hybrid — RAD 复现的可插拔模型后端（HybridBackend）。

论文口径（RAD-Sim-14B）：所有模型操作由 Qwen2.5-14B-Instruct（rev cf98f3b）完成。
本机 Mac M4 24GB 放不下 bf16（~28GB），采用混合后端：
  - generate（Extract 步骤，开销最大）→ DashScope API 同款模型（可切换 mlx 本地兜底）
  - conditional_log_likelihood / harmful_probabilities（Rerank/Classify，需要 logits，
    API 不提供 prompt 打分）→ 本地 MLX 4-bit 量化权重前向

已知偏差（记录进 PROJECT.md 指标口径）：
  D1. 打分步骤在 4-bit 量化权重上计算（论文为 bf16）；rerank 是相对排序，预期影响小
  D2. 若 generate 走本地 MLX，生成同样来自 4-bit 权重
  D3. API 生成 serving 与论文 A100 逐位生成可能存在非确定性差异（temperature=0）

实现严格对齐 benchmarks/RAD-code/src/rad_defense/models.py 的
TransformersLanguageModel：同样的 context/continuation 拼接、同样的 chat 模板、
同样的 Bad/bad/Good/good 单 token softmax。
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".llm_cache"
MLX_MODEL_PATH = ROOT / "models" / "Qwen2.5-14B-Instruct-4bit"

# OpenAI 兼容 API 生成器默认指向 DashScope；百炼已下架 qwen2.5-14b-instruct，
# 可经环境变量整体改指向其他兼容渠道（ModelScope/SiliconFlow 等）：
#   RAD_API_BASE / RAD_API_MODEL / RAD_API_KEY_ENV（默认读 QWEN_KEY）
DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_MODEL = "qwen2.5-14b-instruct"

_cache_lock = threading.Lock()


def _cache_get(key: str) -> str | None:
    path = CACHE_DIR / f"{key}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))["content"]
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def _cache_put(key: str, content: str) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    tmp = CACHE_DIR / f"{key}.tmp"
    final = CACHE_DIR / f"{key}.json"
    with _cache_lock:
        tmp.write_text(json.dumps({"content": content}, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, final)


def _key(kind: str, *parts: object) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\0")
    return f"{kind}-{h.hexdigest()[:40]}"


class MLXQwenScorer:
    """本地 MLX 打分器：条件对数似然 + Bad/Good 下一 token 概率。

    数学上复刻 TransformersLanguageModel：
      - cll: tokenizer(context, add_special_tokens=True) + tokenizer(continuation,
        add_special_tokens=False) 拼接前向，取 continuation 段 logprob 之和（可长度归一）
      - harmful_probabilities: chat 模板（system=You are a helpful assistant.）+
        add_generation_prompt=True，取最后位置 logits，在 {Bad,bad} vs {Good,good}
        单 token 集合上 softmax，P = P(Bad 组)
    """

    def __init__(self, model_path: str | Path = MLX_MODEL_PATH) -> None:
        import mlx.core as mx
        from mlx_lm import load

        self._mx = mx
        self.model, self.tokenizer = load(str(model_path))
        self.usage = {"prompt_tokens": 0, "output_tokens": 0,
                      "n_cll": 0, "n_gen": 0, "n_cls": 0}

    def usage_reset(self) -> None:
        for k in self.usage:
            self.usage[k] = 0

    def _logprobs_of(self, ids: list[int], cont_len: int) -> list[float]:
        mx = self._mx
        logits = self.model(mx.array([ids]))  # (1, T, V)
        # 对齐 repo：logits[:, :-1] 与 targets[:, 1:] 对位
        logits = logits[:, :-1, :]
        # 数值稳定 log_softmax（fp16 下必须先减 max，否则 exp 溢出全变 -inf）
        m = mx.max(logits, axis=-1, keepdims=True)
        log_softmax = logits - m - mx.log(mx.sum(mx.exp(logits - m), axis=-1, keepdims=True))
        targets = mx.array([ids[1:]])
        tok_lp = mx.take_along_axis(log_softmax, targets[:, :, None], axis=-1)[0, :, 0]
        start = len(ids) - cont_len - 1
        return [float(v) for v in tok_lp[start : start + cont_len]]

    def conditional_log_likelihood(
        self, context: str, continuation: str, *, length_normalize: bool = False
    ) -> float:
        key = _key("cll", context, continuation, length_normalize)
        cached = _cache_get(key)
        if cached is not None:
            return float(cached)
        ctx_ids = self.tokenizer.encode(context)
        cont_ids = self.tokenizer.encode(continuation)  # Qwen 无 BOS，等价 add_special_tokens=False
        ids = ctx_ids + cont_ids
        self.usage["prompt_tokens"] += len(ids)
        self.usage["n_cll"] += 1
        lps = self._logprobs_of(ids, len(cont_ids))
        score = sum(lps)
        if length_normalize and lps:
            score /= len(lps)
        _cache_put(key, repr(score))
        return score

    def harmful_probabilities(self, prompts: Sequence[str]) -> list[float]:
        results: list[float] = []
        # token 组：对齐 repo（Bad/bad → 组0，Good/good → 组1，仅取单 token 编码）
        groups: dict[int, int] = {}
        for group, variants in enumerate((("Bad", "bad"), ("Good", "good"))):
            for value in variants:
                ids = self.tokenizer.encode(value)
                if len(ids) == 1:
                    groups.setdefault(ids[0], group)
        if set(groups.values()) != {0, 1}:
            raise ValueError("tokenizer 必须把 Bad/Good 各编码为单 token")
        token_ids = list(groups)
        for prompt in prompts:
            key = _key("harm", prompt)
            cached = _cache_get(key)
            if cached is not None:
                results.append(float(cached))
                continue
            text = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            ids = self.tokenizer.encode(text)
            self.usage["prompt_tokens"] += len(ids)
            self.usage["n_cls"] += 1
            mx = self._mx
            logits = self.model(mx.array([ids]))[:, -1, :].reshape(-1)
            picked = mx.array([logits[i] for i in token_ids])
            probs = mx.softmax(picked)
            p_bad = float(sum(probs[i].item() for i, tid in enumerate(token_ids) if groups[tid] == 0))
            _cache_put(key, repr(p_bad))
            results.append(p_bad)
        return results


class DashScopeGenerator:
    """OpenAI 兼容 API 生成器（DashScope/ModelScope/SiliconFlow 通吃，环境变量切换）。"""

    def __init__(self, model: str | None = None, max_new_tokens: int = 1024,
                 retries: int = 4, workers: int = 4) -> None:
        self.base = os.environ.get("RAD_API_BASE", DASHSCOPE_BASE)
        self.model = model or os.environ.get("RAD_API_MODEL", DASHSCOPE_MODEL)
        key_env = os.environ.get("RAD_API_KEY_ENV", "QWEN_KEY")
        self.key = os.environ.get(key_env, "")
        self.max_new_tokens = max_new_tokens
        self.retries = retries
        self.workers = workers

    def _one(self, prompt: str, system_prompt: str) -> str:
        key = _key("gen", self.model, system_prompt, prompt)
        cached = _cache_get(key)
        if cached is not None:
            return cached
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_new_tokens,
            "temperature": 0.0,
        }).encode()
        last_err: Exception | None = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(
                    self.base + "/chat/completions", data=body, method="POST",
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {self.key}"},
                )
                with urllib.request.urlopen(req, timeout=180) as resp:
                    data = json.loads(resp.read().decode())
                content = data["choices"][0]["message"]["content"]
                _cache_put(key, content)
                return content
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code in (429, 500, 502, 503):
                    time.sleep(2 ** attempt)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"API 生成请求失败（{self.retries} 次）: {last_err}")

    def generate(self, prompts: Sequence[str], *, system_prompt: str | None = None) -> list[str]:
        sysmsg = system_prompt or "You are a helpful assistant."
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(lambda p: self._one(p, sysmsg), prompts))


class MLXGenerator:
    """本地 MLX 生成兜底（DashScope 无 14B 权限时）。确定性 temperature=0。"""

    def __init__(self, scorer: MLXQwenScorer, max_new_tokens: int = 1024) -> None:
        self.scorer = scorer
        self.max_new_tokens = max_new_tokens

    def generate(self, prompts: Sequence[str], *, system_prompt: str | None = None) -> list[str]:
        from mlx_lm import generate as mlx_generate

        sysmsg = system_prompt or "You are a helpful assistant."
        out: list[str] = []
        for prompt in prompts:
            key = _key("genmlx", prompt, sysmsg)
            cached = _cache_get(key)
            if cached is not None:
                out.append(cached)
                continue
            text = self.scorer.tokenizer.apply_chat_template(
                [{"role": "system", "content": sysmsg},
                 {"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
            result = mlx_generate(
                self.scorer.model, self.scorer.tokenizer, prompt=text,
                max_tokens=self.max_new_tokens, verbose=False)
            self.scorer.usage["prompt_tokens"] += len(self.scorer.tokenizer.encode(text))
            self.scorer.usage["output_tokens"] += len(self.scorer.tokenizer.encode(result))
            self.scorer.usage["n_gen"] += 1
            _cache_put(key, result)
            out.append(result)
        return out


class HybridBackend:
    """RADModelBackend 协议实现：API/本地生成 + 本地打分。"""

    def __init__(self, generator, scorer: MLXQwenScorer) -> None:
        self.generator = generator
        self.scorer = scorer

    # 协议三方法
    def generate(self, prompts, *, system_prompt=None):
        return self.generator.generate(prompts, system_prompt=system_prompt)

    def conditional_log_likelihood(self, context, continuation, *, length_normalize=False):
        return self.scorer.conditional_log_likelihood(
            context, continuation, length_normalize=length_normalize)

    def harmful_probabilities(self, prompts):
        return self.scorer.harmful_probabilities(prompts)


def build_backend(max_new_tokens: int = 1024) -> HybridBackend:
    """按环境变量 RAD_GENERATOR=dashscope|mlx 选择生成器，打分恒为本地 MLX。"""
    scorer = MLXQwenScorer()
    mode = os.environ.get("RAD_GENERATOR", "dashscope")
    if mode == "dashscope":
        generator = DashScopeGenerator(max_new_tokens=max_new_tokens)
    else:
        generator = MLXGenerator(scorer, max_new_tokens=max_new_tokens)
    return HybridBackend(generator, scorer)
