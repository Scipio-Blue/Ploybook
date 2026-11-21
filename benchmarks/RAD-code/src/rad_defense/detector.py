from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Sequence

from .config import RADConfig
from .data import (
    DEFAULT_DATASET_ID,
    PAPER_DATABASE_PATH,
    PAPER_DATASET_REVISION,
    AttackExample,
    RADDatabase,
    default_cache_dir,
    load_database,
)
from .index import FaissIndex
from .interfaces import RADModelBackend, SimilarityIndex, TextEncoder

NO_STRATEGY = "The user prompt is straightforward and genuine in expressing the query."
NO_STRATEGY_GUIDANCE = "The intent can be taken by directly summarizing the prompt."
EXTRACTOR_SYSTEM = (
    "You are an Intent Extraction Assistant specialized in accurately identifying and "
    "extracting user intents from prompts. Your goal is to carefully analyze the provided "
    "strategies, clearly articulate your reasoning, and explicitly state the user's actual intent."
)


@dataclass(frozen=True)
class EvidenceScore:
    database_index: int | None
    example_id: str
    retrieval_score: float | None
    rerank_score: float
    strategy: str
    guidance: str
    intent: str
    harmful_probability: float
    vote_weight: float


@dataclass(frozen=True)
class RADResult:
    prompt: str
    harmful_probability: float
    threshold: float
    blocked: bool
    response: str | None
    evidence: tuple[EvidenceScore, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    maximum = max(values)
    exponents = [math.exp(value - maximum) for value in values]
    denominator = sum(exponents)
    return [value / denominator for value in exponents]


def _clean_strategy(value: str) -> str:
    strategy = value.strip()
    if strategy.startswith('"'):
        strategy = strategy[1:]
    if strategy.endswith('"'):
        strategy = strategy[:-1]
    if strategy.startswith("Analysis: "):
        strategy = strategy[len("Analysis: ") :]
    replacements = {
        "The prompt": "The user prompt",
        "the prompt": "the user prompt",
        "This prompt": "The user prompt",
        "this prompt": "the user prompt",
    }
    for old, new in replacements.items():
        strategy = strategy.replace(old, new)
    return strategy


def _extract_intent(response: str) -> str:
    value = response.split("</think>")[-1].strip()
    matches = re.findall(r"(?:^|\n)\s*-?\s*Extracted Intent:\s*(.+)", value, flags=re.I)
    if matches:
        return matches[-1].strip()
    if value.startswith("Intent:"):
        value = value[len("Intent:") :].strip()
    if "Summary:" in value:
        value = value.split("Summary:", 1)[0].strip()
    return value or "Ambiguous"


class RADDetector:
    """Retrieve, rerank, extract, classify, and vote over attack examples."""

    def __init__(
        self,
        config: RADConfig,
        database: Sequence[AttackExample],
        encoder: TextEncoder,
        language_model: RADModelBackend,
        index: SimilarityIndex,
    ) -> None:
        if len(database) != index.size:
            raise ValueError("database and vector index must contain the same number of rows")
        self.config = config
        self.database = tuple(database)
        self.encoder = encoder
        self.model_backend = language_model
        # Compatibility for integrations built against the pre-release API.
        self.language_model = language_model
        self.index = index
        prompt_dir = files("rad_defense").joinpath("prompts")
        self.extract_template = prompt_dir.joinpath("extract.txt").read_text(
            encoding="utf-8"
        ).rstrip("\n")
        self.classify_full_template = prompt_dir.joinpath("classify_full.txt").read_text(
            encoding="utf-8"
        ).rstrip("\n")
        self.classify_intent_template = prompt_dir.joinpath("classify_intent.txt").read_text(
            encoding="utf-8"
        ).rstrip("\n")

    @classmethod
    def from_paths(
        cls,
        config: RADConfig,
        database_path: str | Path,
        index_path: str | Path,
        *,
        rebuild_index: bool = False,
    ) -> "RADDetector":
        from .models import ContrieverEncoder, TransformersLanguageModel

        database = load_database(database_path)
        encoder = ContrieverEncoder(config.encoder_id, config.encoder_revision)
        index_file = Path(index_path)
        if rebuild_index or not index_file.exists():
            index = FaissIndex.build(database, encoder)
            index.save(index_file, database, encoder)
        else:
            index = FaissIndex.load(index_file, database, encoder)
        language_model = TransformersLanguageModel(
            config.model_id,
            config.model_revision,
            max_new_tokens=config.max_new_tokens,
        )
        return cls(config, database, encoder, language_model, index)

    @classmethod
    def from_database(
        cls,
        config: RADConfig,
        database: RADDatabase | Sequence[AttackExample] | str | Path,
        *,
        encoder: TextEncoder,
        model_backend: RADModelBackend,
        index: SimilarityIndex | None = None,
        index_path: str | Path | None = None,
        rebuild_index: bool = False,
        batch_size: int = 32,
    ) -> "RADDetector":
        """Construct RAD from a user-supplied database and pluggable backends."""

        if isinstance(database, (str, Path)):
            examples: Sequence[AttackExample] = RADDatabase.from_csv(database)
        else:
            examples = database
        if index is not None and index_path is not None:
            raise ValueError("provide index or index_path, not both")
        if index is None:
            if index_path is None:
                index = FaissIndex.build(examples, encoder, batch_size)
            else:
                target = Path(index_path)
                if rebuild_index or not target.exists():
                    index = FaissIndex.build(examples, encoder, batch_size)
                    index.save(target, examples, encoder)
                else:
                    index = FaissIndex.load(target, examples, encoder)
        return cls(config, examples, encoder, model_backend, index)

    @classmethod
    def from_pretrained(
        cls,
        preset: str | RADConfig = "paper-sim",
        *,
        cache_dir: str | Path | None = None,
        repo_id: str = DEFAULT_DATASET_ID,
        revision: str = PAPER_DATASET_REVISION,
        database_filename: str = PAPER_DATABASE_PATH,
        local_files_only: bool = False,
        rebuild_index: bool = False,
        index_batch_size: int = 32,
        device: str | None = None,
        device_map: str = "auto",
        encoder: TextEncoder | None = None,
        model_backend: RADModelBackend | None = None,
    ) -> "RADDetector":
        """Load a preset or explicit configuration with a Hub database and cached index."""

        from .artifacts import prepare_artifacts
        from .models import ContrieverEncoder, TransformersLanguageModel

        config = preset if isinstance(preset, RADConfig) else RADConfig.from_preset(preset)
        cache_root = (
            Path(cache_dir).expanduser().resolve()
            if cache_dir is not None
            else default_cache_dir().resolve()
        )
        active_encoder = encoder or ContrieverEncoder(
            config.encoder_id,
            config.encoder_revision,
            device=device,
            local_files_only=local_files_only,
            cache_dir=str(cache_root / "hub"),
        )
        artifacts = prepare_artifacts(
            config,
            active_encoder,
            cache_dir=cache_root,
            repo_id=repo_id,
            revision=revision,
            filename=database_filename,
            local_files_only=local_files_only,
            rebuild_index=rebuild_index,
            batch_size=index_batch_size,
        )
        active_backend = model_backend or TransformersLanguageModel(
            config.model_id,
            config.model_revision,
            max_new_tokens=config.max_new_tokens,
            device_map=device_map,
            local_files_only=local_files_only,
            cache_dir=str(cache_root / "hub"),
        )
        return cls(
            config,
            artifacts.database,
            active_encoder,
            active_backend,
            artifacts.index,
        )

    @staticmethod
    def _pmi_context(strategy: str) -> str:
        return (
            "The following strategy accurately describes the patterns of the user prompt:\n"
            f"Strategy: {strategy}\nUser prompt: "
        )

    @staticmethod
    def _likelihood_context(prompt: str) -> str:
        return (
            "The patterns and wordings of the following user prompt is accurately described "
            "and analyzed by the following analysis.\n\n"
            f"User prompt: {prompt}\nAnalysis: "
        )

    def _rerank(
        self, prompt: str, indices: Sequence[int], retrieval_scores: Sequence[float]
    ) -> list[tuple[int, float, float]]:
        ranked: list[tuple[int, float, float]] = []
        if self.config.rerank_mode == "pmi":
            base = self.language_model.conditional_log_likelihood(
                "User prompt: ", prompt, length_normalize=True
            )
            for index, retrieval_score in zip(indices, retrieval_scores):
                strategy = _clean_strategy(self.database[index].analysis)
                score = (
                    self.language_model.conditional_log_likelihood(
                        self._pmi_context(strategy), prompt, length_normalize=True
                    )
                    - base
                )
                if score > 0:
                    ranked.append((index, float(retrieval_score), score))
        else:
            context = self._likelihood_context(prompt)
            base = self.language_model.conditional_log_likelihood(
                context, NO_STRATEGY, length_normalize=True
            )
            for index, retrieval_score in zip(indices, retrieval_scores):
                strategy = _clean_strategy(self.database[index].analysis)
                score = self.language_model.conditional_log_likelihood(
                    context, strategy, length_normalize=True
                )
                if score - base > 0:
                    ranked.append((index, float(retrieval_score), score))
        ranked.sort(key=lambda item: item[2], reverse=True)
        return ranked[: self.config.rerank_k]

    def _no_strategy_score(self, prompt: str) -> float:
        if self.config.rerank_mode == "likelihood":
            return self.language_model.conditional_log_likelihood(
                self._likelihood_context(prompt), NO_STRATEGY, length_normalize=True
            )
        base = self.language_model.conditional_log_likelihood(
                "User prompt: ", prompt, length_normalize=True
            )
        return (
            self.language_model.conditional_log_likelihood(
                self._pmi_context(NO_STRATEGY), prompt, length_normalize=True
            )
            - base
        )

    def _intent_likelihood(self, prompt: str, strategy: str, intent: str) -> float:
        context = (
            "You are an Intent Extraction Assistant. Your primary task is to clearly identify "
            "the user's actual intent based on the provided user prompt.\n\n"
            f"User Prompt:\n{prompt}\n\n"
            "You are given a strategy that explains how the user prompt could be constructed "
            "from the user's actual intent.\n\n"
            f"Strategy:\n{strategy}\n\n"
            "Your Tasks:\n"
            "1. Analyze how the strategy explains the construction of the user prompt.\n"
            "2. Extract the user's actual intent according to the strategy.\n\n"
            "Output:\nExtracted Intent: "
        )
        return self.language_model.conditional_log_likelihood(
            context, intent, length_normalize=True
        )

    def score(self, prompt: str, *, threshold: float | None = None) -> RADResult:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt must not be empty")
        active_threshold = self.config.threshold if threshold is None else threshold
        if not 0 <= active_threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")

        query = self.encoder.encode([prompt])
        retrieval_scores, retrieved_indices = self.index.search(query, self.config.retrieve_k)
        ranked = self._rerank(
            prompt,
            retrieved_indices[0].tolist(),
            retrieval_scores[0].tolist(),
        )

        candidates: list[tuple[int | None, float | None, float, str, str]] = []
        for index, retrieval_score, rerank_score in ranked:
            example = self.database[index]
            candidates.append(
                (
                    index,
                    retrieval_score,
                    rerank_score,
                    _clean_strategy(example.analysis),
                    example.guidance.strip(),
                )
            )
        candidates.append(
            (None, None, self._no_strategy_score(prompt), NO_STRATEGY, NO_STRATEGY_GUIDANCE)
        )

        extraction_prompts = [
            self.extract_template.format(prompt=prompt, strategy=strategy, guidance=guidance)
            for _, _, _, strategy, guidance in candidates
        ]
        intents = [
            _extract_intent(value)
            for value in self.language_model.generate(
                extraction_prompts, system_prompt=EXTRACTOR_SYSTEM
            )
        ]

        if self.config.classifier_mode == "intent_only":
            classifier_prompts = [
                self.classify_intent_template.format(intent=intent) for intent in intents
            ]
        else:
            classifier_prompts = [
                self.classify_full_template.format(
                    prompt=prompt, strategy=item[3], intent=intent
                )
                for item, intent in zip(candidates, intents)
            ]
        harmful_probabilities = self.language_model.harmful_probabilities(classifier_prompts)
        if len(harmful_probabilities) != len(candidates):
            raise ValueError("classifier returned a different number of scores than requested")

        if self.config.vote_mode == "majority":
            weights = [1.0 / len(candidates)] * len(candidates)
        else:
            joint_scores = [
                item[2] + self._intent_likelihood(prompt, item[3], intent)
                for item, intent in zip(candidates, intents)
            ]
            weights = _softmax(joint_scores)
        harmful_probability = float(
            sum(
                weight * probability
                for weight, probability in zip(weights, harmful_probabilities)
            )
        )
        blocked = harmful_probability >= active_threshold
        evidence = tuple(
            EvidenceScore(
                database_index=item[0],
                example_id=(
                    self.database[item[0]].example_id
                    if item[0] is not None
                    else "no-strategy"
                ),
                retrieval_score=item[1],
                rerank_score=item[2],
                strategy=item[3],
                guidance=item[4],
                intent=intent,
                harmful_probability=float(probability),
                vote_weight=float(weight),
            )
            for item, intent, probability, weight in zip(
                candidates, intents, harmful_probabilities, weights
            )
        )
        return RADResult(
            prompt=prompt,
            harmful_probability=harmful_probability,
            threshold=active_threshold,
            blocked=blocked,
            response=self.config.refusal_response if blocked else None,
            evidence=evidence,
        )

    def predict(self, prompt: str, *, threshold: float | None = None) -> bool:
        """Return True when RAD classifies the prompt as harmful."""

        return self.score(prompt, threshold=threshold).blocked

    def score_batch(
        self, prompts: Sequence[str], *, threshold: float | None = None
    ) -> list[RADResult]:
        """Score prompts in input order.

        The stable API permits backends to optimize this operation later; the reference
        backend deliberately uses the exact single-prompt paper path for each item.
        """

        return [self.score(prompt, threshold=threshold) for prompt in prompts]
