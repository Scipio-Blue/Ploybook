from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

RerankMode = Literal["pmi", "likelihood"]
VoteMode = Literal["weighted", "majority"]
ClassifierMode = Literal["prompt_strategy_intent", "intent_only"]

QWEN_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
CONTRIEVER_REVISION = "abe8c1493371369031bcb1e02acb754cf4e162fa"


@dataclass(frozen=True)
class RADConfig:
    """Configuration for the two RAD variants reported in the paper."""

    name: str = "rad-sim-14b"
    model_id: str = "Qwen/Qwen2.5-14B-Instruct"
    model_revision: str = QWEN_REVISION
    encoder_id: str = "facebook/contriever-msmarco"
    encoder_revision: str = CONTRIEVER_REVISION
    retrieve_k: int = 50
    rerank_k: int = 10
    rerank_mode: RerankMode = "likelihood"
    vote_mode: VoteMode = "majority"
    classifier_mode: ClassifierMode = "intent_only"
    threshold: float = 0.5
    max_new_tokens: int = 1024
    refusal_response: str = "I'm sorry, but I can't help with that."

    def __post_init__(self) -> None:
        if self.retrieve_k < 1:
            raise ValueError("retrieve_k must be positive")
        if self.rerank_k < 1 or self.rerank_k > self.retrieve_k:
            raise ValueError("rerank_k must be between 1 and retrieve_k")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if self.rerank_mode not in {"pmi", "likelihood"}:
            raise ValueError(f"unsupported rerank_mode: {self.rerank_mode}")
        if self.vote_mode not in {"weighted", "majority"}:
            raise ValueError(f"unsupported vote_mode: {self.vote_mode}")
        if self.classifier_mode not in {"prompt_strategy_intent", "intent_only"}:
            raise ValueError(f"unsupported classifier_mode: {self.classifier_mode}")

        for field, value in (
            ("name", self.name),
            ("model_id", self.model_id),
            ("model_revision", self.model_revision),
            ("encoder_id", self.encoder_id),
            ("encoder_revision", self.encoder_revision),
        ):
            if not value.strip():
                raise ValueError(f"{field} must not be empty")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RADConfig":
        with Path(path).open(encoding="utf-8") as handle:
            values: dict[str, Any] = yaml.safe_load(handle) or {}
        if "model_id" in values and "model_revision" not in values:
            values["model_revision"] = "main"
        if "encoder_id" in values and "encoder_revision" not in values:
            values["encoder_revision"] = "main"
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def paper_base(cls) -> "RADConfig":
        return cls(
            name="rad-base-14b",
            rerank_mode="pmi",
            vote_mode="weighted",
            classifier_mode="prompt_strategy_intent",
        )

    @classmethod
    def paper_sim(cls) -> "RADConfig":
        return cls()

    @classmethod
    def from_preset(cls, name: str = "paper-sim") -> "RADConfig":
        normalized = name.strip().lower().replace("_", "-")
        if normalized in {"paper-sim", "rad-sim", "rad-sim-14b"}:
            return cls.paper_sim()
        if normalized in {"paper-base", "rad-base", "rad-base-14b"}:
            return cls.paper_base()
        raise ValueError(
            f"unknown RAD preset {name!r}; choose 'paper-sim' or 'paper-base'"
        )
