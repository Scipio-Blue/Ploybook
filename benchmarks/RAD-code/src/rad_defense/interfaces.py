from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class TextEncoder(Protocol):
    model_id: str
    revision: str

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray: ...


class RADModelBackend(Protocol):
    """Model operations required by the five-stage RAD algorithm."""

    def generate(self, prompts: Sequence[str], *, system_prompt: str | None = None) -> list[str]: ...

    def conditional_log_likelihood(
        self, context: str, continuation: str, *, length_normalize: bool = False
    ) -> float: ...

    def harmful_probabilities(self, prompts: Sequence[str]) -> list[float]: ...


# Backwards-compatible name for pre-release integrations.
LanguageModel = RADModelBackend


class SimilarityIndex(Protocol):
    size: int

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...
