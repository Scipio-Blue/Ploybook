from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .data import AttackExample, database_digest
from .interfaces import SimilarityIndex, TextEncoder


def _normalize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("encoder returned a zero-length embedding")
    return values / norms


@dataclass
class NumpyIndex(SimilarityIndex):
    """Small exact index used by tests and CPU smoke runs."""

    embeddings: np.ndarray

    def __post_init__(self) -> None:
        self.embeddings = _normalize(self.embeddings)

    @property
    def size(self) -> int:
        return int(self.embeddings.shape[0])

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        query = _normalize(query)
        k = min(k, self.size)
        similarities = query @ self.embeddings.T
        indices = np.argsort(-similarities, axis=1)[:, :k]
        scores = np.take_along_axis(similarities, indices, axis=1)
        return scores.astype(np.float32), indices.astype(np.int64)


class FaissIndex(SimilarityIndex):
    def __init__(self, index: object) -> None:
        self.index = index

    @property
    def size(self) -> int:
        return int(self.index.ntotal)  # type: ignore[attr-defined]

    @classmethod
    def build(
        cls,
        examples: Sequence[AttackExample],
        encoder: TextEncoder,
        batch_size: int = 32,
    ) -> "FaissIndex":
        import faiss

        embeddings = _normalize(encoder.encode([item.jailbreak for item in examples], batch_size))
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return cls(index)

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        query = _normalize(query)
        k = min(k, self.size)
        return self.index.search(query, k)  # type: ignore[no-any-return,attr-defined]

    def save(
        self,
        path: str | Path,
        examples: Sequence[AttackExample],
        encoder: TextEncoder,
    ) -> None:
        import faiss

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(target))
        metadata = {
            "database_sha256": database_digest(examples),
            "encoder_id": encoder.model_id,
            "encoder_revision": encoder.revision,
            "rows": len(examples),
        }
        target.with_suffix(target.suffix + ".json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        examples: Sequence[AttackExample],
        encoder: TextEncoder,
    ) -> "FaissIndex":
        import faiss

        source = Path(path)
        metadata_path = source.with_suffix(source.suffix + ".json")
        if not metadata_path.exists():
            raise ValueError(f"index metadata not found: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "database_sha256": database_digest(examples),
            "encoder_id": encoder.model_id,
            "encoder_revision": encoder.revision,
            "rows": len(examples),
        }
        if metadata != expected:
            raise ValueError("FAISS index metadata does not match the database or encoder")
        instance = cls(faiss.read_index(str(source)))
        if instance.size != len(examples):
            raise ValueError("FAISS index row count does not match the database")
        return instance

