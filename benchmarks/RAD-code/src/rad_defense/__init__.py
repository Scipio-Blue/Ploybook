"""Public interface for Retrieval-Augmented Defense."""

from .config import RADConfig
from .data import (
    PAPER_DATABASE_DIGEST,
    PAPER_DATASET_REVISION,
    AttackExample,
    RADDatabase,
    download_dataset,
    load_database,
)
from .detector import EvidenceScore, RADDetector, RADResult
from .interfaces import RADModelBackend, SimilarityIndex, TextEncoder

__all__ = [
    "PAPER_DATABASE_DIGEST",
    "PAPER_DATASET_REVISION",
    "AttackExample",
    "EvidenceScore",
    "RADConfig",
    "RADDatabase",
    "RADDetector",
    "RADModelBackend",
    "RADResult",
    "SimilarityIndex",
    "TextEncoder",
    "download_dataset",
    "load_database",
]

__version__ = "0.2.1"
