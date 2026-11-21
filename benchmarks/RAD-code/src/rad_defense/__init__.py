"""Public interface for Retrieval-Augmented Defense."""

from .config import RADConfig
from .curation import (
    CurationConfig,
    CurationResult,
    GenerationBackend,
    RADCurator,
    curate_csv,
    curate_hub,
)
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
    "CurationConfig",
    "CurationResult",
    "EvidenceScore",
    "GenerationBackend",
    "RADConfig",
    "RADCurator",
    "RADDatabase",
    "RADDetector",
    "RADModelBackend",
    "RADResult",
    "SimilarityIndex",
    "TextEncoder",
    "curate_csv",
    "curate_hub",
    "download_dataset",
    "load_database",
]

__version__ = "0.2.1"
