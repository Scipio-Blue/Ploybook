from __future__ import annotations

import fcntl
import hashlib
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .config import RADConfig
from .data import (
    DEFAULT_DATASET_ID,
    PAPER_DATABASE_DIGEST,
    PAPER_DATABASE_PATH,
    PAPER_DATASET_REVISION,
    RADDatabase,
    default_cache_dir,
)
from .index import FaissIndex
from .interfaces import TextEncoder


@dataclass(frozen=True)
class PreparedRADArtifacts:
    database: RADDatabase
    index: FaissIndex
    index_path: Path
    cache_dir: Path


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _index_cache_path(cache_root: Path, database: RADDatabase, encoder: TextEncoder) -> Path:
    identity = f"{database.digest}\0{encoder.model_id}\0{encoder.revision}"
    key = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return cache_root / "indexes" / f"{key}.faiss"


def _build_index_atomically(
    target: Path,
    database: RADDatabase,
    encoder: TextEncoder,
    *,
    batch_size: int,
) -> FaissIndex:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rad-index-", dir=target.parent) as temporary_dir:
        temporary = Path(temporary_dir) / target.name
        index = FaissIndex.build(database, encoder, batch_size)
        index.save(temporary, database, encoder)
        temporary_metadata = temporary.with_suffix(temporary.suffix + ".json")
        target_metadata = target.with_suffix(target.suffix + ".json")
        os.replace(temporary, target)
        os.replace(temporary_metadata, target_metadata)
    return FaissIndex.load(target, database, encoder)


def prepare_artifacts(
    config: RADConfig,
    encoder: TextEncoder,
    *,
    cache_dir: str | Path | None = None,
    repo_id: str = DEFAULT_DATASET_ID,
    revision: str = PAPER_DATASET_REVISION,
    filename: str = PAPER_DATABASE_PATH,
    local_files_only: bool = False,
    rebuild_index: bool = False,
    batch_size: int = 32,
) -> PreparedRADArtifacts:
    """Resolve the paper database and a matching, concurrency-safe FAISS index."""

    cache_root = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir is not None
        else default_cache_dir().resolve()
    )
    expected_digest = (
        PAPER_DATABASE_DIGEST
        if repo_id == DEFAULT_DATASET_ID
        and revision == PAPER_DATASET_REVISION
        and filename == PAPER_DATABASE_PATH
        else None
    )
    database = RADDatabase.from_hub(
        repo_id=repo_id,
        revision=revision,
        filename=filename,
        cache_dir=cache_root,
        local_files_only=local_files_only,
        expected_digest=expected_digest,
    )
    index_path = _index_cache_path(cache_root, database, encoder)
    metadata_path = index_path.with_suffix(index_path.suffix + ".json")
    lock_path = index_path.with_suffix(index_path.suffix + ".lock")
    with _exclusive_lock(lock_path):
        if not rebuild_index and index_path.exists() and metadata_path.exists():
            try:
                index = FaissIndex.load(index_path, database, encoder)
            except (OSError, RuntimeError, ValueError):
                index = _build_index_atomically(
                    index_path, database, encoder, batch_size=batch_size
                )
        else:
            index = _build_index_atomically(index_path, database, encoder, batch_size=batch_size)
    return PreparedRADArtifacts(database, index, index_path, cache_root)
