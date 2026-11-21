from __future__ import annotations

import csv
import hashlib
import os
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATASET_ID = "BruceYg/RAD-Database"
PAPER_DATASET_REVISION = "6e5d5ddcc93a9bedd543a33fc3790470f618eeb7"
PAPER_DATABASE_PATH = "database/full.csv"
PAPER_DATABASE_DIGEST = "559aea650bded65c1df7050cb17ab89d13901710c5327bd8d3282559f24952a0"
REQUIRED_COLUMNS = ("query", "jailbreak", "analysis", "guidance")


def default_cache_dir() -> Path:
    """Return the RAD cache root, honoring standard user cache configuration."""

    override = os.environ.get("RAD_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache).expanduser() / "rad-defense"
    return Path.home() / ".cache" / "rad-defense"


@dataclass(frozen=True)
class AttackExample:
    query: str
    jailbreak: str
    analysis: str
    guidance: str
    example_id: str = ""
    attack_family: str = ""
    split: str = ""


def _generated_example_id(required: dict[str, str], occurrence: int) -> str:
    digest = hashlib.sha256()
    for name in REQUIRED_COLUMNS:
        digest.update(required[name].encode("utf-8"))
        digest.update(b"\0")
    digest.update(str(occurrence).encode("ascii"))
    return f"rad-{digest.hexdigest()[:20]}"


def load_database(path: str | Path) -> list[AttackExample]:
    source = Path(path)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"{source} is missing required columns: {names}")
        examples = []
        occurrences: defaultdict[tuple[str, ...], int] = defaultdict(int)
        seen_ids: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            required = {name: (row.get(name) or "").strip() for name in REQUIRED_COLUMNS}
            empty = [name for name, value in required.items() if not value]
            if empty:
                raise ValueError(f"{source}:{row_number} has empty fields: {', '.join(empty)}")
            content_key = tuple(required[name] for name in REQUIRED_COLUMNS)
            occurrence = occurrences[content_key]
            occurrences[content_key] += 1
            example_id = (row.get("example_id") or "").strip()
            if not example_id:
                example_id = _generated_example_id(required, occurrence)
            if example_id in seen_ids:
                raise ValueError(f"{source}:{row_number} has duplicate example_id: {example_id}")
            seen_ids.add(example_id)
            examples.append(
                AttackExample(
                    **required,
                    example_id=example_id,
                    attack_family=(row.get("attack_family") or "").strip(),
                    split=(row.get("split") or "").strip(),
                )
            )
    if not examples:
        raise ValueError(f"{source} contains no database examples")
    return examples


def database_digest(examples: Iterable[AttackExample]) -> str:
    digest = hashlib.sha256()
    for example in examples:
        for value in (example.query, example.jailbreak, example.analysis, example.guidance):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class RADDatabase(Sequence[AttackExample]):
    """A validated RAD attack-example database and its content identity."""

    examples: tuple[AttackExample, ...]
    digest: str
    source: Path | None = None
    repo_id: str | None = None
    revision: str | None = None

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        expected_digest: str | None = None,
        repo_id: str | None = None,
        revision: str | None = None,
    ) -> "RADDatabase":
        source = Path(path)
        examples = tuple(load_database(source))
        digest = database_digest(examples)
        if expected_digest is not None and digest != expected_digest:
            raise ValueError(
                f"database digest mismatch for {source}: expected {expected_digest}, got {digest}"
            )
        return cls(examples, digest, source.resolve(), repo_id, revision)

    @classmethod
    def from_hub(
        cls,
        *,
        repo_id: str = DEFAULT_DATASET_ID,
        revision: str = PAPER_DATASET_REVISION,
        filename: str = PAPER_DATABASE_PATH,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        expected_digest: str | None = None,
    ) -> "RADDatabase":
        cache_root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
        if (
            expected_digest is None
            and repo_id == DEFAULT_DATASET_ID
            and revision == PAPER_DATASET_REVISION
            and filename == PAPER_DATABASE_PATH
        ):
            expected_digest = PAPER_DATABASE_DIGEST
        snapshot = download_dataset(
            local_dir=None,
            repo_id=repo_id,
            revision=revision,
            allow_patterns=[filename],
            cache_dir=cache_root / "hub",
            local_files_only=local_files_only,
        )
        source = snapshot / filename
        if not source.exists():
            raise FileNotFoundError(
                f"{filename} was not found in {repo_id}@{revision}; "
                "check dataset access and the selected revision"
            )
        return cls.from_csv(
            source,
            expected_digest=expected_digest,
            repo_id=repo_id,
            revision=revision,
        )

    def __getitem__(self, index: int | slice) -> AttackExample | tuple[AttackExample, ...]:
        return self.examples[index]

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[AttackExample]:
        return iter(self.examples)


def download_dataset(
    local_dir: str | Path | None,
    repo_id: str = DEFAULT_DATASET_ID,
    revision: str = PAPER_DATASET_REVISION,
    allow_patterns: list[str] | None = None,
    *,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> Path:
    """Download a pinned dataset snapshot without embedding credentials."""

    from huggingface_hub import snapshot_download

    destination: Path | None = None
    if local_dir is not None:
        destination = Path(local_dir)
        destination.mkdir(parents=True, exist_ok=True)
    resolved = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=destination,
        cache_dir=Path(cache_dir) if cache_dir is not None else None,
        allow_patterns=allow_patterns,
        local_files_only=local_files_only,
    )
    return Path(resolved)
