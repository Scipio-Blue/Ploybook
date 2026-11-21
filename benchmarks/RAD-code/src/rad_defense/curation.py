"""Stable attack-example curation with resumable, local-only CSV output."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .annotation import (
    ANNOTATION_PROMPT,
    ANNOTATION_SYSTEM_PROMPT,
    Annotation,
    parse_annotation,
)

DEFAULT_CURATION_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
DEFAULT_CURATION_MODEL_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
CURATION_SCHEMA_VERSION = "rad-curation-state-v1"
CURATION_PROMPT_VERSION = "rad-annotation-v1"


class GenerationBackend(Protocol):
    """The only model operation needed by the curation interface."""

    def generate(
        self, prompts: Sequence[str], *, system_prompt: str | None = None
    ) -> list[str]: ...


@dataclass(frozen=True)
class CurationConfig:
    """Model identity and runtime settings for deterministic curation."""

    model_id: str = DEFAULT_CURATION_MODEL_ID
    model_revision: str = DEFAULT_CURATION_MODEL_REVISION
    max_new_tokens: int = 512
    device_map: str = "auto"
    batch_size: int = 1
    cache_dir: str | Path | None = None
    offline: bool = False

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
        if (
            self.model_id != DEFAULT_CURATION_MODEL_ID
            and self.model_revision == DEFAULT_CURATION_MODEL_REVISION
        ):
            object.__setattr__(self, "model_revision", "main")
        if not self.model_revision.strip():
            raise ValueError("model_revision must be non-empty")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not self.device_map.strip():
            raise ValueError("device_map must be non-empty")


@dataclass(frozen=True)
class CurationResult:
    """Paths and terminal row counts for a curation run."""

    output_path: Path
    checkpoint_path: Path
    total: int
    preserved: int
    generated: int
    failed: int

    @property
    def output(self) -> Path:
        """Compatibility-friendly alias for ``output_path``."""

        return self.output_path

    @property
    def checkpoint(self) -> Path:
        """Compatibility-friendly alias for ``checkpoint_path``."""

        return self.checkpoint_path


class RADCurator:
    """Generate and validate RAD annotations with an injectable backend."""

    def __init__(
        self, backend: GenerationBackend, config: CurationConfig | None = None
    ) -> None:
        self.backend = backend
        self.config = config or CurationConfig()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_CURATION_MODEL_ID,
        *,
        model_revision: str = DEFAULT_CURATION_MODEL_REVISION,
        max_new_tokens: int = 512,
        device_map: str = "auto",
        batch_size: int = 1,
        cache_dir: str | Path | None = None,
        offline: bool = False,
        config: CurationConfig | None = None,
    ) -> "RADCurator":
        """Load a pinned or custom Hugging Face causal chat model."""

        from .models import TransformersLanguageModel

        resolved = config or CurationConfig(
            model_id=model_id,
            model_revision=model_revision,
            max_new_tokens=max_new_tokens,
            device_map=device_map,
            batch_size=batch_size,
            cache_dir=cache_dir,
            offline=offline,
        )
        backend = TransformersLanguageModel(
            resolved.model_id,
            revision=resolved.model_revision,
            max_new_tokens=resolved.max_new_tokens,
            device_map=resolved.device_map,
            local_files_only=resolved.offline,
            cache_dir=str(resolved.cache_dir)
            if resolved.cache_dir is not None
            else None,
        )
        return cls(backend, resolved)

    def generate(self, pairs: Sequence[tuple[str, str]]) -> list[Annotation]:
        """Generate fully validated annotations for ``(query, jailbreak)`` pairs."""

        prompts = [
            ANNOTATION_PROMPT.format(prompt=jailbreak, intent=query)
            for query, jailbreak in pairs
        ]
        responses = self.backend.generate(
            prompts, system_prompt=ANNOTATION_SYSTEM_PROMPT
        )
        if len(responses) != len(prompts):
            raise ValueError(
                f"curation model returned {len(responses)} outputs for {len(prompts)} inputs"
            )
        return [parse_annotation(response) for response in responses]


@dataclass
class _InputRow:
    index: int
    key: str
    values: dict[str, str]
    preserved: bool


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row_key(query: str, jailbreak: str, occurrence: int) -> str:
    digest = hashlib.sha256()
    digest.update(query.encode("utf-8"))
    digest.update(b"\0")
    digest.update(jailbreak.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(occurrence).encode("ascii"))
    return f"row-{digest.hexdigest()}"


def _load_input(path: Path) -> tuple[list[_InputRow], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        if not fieldnames:
            raise ValueError(f"{path} is empty or has no CSV header")
        duplicates = sorted({name for name in fieldnames if fieldnames.count(name) > 1})
        if duplicates:
            raise ValueError(f"{path} has duplicate columns: {', '.join(duplicates)}")
        if "query" not in fieldnames:
            raise ValueError(f"{path} is missing required column: query")
        has_jailbreak = "jailbreak" in fieldnames
        has_prompt = "prompt" in fieldnames
        if not has_jailbreak and not has_prompt:
            raise ValueError(
                f"{path} is missing required column: jailbreak (or prompt alias)"
            )

        rows: list[_InputRow] = []
        occurrences: defaultdict[tuple[str, str], int] = defaultdict(int)
        explicit_ids: set[str] = set()
        for number, raw in enumerate(reader, start=2):
            if None in raw:
                raise ValueError(f"{path}:{number} has more values than columns")
            query = (raw.get("query") or "").strip()
            jailbreak = (
                (raw.get("jailbreak") or "").strip()
                if has_jailbreak
                else (raw.get("prompt") or "").strip()
            )
            prompt = (raw.get("prompt") or "").strip() if has_prompt else ""
            if not query:
                raise ValueError(f"{path}:{number} requires a non-empty query")
            if not jailbreak:
                raise ValueError(f"{path}:{number} requires a non-empty jailbreak")
            if has_jailbreak and has_prompt and prompt and prompt != jailbreak:
                raise ValueError(
                    f"{path}:{number} has conflicting prompt and jailbreak values"
                )

            values = {name: (raw.get(name) or "") for name in fieldnames}
            values["query"] = query
            values["jailbreak"] = jailbreak
            values.setdefault("analysis", "")
            values.setdefault("guidance", "")

            explicit_id = (
                (raw.get("example_id") or "").strip()
                if "example_id" in fieldnames
                else ""
            )
            if explicit_id:
                if explicit_id in explicit_ids:
                    raise ValueError(
                        f"{path} contains duplicate example_id value: {explicit_id}"
                    )
                explicit_ids.add(explicit_id)
                values["example_id"] = explicit_id
                key = f"id:{explicit_id}"
            else:
                pair = (query, jailbreak)
                occurrence = occurrences[pair]
                occurrences[pair] += 1
                key = _row_key(query, jailbreak, occurrence)

            rows.append(
                _InputRow(
                    index=len(rows),
                    key=key,
                    values=values,
                    preserved=bool(
                        (values.get("analysis") or "").strip()
                        and (values.get("guidance") or "").strip()
                    ),
                )
            )
    if not rows:
        raise ValueError(f"{path} contains no input rows")
    return rows, fieldnames


def _output_fields(input_fields: Sequence[str]) -> list[str]:
    available = list(input_fields)
    if "jailbreak" not in available:
        available.append("jailbreak")
    for annotation_field in ("analysis", "guidance"):
        if annotation_field not in available:
            available.append(annotation_field)
    preferred = (
        "example_id",
        "query",
        "jailbreak",
        "analysis",
        "guidance",
        "attack_family",
        "split",
    )
    ordered = [name for name in preferred if name in available]
    ordered.extend(name for name in available if name not in ordered)
    return ordered


def _checkpoint_path(output: Path) -> Path:
    return Path(f"{output}.state.jsonl")


def _generation_identity(config: CurationConfig) -> dict[str, object]:
    return {
        "model": {"id": config.model_id, "revision": config.model_revision},
        "generation": {
            "max_new_tokens": config.max_new_tokens,
            "device_map": config.device_map,
            "batch_size": config.batch_size,
        },
    }


def _run_metadata(
    input_identity: dict[str, str], config: CurationConfig
) -> dict[str, object]:
    return {
        "event": "run",
        "schema_version": CURATION_SCHEMA_VERSION,
        "prompt_version": CURATION_PROMPT_VERSION,
        "input": input_identity,
        **_generation_identity(config),
    }


def _read_checkpoint(
    path: Path,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    metadata: dict[str, object] | None = None
    events: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{number} contains invalid checkpoint JSON"
                ) from error
            if not isinstance(event, dict):
                raise ValueError(f"{path}:{number} checkpoint event must be an object")
            if metadata is None:
                if event.get("event") != "run":
                    raise ValueError(f"{path} does not start with run metadata")
                metadata = event
            elif event.get("event") == "row" and isinstance(event.get("row_key"), str):
                events[str(event["row_key"])] = event
    if metadata is None:
        raise ValueError(f"{path} contains no run metadata")
    return metadata, events


def _append_events(
    path: Path, events: Sequence[dict[str, object]], *, truncate: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if truncate else "a"
    with path.open(mode, encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _assert_resume_identity(
    actual: dict[str, object], expected: dict[str, object]
) -> None:
    names = ("schema_version", "prompt_version", "input", "model", "generation")
    mismatches = [name for name in names if actual.get(name) != expected.get(name)]
    if mismatches:
        raise ValueError(
            "cannot resume because checkpoint identity differs for "
            f"{', '.join(mismatches)}; choose a new output or use --overwrite"
        )


def _sanitize_error(error: BaseException) -> str:
    message = " ".join(str(error).split()) or error.__class__.__name__
    secrets = {
        value
        for name, value in os.environ.items()
        if value
        and any(
            marker in name.upper()
            for marker in ("TOKEN", "SECRET", "PASSWORD", "API_KEY")
        )
    }
    for secret in sorted(secrets, key=len, reverse=True):
        message = message.replace(secret, "[REDACTED]")
    message = re.sub(r"\bhf_[A-Za-z0-9]{8,}\b", "[REDACTED]", message)
    message = re.sub(r"(?i)bearer\s+\S+", "Bearer [REDACTED]", message)
    return message[:500]


def _row_event(
    row: _InputRow,
    status: str,
    attempts: int,
    *,
    annotation: Annotation | None = None,
    error: BaseException | None = None,
) -> dict[str, object]:
    event: dict[str, object] = {
        "event": "row",
        "row_key": row.key,
        "row_index": row.index,
        "status": status,
        "attempts": attempts,
    }
    if annotation is not None:
        event["analysis"] = annotation.analysis
        event["guidance"] = annotation.guidance
    if error is not None:
        event["error"] = _sanitize_error(error)
    return event


def _write_output_atomic(
    output: Path,
    rows: Sequence[_InputRow],
    fields: Sequence[str],
    events: dict[str, dict[str, object]],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                event = events.get(row.key)
                if event is None or event.get("status") not in {
                    "preserved",
                    "generated",
                }:
                    continue
                values = dict(row.values)
                if event.get("status") == "generated":
                    values["analysis"] = str(event.get("analysis") or "")
                    values["guidance"] = str(event.get("guidance") or "")
                writer.writerow(values)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def _result(
    output: Path,
    state: Path,
    rows: Sequence[_InputRow],
    events: dict[str, dict[str, object]],
) -> CurationResult:
    statuses = [str(events.get(row.key, {}).get("status") or "") for row in rows]
    return CurationResult(
        output_path=output,
        checkpoint_path=state,
        total=len(rows),
        preserved=statuses.count("preserved"),
        generated=statuses.count("generated"),
        failed=statuses.count("failed"),
    )


def _curate_loaded(
    rows: list[_InputRow],
    input_fields: list[str],
    input_identity: dict[str, str],
    output: Path,
    *,
    config: CurationConfig,
    backend: GenerationBackend | None,
    curator: RADCurator | None,
    overwrite: bool,
    retry_failed: bool,
) -> CurationResult:
    if overwrite and retry_failed:
        raise ValueError("overwrite and retry_failed are mutually exclusive")
    state = _checkpoint_path(output)
    expected = _run_metadata(input_identity, config)

    if state.exists() and not overwrite:
        metadata, events = _read_checkpoint(state)
        _assert_resume_identity(metadata, expected)
    else:
        if retry_failed:
            raise ValueError("--retry-failed requires an existing checkpoint")
        if output.exists() and not overwrite:
            raise ValueError(
                f"{output} already exists without a checkpoint; use --overwrite"
            )
        _append_events(state, [expected], truncate=True)
        events = {}

    fields = _output_fields(input_fields)
    initial_events: list[dict[str, object]] = []
    if overwrite or not any(events.values()):
        if not overwrite:
            for row in rows:
                if row.preserved and row.key not in events:
                    event = _row_event(row, "preserved", 0)
                    event["analysis"] = row.values.get("analysis", "")
                    event["guidance"] = row.values.get("guidance", "")
                    events[row.key] = event
                    initial_events.append(event)
    if initial_events:
        _append_events(state, initial_events)
    _write_output_atomic(output, rows, fields, events)

    if retry_failed:
        pending = [
            row for row in rows if events.get(row.key, {}).get("status") == "failed"
        ]
    else:
        pending = [row for row in rows if row.key not in events]
    if not pending:
        return _result(output, state, rows, events)

    active_curator = curator
    if active_curator is None and backend is not None:
        active_curator = RADCurator(backend, config)
    if active_curator is None:
        try:
            active_curator = RADCurator.from_pretrained(config=config)
        except Exception as error:
            failed_events = [
                _row_event(
                    row,
                    "failed",
                    int(events.get(row.key, {}).get("attempts") or 0) + 1,
                    error=error,
                )
                for row in pending
            ]
            _append_events(state, failed_events)
            events.update({str(event["row_key"]): event for event in failed_events})
            _write_output_atomic(output, rows, fields, events)
            return _result(output, state, rows, events)

    for start in range(0, len(pending), config.batch_size):
        batch = pending[start : start + config.batch_size]
        attempts = {
            row.key: int(events.get(row.key, {}).get("attempts") or 0) + 1
            for row in batch
        }
        try:
            prompts = [
                ANNOTATION_PROMPT.format(
                    prompt=row.values["jailbreak"], intent=row.values["query"]
                )
                for row in batch
            ]
            responses = active_curator.backend.generate(
                prompts, system_prompt=ANNOTATION_SYSTEM_PROMPT
            )
            if not isinstance(responses, Sequence) or isinstance(
                responses, (str, bytes)
            ):
                raise ValueError("curation model must return a sequence of strings")
            if len(responses) != len(batch):
                raise ValueError(
                    f"curation model returned {len(responses)} outputs for {len(batch)} inputs"
                )
        except Exception as error:
            batch_events = [
                _row_event(row, "failed", attempts[row.key], error=error)
                for row in batch
            ]
        else:
            batch_events = []
            for row, response in zip(batch, responses):
                try:
                    if not isinstance(response, str):
                        raise ValueError("curation model output must be a string")
                    annotation = parse_annotation(response)
                    batch_events.append(
                        _row_event(
                            row, "generated", attempts[row.key], annotation=annotation
                        )
                    )
                except Exception as error:
                    batch_events.append(
                        _row_event(row, "failed", attempts[row.key], error=error)
                    )
        _append_events(state, batch_events)
        events.update({str(event["row_key"]): event for event in batch_events})
        _write_output_atomic(output, rows, fields, events)

    return _result(output, state, rows, events)


def curate_csv(
    path: str | Path,
    output: str | Path,
    *,
    config: CurationConfig | None = None,
    backend: GenerationBackend | None = None,
    curator: RADCurator | None = None,
    overwrite: bool = False,
    retry_failed: bool = False,
) -> CurationResult:
    """Curate a local CSV and atomically maintain its local checkpoint and output."""

    source = Path(path).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    rows, fields = _load_input(source)
    resolved = (
        curator.config
        if curator is not None and config is None
        else (config or CurationConfig())
    )
    identity = {"kind": "local", "digest": _file_digest(source)}
    return _curate_loaded(
        rows,
        fields,
        identity,
        destination,
        config=resolved,
        backend=backend,
        curator=curator,
        overwrite=overwrite,
        retry_failed=retry_failed,
    )


def curate_hub(
    repo_id: str,
    revision: str,
    filename: str,
    output: str | Path,
    *,
    config: CurationConfig | None = None,
    backend: GenerationBackend | None = None,
    curator: RADCurator | None = None,
    overwrite: bool = False,
    retry_failed: bool = False,
) -> CurationResult:
    """Download one pinned dataset CSV read-only, then curate it to a local path."""

    if not repo_id.strip():
        raise ValueError("repo_id must be non-empty")
    if not revision.strip():
        raise ValueError("an explicit dataset revision is required")
    if not filename.strip():
        raise ValueError("filename must be non-empty")
    from huggingface_hub import hf_hub_download

    resolved = (
        curator.config
        if curator is not None and config is None
        else (config or CurationConfig())
    )
    downloaded = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename=filename,
            cache_dir=str(resolved.cache_dir)
            if resolved.cache_dir is not None
            else None,
            local_files_only=resolved.offline,
            token=os.environ.get("HF_TOKEN") or None,
        )
    )
    rows, fields = _load_input(downloaded)
    identity = {
        "kind": "hub",
        "repo_id": repo_id,
        "revision": revision,
        "filename": filename,
        "digest": _file_digest(downloaded),
    }
    return _curate_loaded(
        rows,
        fields,
        identity,
        Path(output).expanduser().resolve(),
        config=resolved,
        backend=backend,
        curator=curator,
        overwrite=overwrite,
        retry_failed=retry_failed,
    )
