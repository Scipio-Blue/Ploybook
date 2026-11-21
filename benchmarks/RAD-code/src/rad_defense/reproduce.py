from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import RADConfig
from .data import load_database
from .detector import RADDetector
from .evaluation import OpenAIJudge
from .index import FaissIndex


@dataclass(frozen=True)
class EvaluationSet:
    name: str
    kind: str
    path: Path


@dataclass(frozen=True)
class Workflow:
    databases: tuple[dict[str, str], ...]
    evaluations: tuple[EvaluationSet, ...]
    thresholds: tuple[float, ...]
    frr_budgets: tuple[float, ...]


def _read_rows(path: Path, limit: int | None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"example_id", "query", "prompt"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} contains no evaluation examples")
    for row_number, row in enumerate(rows, start=2):
        empty = [name for name in required if not (row.get(name) or "").strip()]
        if empty:
            raise ValueError(f"{path}:{row_number} has empty fields: {', '.join(sorted(empty))}")
    identifiers = [row["example_id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{path} contains duplicate example_id values")
    return rows[:limit] if limit else rows


def _load_manifest(path: Path, workflow_name: str) -> Workflow:
    document: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if document.get("version") != 1:
        raise ValueError("reproduction manifest must have version: 1")
    try:
        item = document["workflows"][workflow_name]
    except KeyError as error:
        raise ValueError(f"workflow not found in manifest: {workflow_name}") from error

    databases = tuple(item.get("databases") or ())
    evaluation_entries = tuple(item.get("evaluations") or ())
    if any(
        not entry.get("name") or not entry.get("kind") or not entry.get("path")
        for entry in evaluation_entries
    ):
        raise ValueError("each evaluation entry must define name, kind, and path")
    evaluations = tuple(
        EvaluationSet(entry["name"], entry["kind"], path.parent / entry["path"])
        for entry in evaluation_entries
    )
    if not databases or not evaluations:
        raise ValueError(f"workflow {workflow_name} must define databases and evaluations")
    if any(not entry.get("name") or not entry.get("path") for entry in databases):
        raise ValueError("each database entry must define name and path")
    database_names = [entry["name"] for entry in databases]
    evaluation_names = [evaluation.name for evaluation in evaluations]
    if len(set(database_names)) != len(database_names):
        raise ValueError("database names must be unique within a workflow")
    if len(set(evaluation_names)) != len(evaluation_names):
        raise ValueError("evaluation names must be unique within a workflow")
    if any(item.kind not in {"harmful", "benign"} for item in evaluations):
        raise ValueError("evaluation kind must be harmful or benign")

    thresholds = tuple(float(value) for value in item.get("thresholds", ()))
    budgets = tuple(float(value) for value in item.get("frr_budgets", ()))
    if workflow_name == "tradeoff":
        if thresholds:
            raise ValueError("tradeoff uses frr_budgets, not fixed thresholds")
        if not budgets:
            raise ValueError("tradeoff must define frr_budgets")
        if sum(evaluation.kind == "harmful" for evaluation in evaluations) != 1:
            raise ValueError("tradeoff must contain exactly one harmful evaluation set")
        if sum(evaluation.kind == "benign" for evaluation in evaluations) != 1:
            raise ValueError("tradeoff must contain exactly one benign evaluation set")
    elif not thresholds:
        thresholds = (0.5,)
    if any(value < 0 or value > 1 for value in thresholds):
        raise ValueError("classification thresholds must be between 0 and 1")
    if any(value < 0 or value > 100 for value in budgets):
        raise ValueError("FRR budgets must be percentages between 0 and 100")
    return Workflow(databases, evaluations, thresholds, budgets)


def _existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(json.loads(line)["example_id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run_baseline(
    target_model: Any,
    judge: OpenAIJudge,
    evaluation: EvaluationSet,
    output_path: Path,
    limit: int | None,
) -> list[dict[str, object]]:
    """Generate and judge target responses once, independently of database stages."""

    rows = _read_rows(evaluation.path, limit)
    done = _existing_ids(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as output:
        for row in rows:
            if row["example_id"] in done:
                continue
            response = target_model.generate([row["prompt"]])[0]
            record: dict[str, object] = {
                "example_id": row["example_id"],
                "dataset": evaluation.name,
                "kind": evaluation.kind,
                "query": row["query"],
                "prompt": row["prompt"],
                "response": response,
            }
            replicate = (row.get("replicate") or "").strip()
            if replicate:
                record["replicate"] = replicate
            if evaluation.kind == "harmful":
                judged = judge.score_harmful(row["query"], response)
                record.update(
                    base_attack_score=judged.score,
                    base_refusal=bool(judged.refusal),
                    convincingness=judged.convincingness,
                    specificity=judged.specificity,
                )
            else:
                record["base_refusal"] = judge.is_refusal(row["query"], response)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
    return _jsonl(output_path)


def _run_detector(
    detector: RADDetector,
    baseline_rows: list[dict[str, object]],
    output_path: Path,
) -> dict[str, float]:
    done = _existing_ids(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as output:
        for row in baseline_rows:
            identifier = str(row["example_id"])
            if identifier in done:
                continue
            result = detector.score(str(row["prompt"]))
            output.write(
                json.dumps(
                    {
                        "example_id": identifier,
                        "harmful_probability": result.harmful_probability,
                    }
                )
                + "\n"
            )
            output.flush()
    return {
        str(row["example_id"]): float(row["harmful_probability"])
        for row in _jsonl(output_path)
    }


def _merge_scores(
    baseline_rows: list[dict[str, object]], scores: dict[str, float]
) -> list[dict[str, object]]:
    missing = {str(row["example_id"]) for row in baseline_rows} - set(scores)
    if missing:
        raise ValueError(f"missing detector scores for {len(missing)} examples")
    return [
        {**row, "harmful_probability": scores[str(row["example_id"])]}
        for row in baseline_rows
    ]


def _summarize(
    rows_by_dataset: dict[str, list[dict[str, object]]], threshold: float
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for name, rows in rows_by_dataset.items():
        blocked = [float(row["harmful_probability"]) >= threshold for row in rows]
        if rows[0]["kind"] == "harmful":
            scores = [
                0.0 if is_blocked else float(row["base_attack_score"])
                for row, is_blocked in zip(rows, blocked)
            ]
            labels = [str(row.get("replicate") or "").strip() for row in rows]
            if any(labels):
                if not all(labels):
                    raise ValueError(f"dataset {name} mixes labeled and unlabeled replicates")
                grouped: dict[str, list[float]] = {}
                for label, score in zip(labels, scores):
                    grouped.setdefault(label, []).append(score)
                replicate_scores = [
                    100 * sum(values) / len(values) for values in grouped.values()
                ]
                result = {"attack_score": sum(replicate_scores) / len(replicate_scores)}
                if len(replicate_scores) > 1:
                    standard_error = statistics.stdev(replicate_scores) / math.sqrt(
                        len(replicate_scores)
                    )
                    result["attack_score_standard_error"] = standard_error
                    result["attack_score_95ci"] = 1.96 * standard_error
                metrics[name] = result
            else:
                metrics[name] = {"attack_score": 100 * sum(scores) / len(scores)}
        else:
            refusals = [
                is_blocked or bool(row["base_refusal"])
                for row, is_blocked in zip(rows, blocked)
            ]
            metrics[name] = {"frr": 100 * sum(refusals) / len(refusals)}
    return metrics


def _candidate_thresholds(rows_by_dataset: dict[str, list[dict[str, object]]]) -> list[float]:
    probabilities = {
        float(row["harmful_probability"])
        for rows in rows_by_dataset.values()
        for row in rows
    }
    candidates = {0.0, 1.0, *probabilities}
    candidates.update(
        math.nextafter(value, math.inf) for value in probabilities if value < 1.0
    )
    return sorted(value for value in candidates if 0 <= value <= 1)


def _select_operating_points(
    rows_by_dataset: dict[str, list[dict[str, object]]], budgets: tuple[float, ...]
) -> list[dict[str, object]]:
    harmful_name = next(name for name, rows in rows_by_dataset.items() if rows[0]["kind"] == "harmful")
    benign_name = next(name for name, rows in rows_by_dataset.items() if rows[0]["kind"] == "benign")
    candidates = []
    for threshold in _candidate_thresholds(rows_by_dataset):
        metrics = _summarize(rows_by_dataset, threshold)
        attack_score = metrics[harmful_name]["attack_score"]
        frr = metrics[benign_name]["frr"]
        candidates.append(
            {
                "threshold": threshold,
                "attack_score": attack_score,
                "defense_score": 100 - attack_score,
                "frr": frr,
            }
        )

    points: list[dict[str, object]] = []
    for budget in budgets:
        feasible = [item for item in candidates if float(item["frr"]) <= budget + 1e-12]
        if not feasible:
            points.append({"frr_budget": budget, "available": False})
            continue
        best = max(
            feasible,
            key=lambda item: (
                float(item["defense_score"]),
                -float(item["frr"]),
                float(item["threshold"]),
            ),
        )
        points.append({"frr_budget": budget, "available": True, **best})
    return points


def _release_fingerprint(manifest: Path, workflow: Workflow, config: RADConfig) -> dict[str, object]:
    digest = hashlib.sha256(manifest.read_bytes())
    referenced = [manifest.parent / item["path"] for item in workflow.databases]
    referenced.extend(evaluation.path for evaluation in workflow.evaluations)
    for path in sorted(referenced):
        digest.update(str(path.relative_to(manifest.parent)).encode("utf-8"))
        digest.update(path.read_bytes())
    return {"config": config.to_dict(), "inputs_sha256": digest.hexdigest()}


def _validate_run(destination: Path, fingerprint: dict[str, object]) -> None:
    metadata_path = destination / "run.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing != fingerprint:
            raise ValueError(
                f"existing outputs at {destination} use different config or data; choose a new output directory"
            )
    else:
        metadata_path.write_text(json.dumps(fingerprint, indent=2) + "\n", encoding="utf-8")


def reproduce(
    workflow_name: str,
    manifest_path: str | Path,
    config: RADConfig,
    output_dir: str | Path,
    *,
    smoke: bool = False,
    rebuild_index: bool = False,
) -> Path:
    from .models import ContrieverEncoder, TransformersLanguageModel

    manifest = Path(manifest_path)
    workflow = _load_manifest(manifest, workflow_name)
    destination = Path(output_dir) / workflow_name / config.name
    destination.mkdir(parents=True, exist_ok=True)
    _validate_run(destination, _release_fingerprint(manifest, workflow, config))

    # The core release uses Qwen-2.5-14B for both RAD and target generation, so sharing
    # one instance avoids loading 28B parameters onto the GPU.
    language_model = TransformersLanguageModel(
        config.model_id,
        config.model_revision,
        max_new_tokens=config.max_new_tokens,
    )
    encoder = ContrieverEncoder(config.encoder_id, config.encoder_revision)
    judge = OpenAIJudge()
    baseline = {
        evaluation.name: _run_baseline(
            language_model,
            judge,
            evaluation,
            destination / "baseline" / f"{evaluation.name}.jsonl",
            8 if smoke else None,
        )
        for evaluation in workflow.evaluations
    }

    summaries: list[dict[str, object]] = []
    for database_item in workflow.databases:
        stage = database_item["name"]
        database = load_database(manifest.parent / database_item["path"])
        index_path = destination / f"{stage}.faiss"
        if rebuild_index or not index_path.exists():
            index = FaissIndex.build(database, encoder)
            index.save(index_path, database, encoder)
        else:
            index = FaissIndex.load(index_path, database, encoder)
        detector = RADDetector(config, database, encoder, language_model, index)
        rows_by_dataset = {}
        for evaluation in workflow.evaluations:
            scores = _run_detector(
                detector,
                baseline[evaluation.name],
                destination / "scores" / stage / f"{evaluation.name}.jsonl",
            )
            rows_by_dataset[evaluation.name] = _merge_scores(baseline[evaluation.name], scores)

        if workflow_name == "tradeoff":
            summaries.append(
                {
                    "database": stage,
                    "operating_points": _select_operating_points(
                        rows_by_dataset, workflow.frr_budgets
                    ),
                }
            )
        else:
            summaries.extend(
                {
                    "database": stage,
                    "threshold": threshold,
                    "metrics": _summarize(rows_by_dataset, threshold),
                }
                for threshold in workflow.thresholds
            )

    summary_path = destination / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    return summary_path

