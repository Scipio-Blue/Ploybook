from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import platform
import sys
import warnings
from dataclasses import replace
from pathlib import Path

from .config import RADConfig
from .data import (
    DEFAULT_DATASET_ID,
    PAPER_DATABASE_PATH,
    PAPER_DATASET_REVISION,
    default_cache_dir,
    download_dataset,
    load_database,
)
from .index import FaissIndex


def _configuration(
    path: str | None,
    preset: str = "paper-sim",
    *,
    model_id: str | None = None,
    model_revision: str | None = None,
) -> RADConfig:
    """Resolve configuration with explicit CLI overrides taking precedence."""

    config = RADConfig.from_yaml(path) if path else RADConfig.from_preset(preset)
    candidate_overrides = {
        "model_id": model_id,
        "model_revision": (
            "main"
            if model_id is not None and model_revision is None
            else model_revision
        ),
    }
    overrides = {
        name: value for name, value in candidate_overrides.items() if value is not None
    }
    return replace(config, **overrides) if overrides else config


def _download(args: argparse.Namespace) -> int:
    target = download_dataset(
        args.local_dir,
        args.repo_id,
        args.revision,
        local_files_only=args.offline,
    )
    print(target)
    return 0


def _build_index(args: argparse.Namespace) -> int:
    from .models import ContrieverEncoder

    config = _configuration(args.config)
    database = load_database(args.database)
    encoder = ContrieverEncoder(
        config.encoder_id, config.encoder_revision, device=args.device
    )
    index = FaissIndex.build(database, encoder, args.batch_size)
    index.save(args.output, database, encoder)
    print(f"indexed {len(database)} examples at {args.output}")
    return 0


def _setup(args: argparse.Namespace) -> int:
    from huggingface_hub import snapshot_download

    from .artifacts import prepare_artifacts
    from .models import ContrieverEncoder

    config = _configuration(
        args.config,
        args.preset,
        model_id=args.model_id,
        model_revision=args.model_revision,
    )
    cache_root = (
        Path(args.cache_dir).expanduser().resolve()
        if args.cache_dir
        else default_cache_dir().resolve()
    )
    encoder = ContrieverEncoder(
        config.encoder_id,
        config.encoder_revision,
        device=args.device,
        local_files_only=args.offline,
        cache_dir=str(cache_root / "hub"),
    )
    artifacts = prepare_artifacts(
        config,
        encoder,
        cache_dir=cache_root,
        repo_id=args.repo_id,
        revision=args.revision,
        filename=args.database_filename,
        local_files_only=args.offline,
        rebuild_index=args.rebuild_index,
        batch_size=args.batch_size,
    )
    if not args.skip_model:
        snapshot_download(
            repo_id=config.model_id,
            revision=config.model_revision,
            cache_dir=cache_root / "hub",
            local_files_only=args.offline,
        )
    print(
        json.dumps(
            {
                "configuration": args.config or args.preset,
                "name": config.name,
                "model_id": config.model_id,
                "model_revision": config.model_revision,
                "encoder_id": config.encoder_id,
                "encoder_revision": config.encoder_revision,
                "cache_dir": str(artifacts.cache_dir),
                "database": str(artifacts.database.source),
                "database_digest": artifacts.database.digest,
                "database_rows": len(artifacts.database),
                "index": str(artifacts.index_path),
                "model_cached": not args.skip_model,
            },
            indent=2,
        )
    )
    return 0


def _input_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt is not None:
        return [args.prompt]
    source = Path(args.input)
    if source.suffix.lower() in {".jsonl", ".ndjson"}:
        prompts: list[str] = []
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict) or not isinstance(
                    value.get("prompt"), str
                ):
                    raise ValueError(
                        f"{source}:{line_number} must be a JSON object with a string prompt"
                    )
                prompts.append(value["prompt"])
        return prompts
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "prompt" not in (reader.fieldnames or []):
            raise ValueError("input CSV must contain a prompt column")
        return [(row.get("prompt") or "") for row in reader]


def _score(args: argparse.Namespace) -> int:
    from .detector import RADDetector

    config = _configuration(
        args.config,
        args.preset,
        model_id=args.model_id,
        model_revision=args.model_revision,
    )
    legacy_paths = any((args.database, args.index))
    if legacy_paths:
        if not args.database or not args.index:
            raise ValueError("--database and --index must be provided together")
        detector = RADDetector.from_paths(
            config,
            args.database,
            args.index,
            rebuild_index=args.rebuild_index,
        )
    else:
        detector = RADDetector.from_pretrained(
            config,
            cache_dir=args.cache_dir,
            repo_id=args.repo_id,
            revision=args.revision,
            database_filename=args.database_filename,
            local_files_only=args.offline,
            rebuild_index=args.rebuild_index,
            index_batch_size=args.batch_size,
            device=args.device,
            device_map=args.device_map,
        )
    results = detector.score_batch(_input_prompts(args), threshold=args.threshold)
    output = Path(args.output) if args.output else None
    handle = output.open("w", encoding="utf-8") if output else sys.stdout
    try:
        for result in results:
            handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
            handle.flush()
    finally:
        if output:
            handle.close()
    return 0


def _doctor(args: argparse.Namespace) -> int:
    cache_root = (
        Path(args.cache_dir).expanduser().resolve()
        if args.cache_dir
        else default_cache_dir().resolve()
    )
    report: dict[str, object] = {
        "python": platform.python_version(),
        "python_supported": (3, 10) <= sys.version_info[:2] < (3, 12),
        "cache_dir": str(cache_root),
        "cache_exists": cache_root.exists(),
        "cached_indexes": len(list((cache_root / "indexes").glob("*.faiss"))),
        "packages": {
            name: importlib.util.find_spec(name) is not None
            for name in ("torch", "transformers", "huggingface_hub", "faiss")
        },
    }
    if importlib.util.find_spec("torch") is not None:
        import torch

        report["cuda"] = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "torch_version": torch.__version__,
        }
    print(json.dumps(report, indent=2))
    return 0


def _reproduce(args: argparse.Namespace) -> int:
    from .reproduce import reproduce

    summary = reproduce(
        args.workflow,
        args.manifest,
        _configuration(args.config),
        args.output_dir,
        smoke=args.smoke,
        rebuild_index=args.rebuild_index,
    )
    print(summary)
    return 0


def _annotate(args: argparse.Namespace) -> int:
    from .annotation import annotate_csv
    from .models import TransformersLanguageModel

    warnings.warn(
        "rad annotate is deprecated and will be removed in a future release; use rad curate",
        FutureWarning,
        stacklevel=2,
    )
    model = TransformersLanguageModel(
        args.model,
        revision=args.revision,
        max_new_tokens=args.max_new_tokens,
        device_map=args.device_map,
    )
    print(annotate_csv(args.input, args.output, model))
    return 0


def _curate(args: argparse.Namespace) -> int:
    from .curation import CurationConfig, curate_csv, curate_hub

    config = CurationConfig(
        model_id=args.model_id,
        model_revision=args.model_revision,
        max_new_tokens=args.max_new_tokens,
        device_map=args.device_map,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        offline=args.offline,
    )
    if args.input is not None:
        if args.dataset_revision is not None or args.filename is not None:
            raise ValueError(
                "--dataset-revision and --filename are only valid with --repo-id"
            )
        result = curate_csv(
            args.input,
            args.output,
            config=config,
            overwrite=args.overwrite,
            retry_failed=args.retry_failed,
        )
    else:
        if args.dataset_revision is None or args.filename is None:
            raise ValueError("--repo-id requires --dataset-revision and --filename")
        result = curate_hub(
            args.repo_id,
            args.dataset_revision,
            args.filename,
            args.output,
            config=config,
            overwrite=args.overwrite,
            retry_failed=args.retry_failed,
        )
    print(
        json.dumps(
            {
                "output": str(result.output_path),
                "checkpoint": str(result.checkpoint_path),
                "total": result.total,
                "preserved": result.preserved,
                "generated": result.generated,
                "failed": result.failed,
            },
            indent=2,
        )
    )
    return 1 if result.failed else 0


def _generate_pap(args: argparse.Namespace) -> int:
    from .attacks import (
        generate_pap,
        load_pap_techniques,
        load_prompt_inputs,
        write_generated_prompts,
    )
    from .models import TransformersLanguageModel

    class LocalClient:
        def __init__(self) -> None:
            self.model = TransformersLanguageModel(
                args.model,
                revision=args.revision,
                max_new_tokens=args.max_new_tokens,
                device_map=args.device_map,
            )

        def complete(self, prompt: str, *, system_prompt: str) -> str:
            return self.model.generate([prompt], system_prompt=system_prompt)[0]

    prompts = generate_pap(
        load_prompt_inputs(args.input),
        LocalClient(),
        load_pap_techniques(args.taxonomy),
    )
    print(write_generated_prompts(args.output, prompts))
    return 0


def _generate_pair(args: argparse.Namespace) -> int:
    from .attacks import (
        OpenAIChatClient,
        generate_pair,
        load_prompt_inputs,
        write_generated_prompts,
    )

    prompts = generate_pair(
        load_prompt_inputs(args.input),
        OpenAIChatClient(args.attacker_model, temperature=args.attacker_temperature),
        OpenAIChatClient(args.target_model, temperature=args.target_temperature),
        OpenAIChatClient(args.judge_model, temperature=0.0),
        iterations=args.iterations,
        target_prefix=args.target_prefix,
    )
    print(write_generated_prompts(args.output, prompts))
    return 0


def _add_artifact_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--preset", choices=("paper-sim", "paper-base"), default="paper-sim"
    )
    parser.add_argument(
        "--config",
        help=(
            "RAD YAML configuration; values replace the preset and explicit model flags "
            "override the YAML"
        ),
    )
    parser.add_argument(
        "--model-id",
        help="Hugging Face defender model repository (revision defaults to main)",
    )
    parser.add_argument(
        "--model-revision",
        help="model commit, tag, or branch (pin a commit for reproducible use)",
    )
    parser.add_argument("--repo-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--revision", default=PAPER_DATASET_REVISION)
    parser.add_argument("--database-filename", default=PAPER_DATABASE_PATH)
    parser.add_argument("--cache-dir")
    parser.add_argument("--offline", action="store_true", help="use cached files only")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rebuild-index", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rad", description="Retrieval-Augmented Defense"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup = subparsers.add_parser(
        "setup",
        help="download a detector configuration and build its cached retrieval index",
    )
    _add_artifact_options(setup)
    setup.add_argument(
        "--skip-model",
        action="store_true",
        help="prepare only the database, encoder, and index",
    )
    setup.set_defaults(handler=_setup)

    score = subparsers.add_parser("score", help="score one prompt or a CSV/JSONL batch")
    _add_artifact_options(score)
    source = score.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--input")
    score.add_argument("--output")
    score.add_argument("--threshold", type=float)
    score.add_argument("--device-map", default="auto")
    score.add_argument("--database", help=argparse.SUPPRESS)
    score.add_argument("--index", help=argparse.SUPPRESS)
    score.set_defaults(handler=_score)

    doctor = subparsers.add_parser("doctor", help="report runtime and cache readiness")
    doctor.add_argument("--cache-dir")
    doctor.set_defaults(handler=_doctor)

    download = subparsers.add_parser(
        "download", help="legacy explicit dataset downloader"
    )
    download.add_argument("--repo-id", default=DEFAULT_DATASET_ID)
    download.add_argument("--revision", default=PAPER_DATASET_REVISION)
    download.add_argument("--local-dir", default="artifacts/data")
    download.add_argument("--offline", action="store_true")
    download.set_defaults(handler=_download)

    index = subparsers.add_parser("build-index", help="build an explicit FAISS index")
    index.add_argument("--config", default="configs/rad-sim.yaml")
    index.add_argument("--database", required=True)
    index.add_argument("--output", default="artifacts/database.faiss")
    index.add_argument("--batch-size", type=int, default=32)
    index.add_argument("--device", choices=("cpu", "cuda"), default=None)
    index.set_defaults(handler=_build_index)

    reproduce = subparsers.add_parser(
        "reproduce", help="[experimental] run a paper-core reproduction workflow"
    )
    reproduce.add_argument(
        "workflow", choices=("performance", "tradeoff", "adaptation")
    )
    reproduce.add_argument("--manifest", required=True)
    reproduce.add_argument("--config", default="configs/rad-sim.yaml")
    reproduce.add_argument("--output-dir", default="outputs")
    reproduce.add_argument("--smoke", action="store_true")
    reproduce.add_argument("--rebuild-index", action="store_true")
    reproduce.set_defaults(handler=_reproduce)

    curate = subparsers.add_parser(
        "curate", help="generate detector-ready analysis and guidance for attack pairs"
    )
    curate_source = curate.add_mutually_exclusive_group(required=True)
    curate_source.add_argument(
        "--input", help="local CSV containing query,jailbreak pairs"
    )
    curate_source.add_argument(
        "--repo-id", help="read-only Hugging Face dataset repository"
    )
    curate.add_argument(
        "--dataset-revision", help="required immutable dataset revision"
    )
    curate.add_argument("--filename", help="CSV path within the dataset repository")
    curate.add_argument("--output", required=True, help="local detector-ready CSV")
    curate.add_argument("--model-id", default="Qwen/Qwen2.5-14B-Instruct")
    curate.add_argument(
        "--model-revision",
        default="cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
    )
    curate.add_argument("--max-new-tokens", type=int, default=512)
    curate.add_argument("--device-map", default="auto")
    curate.add_argument("--batch-size", type=int, default=1)
    curate.add_argument("--cache-dir")
    curate.add_argument(
        "--offline", action="store_true", help="use cached Hub files only"
    )
    recovery = curate.add_mutually_exclusive_group()
    recovery.add_argument(
        "--overwrite",
        action="store_true",
        help="start a new run and recompute all rows",
    )
    recovery.add_argument(
        "--retry-failed",
        action="store_true",
        help="retry only rows previously marked failed",
    )
    curate.set_defaults(handler=_curate)

    annotate = subparsers.add_parser(
        "annotate",
        help="[deprecated] generate RAD analysis and guidance fields; use curate",
    )
    annotate.add_argument(
        "--input", required=True, help="CSV containing query and jailbreak columns"
    )
    annotate.add_argument("--output", required=True)
    annotate.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    annotate.add_argument(
        "--revision", default="cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
    )
    annotate.add_argument("--max-new-tokens", type=int, default=512)
    annotate.add_argument("--device-map", default="auto")
    annotate.set_defaults(handler=_annotate)

    generate = subparsers.add_parser(
        "generate-attacks",
        help="[experimental] generate authorized PAIR or PAP prompts",
    )
    generate_subparsers = generate.add_subparsers(dest="attack", required=True)

    pap = generate_subparsers.add_parser(
        "pap", help="generate PAP prompts from a licensed taxonomy"
    )
    pap.add_argument(
        "--input",
        required=True,
        help="CSV containing example_id,query or forbidden_prompt",
    )
    pap.add_argument(
        "--taxonomy", required=True, help="licensed JSONL taxonomy with ss_* fields"
    )
    pap.add_argument("--output", required=True)
    pap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    pap.add_argument("--revision", default="cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8")
    pap.add_argument("--max-new-tokens", type=int, default=512)
    pap.add_argument("--device-map", default="auto")
    pap.set_defaults(handler=_generate_pap)

    pair = generate_subparsers.add_parser(
        "pair", help="run a PAIR search against a target you control"
    )
    pair.add_argument(
        "--input",
        required=True,
        help="CSV containing example_id,query or forbidden_prompt",
    )
    pair.add_argument("--output", required=True)
    pair.add_argument("--attacker-model", required=True)
    pair.add_argument("--target-model", required=True)
    pair.add_argument("--judge-model", required=True)
    pair.add_argument("--iterations", type=int, default=5)
    pair.add_argument("--target-prefix", default="Sure, I'm happy to help")
    pair.add_argument("--attacker-temperature", type=float, default=1.0)
    pair.add_argument("--target-temperature", type=float, default=0.0)
    pair.set_defaults(handler=_generate_pair)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
