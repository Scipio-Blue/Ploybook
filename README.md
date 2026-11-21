# Ploybook

Research code for **Ploybook: Strategy-Guided Intent Recovery for Jailbreak Defense**,
accepted by the 12th International Conference on Computer and Communications
(ICCC 2026), paper IC473. Authors: Jun Tang and Jiaming Liu.

Ploybook uses a maintained strategy catalog to navigate jailbreak disguises,
recover an intent under each selected strategy, and assess the recovered intent
with a typed harm/fidelity/severity decision battery.

## Reproduce the reported measurements without model calls

Python 3.10 or 3.11 is recommended. This path needs no GPU and no credentials.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-offline.txt
python scripts/reproduce.py
python scripts/replay_decisions.py
python -m unittest discover -s tests
```

The first command recomputes the primary four-arm means, concealed comparison,
navigation recall, ablation, target replications, injection, adaptive attacks,
maintenance and timing summaries from per-sample records. Its output is
`reproduced/data-validation.json`. Internal historical arm names `unmask` and
`wiki` refer to Ploybook; all recorded scores retain their original scale.

The replay command recomputes the 25-policy threshold grid from exported gate,
core and severity scores. These scores were reconstructed from 3,435 cached
assessments and reproduce all 1,294 primary scores and verdicts exactly. It
does not infer response scores for policies that were not run against a target.

## Screen a new request

The portable entry point follows the paper's embedding-navigation path:

```bash
pip install -r requirements-inference.txt
export JEV_KEY='your-own-typesafe-key'
python scripts/screen_prompt.py --prompt 'A request to inspect' --allow-api
```

Fresh inference loads `facebook/contriever-msmarco` and
`Qwen/Qwen2.5-14B-Instruct` through Hugging Face and uses bfloat16 on CUDA.
The 14B extraction model requires suitable accelerator memory. The hosted
decision model is pinned to `jev-1.14.0`; availability and charges depend on
the provider. The command explicitly requires `--allow-api` for a hosted call.
`--encoder-path`, `--model-path`, and `--device` accept locally cached models.
The gate is evaluated before model loading, so immediately blocked requests
need no extraction model. The returned trace includes the recovered intent,
strategy page, core score, severity score, final score and block verdict.

## Evaluation implementation and provenance

- `scripts/runs/server_phase1.py`: original GPU-side navigation and extraction,
  plus the RAD-Sim baseline.
- `scripts/runs/run_phase2.py`: primary four-arm evaluation and the **binary
  severity** aggregation rule used for all reported primary results.
- `scripts/runs/server_gpu_window.py` and `run_phase2_ext.py`: concealed extension.
- `ablation_no_extraction.py`, `adaptive_attack.py`, `maint_experiment.py`,
  `target2_eval.py`, and `qwen38_target3_eval.py`: corresponding extension studies.
- `src/wiki_rad.py`: additional development paths; its optional choice navigator,
  second hop and ordinal diagnostic questions are not the primary reported path.
- `src/jev_client.py`: typed question wording and hosted client. Its older
  exploratory `aggregate` helper uses ordinal severity; use
  `run_phase2.aggregate` for the paper's binary-severity results.

The archived experiment drivers retain their original append/resume behavior.
Run new experiments in a copy and direct outputs to separate files. They need
the attack-input CSVs and precomputed extractions indicated by their paths,
plus user-supplied `JEV_KEY`/`OR_KEY`. New hosted runs incur provider charges.
Saved score archives verify the reported results; they are not a substitute
for obtaining the original benchmark inputs for a fresh full evaluation.

The released catalog includes all 55 pages' runtime fields (name, trigger,
strategy, guidance), original counts and cluster IDs, and the exact page
embeddings. Source-example cues and membership lists are omitted. The gated
RAD attack database and model weights are not redistributed. Obtain the RAD
database with its authors' access approval:
https://huggingface.co/datasets/BruceYg/RAD-Database

The RAD baseline implementation and StrongREJECT rubric templates in
`benchmarks/RAD-code` retain their upstream MIT license and attribution:
https://github.com/BruceYg/Retrieval-Augmented-Defense

`PROVENANCE.json` records source and release hashes. Deployment-location
constants were made relative to this checkout; offline guards were added to
the hosted clients. The public archive contains numeric evaluation records,
not target-generated harmful answers, account credentials or private materials.

## Decision policy

Screening blocks when both jailbreak and raw-harm scores are at least 0.85.
Otherwise `core = max(fidelity * intent_harm)` across hypotheses. If
`0.35 <= core < 0.70` and `max(binary_severity) >= 0.70`, the final score is
raised to 0.70. A score of at least 0.5 blocks. The 0.35/0.70 bands follow the
strict guardrails policy, adapted here to a product score and binary severity:
https://docs.typesafe.ai/cookbooks/llm_guardrails.md

The post-hoc replay varies the lower bound from 0.25 to 0.45 in steps of 0.05
and severity cutoff from 0.60 to 0.80 in steps of 0.05. It blocks 1,174–1,198
of 1,284 harmful prompts and none of the ten primary benign prompts. The
original default blocks 1,182, including eight severity-escalated cases.

## License

The authors' code, authored runtime strategy fields and numeric score records
are released under the MIT license. The vendored RAD implementation retains its
separate MIT notice in `benchmarks/RAD-code/LICENSE`. Model weights and source
benchmarks remain subject to their respective owners' terms.
