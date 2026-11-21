# Retrieval-Augmented Defense

Reusable reference implementation of **Retrieval-Augmented Defense: Adaptive and Controllable
Jailbreak Prevention for Large Language Models** (ACL 2026).

[Project website](https://bruceyg.github.io/rad/) · [Database on Hugging Face](https://huggingface.co/datasets/BruceYg/RAD-Database)

RAD is a standalone detector. It retrieves known attack examples, reranks their inferred
strategies, extracts candidate user intents, classifies each intent, and votes to produce
`P(harmful | prompt)`. It does not call or modify the model being protected: an application
forwards an accepted prompt to its own model and handles blocked prompts according to its policy.

> **Content warning:** the companion dataset contains adversarial prompts and harmful requests
> for defensive safety research. Follow its access terms and handle it in a controlled
> environment.

## Install

The paper environment uses Python 3.10.4, PyTorch 2.5.1, CUDA 12.1, Transformers 4.46.1, and
FAISS GPU 1.7.2:

```bash
conda env create -f environment.yml
conda activate rad
```

For CPU-only development:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[cpu,test]'
```

Full inference loads Qwen-2.5-14B-Instruct and normally requires a suitable accelerator. The
paper used an NVIDIA A100.

## Quickstart

> **Hugging Face access required:** The default paper presets (`paper-sim` and `paper-base`)
> use the gated [`BruceYg/RAD-Database`](https://huggingface.co/datasets/BruceYg/RAD-Database)
> dataset. Request and obtain access on Hugging Face before running `rad setup`, then
> authenticate with `hf auth login` or `HF_TOKEN`. Authentication alone is not sufficient
> without approval.

Prepare the default RAD-Sim-14B preset. This downloads the immutable paper database and pinned
model snapshots, validates the database, and builds a matching local FAISS index:

```bash
rad setup --preset paper-sim
rad score --preset paper-sim --prompt "A user prompt to inspect"
```

The setup output records the database digest, row count, and index location. After setup,
cached execution can be enforced with `--offline`:

```bash
rad score --preset paper-sim --offline --prompt "A user prompt to inspect"
```

Use `rad doctor` to inspect Python, package, CUDA, and cache readiness. Batch input may be CSV
with a `prompt` column or JSONL with one `{"prompt": "..."}` object per line:

```bash
rad score --input prompts.csv --output decisions.jsonl
rad score --input prompts.jsonl --output decisions.jsonl
```

Curate supplied attack pairs into a detector-ready local database:

```bash
rad curate --input pairs.csv --output curated.csv

rad curate \
  --repo-id my-org/raw-attacks \
  --dataset-revision <immutable-commit> \
  --filename data/pairs.csv \
  --output curated.csv
```

Curation never writes to the Hub. It validates input before loading a model, preserves extra
columns and completed annotations, and maintains a resumable local checkpoint. See the
[curation guide](docs/CURATION.md) for schemas, recovery modes, model configuration, exit codes,
and the separate optional user-controlled upload workflow.

RAD-Sim-14B is the default paper configuration: strategy-likelihood reranking, intent-only
classification, and majority voting. The generally stronger but slightly higher-FRR paper
configuration uses PMI reranking and posterior-weighted voting:

```bash
rad setup --preset paper-base
rad score --preset paper-base --prompt "A user prompt to inspect"
```

## Python API

```python
from rad_defense import RADDetector

detector = RADDetector.from_pretrained("paper-sim")
result = detector.score("A user prompt to inspect")

print(result.harmful_probability)
print(result.blocked)
if result.blocked:
    print(result.response)

for item in result.evidence:
    print(item.example_id, item.strategy, item.intent, item.vote_weight)
```

`score_batch(...)` returns results in input order. `predict(...)` returns only the boolean
block decision. Threshold comparison uses `>=`, as specified in the paper:

```python
results = detector.score_batch(["first prompt", "second prompt"], threshold=0.65)
blocked = detector.predict("another prompt", threshold=0.65)
```

The result trace contains stable database example IDs, retrieval and reranking scores, inferred
strategies and intents, classifier probabilities, and vote weights. RAD returns a refusal string
when blocked but deliberately does not invoke the protected target model.

### Use a custom Hugging Face database

The paper presets default to the pinned `BruceYg/RAD-Database` snapshot. To use another
Hugging Face dataset repository, pass the same dataset identity to both setup and scoring:

```bash
rad setup \
  --repo-id my-org/my-rad-database \
  --revision <immutable-dataset-commit> \
  --database-filename database/full.csv

rad score \
  --repo-id my-org/my-rad-database \
  --revision <immutable-dataset-commit> \
  --database-filename database/full.csv \
  --prompt "A user prompt to inspect"
```

Here, `--revision` always identifies the **dataset** revision. Use a commit hash rather than a
moving branch for reproducibility. The repository may be public or gated; gated repositories use
the normal Hugging Face authentication described below.

A database CSV requires nonempty `query`, `jailbreak`, `analysis`, and `guidance` fields.
Optional fields are `example_id`, `attack_family`, and `split`. Missing IDs are generated
deterministically. RAD validates the schema and builds a new index whose cache key includes the
database contents and encoder identity.

### Change the defender model

The defender model performs RAD's strategy reranking, intent extraction, and harmfulness
classification. It is separate from the protected target model, which RAD never invokes.
Override the defender model while retaining a paper algorithm preset:

```bash
rad setup \
  --preset paper-sim \
  --model-id my-org/my-defender-model \
  --model-revision <immutable-model-commit>

rad score \
  --preset paper-sim \
  --model-id my-org/my-defender-model \
  --model-revision <immutable-model-commit> \
  --prompt "A user prompt to inspect"
```

`--model-revision` identifies the **model** revision. If `--model-id` is supplied without it,
RAD uses `main`; pinning an immutable commit is strongly recommended for experiments and
deployments. The built-in Transformers backend expects a causal language model with a chat
template and at least one single-token spelling of each `Bad` and `Good` classifier label.
Other models can be supported through `RADModelBackend`.

### Use a custom YAML configuration

Copy [`configs/custom.example.yaml`](configs/custom.example.yaml), edit it, and use the same
file for setup and scoring:

```bash
cp configs/custom.example.yaml custom.yaml
rad setup --config custom.yaml --repo-id my-org/my-rad-database \
  --revision <immutable-dataset-commit>
rad score --config custom.yaml --repo-id my-org/my-rad-database \
  --revision <immutable-dataset-commit> --prompt "A user prompt to inspect"
```

A YAML file replaces the selected preset. Explicit `--model-id` and `--model-revision` flags
override YAML values. YAML can configure the model and encoder identities, retrieval and reranking
sizes, reranking/classification/voting modes, threshold, generation limit, and refusal response.

The equivalent Python API accepts a `RADConfig` directly:

```python
from rad_defense import RADConfig, RADDetector

config = RADConfig.from_yaml("custom.yaml")
detector = RADDetector.from_pretrained(
    config,
    repo_id="my-org/my-rad-database",
    revision="<immutable-dataset-commit>",
    database_filename="database/full.csv",
)
```

### Custom backends and local databases

The built-in implementations use Hugging Face Transformers, Qwen, Contriever, and FAISS.
Applications may instead inject implementations of `RADModelBackend`, `TextEncoder`, and
`SimilarityIndex`:

```python
from rad_defense import RADConfig, RADDatabase, RADDetector

database = RADDatabase.from_csv("my-rad-database.csv")
detector = RADDetector.from_database(
    RADConfig.from_yaml("custom.yaml"),
    database,
    encoder=my_encoder,
    model_backend=my_backend,
    index_path="my-rad-database.faiss",
)
```

The same CSV schema is produced by the stable v0.2 attack-example curation interface.

## Data identity and cache behavior

The default preset pins
[`BruceYg/RAD-Database`](https://huggingface.co/datasets/BruceYg/RAD-Database) at commit
`6e5d5ddcc93a9bedd543a33fc3790470f618eeb7`. Its 10,991-row field-level digest is
`559aea650bded65c1df7050cb17ab89d13901710c5327bd8d3282559f24952a0`.

RAD caches Hub snapshots and indexes below `$RAD_CACHE_DIR`, then `$XDG_CACHE_HOME/rad-defense`,
or the platform user cache directory. An index key includes the database digest, encoder ID, and
encoder revision. Index builds are locked and atomically installed, so concurrent cluster jobs
cannot consume a partial index.

Credentials are never stored by RAD. For a gated snapshot, authenticate with the current
Hugging Face CLI (`hf auth login`) or provide `HF_TOKEN` through the normal Hub mechanism.

## Apptainer on HPC

Apptainer is an optional deployment artifact, not the primary API. It is useful on clusters
where reproducing the Python/CUDA stack is difficult:

```bash
apptainer build rad.sif containers/rad.def
apptainer exec --nv \
  --bind "${RAD_CACHE_DIR:-$HOME/.cache/rad-defense}:/cache" \
  --env RAD_CACHE_DIR=/cache \
  rad.sif rad doctor
```

The image contains code and dependencies only. Model weights, the gated database, credentials,
and runtime caches remain outside it. See [the Apptainer guide](docs/APPTAINER.md).

## Experimental research workflows

`rad curate` is stable in v0.2. `rad annotate` remains available for compatibility but is
deprecated in favor of `rad curate`. The `rad reproduce` and `rad generate-attacks` commands
remain experimental research workflows; PAIR/PAP generation is not part of stable curation.
See the [curation guide](docs/CURATION.md),
[attack-generation safety guidance](docs/ATTACK_GENERATION.md), and the
[release plan](docs/PUBLIC_RELEASE_PLAN.md).

## Testing, licensing, and citation

Unit tests use deterministic fake backends and do not download model weights or call hosted APIs:

```bash
python -m pytest -q
python -m compileall -q src tests
```

Original code is released under the [MIT License](LICENSE). The companion dataset has separate
terms because its queries and jailbreak prompts include third-party material; the code license
does not relicense that content. Consult [data provenance](docs/DATA_PROVENANCE.md) and the
[dataset card](docs/HUGGINGFACE_DATASET_CARD.md) before redistribution.

```bibtex
@inproceedings{yang-etal-2026-retrieval,
    title = "Retrieval-Augmented Defense: Adaptive and Controllable Jailbreak Prevention for Large Language Models",
    author = "Yang, Guangyu  and
      Chen, Jinghong  and
      Mei, Jingbiao  and
      Lin, Weizhe  and
      Byrne, Bill",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.1895/",
    doi = "10.18653/v1/2026.acl-long.1895",
    pages = "40849--40868",
    ISBN = "979-8-89176-390-6",
    abstract = "Large Language Models (LLMs) remain vulnerable to jailbreak attacks, which attempt to elicit harmful responses from LLMs. The evolving nature and diversity of these attacks pose many challenges for defense systems, including (1) adaptation to counter emerging attack strategies without costly retraining, and (2) control of the trade-off between safety and utility. To address these challenges, we propose Retrieval-Augmented Defense (RAD), a novel framework for jailbreak detection that incorporates a database of known attack examples into Retrieval-Augmented Generation, which is used to infer the underlying, malicious user query and jailbreak strategy used to attack the system. RAD enables training-free updates for newly discovered jailbreak strategies and provides a mechanism to balance safety and utility. Experiments on StrongREJECT show that RAD substantially reduces the effectiveness of strong jailbreak attacks such as PAP and PAIR while maintaining low rejection rates for benign queries. We propose a novel evaluation scheme and show that RAD achieves a robust safety-utility trade-off across a range of operating points in a controllable manner."
}
```
