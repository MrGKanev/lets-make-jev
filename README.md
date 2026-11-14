# Jev: let the model choose

Jev tests a simple idea: when an LLM only needs to choose a category, score the possible first answer tokens instead of generating a text response. The client maps single-token labels (`A`, `B`, …) back to application categories and routes uncertain decisions to human review.

For example, a banking message can become `card_issues`, `payments_transfers`, `billing_fees`, `account_access`, or `other`. The experiment compares accuracy, latency, throughput, calibration and sensitivity to category order.

## Current status

The client, benchmark, LoRA training script, demo and human-review CLI are implemented. CPU tests use a fake inference server; they do **not** establish model quality or compatibility with a real SGLang deployment. No real GPU benchmark results or trained model artifacts are included in this checkout. The article is a draft pending those results.

See [PLAN.md](PLAN.md) for remaining work and [NOTES.md](NOTES.md) for development history and design decisions.

## Start on Windows

You need Docker Desktop running with the WSL2 backend and Linux containers, plus an NVIDIA GPU with a WSL-compatible driver. Host Python and Python packages are not required.

From Command Prompt in this directory:

```bat
run.bat smoke
```

This builds the container, runs CPU tests and data validation, then performs a small real-model benchmark using Qwen 0.5B. Once that succeeds, run the full experiment:

```bat
run.bat
```

Double-clicking `run.bat` also runs the full experiment. It performs:

1. CPU tests and dataset validation.
2. A smoke benchmark on the first configured model.
3. LoRA training of Qwen 0.5B, unless an existing training log is present or training is disabled.
4. Full benchmarks on Qwen 0.5B, Qwen 4B and the trained model.
5. JSON results, readable reports, policies when eligible, and article figures.

The first run automatically downloads the SGLang Docker image, installs container dependencies and downloads model weights from Hugging Face. The dataset files are already included. Internet access and sufficient disk space are needed; later runs reuse the Docker image layers and model cache. The configured public models do not normally need an `HF_TOKEN`; set one if using a gated/private model.

To start with only the small model and skip training:

```bat
set MODELS=Qwen/Qwen2.5-0.5B-Instruct
set SKIP_TRAIN=1
run.bat
```

| Setting | Default | Purpose |
| --- | --- | --- |
| `MODELS` | Qwen 0.5B and Qwen 4B | Space-separated Hugging Face model IDs |
| `SKIP_TRAIN` | `0` | Set to `1` to skip LoRA and the trained-model benchmark |
| `HF_TOKEN` | Empty | Optional Hugging Face access token |
| `JEV_NO_PAUSE` | Unset | Set to `1` to disable the Windows script's final pause |

`run.bat test` runs CPU checks and data validation without downloading model weights. It uses the same GPU-enabled Compose service, so Docker still needs NVIDIA device support. All modes stop on failure and return a nonzero exit code.

## Linux and direct Docker usage

```bash
bash run.sh
MODELS=Qwen/Qwen2.5-0.5B-Instruct SKIP_TRAIN=1 bash run.sh
JEV_MODE=smoke docker compose run --rm --build jev
```

The image contains the project source. Compose mounts the checkout so current source and generated outputs are available on the host. `.gitattributes` keeps shell scripts in LF format on Windows.

## Where outputs live

| Location | Contents |
| --- | --- |
| Docker volume `hf-cache` | Downloaded model weights and tokenizers |
| `models/jev-Qwen2.5-0.5B-Instruct/` | Merged trained model and training log |
| `bench/results/<run-id>/` | Per-experiment snapshots, manifest, metrics, reports and figures |

Each benchmark invocation creates its own experiment directory; smoke and full results are separate. Existing trained models are reused when `train_log.json` exists. Delete the relevant `models/jev-*` directory intentionally if you want to retrain. Removing the Docker cache volume causes weights to be downloaded again.

## What the benchmark compares

- **Score:** probabilities for the allowed label tokens.
- **Label:** the same prompt, generating one answer token; the closest matched baseline.
- **Generate:** a text answer parsed back to a category.
- **Structured:** JSON-schema output, skipped when unsupported by the server.
- **Batched score:** several inputs scored in one request.

The two domains are banking support (Banking77 plus CLINC out-of-scope) and news (AG News plus CLINC out-of-scope). Training, calibration and headline evaluation use separate splits. Additional synthetic challenges are diagnostic only.

If `/v1/score` rejects a request, the client falls back to native `/generate` with `token_ids_logprob`; results record the backend. Confidence is relative to the supplied categories, so an `other` option and calibrated review thresholds matter.

## Live demo

```bash
docker compose run --rm --build --service-ports jev python3 -m ui.serve --model Qwen/Qwen2.5-0.5B-Instruct --launch
```

Open http://localhost:8000 for the score-versus-generate comparison. Stop it with Ctrl+C.

## Troubleshooting and limitations

- **Docker connection error:** start Docker Desktop and select Linux containers with WSL2.
- **No GPU / CUDA unavailable:** verify NVIDIA GPU access from Docker before running inference. CPU tests do not validate GPU support.
- **Out of GPU memory:** begin with the 0.5B-only command above. The default 4B configuration has not yet been validated on a 12 GB RTX 4070.
- **Server startup fails:** inspect `bench/results/<run-id>/*.server.log` and the failed run's `manifest.json`. Shutdown has a bounded graceful wait, followed by forced termination if needed.
- **No policy file:** the dev set did not yield an eligible threshold; smoke runs can be too small. This is not necessarily a broken run.

The Docker base currently uses `lmsysorg/sglang:latest`; dependency versions are not fully pinned. Record the first successful environment and pin it before treating runs as reproducible. Runtime and speedup claims require real measurements; CPU/fake-server results cannot establish them.

## Data and licences

Banking77 (CC-BY-4.0, Casanueva et al. 2020; copy via `mteb/banking77`), CLINC150 (CC-BY-3.0, Larson et al. 2019) and AG News (Zhang et al. 2015, research use; copy via `fancyzhx/ag_news`).

## Matched baseline, diagnostics and calibrated decisions

The `label` lane sends exactly the same user prompt as scoring and generates **one token** (A/B/C…), without a reason. This separates the scoring technique from the cost of generating explanations. Invalid generated labels remain errors. Solo lanes are shuffled per repetition and all test prompts are primed before each lane; actual cache residency depends on server capacity. Add `--cold-cache` to measure score, label and prose generation separately with the SGLang KV cache flushed before each request. This requires a dedicated, idle server and fails if cache flushing is rejected. Flush time is excluded from request latency. See the [SGLang benchmark cache-flush documentation](https://docs.sglang.ai/developer_guide/bench_serving).

```bash
docker compose run --rm jev python3 -m bench.full \
  --models Qwen/Qwen2.5-0.5B-Instruct --cold-cache \
  --min-auto 60 --policy-criterion wilson
```

Thresholds and temperature are selected on dev only. The default minimum is 30 automated dev examples. `--policy-criterion observed` (default) requires empirical accuracy to meet `--target-acc`; `wilson` requires the lower bound of a two-sided 95% Wilson interval to meet it. A threshold may be unreachable, especially with `--smoke` (20 dev cases); in that case no policy file is exported. The interval does not account for threshold search, correlated examples, or distribution shift and is not a guarantee of future accuracy.

Each run writes:

- `<model>__<domain>.json`: per-request latency, predictions and stable case IDs; confusion matrix, precision/recall/F1 per class, confidence-ranked errors, and risk/coverage sweep.
- `<model>__<domain>.report.md` and `.report.risk.png`: readable diagnostics, latency comparison, test automation counts and 95% intervals, challenge results, and coverage/error plot. The test sweep is exploratory; never choose thresholds from it.
- `<model>__<domain>.policy.json`: fitted temperature and thresholds, model, prompt version and task/category fingerprint. A stale policy is removed when no eligible threshold exists.

Use the exported policy in the client or demo:

```python
from jev.decide import Jev
engine = Jev("Qwen/Qwen2.5-0.5B-Instruct", policy="bench/results/<run-id>/Qwen__Qwen2.5-0.5B-Instruct__support.policy.json")
# result = engine.decide(task, text, choices)
# result contains probabilities, raw_probabilities, calibrated, route, review_reasons
```

```bash
docker compose run --rm --service-ports jev python3 -m ui.serve \
  --model Qwen/Qwen2.5-0.5B-Instruct --launch \
  --policy bench/results/<run-id>/Qwen__Qwen2.5-0.5B-Instruct__support.policy.json
```

The policy validates the model and exact task, category descriptions/order and prompt version. Review reasons are `other`, `low_confidence`, and `small_margin`. Without a policy, the existing 0.80/0.20 routing defaults apply; the demo's explicit threshold flags still work.

## Evaluation-only challenges

`data/{support,news}_challenge.jsonl` contain 30 hand-authored synthetic examples per domain: Bulgarian, mixed language, typos, near-domain out-of-scope requests, multiple intents, prompt injection, and long text. Multi-intent examples state a primary intent. They are diagnostic examples, not a representative accuracy benchmark. They are excluded from training, calibration, threshold selection, and headline test metrics. The benchmark evaluates them separately, including results per tag. Rebuild deterministically with `python3 -m data.challenges`.

CPU checks (no model downloads):

```bash
python3 -m tests.test_offline
python3 -m tests.test_features
python3 -m tests.smoke_fake
```

## Experiment history and data checks

Every `bench.full` invocation now creates a unique directory:

```text
bench/results/<UTC timestamp>-<random id>/
  manifest.json                 # configuration, start/end, complete/failed status
  data/                         # exact dataset snapshots used by this run
  data-quality.json             # schema, duplicates, leakage and shared-domain warnings
  <model>__<domain>.json
  <model>__<domain>.policy.json  # only when a dev threshold is eligible
  <model>__<domain>.report.md
  <model>__<domain>.report.risk.png
  article/                      # figures and tables for this run (unless --no-plot)
```

`--out` selects the parent directory; it never reuses a previous run. Dataset SHA-256 hashes and snapshots are saved before inference. The benchmark reads the snapshots. Failed runs retain their manifest and available evidence. No Git commit or Git metadata is read or recorded. Use the printed experiment path when loading policies or regenerating article figures.

```bash
python3 -m data.validate --out data-quality.json
python3 -m article.figures bench/results/<run-id>
```

Validation covers empty/invalid records, label membership, duplicate IDs, normalized duplicate texts, and task/category description/order consistency per domain. It compares all train/dev/test/challenge files across both domains. Same text across different splits is an error; shared text across domains in the same split is a warning (the bundled data has 225 such shared texts). Normalization uses Unicode NFKC, case folding and collapsed whitespace; it does not detect semantic paraphrases. `sample.jsonl` is a demo fixture, outside the split audit. Checks run before the benchmark and before direct LoRA training.

## JSONL CLI and human review

Input records need `task`, `text` and `choices` (2–10 category descriptions); `id` and reference `label` are optional. Missing IDs are assigned by row position. Use explicit IDs to join results across reordered inputs. The client reuses HTTP connections per thread and retries connection failures, timeouts and HTTP 429/500/502/503/504 up to two times with bounded backoff. Timed-out inference can compute twice; measured latency includes retries. Successful responses are checked for item/label counts, numeric finite probabilities, range and sum. Malformed responses fail rather than silently dropping categories.

Run inside the container, or in a Python environment with the inference dependencies installed:

```bash
python3 -m jev predict --input requests.jsonl --output predictions.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --policy bench/results/<run-id>/Qwen__Qwen2.5-0.5B-Instruct__support.policy.json \
  --batch 16 --review review-queue.jsonl
```

Each output preserves the input and adds the decision, calibrated/raw probabilities, routing and review reasons. Batches combine consecutive rows only when task and category order/descriptions match. `--url` selects the inference server. `--input -` and `--output -` support stdin/stdout; summaries go to stderr. Existing output files are refused. Without `--policy`, default thresholds apply and probabilities are marked uncalibrated.

Export a queue from an existing predictions file:

```bash
python3 -m jev review export --predictions predictions.jsonl --output review-queue.jsonl
```

Edit only `review_label` and optional `review_notes` in that queue. Set the label to one of the record's category keys; leave it `null` to keep the item pending. Original inputs and predictions are fingerprinted, so applying a review rejects mismatched or altered source records and duplicate/unknown IDs.

```bash
python3 -m jev review apply --predictions predictions.jsonl \
  --reviews review-queue.jsonl --output reviewed-labels.jsonl

python3 -m jev predict --input reviewed-labels.jsonl --output reevaluated.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct --policy path/to/policy.json
```

Applying reviews creates a **new labeled dataset** and reports the original model's agreement with the human labels. Re-running `predict` with those labels reports accuracy for the new model/policy on stderr. These are metrics on the reviewed subset, not an estimate for all traffic. Reviewed examples are never automatically added to training or used to tune thresholds.

Additional CPU workflow checks: `python3 -m tests.test_workflows`.
