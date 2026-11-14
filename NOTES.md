# Development history and design decisions

This file describes the project's evolution, not measured model performance. See [PLAN.md](PLAN.md) for remaining work and [README.md](README.md) for commands.

## 1. From an idea to a client

The first version compared first-token scoring with text generation for fixed-choice decisions. Categories use single-token labels A–J and map back to meaningful application keys. A confidence threshold and minimum margin between the two leading choices control automatic decisions versus human review.

A native `/generate` fallback was added for servers without a usable `/v1/score` endpoint. Compatibility with a real SGLang deployment still needs GPU validation.

## 2. Data and matched comparisons

The experiment covers banking support and news. Public datasets provide separate train/dev/test splits, with checks for duplicates and leakage. The `other` class comes from CLINC out-of-scope examples; some are borderline cases for the news domain.

The benchmark includes single-token generation with the same prompt, prose and structured generation, batching, repeated measurements and category permutations. This helps distinguish the cost of explanations from the difference between scoring and generating one label.

## 3. Calibration and diagnostics

Temperature and thresholds are selected on dev only. Test evaluates the selected policy. Diagnostics include confusion matrices, per-class metrics, risk/coverage and synthetic challenges. Challenges are excluded from training and threshold selection.

Confidence is relative to the allowed categories and does not guarantee accuracy on future traffic. Exported policies validate the model, prompt version and exact category configuration.

## 4. Training and workflows

The LoRA script trains the first answer token while reshuffling options every epoch. The resulting model is benchmarked against the base model. This path is implemented but has not been validated on a GPU.

A web demo, JSONL CLI and human-review queue support practical use. Reviewed examples become a new labeled dataset without automatic training inclusion. Each experiment stores its own manifest, dataset snapshots and hashes.

## 5. Startup and reliability improvements — 2026-11-14

The Windows launcher supports full/smoke/test modes, checks Docker and Compose, and propagates failures. SKIP_TRAIN is passed into the container. Outputs and downloaded model weights persist between runs.

Server cleanup handles interruption during startup. A stalled graceful shutdown is followed by forced termination. The CLI rejects incomplete batch responses instead of silently writing partial results.

README now explains the idea, first run, automatic downloads, output locations and limitations. Documentation is in English; multilingual evaluation fixtures retain their original languages.

## 6. Public repository preparation — 2026-11-14

The public history starts from a single descriptive initial commit containing the current implementation. Downloaded third-party reference posts, images and videos in `source/` are excluded from the publication tree and retained locally. The previous history and working changes are backed up outside the repository before replacement.

## Remaining unknowns

No real GPU measurements are included. Speed, accuracy, LoRA improvements, 4B memory requirements and real SGLang endpoint behavior remain unverified. Pin the Docker image and dependency versions after the first successful real run. The article remains a draft until results are available.
