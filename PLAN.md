# Roadmap to the first completed experiment

## Implemented

- [x] Scoring client, single-token labels and fallback backend.
- [x] Prose, single-token, structured and batch comparison lanes.
- [x] Two domains with train/dev/test splits and separate synthetic challenges.
- [x] Data validation, dev calibration and test diagnostics.
- [x] Isolated experiments with manifests and dataset snapshots.
- [x] LoRA script, JSONL CLI, human review and web demo.
- [x] Docker and Windows launcher with full/smoke/test modes.
- [x] CPU tests and a fake inference server for logic checks.

Implemented does not mean validated against a real model. GPU milestones are tracked separately below.

## Next: real smoke test on RTX 4070

- [ ] Verify Docker Desktop/WSL2 and NVIDIA GPU access.
- [ ] Run `run.bat smoke`: successful Qwen 0.5B startup and real endpoint responses.
- [ ] Inspect sanity checks, backend selection, server logs and manifest.
- [ ] Record and pin the working Docker image and dependency versions.

## Full experiment

- [ ] Complete the 0.5B benchmark without training.
- [ ] Complete LoRA training and benchmark the trained model.
- [ ] Check whether 4B fits and runs on the selected GPU configuration.
- [ ] Review accuracy, p50/p95, throughput, calibration and order bias.
- [ ] Evaluate auto/review decisions on test using thresholds selected on dev.
- [ ] Retain results, configuration, versions and limitations.

## Publication

- [ ] Populate the article using measured results only.
- [ ] Review figures, source attribution and dataset licenses.
- [ ] Record the demo with a real model.
- [ ] Verify the final reproduction instructions on Windows.

The experiment is complete when real runs succeed, results are inspectable and the article is supported by those results. Commands are in [README.md](README.md); history and decisions are in [NOTES.md](NOTES.md).
