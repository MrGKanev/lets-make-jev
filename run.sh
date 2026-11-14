#!/usr/bin/env bash
# Everything, A to Z, in the container: tests -> data -> smoke -> fine-tune -> full benchmark -> figures.
# On the host this just starts the container; inside it, it does the work.
# Usage: bash run.sh      |      MODELS="Qwen/Qwen2.5-0.5B-Instruct" bash run.sh
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f /.dockerenv ]; then
    exec docker compose run --rm --build jev
fi

JEV_MODE="${JEV_MODE:-full}"
case "$JEV_MODE" in
    full|smoke|test) ;;
    *) echo "Invalid JEV_MODE: $JEV_MODE (use full, smoke or test)"; exit 2 ;;
esac

MODELS="${MODELS:-Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen3-4B-Instruct-2507}"

python3 -m tests.test_offline
python3 -m tests.test_features
python3 -m tests.test_workflows
python3 -m tests.smoke_fake
if [ "$JEV_MODE" = test ]; then
    python3 -m data.validate --out /tmp/jev-data-quality.json
    echo "CPU tests and data validation passed."
    exit 0
fi
nvidia-smi || { echo "No GPU in the container: check NVIDIA Container Toolkit / Docker Desktop GPU support."; exit 1; }
python3 -c 'import torch; assert torch.cuda.is_available(), "PyTorch cannot access CUDA"; print("CUDA:", torch.cuda.get_device_name(0))'

for domain in support news; do
    for split in train dev test; do
        if [ ! -s "data/${domain}_${split}.jsonl" ]; then
            python3 -m data.prepare
            break 2
        fi
    done
done
python3 -m data.validate --out /tmp/jev-data-quality.json

python3 -m bench.full --models ${MODELS%% *} --smoke --no-plot   # fail fast on API problems, first model only
if [ "$JEV_MODE" = smoke ]; then
    echo "GPU smoke test passed. Results: bench/results/<run-id>/"
    exit 0
fi
# The "Jev-like" model: the small base model fine-tuned to answer with the label token (skip: SKIP_TRAIN=1).
TUNED=models/jev-Qwen2.5-0.5B-Instruct
if [ "${SKIP_TRAIN:-0}" != 1 ]; then
    [ -f $TUNED/train_log.json ] || python3 -m train.lora
    MODELS="$MODELS $TUNED"
fi

python3 -m bench.full --models $MODELS
echo "Done: results, policies, reports and article figures are in bench/results/<run-id>/"
