#!/usr/bin/env bash
# Runs ON the VM (inside tmux): the full training pipeline end to end.
# Log: ~/physio-ai/outputs/pipeline.log
set -euo pipefail
cd "$HOME/physio-ai"
source .venv/bin/activate
export PHYSIO_LLM_PROVIDER=template
mkdir -p outputs
exec > >(tee -a outputs/pipeline.log) 2>&1
echo "=== $(date -u) build manifest"
python scripts/build_manifest.py
echo "=== $(date -u) extract poses"
python scripts/extract_poses.py --workers "${WORKERS:-8}" ${MAX_VIDEOS:+--max-videos $MAX_VIDEOS}
echo "=== $(date -u) baselines"
python scripts/train_baselines.py
echo "=== $(date -u) temporal models"
python scripts/train_temporal.py --epochs "${EPOCHS:-60}"
echo "=== $(date -u) grouped CV for TCN"
python scripts/train_temporal.py --cv tcn:cls --epochs "${EPOCHS:-60}"
python scripts/train_temporal.py --cv tcn:reg --epochs "${EPOCHS:-60}"
echo "=== $(date -u) compare + select"
python scripts/compare_models.py
echo "=== $(date -u) reference stats"
python scripts/build_reference_stats.py
echo "=== $(date -u) final test evaluation"
python scripts/evaluate_test.py
echo "=== $(date -u) tests"
python -m pytest -q tests
echo "=== $(date -u) DONE"
