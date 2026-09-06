#!/usr/bin/env bash
# Runs ON the VM (inside tmux): resume after a pause. Skips pose extraction entirely
# (all clips are cached in data/poses) and skips temporal runs that already finished.
set -euo pipefail
cd "$HOME/physio-ai"
source .venv/bin/activate
export PHYSIO_LLM_PROVIDER=template
mkdir -p outputs
exec > >(tee -a outputs/pipeline.log) 2>&1
echo "=== $(date -u) RESUME: temporal models (skipping completed runs)"
python scripts/train_temporal.py --epochs "${EPOCHS:-60}" --skip-existing
echo "=== $(date -u) grouped CV for TCN"
python scripts/train_temporal.py --cv tcn:cls --epochs "${EPOCHS:-60}" --skip-existing
python scripts/train_temporal.py --cv tcn:reg --epochs "${EPOCHS:-60}" --skip-existing
echo "=== $(date -u) compare + select"
python scripts/compare_models.py
echo "=== $(date -u) reference stats"
python scripts/build_reference_stats.py
echo "=== $(date -u) final test evaluation"
python scripts/evaluate_test.py
echo "=== $(date -u) seed demo database"
python scripts/seed_demo.py --reset
echo "=== $(date -u) tests"
python -m pytest -q tests
echo "=== $(date -u) DONE"
