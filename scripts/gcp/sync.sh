#!/usr/bin/env bash
# Copy code to the VM (push) or results back (pull).
# Usage: PROJECT=... ./scripts/gcp/sync.sh push|pull
set -euo pipefail
PROJECT="${PROJECT:?set PROJECT}"
ZONE="${ZONE:-us-central1-a}"
VM="${VM:-physio-vm}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
case "${1:-}" in
  push)
    gcloud compute ssh "$VM" --zone "$ZONE" --project "$PROJECT" --command "mkdir -p ~/physio-ai"
    tar -C "$ROOT" --exclude=.git --exclude=.venv --exclude='data/poses' --exclude='data/tmp' --exclude='data/raw/*.mp4' \
        --exclude='outputs/models/*.pt' --exclude='__pycache__' --exclude='.pytest_cache' -czf /tmp/physio-ai.tgz .
    gcloud compute scp /tmp/physio-ai.tgz "$VM:~/physio-ai.tgz" --zone "$ZONE" --project "$PROJECT"
    gcloud compute ssh "$VM" --zone "$ZONE" --project "$PROJECT" --command "tar -C ~/physio-ai -xzf ~/physio-ai.tgz && rm ~/physio-ai.tgz && echo pushed"
    ;;
  pull)
    gcloud compute ssh "$VM" --zone "$ZONE" --project "$PROJECT" --command \
      "cd ~/physio-ai && tar -czf ~/results.tgz outputs data/manifest data/poses/_progress.jsonl"
    gcloud compute scp "$VM:~/results.tgz" /tmp/results.tgz --zone "$ZONE" --project "$PROJECT"
    tar -C "$ROOT" -xzf /tmp/results.tgz && echo pulled
    ;;
  *) echo "usage: $0 push|pull"; exit 1;;
esac
