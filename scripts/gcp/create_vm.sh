#!/usr/bin/env bash
# Create the CPU VM used for pose extraction and training.
# Usage: PROJECT=my-project ./scripts/gcp/create_vm.sh
set -euo pipefail
PROJECT="${PROJECT:?set PROJECT=<gcp project id>}"
ZONE="${ZONE:-us-central1-a}"
VM="${VM:-physio-vm}"
MACHINE="${MACHINE:-n2-standard-8}"
DISK_GB="${DISK_GB:-300}"

gcloud config set project "$PROJECT" >/dev/null
gcloud services enable compute.googleapis.com >/dev/null
if ! gcloud compute instances describe "$VM" --zone "$ZONE" >/dev/null 2>&1; then
  gcloud compute instances create "$VM" \
    --zone "$ZONE" --machine-type "$MACHINE" \
    --image-family debian-12 --image-project debian-cloud \
    --boot-disk-size "${DISK_GB}GB" --boot-disk-type pd-balanced \
    --scopes cloud-platform \
    --metadata enable-oslogin=false
  echo "waiting for SSH..."
  for i in $(seq 1 30); do
    if gcloud compute ssh "$VM" --zone "$ZONE" --command "true" >/dev/null 2>&1; then break; fi
    sleep 10
  done
fi
gcloud compute instances describe "$VM" --zone "$ZONE" --format='value(status,networkInterfaces[0].accessConfigs[0].natIP)'
