#!/usr/bin/env bash
# Runs ON the VM: system packages, Python venv, project dependencies, pose model.
set -euo pipefail
cd "$HOME/physio-ai"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip ffmpeg libgl1 libegl1 libgles2 libopengl0 libglib2.0-0 tmux git >/dev/null
if [ ! -d .venv ]; then python3 -m venv .venv; fi
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
python - <<'EOF'
from physio import config
from physio.data.download import download_pose_model
config.ensure_dirs()
p = download_pose_model()
print("pose model:", p, p.stat().st_size, "bytes")
import mediapipe, cv2, torch, sklearn
print("mediapipe", mediapipe.__version__, "opencv", cv2.__version__, "torch", torch.__version__, "sklearn", sklearn.__version__)
EOF
echo "bootstrap done"
