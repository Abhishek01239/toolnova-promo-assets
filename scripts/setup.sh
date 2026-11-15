#!/usr/bin/env bash
# One-time setup: system packages, Python deps, and the free Kokoro voice model.
set -euo pipefail
cd "$(dirname "$0")/.."
if command -v apt-get >/dev/null; then
  sudo_cmd=""; [ "$(id -u)" -ne 0 ] && sudo_cmd="sudo"
  $sudo_cmd apt-get update -qq
  $sudo_cmd apt-get install -y -qq ffmpeg libegl1 libegl-mesa0 fonts-inter fonts-noto-color-emoji
fi
python3 -m pip install -q -r requirements.txt
bash scripts/get_models.sh
