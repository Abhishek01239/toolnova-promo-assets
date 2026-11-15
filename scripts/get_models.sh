#!/usr/bin/env bash
# Download the Kokoro TTS model (Apache-2.0, ~350 MB) into models/ (skipped if present).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
base=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
for f in kokoro-v1.0.onnx voices-v1.0.bin; do
  [ -s "models/$f" ] || curl -fsSL -o "models/$f" "$base/$f"
done
echo "models ready"
