#!/usr/bin/env bash
#
# Download the unsloth/Qwen3.6-27B-NVFP4 model.
#
# Usage:
#   bash scripts/setup.sh
#
# Env vars (optional):
#   MODEL_DIR      Where to place the model. Default: ~/.cache/huggingface/hub
#   HF_TOKEN       HF token (public model, so usually unnecessary)
#

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODEL_REPO="unsloth/Qwen3.6-27B-NVFP4"

echo "=== Downloading ${MODEL_REPO} ==="
echo ""

# Use HF default cache (avoids duplicating if already downloaded by another tool)
if command -v huggingface-cli >/dev/null 2>&1; then
  echo "Using huggingface-cli ..."
  HF_HUB_ENABLE_HF_TRANSFER=1 \
    huggingface-cli download "${MODEL_REPO}"
elif command -v hf >/dev/null 2>&1; then
  echo "Using hf CLI ..."
  HF_HUB_ENABLE_HF_TRANSFER=1 \
    hf download "${MODEL_REPO}"
else
  echo "ERROR: neither 'huggingface-cli' nor 'hf' found." >&2
  echo "  Install: pip install 'huggingface-hub[hf_transfer]'" >&2
  exit 1
fi

echo ""
echo "=== Model download complete ==="
echo "Model cached in HF hub directory (~14 GB for NVFP4)."
echo ""
echo "You can now start the server with:"
echo "  bash scripts/serve.sh"
