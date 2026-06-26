#!/usr/bin/env bash
#
# Install vLLM nightly (cu130) with NVFP4 support for RTX 5090 (SM 12.0).
#
# Installs into a Python venv at ./venv (or uses existing one).
#   - CUDA 13 PyTorch nightly (torch >= 2.9, for Blackwell SM 12.0)
#   - vLLM nightly (cu130, with native NVFP4 via compressed-tensors)
#   - hf_transfer for fast model downloads
#
# Usage:
#   bash scripts/install.sh
#   source venv/bin/activate
#   vllm serve unsloth/Qwen3.6-27B-NVFP4 ...
#
# Env vars (optional):
#   VENV_DIR       Where to create venv. Default: ./venv
#   UV_EXTRA       Extra pip/uv args (e.g. --index-url ...)
#

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

VENV_DIR="${VENV_DIR:-${ROOT_DIR}/venv}"

# ---------- Pre-flight ----------
echo "=== Pre-flight checks ==="

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found." >&2; exit 1
fi
PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "  Python: $(python3 --version 2>&1)"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found — NVIDIA driver missing." >&2; exit 1
fi
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "  GPU: $GPU"

# ---------- Create venv ----------
if [[ ! -d "${VENV_DIR}" ]]; then
  echo ""
  echo "Creating venv at ${VENV_DIR} ..."
  python3 -m venv "${VENV_DIR}"
fi

echo "Activating venv ..."
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# ---------- Upgrade pip ----------
echo ""
echo "=== Upgrading pip ==="
pip install --upgrade pip setuptools wheel -q

# ---------- Install PyTorch nightly (CUDA 13) ----------
echo ""
echo "=== Installing PyTorch nightly (CUDA 13) ==="
pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu130 \
  --extra-index-url https://pypi.org/simple \
  -q ${UV_EXTRA:-}

# ---------- Install vLLM nightly (cu130) ----------
echo ""
echo "=== Installing vLLM nightly (cu130) ==="
pip install --pre vllm \
  --index-url https://wheels.vllm.ai/nightly/cu130 \
  --extra-index-url https://pypi.org/simple \
  -q ${UV_EXTRA:-}

# ---------- Install hf_transfer ----------
echo ""
echo "=== Installing hf_transfer ==="
pip install "huggingface-hub[hf_transfer]" -q

# ---------- Verify ----------
echo ""
echo "=== Verification ==="
python3 -c "
import torch
print(f'  torch: {torch.__version__}')
print(f'  CUDA:  {torch.version.cuda}')
print(f'  cuDNN: {torch.backends.cudnn.version()}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  Device: {torch.cuda.get_device_name(0)}')
    cap = torch.cuda.get_device_capability(0)
    print(f'  Capability: SM {cap[0]}.{cap[1]}')
    if cap[0] < 12:
        print('  WARNING: NVFP4 requires Blackwell (SM 12.x). Your GPU may not support it.')
    else:
        print('  ✓ SM 12.x detected — NVFP4 supported.')
"

python3 -c "
import vllm
print(f'  vLLM: {vllm.__version__}')
"

echo ""
echo "=== Install complete ==="
echo ""
echo "Next steps:"
echo "  source venv/bin/activate"
echo "  bash scripts/setup.sh          # download model"
echo "  bash scripts/serve.sh          # start server"
echo ""
