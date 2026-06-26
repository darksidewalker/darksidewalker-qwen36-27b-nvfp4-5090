#!/usr/bin/env bash
#
# One-liner: install venv + download model + start server.
#
# Usage:
#   bash scripts/go.sh
#
# Env vars (optional):
#   PORT           Server port. Default: 8000
#   MAX_MODEL_LEN  Context length. Default: 262144 (260K)
#   GPU_MEM_UTIL   GPU memory utilization. Default: 0.89
#   NVFP4_BACKEND  GEMM backend: marlin (default) or flashinfer-cutlass
#   VISION         Set to 0 for text-only. Default: full model (vision ON)
#
# Idempotent: skips install/download if already done.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.89}"
NVFP4_BACKEND="${NVFP4_BACKEND:-marlin}"
VISION="${VISION:-1}"
VENV="$ROOT/venv"

# ── 1. Create venv ──────────────────────────────────────────────
if [[ ! -d "$VENV" ]]; then
  echo "[1/4] Creating venv..."
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# ── 2. Install deps (idempotent — pip is fast if already done) ──
if ! python3 -c "import vllm" 2>/dev/null; then
  echo "[2/4] Installing PyTorch nightly (cu130) + vLLM nightly..."
  pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130 --extra-index-url https://pypi.org/simple -q
  pip install --pre vllm --index-url https://wheels.vllm.ai/nightly/cu130 --extra-index-url https://pypi.org/simple -q
  pip install "huggingface-hub[hf_transfer]" -q
else
  echo "[2/4] vLLM already installed."
fi

# ── 3. Download model ──────────────────────────────────────────
MODEL_REPO="unsloth/Qwen3.6-27B-NVFP4"
if ! python3 -c "
from huggingface_hub import scan_cache_dir
cache = scan_cache_dir()
repos = [r.repo_id for rev in cache.repos for r in [rev]]
exit(0 if '$MODEL_REPO' in repos else 1)
" 2>/dev/null; then
  echo "[3/4] Downloading $MODEL_REPO (~14 GB)..."
  HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download "$MODEL_REPO"
else
  echo "[3/4] Model already cached."
fi

# ── 4. Start server ────────────────────────────────────────────
export VLLM_NVFP4_GEMM_BACKEND="$NVFP4_BACKEND"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_CUMEM_ENABLE=0
export NCCL_P2P_DISABLE=1
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export VLLM_NO_USAGE_STATS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export VLLM_FLOAT32_MATMUL_PRECISION=high
export VLLM_USE_FLASHINFER_SAMPLER=1
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=8
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_MARLIN_USE_ATOMIC_ADD=1

echo ""
echo "[4/4] Starting vLLM on port $PORT (ctx=${MAX_MODEL_LEN}, mem=${GPU_MEM_UTIL}, backend=${NVFP4_BACKEND}, vision=${VISION})"
echo ""

CMD=(
  vllm serve "$MODEL_REPO"
  --host 0.0.0.0 --port "$PORT"
  --dtype bfloat16
  --tensor-parallel-size 1
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --max-num-seqs 4
  --max-num-batched-tokens 4096
  --kv-cache-dtype fp8
  --trust-remote-code
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --enable-prefix-caching
  --enable-chunked-prefill
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
)

[[ "$VISION" == "0" ]] && CMD+=(--language-model-only)

exec "${CMD[@]}"
