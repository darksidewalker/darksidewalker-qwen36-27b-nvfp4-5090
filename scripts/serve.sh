#!/usr/bin/env bash
#
# Serve unsloth/Qwen3.6-27B-NVFP4 on a single RTX 5090 (32 GB) with 260K context.
#
# Uses the venv created by scripts/install.sh.
#
# Config:
#   - NVFP4 quantization (auto-detected from model config)
#   - MTP speculative decoding (n=3)
#   - fp8 KV cache (massive VRAM savings at 260K context)
#   - Marlin GEMM backend (best decode throughput ~80 TPS)
#   - Language-model-only (text, no vision encoder)
#   - Qwen3 reasoning parser + tool calling
#
# Usage:
#   bash scripts/serve.sh
#
# Env vars (optional):
#   PORT             Server port. Default: 8000
#   MAX_MODEL_LEN    Context length. Default: 262144 (260K)
#   GPU_MEM_UTIL     GPU memory utilization. Default: 0.89
#   MAX_NUM_SEQS     Max concurrent sequences. Default: 4
#   NVFP4_BACKEND    GEMM backend: marlin (default) or flashinfer-cutlass
#   MODEL            HF model path. Default: unsloth/Qwen3.6-27B-NVFP4
#   CHAT_TEMPLATE    Path to chat template jinja file. Default: repo's compose/qwen3.5-enhanced.jinja
#   VISION           Set to 1 to enable vision (loads visual encoder in BF16)
#

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# ---------- Defaults ----------
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.89}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
NVFP4_BACKEND="${NVFP4_BACKEND:-marlin}"
MODEL="${MODEL:-unsloth/Qwen3.6-27B-NVFP4}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-${ROOT_DIR}/compose/qwen3.5-enhanced.jinja}"
VISION="${VISION:-0}"

# ---------- Activate venv ----------
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/venv}"
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "ERROR: venv not found at ${VENV_DIR}. Run 'bash scripts/install.sh' first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# ---------- Export env for vLLM ----------
export VLLM_NVFP4_GEMM_BACKEND="${NVFP4_BACKEND}"
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

# ---------- Build command ----------
echo "=== Qwen3.6-27B-NVFP4 Server ==="
echo "  Model:         ${MODEL}"
echo "  Context:       ${MAX_MODEL_LEN} (~$(( MAX_MODEL_LEN / 1024 ))K tokens)"
echo "  GPU mem util:  ${GPU_MEM_UTIL}"
echo "  NVFP4 backend: ${NVFP4_BACKEND}"
echo "  Port:          ${PORT}"
echo "  Vision:        ${VISION}"
echo ""

CMD=(
  vllm serve "${MODEL}"
  --host 0.0.0.0
  --port "${PORT}"
  --dtype bfloat16
  --tensor-parallel-size 1
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens 4096
  --kv-cache-dtype fp8
  --trust-remote-code
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --chat-template "${CHAT_TEMPLATE}"
  --enable-prefix-caching
  --enable-chunked-prefill
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
)

# Vision: Qwen3.6 is multimodal. Default is full model (vision + text).
# The visual encoder is kept in BF16 (excluded from NVFP4 quantization).
# Set VISION=0 to run text-only (--language-model-only) for lower VRAM usage.
if [[ "${VISION}" == "0" ]]; then
  CMD+=(--language-model-only)
fi

echo "Starting server on port ${PORT} ..."
echo "Press Ctrl+C to stop."
echo ""

exec "${CMD[@]}"
