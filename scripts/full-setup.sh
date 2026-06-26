#!/usr/bin/env bash
#
# Full end-to-end setup: install venv, download model, start server, sanity test.
#
# Usage:
#   bash scripts/full-setup.sh
#
# Env vars (optional):
#   SKIP_INSTALL   Set to 1 to skip venv creation (use existing)
#   SKIP_MODEL     Set to 1 to skip model download
#   SKIP_SERVER    Set to 1 to stop after install+download
#   VENV_DIR       Override venv location. Default: ./venv
#   PORT           Server port. Default: 8000
#   HEALTH_TIMEOUT Max seconds to wait for server. Default: 300

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# ---------- Color helpers ----------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

pass() { echo -e "  ${GREEN}✓${NC} $*"; }
fail() { echo -e "  ${RED}✗${NC} $*" >&2; }
warn() { echo -e "  ${YELLOW}⚠${NC} $*"; }
info() { echo -e "  ${CYAN}→${NC} $*"; }

FAIL=0

# ---------- Dependency checks ----------
echo ""
echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Dependency checks${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
echo ""

for cmd in python3 curl git; do
  if command -v "$cmd" >/dev/null 2>&1; then
    pass "$cmd"
  else
    fail "Missing: $cmd (required)"
    FAIL=1
  fi
done

if command -v nvidia-smi >/dev/null 2>&1; then
  DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
  pass "NVIDIA driver ($DRIVER_VER) — $GPU_NAME"

  VRAM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 | grep -oP '\d+' || echo "0")
  if [[ "$VRAM_TOTAL" -lt 24000 ]]; then
    warn "GPU has ${VRAM_TOTAL} MiB VRAM — 32 GB recommended"
  fi
else
  fail "Missing: nvidia-smi (NVIDIA driver not found)"
  FAIL=1
fi

echo ""
if [[ "$FAIL" -ne 0 ]]; then
  echo -e "${RED}Dependency checks failed. Fix the issues above and re-run.${NC}"
  exit 1
fi

echo -e "${GREEN}All dependency checks passed.${NC}"
echo ""

# ---------- Install ----------
if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
  echo -e "${CYAN}  Installing vLLM nightly + PyTorch${NC}"
  echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
  echo ""

  info "Running install.sh ..."
  bash "${ROOT_DIR}/scripts/install.sh"
  pass "Install complete"
  echo ""
else
  info "SKIP_INSTALL=1 — skipping install"
  echo ""
fi

# ---------- Model download ----------
if [[ "${SKIP_MODEL:-0}" != "1" ]]; then
  echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
  echo -e "${CYAN}  Model download${NC}"
  echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
  echo ""

  info "Running setup.sh ..."
  bash "${ROOT_DIR}/scripts/setup.sh"
  pass "Model download complete"
  echo ""
else
  info "SKIP_MODEL=1 — skipping model download"
  echo ""
fi

if [[ "${SKIP_SERVER:-0}" == "1" ]]; then
  pass "SKIP_SERVER=1 — stopping here."
  exit 0
fi

# ---------- Start server ----------
echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Starting server${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
echo ""

PORT="${PORT:-8000}"
URL="http://localhost:${PORT}"

info "Starting vLLM server on port ${PORT} ..."

# Kill any existing server on this port
if command -v lsof >/dev/null 2>&1; then
  EXISTING_PID=$(lsof -ti :${PORT} 2>/dev/null || true)
  if [[ -n "$EXISTING_PID" ]]; then
    info "Killing existing process on port ${PORT} (PID: $EXISTING_PID) ..."
    kill "$EXISTING_PID" 2>/dev/null || true
    sleep 2
  fi
fi

# Start server in background
bash "${ROOT_DIR}/scripts/serve.sh" &
SERVER_PID=$!
info "Server PID: ${SERVER_PID}"

# ---------- Wait for readiness ----------
echo ""
echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Waiting for server startup${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
echo ""

HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-300}"
ELAPSED=0
INTERVAL=5

while [[ "$ELAPSED" -lt "$HEALTH_TIMEOUT" ]]; do
  if curl -sf "${URL}/v1/models" >/dev/null 2>&1; then
    pass "Server endpoint reachable at ${URL}"
    break
  fi

  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    fail "Server process exited unexpectedly (PID: ${SERVER_PID})"
    exit 1
  fi

  sleep "$INTERVAL"
  ELAPSED=$((ELAPSED + INTERVAL))
  if [[ $((ELAPSED % 30)) -eq 0 ]]; then
    info "Still waiting... (${ELAPSED}s / ${HEALTH_TIMEOUT}s)"
  fi
done

if [[ "$ELAPSED" -ge "$HEALTH_TIMEOUT" ]]; then
  fail "Server did not start within ${HEALTH_TIMEOUT}s"
  kill "$SERVER_PID" 2>/dev/null || true
  exit 1
fi

echo ""

# ---------- Sanity test ----------
echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Sanity test${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
echo ""

info "Sending test request ..."
RESPONSE=$(curl -sf "${URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b-nvfp4","messages":[{"role":"user","content":"Capital of France? Answer in one word."}],"max_tokens":10}')

if [[ -n "$RESPONSE" ]]; then
  CONTENT=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'].strip())" 2>/dev/null || echo "<parse error>")
  TOKENS=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null || echo "?")
  pass "Server responded: \"$CONTENT\" ($TOKENS tokens)"
else
  fail "Empty response from server"
  exit 1
fi

echo ""

# ---------- GPU state ----------
if command -v nvidia-smi >/dev/null 2>&1; then
  echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
  echo -e "${CYAN}  GPU state${NC}"
  echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
  echo ""
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
             --format=csv,noheader
  echo ""
fi

# ---------- Summary ----------
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Setup complete ✓${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
echo ""
echo "  Endpoint:  ${URL}/v1/*"
echo "  Server PID: ${SERVER_PID}"
echo "  Model:     unsloth/Qwen3.6-27B-NVFP4"
echo ""
echo "  Bench:     bash scripts/bench.sh"
echo "  Stop:      kill ${SERVER_PID}"
echo ""

# Keep the script alive so the server keeps running
wait "$SERVER_PID" 2>/dev/null || true
