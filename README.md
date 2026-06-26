# Qwen3.6-27B-NVFP4 on a single RTX 5090, 260K context

**Serve `unsloth/Qwen3.6-27B-NVFP4` on a single 32 GB RTX 5090** — native NVFP4 quantization via vLLM nightly, MTP speculative decoding, fp8 KV cache, tool calling, reasoning parser. No Docker, no patches.

NVFP4 (NVIDIA FP4) is the native 4-bit quantization format for Blackwell GPUs. It uses FP4 tensor cores for ~1.6x throughput over BF16 with only 2-4% quality loss. Requires SM 12.0 (RTX 5090) or SM 10.x (datacenter Blackwell).

## Requirements

- **GPU:** 1x NVIDIA RTX 5090 (32 GB, Blackwell GB202, SM 12.0)
- **Driver:** 580.x+ (for CUDA 13 runtime)
- **CUDA toolkit:** 13.x installed on system
- **Disk:** ~14 GB for model weights
- **Python:** 3.12 or 3.13

## Quick start

```bash
bash <(curl -s https://raw.githubusercontent.com/darksidewalker/qwen36-27b-nvfp4-5090/main/scripts/go.sh)
```

That's it — clones nothing, installs venv + vLLM nightly, downloads the model (~14 GB), and starts the server on port 8000. Re-runs skip steps already done.

Or clone first:

```bash
git clone https://github.com/darksidewalker/qwen36-27b-nvfp4-5090.git && cd qwen36-27b-nvfp4-5090 && bash scripts/go.sh
```

Server runs on `http://localhost:8000/v1/*` — drop-in OpenAI-compatible API.

## Configuration

All served via `scripts/serve.sh` with env var overrides:

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Server port |
| `MAX_MODEL_LEN` | `262144` | Context length (260K tokens) |
| `GPU_MEM_UTIL` | `0.89` | GPU memory utilization (~28.5 GB of 32 GB) |
| `MAX_NUM_SEQS` | `4` | Max concurrent sequences |
| `NVFP4_BACKEND` | `marlin` | GEMM backend (`marlin` = better decode ~80 TPS, `flashinfer-cutlass` = better prefill) |
| `VISION` | `0` | Set to `1` to load visual encoder (BF16, uses more VRAM) |
| `MODEL` | `unsloth/Qwen3.6-27B-NVFP4` | HF model path |

## What's different from the AutoRound INT4 setup

| Aspect | AutoRound INT4 | NVFP4 |
|---|---|---|
| Docker | Yes (pinned nightly + Genesis patches) | **No** (native vLLM nightly) |
| Patches | Genesis + tolist cudagraph fix | **None** |
| Quantization | `--quantization auto_round` | Auto-detected (compressed-tensors) |
| GEMM backend | N/A | `marlin` (env: `VLLM_NVFP4_GEMM_BACKEND`) |
| Model size | ~20 GB | **~14 GB** |
| KV cache | `fp8_e4m3` | `fp8` |
| Expected decode TPS | ~160 | ~80-100 |
| Expected prefill pp2048 | ~4000+ | ~4000+ |

NVFP4 has smaller model footprint and is the "native" Blackwell quantization path. AutoRound INT4 gives higher decode TPS but needs more VRAM for the model itself.

## Vision support

Qwen3.6-27B is multimodal (vision + text). **Vision is loaded by default** — the visual encoder runs in BF16 (excluded from NVFP4 quantization) alongside the NVFP4 language model.

To run text-only (slightly lower VRAM), disable vision:

```bash
VISION=0 bash scripts/serve.sh
```

## Benchmark

```bash
bash scripts/bench.sh
```

Expected on RTX 5090 with Marlin backend:
- Decode: ~80 TPS (single sequence, short context)
- Prefill: ~4000 tokens/sec (2048 token prompt)
- Time to first token: ~1.5s (4K context)
