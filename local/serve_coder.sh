#!/usr/bin/env bash
# Serve the abliterated Qwen3.8-27B as a LOCAL coding model (OpenAI-compatible API) using the
# ROCmFPX build. This is the "when I hit my Claude limit" fallback backend.
#
# IMPORTANT sizing for the RX 9060 XT (16GB):
#   The Q8_0 (28.6GB) is the NODE teacher only - it does NOT fit 16GB. For local use pick a
#   quant that leaves room for KV cache:
#     Q3_K_M  13.3GB  <- recommended: best quality that leaves ~2.5GB for context
#     Q3_K_S  12.1GB     safer, more context room
#     Q2_K    10.7GB     most context room, lowest quality
#   Q4_K_S (15.6GB) technically loads but leaves almost no room for KV -> tiny context only.
#
# NOTE: this 27B is DENSE (no MoE, no MTP) - it will be SLOWER than your distilled A3B. It's a
# capable fallback, not the fast path. The distilled A3B is the one that hits 100-200 tok/s.
#
#   ./serve_coder.sh                       # uses ROCMFPX_DIR + MODEL below
#   MODEL=~/models/Qwen3.8-27B-ABLITERATED-Q3_K_M.gguf ./serve_coder.sh
set -euo pipefail

MODEL="${MODEL:-$HOME/models/Qwen3.8-27B-ABLITERATED-Q3_K_M.gguf}"
ROCMFPX_DIR="${ROCMFPX_DIR:-$HOME/ROCmFPX}"     # your local Vulkan build, or the node HIP build
PORT="${PORT:-8080}"
CTX="${CTX:-16384}"                             # dense model; keep modest to fit 16GB KV
NGL="${NGL:-99}"
SERVER="${SERVER:-$ROCMFPX_DIR/build/bin/llama-server}"

[ -f "$MODEL" ] || { echo "model not found: $MODEL"; echo "download a quant that fits 16GB (Q3_K_M recommended):"; \
  echo "  huggingface-cli download Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF Qwen3.8-27B-ABLITERATED-Q3_K_M.gguf --local-dir ~/models"; exit 1; }
[ -x "$SERVER" ] || { echo "llama-server not found at $SERVER (build ROCmFPX first)"; exit 1; }

echo "serving $(basename "$MODEL") on :$PORT (OpenAI API at /v1)"
exec "$SERVER" -m "$MODEL" --host 127.0.0.1 --port "$PORT" \
  -ngl "$NGL" -c "$CTX" -fa on --no-mmap \
  --alias qwen38-coder-abliterated --jinja
