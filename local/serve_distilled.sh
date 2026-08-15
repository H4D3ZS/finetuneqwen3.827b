#!/usr/bin/env bash
# Serve the DISTILLED Qwen3.8-Distill-35B-A3B-Coder (fast MoE + MTP, 12GB) as a local coding
# backend for Claude Code. This is the FAST path (A3B, ~100+ tok/s target on 16GB) - distinct
# from serve_coder.sh which serves the DENSE 27B fallback.
#
#   ./serve_distilled.sh                       # uses ROCMFPX_DIR + MODEL below
#   MODEL=~/models/...Q2KXL_ROCMFPX.gguf ./serve_distilled.sh
#
# One-time download (12.3GB, fits 16GB with room for context):
#   hf download Lord-H4D3ZS/Qwen3.8-Distill-35B-A3B-Coder-Abliterated \
#     Qwen3.8-Distill-35B-A3B-Coder-Abliterated-Q2KXL_ROCMFPX.gguf --local-dir ~/models
set -euo pipefail

MODEL="${MODEL:-$HOME/models/Qwen3.8-Distill-35B-A3B-Coder-Abliterated-Q2KXL_ROCMFPX.gguf}"
ROCMFPX_DIR="${ROCMFPX_DIR:-$HOME/ROCmFPX}"     # your LOCAL RDNA4 build (gfx1200)
PORT="${PORT:-8080}"
CTX="${CTX:-32768}"                             # 12GB model on 16GB leaves ~4GB KV -> ~32k single-stream
NGL="${NGL:-99}"
PARALLEL="${PARALLEL:-1}"                        # 16GB: pick context (np=1) OR concurrency (np>1,-c lower)
SERVER="${SERVER:-$ROCMFPX_DIR/build/bin/llama-server}"

[ -f "$MODEL" ] || { echo "model not found: $MODEL"; echo "download it:"; \
  echo "  hf download Lord-H4D3ZS/Qwen3.8-Distill-35B-A3B-Coder-Abliterated \\"; \
  echo "    $(basename "$MODEL") --local-dir ~/models"; exit 1; }
[ -x "$SERVER" ] || { echo "llama-server not found at $SERVER - build ROCmFPX (see repo BUILD.md)"; exit 1; }

echo "serving DISTILLED A3B $(basename "$MODEL") on :$PORT (OpenAI API at /v1)"
echo "  ctx=$CTX parallel=$PARALLEL  (16GB: raise -np only if you lower -c)"
exec "$SERVER" -m "$MODEL" --host 127.0.0.1 --port "$PORT" \
  -ngl "$NGL" -c "$CTX" -np "$PARALLEL" -fa on --jinja \
  --alias qwen38-distill-a3b-coder
