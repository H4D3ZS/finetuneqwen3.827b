#!/usr/bin/env bash
# LOCAL (RX 9060 XT, 16GB). Serve the distilled A3B on :8080.
#
# MUST be the ROCmFPX fork's llama-server: the GGUF's output.weight is fork-native
# ggml type 102 (Q4_0_ROCMFP4_COHERENT). Upstream llama.cpp — including every
# llamacpp backend lemonade ships — fails with "invalid ggml type 102".
#
#   bash local/serve_distilled.sh
set -euo pipefail

ROCMFPX="${ROCMFPX_DIR:-$HOME/Desktop/ROCmFPX}"
SERVER="$ROCMFPX/build/bin/Release/llama-server.exe"
GGUF="${GGUF:-$HOME/.cache/huggingface/hub/models--Lord-H4D3ZS--Qwen3.8-Distill-35B-A3B-Coder-Abliterated/snapshots/75914ba5f4bbb059d2d54f22eceb626b5481cca9/Qwen3.8-Distill-35B-A3B-Coder-Abliterated-Q2KXL_ROCMFPX.gguf}"
CTX="${CTX:-32768}"   # 32k fits alongside the 12.3GB weights on 16GB; 98304 will OOM

[ -x "$SERVER" ] || { echo "no ROCmFPX llama-server at $SERVER"; exit 1; }
[ -f "$GGUF" ]   || { echo "no GGUF at $GGUF"; exit 1; }

# --jinja + deepseek reasoning-format are REQUIRED: without them this model leaks malformed
# <think> tags into content and drops tool calls. With them: clean output, tool calls 3/3.
exec "$SERVER" -m "$GGUF" --alias qwen38d \
  -c "$CTX" -fa on -ctk q8_0 -ctv q8_0 -ngl 99 -np 1 --no-mmap \
  --jinja --reasoning-format deepseek \
  --port 8080
