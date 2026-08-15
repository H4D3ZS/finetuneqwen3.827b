#!/usr/bin/env bash
# LOCAL (RX 9060 XT, 16GB). Serve the DENSE Qwen3.8-27B on :8080, tuned to fit 16GB.
#
# QUANT CHOICE (measured on this card, Vulkan backend):
#   Q2_0_ROCMFPX  8.6GB  28 tok/s  but INCOHERENT (2-bit collapses a dense model) -> unusable
#   Q3_K_XL      12.5GB   9.3 tok/s coherent, but slow (generic Vulkan K-quant kernels)
#   Q3_0_ROCMFPX ~11.5GB  ~18-22 t/s coherent + fast fork-native kernels  <- BUILD THIS (see
#                                    local/build_dense27b_fast.sh); best dense option here.
# Default below is the coherent one you already have (Q3_K_XL). Override GGUF to swap.
#
#   bash local/serve_dense27b.sh
set -euo pipefail

ROCMFPX="${ROCMFPX_DIR:-$HOME/Desktop/ROCmFPX}"
SERVER="$ROCMFPX/build/bin/Release/llama-server.exe"
GGUF="${GGUF:-$ROCMFPX/Qwen3.8-27B-UD-Q3_K_XL.gguf}"
CTX="${CTX:-16384}"   # 12.5GB weights leave ~3GB; hybrid attn (16 full layers) keeps KV cheap

[ -x "$SERVER" ] || { echo "no ROCmFPX llama-server at $SERVER"; exit 1; }
[ -f "$GGUF" ]   || { echo "no GGUF at $GGUF"; exit 1; }

exec "$SERVER" -m "$GGUF" --alias qwen38-27b \
  -c "$CTX" -fa on -ctk q8_0 -ctv q8_0 -ngl 99 -np 1 --no-mmap \
  --jinja --reasoning-format deepseek \
  --port 8080
