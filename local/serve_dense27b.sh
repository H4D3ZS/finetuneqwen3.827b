#!/usr/bin/env bash
# LOCAL (RX 9060 XT, 16GB). Serve the DENSE Qwen3.8-27B on :8080, tuned to fit 16GB.
#
# QUANT CHOICE (all measured on this card, Vulkan backend):
#   Q2_0_ROCMFPX  8.6GB  28 t/s  INCOHERENT (2-bit collapses a dense model) -> deleted
#   Q3_K_XL      13.4GB   9.3 t/s coherent but flat quant, no vision
#   Ridge 3.7bpw 11.7GB   9.0 t/s coherent, GDN-aware (DeltaNet state @Q8_0), +vision  <- DEFAULT
# Ridge is the best coherent dense-27B here. MTP self-spec gave NO speedup on Vulkan (needs
# ROCm/HIP kernels this RDNA4 card lacks), so it's left OFF. ~9 t/s is the bandwidth wall for
# a coherent dense 27B on 16GB; only the ~3B-active class goes faster (see MOE_UPCYCLE_PLAN).
# repeat-penalty is REQUIRED or the tail degenerates into `pass pass pass`.
#
#   bash local/serve_dense27b.sh                 # serve Ridge
#   GGUF=.../Qwen3.8-27B-UD-Q3_K_XL.gguf bash local/serve_dense27b.sh   # override
set -euo pipefail

ROCMFPX="${ROCMFPX_DIR:-$HOME/Desktop/ROCmFPX}"
SERVER="$ROCMFPX/build/bin/Release/llama-server.exe"
GGUF="${GGUF:-$ROCMFPX/ridge/Qwen3.8-27B-Ridge-3.7bpw.gguf}"
CTX="${CTX:-16384}"   # 11.7GB weights leave ~4GB; hybrid attn (16 full layers) keeps KV cheap

[ -x "$SERVER" ] || { echo "no ROCmFPX llama-server at $SERVER"; exit 1; }
[ -f "$GGUF" ]   || { echo "no GGUF at $GGUF"; exit 1; }

exec "$SERVER" -m "$GGUF" --alias qwen38-27b \
  -c "$CTX" -fa on -ctk q8_0 -ctv q8_0 -ngl 99 -np 1 --no-mmap \
  --jinja --reasoning-budget 0 --repeat-penalty 1.1 \
  --port 8080
