#!/usr/bin/env bash
# LOCAL (RX 9060 XT). Build the FAST coherent dense 27B: a fork-native Q3_0_ROCMFPX_AGENT.
#
# WHY: on this Vulkan card the dense 27B is stuck between two bad options —
#   Q2_0_ROCMFPX  = fast (28 t/s) but INCOHERENT (2-bit collapses a dense model)
#   Q3_K_XL       = coherent but SLOW (9.3 t/s; generic Vulkan K-quant kernels)
# The fork's NATIVE 3-bit (Q3_0_ROCMFPX_AGENT, type 113) gives coherent 3.5-bit weights on
# the SAME fast kernels that made Q2 hit 28 t/s -> expected ~18-22 t/s AND coherent, with the
# AGENT recipe biased toward tool-call coherence (ideal for Claude Code).
#
# DISK: needs ~40GB more free than you have now (C: was 69GB free / 92% full). The 54GB
# download + a q8_0 intermediate won't fit alongside your media. Free space FIRST, or this
# aborts before downloading anything. Rough peak: safetensors(54) + q8_0(29) = ~83GB.
#
#   bash local/build_dense27b_fast.sh
set -euo pipefail
cd "$(dirname "$0")/.."

ROCMFPX="${ROCMFPX_DIR:-$HOME/Desktop/ROCmFPX}"
SRC="${SRC:-$HOME/models/Qwen3.8-27B}"            # safetensors download target
Q8="$ROCMFPX/Qwen3.8-27B-q8_0.gguf"               # intermediate (deleted at the end)
OUT="$ROCMFPX/Qwen3.8-27B-Q3_0_ROCMFPX_AGENT.gguf"
QUANT="${QUANT:-Q3_0_ROCMFPX_AGENT}"              # or Q3_0_ROCMFPX (no agent bias)

need_gb() {  # abort if free space on C: is below $1 GB
  local free; free=$(df -BG /c | awk 'NR==2{gsub("G","",$4);print $4}')
  [ "$free" -ge "$1" ] || { echo "!! only ${free}GB free on C:, need >=${1}GB. Free space first."; exit 1; }
}

echo "== preflight: disk"
need_gb 90

echo "== 1. download base Qwen3.8-27B (safetensors, ~54GB, NOT gated)"
hf download Qwen/Qwen3.8-27B --local-dir "$SRC"

echo "== 2. convert HF -> q8_0 GGUF (smaller intermediate than bf16; keeps disk in budget)"
python "$ROCMFPX/convert_hf_to_gguf.py" "$SRC" --outfile "$Q8" --outtype q8_0

echo "== 3. reclaim: drop the 54GB safetensors now that the GGUF exists"
rm -rf "$SRC" && echo "  removed $SRC"

echo "== 4. quantize q8_0 -> $QUANT (fork-native fast 3-bit)"
"$ROCMFPX/build/bin/Release/llama-quantize.exe" "$Q8" "$OUT" "$QUANT" "$(nproc)"
rm -f "$Q8" && echo "  removed q8_0 intermediate"

SZ=$(python3 -c "import os;print(f'{os.path.getsize(r\"$OUT\")/1e9:.1f}')")
echo "== ${SZ}GB -> $OUT"
echo
echo "== serve it (coherent + fast dense 27B):"
echo "   GGUF=\"$OUT\" bash local/serve_dense27b.sh"
echo "   # then Claude Code -> qwen38-27b-dense (gateway already routes it)"
echo "== BENCH quality + tok/s before trusting it; if 3-bit still degrades logic, the dense"
echo "   27B simply can't be both fast and smart on 16GB, and the A3B class is the only 100+ path."
