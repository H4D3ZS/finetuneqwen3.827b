#!/usr/bin/env bash
# OPTIONAL, on the MI300X. Build the ROCmFPX fork with the HIP/ROCm backend (gfx942) so you
# can sanity-quant + inference-test the distilled model ON the node before downloading 70GB.
#
# The MI300X can build the HIP backend that the local RX 9060 XT build could NOT (it's
# Vulkan-only, no ROCm SDK). So this is the one place ROCmFPX gets its native ROCm kernels.
#
# COST NOTE: building llama.cpp+HIP takes ~10-20 min of billed time. SKIP this on the smoke
# and PoC runs to save money - the FINAL quant happens locally anyway (after local
# abliteration). Only build here if you specifically want on-node GGUF validation.
#
#   bash build_rocmfpx.sh
#   # then: ./ROCmFPX/build/bin/llama-quantize student-merged.gguf out.gguf Q2_0_ROCMFPX
set -euo pipefail
cd "$(dirname "$0")"

REPO="${ROCMFPX_REPO:-https://github.com/charlie12345/ROCmFPX.git}"
# PINNED to the exact commit the local RX 9060 XT setup was validated on. This is the real
# time-saver: the node builds a byte-identical ROCmFPX, so behavior matches local and you
# don't debug a version-drift mismatch. Bump only deliberately.
COMMIT="${ROCMFPX_COMMIT:-b2f5829db8beefc22b49481247d180a48b06793a}"
DIR="${ROCMFPX_DIR:-$PWD/ROCmFPX}"
GFX="${GFX:-gfx942}"   # MI300X

echo "== deps"
which cmake >/dev/null || { echo "install cmake"; sudo apt-get update -q && sudo apt-get install -y cmake git; }

echo "== clone $REPO @ ${COMMIT:0:12}"
if [ ! -d "$DIR/.git" ]; then
  git clone "$REPO" "$DIR"
fi
cd "$DIR"
git fetch --all -q || true
git checkout -q "$COMMIT" || { echo "commit $COMMIT not found - bump ROCMFPX_COMMIT"; exit 1; }
echo "  at $(git rev-parse --short HEAD)"

echo "== configure with HIP backend for $GFX"
cmake -S . -B build \
  -DGGML_HIP=ON -DGGML_HIP_FORCE_MMQ=ON \
  -DGPU_TARGETS="$GFX" -DCMAKE_HIP_ARCHITECTURES="$GFX" \
  -DCMAKE_BUILD_TYPE=Release

echo "== build (llama-quantize + llama-server + convert)"
cmake --build build --config Release -j"$(nproc)" --target llama-quantize llama-server

echo
echo "ROCmFPX (HIP) built at $DIR/build/bin/"
echo "on-node quant test:"
echo "  python $DIR/convert_hf_to_gguf.py /scratch/distill/student-merged --outfile /scratch/s.gguf --outtype bf16"
echo "  $DIR/build/bin/llama-quantize /scratch/s.gguf /scratch/s-fp2.gguf Q2_0_ROCMFPX \$(nproc)"
echo "  $DIR/build/bin/llama-server -m /scratch/s-fp2.gguf -ngl 99 --port 8080   # smoke it"
echo
echo "NOTE: this is a VALIDATION quant only. The SHIPPED model is quantized LOCALLY, after"
echo "local abliteration (abliteration needs bf16, and we keep it off the cloud on purpose)."
