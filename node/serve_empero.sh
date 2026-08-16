#!/usr/bin/env bash
# Serve the Empero Qwen3.8-9B (Apache-2.0, full-parameter distilled from the Qwen3.8 2.4T-A95B
# frontier teacher) as an OpenAI-compatible API on :8082 - the GENERAL-KNOWLEDGE teacher in the
# two-teacher blend (abliterated 27B = offensive/agentic, Empero = general/math/code/reasoning).
#
# WHY A SEPARATE BUILD (not build_rocmfpx.sh's fork): Empero is a Qwen3.5 "Gated DeltaNet"
# (linear-attention hybrid) architecture. The ROCmFPX fork is pinned to an OLD commit and will
# FAIL to load it ("unknown architecture"). Stock llama.cpp master has DeltaNet support, so we
# build that here - ONLY for teacher generation on the MI300X, never for the local RX 9060 XT
# (hybrid SSM/DeltaNet archs crawl at 2-5 tok/s on that ROCm build).
#
# Empero is NOT abliterated (it is censored). It MUST only ever be fed 'general' prompts:
#     02_teacher_gguf.py --url http://127.0.0.1:8082/v1/chat/completions --route general
# Feeding it offensive-security prompts would produce refusals that re-poison the compliance we
# distil from the abliterated teacher. 01_build_corpus.py tags every prompt with route= for this.
#
#   bash serve_empero.sh          # foreground
#   bash serve_empero.sh &        # background, then 02_teacher_gguf.py --route general
set -euo pipefail

REPO="${LLAMACPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"
LCPP="${EMPERO_LCPP:-$PWD/llama.cpp-stock}"
PORT="${EMPERO_PORT:-8082}"
CTX="${CTX:-8192}"
PARALLEL="${PARALLEL:-16}"        # concurrent slots - what makes generation cheap
GFX="${GFX:-gfx942}"             # MI300X
GGUF="${EMPERO_GGUF:-/scratch/empero/Qwen3.8-9B-Q8_0.gguf}"
HF_REPO="${EMPERO_HF:-empero-ai/Qwen3.8-9B-GGUF}"

# 1. build stock llama.cpp (HIP) if absent - REQUIRED for Qwen3.5/Gated-DeltaNet support
if [ ! -x "$LCPP/build/bin/llama-server" ]; then
  echo "== building stock llama.cpp (HIP/$GFX) for DeltaNet support"
  [ -d "$LCPP/.git" ] || git clone --depth 1 "$REPO" "$LCPP"
  ( cd "$LCPP" && cmake -S . -B build -DGGML_HIP=ON \
      -DGPU_TARGETS="$GFX" -DCMAKE_HIP_ARCHITECTURES="$GFX" \
      -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_WEBUI=OFF \
    && cmake --build build --config Release -j"$(nproc)" --target llama-server )
fi
[ -x "$LCPP/build/bin/llama-server" ] || { echo "stock llama.cpp build failed at $LCPP"; exit 1; }

# 2. provision the Empero GGUF (Q8_0, ~9.8GB) if missing
if [ ! -f "$GGUF" ]; then
  echo "== downloading Empero Q8_0 -> $(dirname "$GGUF")"
  mkdir -p "$(dirname "$GGUF")"
  hf download "$HF_REPO" --include "*Q8_0*.gguf" --local-dir "$(dirname "$GGUF")" || \
  huggingface-cli download "$HF_REPO" --include "*Q8_0*.gguf" --local-dir "$(dirname "$GGUF")"
  found="$(ls "$(dirname "$GGUF")"/*Q8_0*.gguf 2>/dev/null | head -1)"
  [ -n "$found" ] || { echo "Empero download failed - check repo/filename"; exit 1; }
  [ "$found" = "$GGUF" ] || ln -sf "$found" "$GGUF"
fi

export ROCM_PATH="${ROCM_PATH:-/opt/rocm-7.2.4}"
echo "serving Empero $(basename "$GGUF") on :$PORT  (ctx=$CTX, parallel=$PARALLEL)"

# total KV = CTX * PARALLEL, sized for 192GB. --jinja for the model's chat template.
exec "$LCPP/build/bin/llama-server" \
  -m "$GGUF" \
  --host 127.0.0.1 --port "$PORT" \
  -ngl 99 \
  -c $((CTX * PARALLEL)) \
  -np "$PARALLEL" \
  -fa on \
  --no-mmap \
  --jinja \
  --alias empero-teacher
