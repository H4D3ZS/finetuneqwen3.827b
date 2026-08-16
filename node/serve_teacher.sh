#!/usr/bin/env bash
# Serve the ABLITERATED Qwen3.8-27B teacher via llama.cpp (HIP/gfx942) as an OpenAI-compatible
# API on :8081. The teacher runs in its OWN process - there is no shared torch env, so the
# CUDA-wheel-clobbers-ROCm-torch failure that killed smoke run 1 cannot recur.
#
#   bash serve_teacher.sh          # foreground
#   bash serve_teacher.sh &        # background, then 02_teacher_gguf.py
set -euo pipefail

GGUF="${GGUF:-/scratch/teacher/Qwen3.8-27B-ABLITERATED-Q8_0.gguf}"
PORT="${PORT:-8081}"
CTX="${CTX:-8192}"
PARALLEL="${PARALLEL:-16}"        # concurrent slots - this is what makes generation cheap
LCPP="${LCPP:-/root/llama.cpp}"

[ -f "$GGUF" ] || { echo "teacher GGUF missing: $GGUF"; exit 1; }
[ -x "$LCPP/build/bin/llama-server" ] || { echo "llama-server not built at $LCPP"; exit 1; }

export ROCM_PATH=/opt/rocm-7.2.4
echo "serving $(basename "$GGUF") on :$PORT  (ctx=$CTX, parallel=$PARALLEL)"

# -np PARALLEL gives PARALLEL concurrent slots; total KV = CTX * PARALLEL, sized for 192GB.
exec "$LCPP/build/bin/llama-server" \
  -m "$GGUF" \
  --host 127.0.0.1 --port "$PORT" \
  -ngl 99 \
  -c $((CTX * PARALLEL)) \
  -np "$PARALLEL" \
  -fa on \
  --no-mmap \
  --alias teacher-abliterated
