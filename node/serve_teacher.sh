#!/usr/bin/env bash
# Download the abliterated Qwen3.8-27B GGUF (Blackfrost-AI) and serve it via ROCmFPX
# llama-server (HIP backend) so 02_teacher_gguf.py can generate distillation targets.
# The teacher is ABLITERATED, so its outputs are uncensored -> the student inherits that.
#
#   bash build_rocmfpx.sh        # once, builds llama-server with HIP (gfx942)
#   bash serve_teacher.sh &      # downloads Q8_0 + serves on :8080 (background)
#   # wait for "server is listening", then run 02_teacher_gguf.py
set -euo pipefail
cd "$(dirname "$0")"

REPO="${TEACHER_REPO:-Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF}"
FILE="${TEACHER_FILE:-Qwen3.8-27B-ABLITERATED-Q8_0.gguf}"   # near-lossless teacher
DIR="${SCRATCH:-/scratch/distill}/teacher"
SERVER="${ROCMFPX_DIR:-$PWD/ROCmFPX}/build/bin/llama-server"
PORT="${PORT:-8080}"
NP="${NP:-8}"            # parallel slots = match 02_teacher_gguf.py --concurrency

mkdir -p "$DIR"
if [ ! -f "$DIR/$FILE" ]; then
  echo "== downloading $REPO / $FILE (~29GB) to scratch"
  URL="https://huggingface.co/$REPO/resolve/main/$FILE"
  curl -L -C - --retry 5 -o "$DIR/$FILE" \
    ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} "$URL"
fi

[ -x "$SERVER" ] || { echo "llama-server not built. Run: bash build_rocmfpx.sh"; exit 1; }
echo "== serving abliterated teacher on :$PORT (np=$NP)"
# --jinja so Qwen3.8's thinking template applies; -ngl 99 = all layers on the MI300X.
exec "$SERVER" -m "$DIR/$FILE" --alias teacher \
  -c 8192 -ngl 99 -fa on -np "$NP" --jinja --host 127.0.0.1 --port "$PORT"
