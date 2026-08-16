#!/usr/bin/env bash
# Inference-side engineering for the TWO speed goals, run on the quantized GGUF (your PC):
#   (1) the 64K context CLIFF  - regression check across the -c 61440 / 65536 boundary
#   (2) 100-200 tok/s          - n-max sweep to find the throughput knee
#
# This is the empirical half of throughput-plan (the artifact). It does NOT retrain anything;
# it measures the model you already have so every later decision is grounded in numbers.
#
#   ./bench_mtp.sh /path/to/Qwen3.8-Coder-A3B.gguf
#
# Requires a llama-server with --spec-type draft-mtp (your ROCmFPX build).
set -euo pipefail

MODEL="${1:?usage: bench_mtp.sh <model.gguf>}"
SERVER="${LLAMA_SERVER:-llama-server}"
PORT="${PORT:-8080}"
PROMPT="${PROMPT:-Write a Python function that merges two sorted lists. Explain briefly.}"
NPREDICT="${NPREDICT:-200}"
REPS="${REPS:-3}"

bench() {  # $1=ctx  $2=n_max
  local ctx="$1" nmax="$2"
  "$SERVER" -m "$MODEL" --port "$PORT" -ngl 99 -fa on -ctk q8_0 -ctv q8_0 \
    -c "$ctx" -np 1 --no-mmap \
    --spec-type draft-mtp --spec-draft-ngl all \
    --spec-draft-n-max "$nmax" --spec-draft-p-min 0.1 \
    > /tmp/srv.log 2>&1 &
  local pid=$!
  for i in $(seq 1 60); do curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok && break; sleep 1; done
  local sum=0
  for r in $(seq 1 "$REPS"); do
    local tps
    tps=$(curl -s "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
      -d "{\"prompt\":\"$PROMPT\",\"n_predict\":$NPREDICT,\"temperature\":0,\"cache_prompt\":false}" \
      | python3 -c "import json,sys;d=json.load(sys.stdin);t=d.get('timings',{});print(f\"{t.get('predicted_per_second',0):.1f}\")")
    sum=$(python3 -c "print($sum+$tps)")
    printf "    rep%d: %s tok/s\n" "$r" "$tps"
  done
  kill "$pid" 2>/dev/null || true; sleep 2
  python3 -c "print(f'  ctx={$ctx} n_max={$nmax}  MEAN {$sum/$REPS:.1f} tok/s')"
}

echo "== GOAL 2: n-max sweep (find the throughput knee) - hold ctx=61440, under the cliff"
for nm in 3 4 6 8 12; do echo "-- n_max=$nm"; bench 61440 "$nm"; done

echo
echo "== GOAL 1: 64K cliff regression - same n_max, cross the boundary"
BEST_NM="${BEST_NM:-8}"
for c in 61440 65536; do echo "-- ctx=$c"; bench "$c" "$BEST_NM"; done
echo
echo "If ctx=65536 is within ~10% of 61440, the cliff is fixed on this build. If it drops ~2.7x,"
echo "the MTP verify-path threshold is still present -> keep serving at -c 61440 (usable ceiling)."
