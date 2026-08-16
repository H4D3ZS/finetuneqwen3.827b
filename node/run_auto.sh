#!/usr/bin/env bash
# FULLY AUTONOMOUS node-side PoC pipeline - survives SSH drop AND Claude session exit.
# Caps teacher-gen at $TARGET completions, then auto-runs train -> merge -> eval. No babysitting.
#   tmux new -s auto 'bash run_auto.sh'
set -uo pipefail
cd "$(dirname "$0")"
source /root/.unsloth/studio/unsloth_studio/bin/activate

export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN}"
export HF_HOME="${HF_HOME:-/scratch/hf}"
export SCRATCH="${SCRATCH:-/scratch/distill}"
export ATTN_IMPL="${ATTN_IMPL:-sdpa}"
export ROCM_PATH=/opt/rocm-7.2.4
TARGET="${TARGET:-850}"        # ~800 usable after a few failures - the budget cap you chose
STUDENT="${STUDENT:-Qwen/Qwen3.6-35B-A3B}"
TEACHER_URL="http://127.0.0.1:8081/v1/chat/completions"
LOG=/root/finetune/node/auto_run.log
say(){ echo "[$(date +%H:%M:%S)] $*"; }

{
say "=== AUTONOMOUS PoC start (TARGET=$TARGET completions) ==="

# 0. ensure teacher server is up (reuse if warm, else start)
if ! curl -s http://127.0.0.1:8081/health 2>/dev/null | grep -q ok; then
  say "teacher not up - starting serve_teacher.sh"
  GGUF=/scratch/teacher/Qwen3.8-27B-ABLITERATED-Q8_0.gguf PORT=8081 CTX=8192 PARALLEL=32 \
    bash serve_teacher.sh > "$SCRATCH/teacher_serve.log" 2>&1 &
  for i in $(seq 1 120); do curl -s http://127.0.0.1:8081/health 2>/dev/null | grep -q ok && break; sleep 1; done
fi
say "teacher up"

# 1. generate up to TARGET (incremental + resumable; naturally stops at --n TARGET prompts)
say "=== teacher generation (cap $TARGET) ==="
python3 02_teacher_gguf.py --url "$TEACHER_URL" --prompts "$SCRATCH/prompts.jsonl" \
  --out "$SCRATCH/teacher_data/" --n "$TARGET" --think --max_new 4096 --concurrency 32
NGOT=$(wc -l < "$SCRATCH/teacher_data/stream.jsonl" 2>/dev/null || echo 0)
say "generation done: $NGOT completions saved"
[ "$NGOT" -lt 50 ] && { say "FATAL: too few completions ($NGOT) - aborting"; exit 1; }

# 2. free VRAM: stop the teacher before training
say "=== stopping teacher to free VRAM ==="
pkill -f "llama-server.*ABLITERATED" 2>/dev/null; sleep 5

# 3. distill (batch 8, completion-only loss, per-step logging)
say "=== SFT distill (1 epoch) ==="
python3 03_unsloth_sft.py --student "$STUDENT" --data "$SCRATCH/teacher_data/" \
  --out "$SCRATCH/student-distilled/" --epochs 1 --batch 8 --grad_accum 2
say "training done"

# 4. merge -> bf16
say "=== merge LoRA -> bf16 ==="
python3 merge_lora.py --adapter "$SCRATCH/student-distilled/" --base "$STUDENT" \
  --out "$SCRATCH/student-merged/"
say "merge done"

# 5. eval gate
say "=== eval distilled vs base ==="
python3 eval.py --distilled "$SCRATCH/student-distilled/" --base "$STUDENT" --lora
say "=== AUTONOMOUS PoC COMPLETE. merged bf16 -> $SCRATCH/student-merged ==="
say "next (when you're back): on-node ROCmFPX quant + bench_mtp, or scp home for local Escha quant."
} 2>&1 | tee "$LOG"
