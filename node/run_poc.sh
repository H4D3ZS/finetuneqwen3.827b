#!/usr/bin/env bash
# PoC run with the ABLITERATED GGUF teacher path (b0f75f7 architecture, built on-node).
#   corpus -> serve abliterated 27B -> concurrent API gen -> free VRAM -> Unsloth SFT -> merge -> eval
# Everything runs under tmux; teacher is a separate llama.cpp process (no torch to clobber).
set -euo pipefail
cd "$(dirname "$0")"

N="${N:-8000}"; EPOCHS="${EPOCHS:-1}"
TEACHER_GGUF="${TEACHER_GGUF:-/scratch/teacher/Qwen3.8-27B-ABLITERATED-Q8_0.gguf}"
STUDENT="${STUDENT:-Qwen/Qwen3.6-35B-A3B}"
PARALLEL="${PARALLEL:-32}"     # MI300X 192GB: teacher is 27GB, room for many slots
MAXNEW="${MAXNEW:-4096}"        # headroom so reasoning doesn't truncate mid-<think>
BATCH="${BATCH:-8}"            # 192GB fits large batch; 8x fewer forward passes than batch=1
GRAD_ACCUM="${GRAD_ACCUM:-2}"  # batch 8 x accum 2 = effective 16 (same as old 1x16)
export HF_HOME="${HF_HOME:-/scratch/hf}"
export SCRATCH="${SCRATCH:-/scratch/distill}"
export ATTN_IMPL="${ATTN_IMPL:-sdpa}"
export ROCM_PATH=/opt/rocm-7.2.4
mkdir -p "$SCRATCH"

python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "torch/ROCm bad"; exit 1; }
echo "== torch $(python3 -c 'import torch;print(torch.__version__)') | N=$N epochs=$EPOCHS"

echo "== 1. corpus ($N prompts)"
python3 01_build_corpus.py --out "$SCRATCH/prompts.jsonl" --n "$N" \
  --sources swebench,toolcalls,local_transcripts,seed \
  --local_glob "$SCRATCH/my_transcripts/**/*.jsonl"
echo "   built $(wc -l < "$SCRATCH/prompts.jsonl") prompts"

echo "== 2a. serve abliterated teacher"
pkill -f "llama-server.*ABLITERATED" 2>/dev/null || true; sleep 2
GGUF="$TEACHER_GGUF" PORT=8081 CTX=8192 PARALLEL="$PARALLEL" bash serve_teacher.sh \
  > "$SCRATCH/teacher_serve.log" 2>&1 &
SERVE_PID=$!
echo "   teacher pid=$SERVE_PID; waiting for health"
for i in $(seq 1 120); do
  curl -s http://127.0.0.1:8081/health 2>/dev/null | grep -q '"ok"' && { echo "   teacher up after ${i}s"; break; }
  sleep 1
done

echo "== 2b. concurrent API generation (think ON, max_new=$MAXNEW)"
python3 02_teacher_gguf.py --prompts "$SCRATCH/prompts.jsonl" \
  --out "$SCRATCH/teacher_data/" --n "$N" --think \
  --max_new "$MAXNEW" --concurrency "$PARALLEL"

echo "== 2c. stop teacher, free VRAM before training"
kill "$SERVE_PID" 2>/dev/null || true
pkill -f "llama-server.*ABLITERATED" 2>/dev/null || true
sleep 5

echo "== 3. Unsloth SFT-distill ($EPOCHS epoch, completion-only loss)"
ATTN_IMPL="$ATTN_IMPL" python3 03_unsloth_sft.py --student "$STUDENT" \
  --data "$SCRATCH/teacher_data/" --out "$SCRATCH/student-distilled/" --epochs "$EPOCHS" \
  --batch "$BATCH" --grad_accum "$GRAD_ACCUM" --resume

echo "== 4. merge LoRA -> bf16"
python3 merge_lora.py --adapter "$SCRATCH/student-distilled/" --base "$STUDENT" \
  --out "$SCRATCH/student-merged/"

echo "== 5. GATE: eval distilled vs base"
python3 eval.py --distilled "$SCRATCH/student-distilled/" --base "$STUDENT" --lora || true

echo "== POC COMPLETE. merged bf16 -> $SCRATCH/student-merged (scp home, then local Escha quant)"
