#!/usr/bin/env bash
# One-shot: re-distill onto Qwen3.8-35B-A3B the moment its weights land on HF.
# REUSES the teacher data we already paid to generate (stream.jsonl) - NO regeneration.
# Same architecture as 3.6-35B-A3B (256 experts / ~3B active / MTP), so the pipeline is identical;
# only the base changes. Run AFTER the 3.6 PoC has finished and given us a real tok/s number.
#   tmux new -s swap 'bash swap_to_38.sh'
set -uo pipefail
cd "$(dirname "$0")"
source /root/.unsloth/studio/unsloth_studio/bin/activate
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN}"
export HF_HOME="${HF_HOME:-/scratch/hf}"
export SCRATCH="${SCRATCH:-/scratch/distill}"
export ROCM_PATH=/opt/rocm-7.2.4
STUDENT="Qwen/Qwen3.8-35B-A3B"
LOG=/root/finetune/node/swap38_run.log
say(){ echo "[$(date +%H:%M:%S)] $*"; }

{
say "=== re-distill onto $STUDENT (reusing existing teacher data) ==="
NGOT=$(wc -l < "$SCRATCH/teacher_data/stream.jsonl" 2>/dev/null || echo 0)
say "reusing $NGOT teacher completions (no regeneration)"
[ "$NGOT" -lt 50 ] && { say "FATAL: teacher_data missing/too small ($NGOT)"; exit 1; }

# confirm the weights actually exist before spending training time
python3 - "$STUDENT" <<'PY' || { echo "3.8-35B-A3B not on HF yet - aborting, try again when weights drop"; exit 2; }
import sys
from huggingface_hub import HfApi
m = sys.argv[1]
try:
    HfApi().model_info(m)
    print(f"OK: {m} exists on HF")
except Exception as e:
    print(f"NOT FOUND: {m}: {e}"); sys.exit(1)
PY

say "=== SFT distill onto 3.8 ==="
python3 03_unsloth_sft.py --student "$STUDENT" --data "$SCRATCH/teacher_data/" \
  --out "$SCRATCH/student38-distilled/" --epochs 1 --batch 8 --grad_accum 2
say "=== merge -> bf16 ==="
python3 merge_lora.py --adapter "$SCRATCH/student38-distilled/" --base "$STUDENT" \
  --out "$SCRATCH/student38-merged/"
say "=== eval 3.8 distilled vs base ==="
python3 eval.py --distilled "$SCRATCH/student38-distilled/" --base "$STUDENT" --lora
say "=== DONE. merged 3.8-A3B bf16 -> $SCRATCH/student38-merged ==="
say "next: ROCmFPX ROCMFP4 quant that fits 16GB + bench_mtp for the real tok/s number."
} 2>&1 | tee "$LOG"
