#!/usr/bin/env bash
# Resume the smoke test from step 3, reusing the teacher_data already generated
# before the host reboot. Steps 3-5 are unchanged by the abliterated-teacher
# rearchitecture (only 01/02 teacher-sourcing changed), so this validates the
# remaining unknowns: LoRA targets, one train step, merge, eval table.
set -euo pipefail
cd "$(dirname "$0")"

export HF_HOME="${HF_HOME:-/scratch/hf}"
export SCRATCH="${SCRATCH:-/scratch/distill}"
export ATTN_IMPL="${ATTN_IMPL:-sdpa}"
STUDENT="${STUDENT:-Qwen/Qwen3.6-35B-A3B}"
EPOCHS="${EPOCHS:-1}"

python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "torch/ROCm not usable - activate the unsloth venv first"; exit 1; }
echo "== resuming with $(python3 -c 'import torch;print(torch.__version__)')"
echo "== teacher pairs: $(cat "$SCRATCH"/teacher_data/*.jsonl | wc -l)"

echo "== 3. Unsloth SFT-distill into fast A3B ($EPOCHS epoch)"
ATTN_IMPL="$ATTN_IMPL" python3 03_unsloth_sft.py --student "$STUDENT" \
  --data "$SCRATCH/teacher_data/" --out "$SCRATCH/student-distilled/" --epochs "$EPOCHS"

echo "== 4. merge LoRA -> full bf16"
python3 merge_lora.py --adapter "$SCRATCH/student-distilled/" --base "$STUDENT" \
  --out "$SCRATCH/student-merged/"

echo "== 5. GATE: eval distilled vs base"
python3 eval.py --distilled "$SCRATCH/student-distilled/" --base "$STUDENT" --lora || true

echo "== RESUME COMPLETE"
