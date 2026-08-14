#!/usr/bin/env bash
# ON THE MI300X (AMD Developer Cloud, ROCm/gfx942). Runs the CLEAN half of the pipeline:
# corpus -> teacher -> distill -> merge -> eval, and leaves a bf16 merged model to download.
# Abliteration is deliberately NOT here - it runs on the local machine (off AMD's cloud).
#
#   export HF_TOKEN=hf_xxx
#   bash run_node.sh --smoke     # ~$5 end-to-end shakedown (ALWAYS run first)
#   bash run_node.sh --poc       # ~$25
#   bash run_node.sh --full      # ~$40
set -euo pipefail
cd "$(dirname "$0")"

N=8000; EPOCHS=1; LORA="--lora"; SMOKE=0
TEACHER="Qwen/Qwen3.8-27B"; STUDENT="Qwen/Qwen3.6-35B-A3B"
while [ $# -gt 0 ]; do case "$1" in
  --smoke) SMOKE=1; N=50; EPOCHS=1; shift ;;
  --poc)  N=8000;  EPOCHS=1; shift ;;
  --full) N=30000; EPOCHS=2; shift ;;
  --n) N="$2"; shift 2 ;;
  --epochs) EPOCHS="$2"; shift 2 ;;
  --no-lora) LORA=""; shift ;;
  --teacher) TEACHER="$2"; shift 2 ;;
  --student) STUDENT="$2"; shift 2 ;;
  *) echo "unknown: $1"; exit 2 ;;
esac; done

export HF_HOME="${HF_HOME:-/scratch/hf}"
export SCRATCH="${SCRATCH:-/scratch/distill}"
export ATTN_IMPL="${ATTN_IMPL:-sdpa}"      # safe on ROCm; flash-attn is finicky on gfx942
mkdir -p "$HF_HOME" "$SCRATCH"

echo "== 0. deps (ROCm torch + vLLM ROCm + training libs)"
pip install -q -U torch --index-url https://download.pytorch.org/whl/rocm6.2 || true
pip install -q -U "transformers>=4.44" accelerate datasets peft "bitsandbytes>=0.44" || true
# vLLM is COST-CRITICAL for step 2 (67x cheaper than HF). ROCm build:
pip install -q -U vllm || echo "  WARNING: vLLM install failed - step 2 will refuse >200 prompts. Fix before --poc."

python - <<'PY'
import torch
print("torch", torch.__version__, "| hip", getattr(torch.version,'hip',None),
      "| avail", torch.cuda.is_available(), "| n", torch.cuda.device_count())
if torch.cuda.is_available():
    p=torch.cuda.get_device_properties(0); print(f"  {p.name} {p.total_memory/1e9:.0f}GB")
PY

echo "== 1. corpus ($N prompts; add your transcripts at \$SCRATCH/my_transcripts/)"
python 01_build_corpus.py --out "$SCRATCH/prompts.jsonl" --n "$N" \
  --sources swebench,toolcalls,local_transcripts,seed \
  --local_glob "$SCRATCH/my_transcripts/**/*.jsonl"

echo "== 2. teacher generates targets (vLLM batched, thinking ON)"
python 02_teacher_generate.py --teacher "$TEACHER" --prompts "$SCRATCH/prompts.jsonl" \
  --out "$SCRATCH/teacher_data/" --n "$N" --think

echo "== 3. distill into fast A3B ($EPOCHS epoch, $LORA)"
ATTN_IMPL="$ATTN_IMPL" python 03_distill_train.py \
  --teacher "$TEACHER" --student "$STUDENT" \
  --data "$SCRATCH/teacher_data/" --out "$SCRATCH/student-distilled/" \
  --epochs "$EPOCHS" $LORA --resume

echo "== 4. merge LoRA -> full bf16 model (so the download is a real model, not an adapter)"
if [ -n "$LORA" ]; then
  python merge_lora.py --adapter "$SCRATCH/student-distilled/" --base "$STUDENT" \
    --out "$SCRATCH/student-merged/"
  MERGED="$SCRATCH/student-merged"
else
  MERGED="$SCRATCH/student-distilled"
fi

echo "== 5. GATE: eval distilled vs base"
python eval.py --distilled "$SCRATCH/student-distilled/" --base "$STUDENT" $LORA || true

echo
if [ "$SMOKE" = "1" ]; then
  echo "== SMOKE PASSED. R1-R9 exercised end-to-end on $N prompts. Review eval, then --poc."
  exit 0
fi
echo "== DONE on node. Merged bf16 model at: $MERGED  (~54-70GB)"
echo "Pull it home, then abliterate + quantize LOCALLY:"
echo "    scp -r <node>:$MERGED ~/Desktop/finetuneqwen3.827b/student-merged"
echo "    cd ~/Desktop/finetuneqwen3.827b/local && ./abliterate.sh ../student-merged"
echo "STOP THE DROPLET once scp finishes."
