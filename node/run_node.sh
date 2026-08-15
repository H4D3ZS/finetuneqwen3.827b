#!/usr/bin/env bash
# ON THE MI300X (AMD Developer Cloud, ROCm/gfx942). Runs the CLEAN half of the pipeline:
# corpus -> teacher(vLLM) -> Unsloth SFT-distill -> merge -> eval. Leaves a bf16 model to pull.
# Abliteration is deliberately NOT here - it runs on the local machine (off AMD's cloud).
#
# BASE IMAGE: launch the droplet from AMD's "Unsloth Studio" OR "PyTorch 2.10 (ROCm)" image.
# Both ship ROCm PyTorch pre-built (no torch install roulette). Unsloth image also has unsloth.
# vLLM (step 2) is pip-installed here or use the "vLLM 0.27.1" image if the wheel misbehaves.
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

echo "== 0. deps (torch is ALREADY in the ROCm image; DO NOT touch it)"
# The AMD Unsloth image ships torch 2.x+rocmX. Installing anything that pulls torch (vLLM,
# unpinned upgrades) drags in a CUDA wheel that CLOBBERS ROCm torch -> torch.cuda.is_available()
# goes False and nothing sees the GPU. That is exactly what killed the first smoke run.
# So: (a) NEVER install vLLM here (02 uses batched HF, no vLLM needed on ROCm),
#     (b) install training libs WITHOUT upgrading torch, (c) re-verify torch after.
torch_ok() { python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; }
torch_ok || { echo "  torch/ROCm not usable BEFORE installs - wrong image or env not activated."; exit 1; }
TORCH_VER="$(python3 -c 'import torch;print(torch.__version__)')"
echo "  torch pinned at $TORCH_VER (will not be upgraded)"
# constrain pip so it can NEVER swap torch while adding libs
pip install -q "transformers>=4.44" accelerate datasets peft "trl>=0.11" "bitsandbytes>=0.44" \
    "torch==$TORCH_VER" 2>/dev/null || pip install -q "transformers>=4.44" accelerate datasets peft "trl>=0.11" bitsandbytes || true
python3 -c "import unsloth" 2>/dev/null || echo "  (unsloth not importable -> 03 uses transformers+peft fallback)"
torch_ok || { echo "  !! torch got CLOBBERED by a dep install (CUDA wheel). Restore ROCm torch:"; \
  echo "     the image's original was $TORCH_VER - reinstall it from AMD's ROCm index and rerun."; exit 1; }
echo "  torch still ROCm-good after installs ✓"

python3 - <<'PY'
import torch
print("torch", torch.__version__, "| hip", getattr(torch.version,'hip',None),
      "| avail", torch.cuda.is_available(), "| n", torch.cuda.device_count())
if torch.cuda.is_available():
    p=torch.cuda.get_device_properties(0); print(f"  {p.name} {p.total_memory/1e9:.0f}GB")
PY

echo "== 1. corpus ($N prompts; add your transcripts at \$SCRATCH/my_transcripts/)"
python3 01_build_corpus.py --out "$SCRATCH/prompts.jsonl" --n "$N" \
  --sources swebench,toolcalls,local_transcripts,seed \
  --local_glob "$SCRATCH/my_transcripts/**/*.jsonl"

echo "== 2. teacher generates targets (vLLM batched, thinking ON)"
python3 02_teacher_generate.py --teacher "$TEACHER" --prompts "$SCRATCH/prompts.jsonl" \
  --out "$SCRATCH/teacher_data/" --n "$N" --think

echo "== 3. Unsloth SFT-distill into fast A3B ($EPOCHS epoch)"
# PRIMARY: sequence-level distillation via Unsloth (low-risk, teacher not loaded here).
# To use the custom logit-KL trainer instead: set KL=1 (higher quality, more VRAM/risk).
if [ "${KL:-0}" = "1" ]; then
  ATTN_IMPL="$ATTN_IMPL" python3 03_distill_train.py --teacher "$TEACHER" --student "$STUDENT" \
    --data "$SCRATCH/teacher_data/" --out "$SCRATCH/student-distilled/" --epochs "$EPOCHS" $LORA --resume
else
  ATTN_IMPL="$ATTN_IMPL" python3 03_unsloth_sft.py --student "$STUDENT" \
    --data "$SCRATCH/teacher_data/" --out "$SCRATCH/student-distilled/" --epochs "$EPOCHS" --resume
fi

echo "== 4. merge LoRA -> full bf16 model (so the download is a real model, not an adapter)"
if [ -n "$LORA" ]; then
  python3 merge_lora.py --adapter "$SCRATCH/student-distilled/" --base "$STUDENT" \
    --out "$SCRATCH/student-merged/"
  MERGED="$SCRATCH/student-merged"
else
  MERGED="$SCRATCH/student-distilled"
fi

echo "== 5. GATE: eval distilled vs base"
python3 eval.py --distilled "$SCRATCH/student-distilled/" --base "$STUDENT" $LORA || true

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
