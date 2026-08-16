#!/usr/bin/env bash
# ON THE MI300X (AMD Developer Cloud, Unsloth Studio ROCm image). Distill the ABLITERATED
# Qwen3.8-27B into the fast Qwen3.6-35B-A3B MoE student. The teacher is a GGUF served by
# llama.cpp (ROCmFPX/HIP), so:
#   - no vLLM, no CUDA-wheel-clobbers-ROCm-torch failure (teacher runs in a separate process),
#   - the student inherits the abliteration through distillation (no separate abliterate step).
#
#   export HF_TOKEN=hf_xxx
#   bash run_node.sh --smoke     # ~$5 end-to-end shakedown (ALWAYS run first)
#   bash run_node.sh --poc       # ~$25
#   bash run_node.sh --full      # ~$40
#
# TEACHER_MODE=hf uses the base (non-abliterated) 3.8 via batched HF instead - then you must
# abliterate locally afterward. Default is the abliterated GGUF teacher (simpler + baked in).
set -euo pipefail
cd "$(dirname "$0")"

N=8000; EPOCHS=1; LORA="--lora"; SMOKE=0; CONC="${CONC:-8}"
TEACHER_MODE="${TEACHER_MODE:-gguf}"     # gguf (abliterated) | hf (base + local abliterate)
STUDENT="Qwen/Qwen3.6-35B-A3B"; TEACHER_HF="Qwen/Qwen3.8-27B"
while [ $# -gt 0 ]; do case "$1" in
  --smoke) SMOKE=1; N=50; EPOCHS=1; CONC=4; shift ;;
  --poc)  N=8000;  EPOCHS=1; shift ;;
  --full) N=30000; EPOCHS=2; shift ;;
  --n) N="$2"; shift 2 ;;
  --epochs) EPOCHS="$2"; shift 2 ;;
  --no-lora) LORA=""; shift ;;
  --student) STUDENT="$2"; shift 2 ;;
  *) echo "unknown: $1"; exit 2 ;;
esac; done

export HF_HOME="${HF_HOME:-/scratch/hf}"
export SCRATCH="${SCRATCH:-/scratch/distill}"
export ATTN_IMPL="${ATTN_IMPL:-sdpa}"
mkdir -p "$HF_HOME" "$SCRATCH"

echo "== 0. deps (torch is in the ROCm image; NEVER clobber it - no vLLM here)"
torch_ok() { python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; }
torch_ok || { echo "  torch/ROCm not usable - activate the image's env (conda) first."; exit 1; }
TORCH_VER="$(python3 -c 'import torch;print(torch.__version__)')"
echo "  torch pinned at $TORCH_VER"
pip install -q "transformers>=4.44" accelerate datasets peft "trl>=0.11" bitsandbytes "torch==$TORCH_VER" 2>/dev/null \
  || pip install -q "transformers>=4.44" accelerate datasets peft "trl>=0.11" bitsandbytes || true
python3 -c "import unsloth" 2>/dev/null || echo "  (unsloth not importable -> 03 uses transformers+peft fallback)"
torch_ok || { echo "  !! a dep install clobbered ROCm torch. Reinstall $TORCH_VER from AMD's ROCm index and rerun."; exit 1; }
echo "  torch still ROCm-good ✓"

echo "== 1. corpus ($N prompts; add your transcripts at \$SCRATCH/my_transcripts/)"
python3 01_build_corpus.py --out "$SCRATCH/prompts.jsonl" --n "$N" \
  --sources swebench,toolcalls,local_transcripts,seed \
  --local_glob "$SCRATCH/my_transcripts/**/*.jsonl"

if [ "$TEACHER_MODE" = "gguf" ]; then
  echo "== 2a. build ROCmFPX (HIP) if needed, and serve the ABLITERATED teacher GGUF"
  [ -x "ROCmFPX/build/bin/llama-server" ] || bash build_rocmfpx.sh
  # serve_teacher.sh does NOT download and defaults LCPP=/root/llama.cpp. Point it at the
  # fork we just built and provision the abliterated Q8_0 (Blackfrost-AI) here if missing.
  export LCPP="${LCPP:-$PWD/ROCmFPX}"
  export GGUF="${GGUF:-/scratch/teacher/Qwen3.8-27B-ABLITERATED-Q8_0.gguf}"
  if [ ! -f "$GGUF" ]; then
    echo "  downloading abliterated teacher (Blackfrost-AI, Q8_0) -> $(dirname "$GGUF")"
    mkdir -p "$(dirname "$GGUF")"
    # 'hf' is the current huggingface_hub CLI (Unsloth image); swap to 'huggingface-cli' if absent.
    hf download Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF --include "*Q8_0*.gguf" \
      --local-dir "$(dirname "$GGUF")" || \
    huggingface-cli download Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF --include "*Q8_0*.gguf" \
      --local-dir "$(dirname "$GGUF")"
    found="$(ls "$(dirname "$GGUF")"/*Q8_0*.gguf 2>/dev/null | head -1)"
    [ -n "$found" ] || { echo "  teacher download failed - check repo/filename"; exit 1; }
    [ "$found" = "$GGUF" ] || ln -sf "$found" "$GGUF"
  fi
  # serve in the background; wait for health
  ( bash serve_teacher.sh >"$SCRATCH/teacher-serve.log" 2>&1 & echo $! >"$SCRATCH/teacher.pid" )
  echo "  waiting for teacher /health ..."
  # serve_teacher.sh serves on :8081 (PORT default); poll and call THAT port. The
  # 02_teacher_gguf.py arg is --url and wants the FULL chat/completions endpoint.
  for _ in $(seq 1 300); do
    curl -sf -m 3 http://127.0.0.1:8081/health >/dev/null 2>&1 && break; sleep 5; done
  curl -sf -m 3 http://127.0.0.1:8081/health >/dev/null 2>&1 || { echo "  teacher didn't come up - see $SCRATCH/teacher-serve.log"; exit 1; }

  echo "== 2b. teacher generates targets via API (abliterated, thinking on)"
  python3 02_teacher_gguf.py --url http://127.0.0.1:8081/v1/chat/completions \
    --prompts "$SCRATCH/prompts.jsonl" --out "$SCRATCH/teacher_data/" --n "$N" \
    --concurrency "$CONC" --think

  # free the teacher's VRAM before training the student
  kill "$(cat "$SCRATCH/teacher.pid" 2>/dev/null)" 2>/dev/null || true
  pkill -f llama-server 2>/dev/null || true; sleep 3
else
  echo "== 2. base 3.8 teacher via batched HF (you will abliterate LOCALLY afterward)"
  python3 02_teacher_generate.py --teacher "$TEACHER_HF" --prompts "$SCRATCH/prompts.jsonl" \
    --out "$SCRATCH/teacher_data/" --n "$N" --think --batch 16
fi

echo "== 3. Unsloth SFT-distill into fast A3B ($EPOCHS epoch)"
ATTN_IMPL="$ATTN_IMPL" python3 03_unsloth_sft.py --student "$STUDENT" \
  --data "$SCRATCH/teacher_data/" --out "$SCRATCH/student-distilled/" --epochs "$EPOCHS" --resume

echo "== 4. merge LoRA -> full bf16 model"
if [ -n "$LORA" ]; then
  python3 merge_lora.py --adapter "$SCRATCH/student-distilled/" --base "$STUDENT" --out "$SCRATCH/student-merged/"
  MERGED="$SCRATCH/student-merged"
else MERGED="$SCRATCH/student-distilled"; fi

echo "== 5. GATE: eval distilled vs base"
python3 eval.py --distilled "$SCRATCH/student-distilled/" --base "$STUDENT" $LORA || true

echo
if [ "$SMOKE" = "1" ]; then
  echo "== SMOKE PASSED end-to-end on $N prompts. Review eval, then --poc."; exit 0
fi
echo "== DONE. Merged bf16 at: $MERGED"
echo "If TEACHER_MODE=gguf (abliterated teacher): the student is ALREADY abliteration-transferred."
echo "Pull home + quantize (skip local abliterate unless TEACHER_MODE=hf):"
echo "    scp -r <node>:$MERGED ~/Desktop/finetuneqwen3.827b/student-merged"
echo "    cd ~/Desktop/finetuneqwen3.827b/local && ./convert_quant.sh ../student-merged"
echo "STOP THE DROPLET once scp finishes."
