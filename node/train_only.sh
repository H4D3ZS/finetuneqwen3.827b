#!/usr/bin/env bash
# MI300X, TRAIN-ONLY. Generation is now done OFF-node (ModelScope API teachers), so the
# droplet is rented ONLY for the LoRA SFT distill + merge + eval. This is the cheap, fast
# part (~$5-25 depending on corpus size). Upload the teacher_* dirs first.
#
#   # on the workstation, after seed+bulk finish generating:
#   tar czf teacher_data.tgz -C node teacher_seed_max teacher_bulk_235b
#   scp teacher_data.tgz prompts_bulk.jsonl "$NODE":~/finetuneqwen3.827b/node/
#   scp -r ~/Desktop/finetuneqwen3.827b "$NODE":~/finetuneqwen3.827b   # code
#   # on the node:
#   cd ~/finetuneqwen3.827b/node && tar xzf teacher_data.tgz && bash train_only.sh
set -euo pipefail
cd "$(dirname "$0")"

STUDENT="${STUDENT:-Qwen/Qwen3.6-35B-A3B}"   # the trainable base (escha W2 / GGUFs are NOT trainable)
DATA="${DATA:-.}"                             # parent dir; load_pairs gathers teacher_* recursively
OUT="${OUT:-/scratch/distill/student-distilled}"
EPOCHS="${EPOCHS:-2}"

echo "== env (do NOT let any pip install clobber the ROCm torch)"
python3 -c "import torch;assert torch.cuda.is_available();print('ROCm torch OK',torch.__version__)"

echo "== 3. distill (LoRA SFT on the merged Qwen3.8-Max seed + Qwen3-235B bulk)"
python3 03_unsloth_sft.py --student "$STUDENT" --data "$DATA" --out "$OUT" \
  --epochs "$EPOCHS" --seq_len 4096 --resume

echo "== merge LoRA -> full weights"
python3 merge_lora.py --adapter "$OUT" --base "$STUDENT" --out "${OUT}-merged"

echo "== eval vs base (does it beat escha on coding + comply on security?)"
python3 eval.py --distilled "${OUT}-merged" --base "$STUDENT" || true

echo
echo "== DONE on the node. Pull home + STOP THE DROPLET:"
echo "   scp -r \"\$NODE\":${OUT}-merged ~/Desktop/student-final"
echo "== then LOCAL (free): abliterate + quantize to your ROCmFPX 2-bit ->"
echo "   python local/abliterate.py --model ~/Desktop/student-final --out ~/Desktop/student-ablit"
echo "   ROCMFPX_DIR=~/Desktop/ROCmFPX ./local/convert_quant.sh ~/Desktop/student-ablit"
echo "   -> your own escha-class fast coder, frontier-seeded. Serve like serve_distilled.sh."
