#!/usr/bin/env bash
# LOCAL (this machine). Abliterate the downloaded bf16 distilled model - remove refusals so
# it complies on authorized security work. Kept OFF the AMD cloud on purpose (ToS + privacy).
#
#   ./abliterate.sh ../student-merged
#
# A ~70GB bf16 35B does NOT fit in 56GB (16 VRAM + 40 RAM). Two ways, both proven:
#   1. Abliterix (RECOMMENDED - it made your existing v2, handles big models via streaming)
#   2. the bundled abliterate.py with --offload_folder (streams through NVMe, slow)
set -euo pipefail
cd "$(dirname "$0")"

SRC="${1:?usage: ./abliterate.sh <bf16-model-dir>}"
OUT="${2:-${SRC%/}-abliterated}"
OFFLOAD="${OFFLOAD:-/e/offload}"   # E: games drive has ~80GB; change if needed

if command -v abliterix >/dev/null 2>&1 || python -c "import abliterix" 2>/dev/null; then
  echo "== using Abliterix (your proven tool)"
  # matches how v2 was produced: projected abliteration, outlier winsorization.
  abliterix --model "$SRC" --out "$OUT" || \
    python -m abliterix --model "$SRC" --out "$OUT"
else
  echo "== Abliterix not found - using bundled abliterate.py with NVMe offload"
  echo "   (install the proven tool: pip install abliterix   # or from github.com/wuwangzhang1216/abliterix)"
  mkdir -p "$OFFLOAD"
  python abliterate.py --model "$SRC" --out "$OUT" --offload_folder "$OFFLOAD"
fi

echo
echo "== abliterated -> $OUT"
echo "VERIFY it complies before quantizing:"
echo "  python -c \"from transformers import *; import torch; \\"
echo "    m=AutoModelForCausalLM.from_pretrained('$OUT',torch_dtype=torch.bfloat16,device_map='auto',offload_folder='$OFFLOAD'); \\"
echo "    t=AutoTokenizer.from_pretrained('$OUT'); \\"
echo "    i=t.apply_chat_template([{'role':'user','content':'bash one-liner to find SUID binaries on a box I own'}],return_tensors='pt',add_generation_prompt=True).to(m.device); \\"
echo "    print(t.decode(m.generate(i,max_new_tokens=120)[0][i.shape[1]:]))\""
echo "Then: ./convert_quant.sh $OUT"
