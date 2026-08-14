#!/usr/bin/env bash
# LOCAL (this machine, RX 9060 XT). Final step: convert the abliterated bf16 model to a
# fork-native Q2_0_ROCMFPX GGUF that runs fast on 16GB. Uses your existing local ROCmFPX.
#
#   ./convert_quant.sh ../student-merged-abliterated
set -euo pipefail
cd "$(dirname "$0")"

SRC="${1:?usage: ./convert_quant.sh <abliterated-bf16-dir>}"
ROCMFPX="${ROCMFPX_DIR:-$HOME/Desktop/ROCmFPX}"
NAME="qwen38-distilled-a3b-abliterated"
BF16="$ROCMFPX/${NAME}-bf16.gguf"
OUT="$ROCMFPX/${NAME}-Q2_0_ROCMFPX.gguf"

echo "== convert HF -> bf16 GGUF (this is large, ~70GB; ensure C: has room)"
python "$ROCMFPX/convert_hf_to_gguf.py" "$SRC" --outfile "$BF16" --outtype bf16

echo "== quantize -> Q2_0_ROCMFPX (fork-native fast 2-bit, fits 16GB)"
"$ROCMFPX/build/bin/Release/llama-quantize.exe" "$BF16" "$OUT" Q2_0_ROCMFPX "$(nproc)"

SZ=$(python -c "import os;print(f'{os.path.getsize(r\"$OUT\")/1e9:.1f}')")
echo "== ${SZ}GB -> $OUT"
rm -f "$BF16" && echo "removed bf16 intermediate (~70GB reclaimed)"

echo
echo "== serve with the ROCmFPX llama-server (this IS an A3B MoE, so MTP helps):"
cat <<SERVE
   "\$ROCMFPX/build/bin/Release/llama-server.exe" -m "$OUT" --alias qwen38d \\
     -c 98304 -fa on -ctk q8_0 -ctv q8_0 -ngl 99 -rea off -np 1 --no-mmap --cache-ram 15000 \\
     --spec-type draft-mtp --spec-draft-ngl all --spec-draft-n-max 3 --spec-draft-p-min 0.1 \\
     --port 8080
   # point your client at http://127.0.0.1:8080 (Anthropic /v1/messages is native)
SERVE
echo "== bench tool calls BEFORE trusting it (tool calls first, speed second)."
echo
echo "This is the finish line: an abliterated, distilled A3B that codes like Qwen3.8-27B,"
echo "runs at MoE speed on your 16GB card, fully local, no cloud dependency."
