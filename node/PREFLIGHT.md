# Finetune Pre-Flight — dense Qwen3.8-27B SFT (verified 2026-08-21)

Goal: LoRA-SFT the **dense Qwen3.8-27B** on our 15,574 teacher pairs so it stays sharp on long
agentic work, then merge → download → re-run the local quant pipeline → new calibrated W2.
This is the "rent the GPU only for training" path (`train_only.sh`), NOT the old distill-into-A3B flow.

## ✅ Verified locally (no rental spent) — the things that usually blow up a run
| Check | Result |
|---|---|
| Training data | **15,574 valid {prompt,completion} pairs** across all `teacher_*` dirs; 0 malformed |
| Loader | `03_unsloth_sft.py load_pairs` reads them recursively from `--data node/` ✅ |
| **Arch support** | base is `model_type: qwen3_5` (arch `Qwen3_5ForConditionalGeneration`). **Only transformers 5.x knows it** (local env = `5.16.0.dev0`, works). `AutoModelForCausalLM` maps `qwen3_5 → Qwen3_5ForCausalLM` (loads the text tower) ✅ |
| Seq-len truncation | completions: median ~292 tok, **p99 ~2176, max ~3035 → 0% exceed seq_len=4096** ✅ |
| merge_lora.py | same safe `AutoModelForCausalLM` path ✅ |
| Scripts | support `--student` / `--base` override so we target Qwen3.8-27B ✅ |

## ⚠️ Fixes applied this pre-flight
1. **`requirements.txt`: `transformers>=4.44` → `>=5.16`** (4.x dies with `KeyError: qwen3_5`). If 5.16
   isn't on PyPI yet, install the dev build on the node: `pip install "git+https://github.com/huggingface/transformers@v5.16.0"`.
2. **Added `--max_steps` to `03_unsloth_sft.py`** for a cheap on-node smoke test (see below).

## GPU choice — recommendation
- **H100 80GB (CUDA) is the LOWER-risk pick for this LoRA run**: bitsandbytes 4-bit "just works", unsloth
  is best-supported on CUDA. QLoRA 4-bit of 27B ≈ 24–30 GB → fits comfortably.
- MI300X 192GB (ROCm) also fine, but bnb 4-bit on ROCm is flaky → script falls back to **bf16 LoRA**
  (~70 GB, fits 192 GB). Use it only if you specifically want ROCm.
- Expect **unsloth to NOT support qwen3_5 yet** → auto-fallback to transformers+peft. That's fine and expected.

## Cost / time estimate (why the unease is overblown)
- 15,574 pairs × 2 epochs ÷ eff-batch 16 ≈ **~1,950 optimizer steps**.
- H100 QLoRA seq-4096 batch-1 ≈ ~1 s/step → **~35–60 min compute** + ~20–40 min setup/model-load + ~10 min merge.
- **Total wall ~1.5–3 h; spot H100 ~$2–3/h → ~$5–10 for the whole run.** Cheap.

## The run — exact commands (do the SMOKE first)
```bash
# on the node, after: pip install -r requirements.txt  (transformers>=5.16!)
cd node

# 1) SMOKE (~$0.20): prove load -> step -> checkpoint before the real run
python3 03_unsloth_sft.py --student Qwen/Qwen3.8-27B --data . \
  --out /scratch/smoke --max_steps 5 --save_every 5
#   PASS = it prints "N training pairs", picks unsloth OR fallback, logs 5 loss lines,
#   writes /scratch/smoke. If it dies here, we lost 12 minutes, not the whole run.

# 2) FULL run
STUDENT=Qwen/Qwen3.8-27B DATA=. OUT=/scratch/student-27b \
  python3 03_unsloth_sft.py --student Qwen/Qwen3.8-27B --data . \
  --out /scratch/student-27b --epochs 2 --save_every 200 --resume

# 3) merge
python3 merge_lora.py --adapter /scratch/student-27b --base Qwen/Qwen3.8-27B \
  --out /scratch/student-27b-merged

# 4) download student-27b-merged, then LOCALLY re-run our quant pipeline:
#    convert_hf_to_gguf.py --outtype f16 -> llama-imatrix -> llama-quantize --imatrix
#    --tensor-type "blk.64=q8_0" Q2_0_ROCMFPX  (see [[local-quant-pipeline]])
```

## Failure modes & mitigations (so nothing is a surprise)
| Failure | Cause | Mitigation |
|---|---|---|
| `KeyError: qwen3_5` | transformers < 5.x | requirements now pins >=5.16 / dev build |
| unsloth import/arch error | qwen3_5 not in unsloth yet | script auto-falls back to transformers+peft (expected) |
| bnb 4-bit error on ROCm | ROCm bnb flaky | script falls back to bf16 LoRA (needs 80GB+); or use CUDA H100 |
| OOM | card too small / seq too long | 4-bit path, or bump to 80GB+ card; seq_len already safe at 4096 |
| crash mid-run | anything | `--resume` + `save_every 200` (save_total_limit 3) resumes from last checkpoint |
| trained model "same as base" | forgot to merge | merge_lora.py is step 3; it warns if adapter is a no-op |

## Still-uneasy checklist (tick these on the node, in order)
- [ ] `pip install -r requirements.txt` → `python -c "import transformers;print(transformers.__version__)"` shows 5.x
- [ ] 5-step **smoke** passes (writes a checkpoint)
- [ ] full run starts, loss trends DOWN over first ~50 steps
- [ ] merge produces `student-27b-merged/` with safetensors
- [ ] download + local `convert_hf_to_gguf.py` loads it without arch error
- [ ] new W2 smoke-tests coherent (the merge task) before replacing the current backend
