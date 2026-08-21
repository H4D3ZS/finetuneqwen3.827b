# finetuneqwen3.827b — a fast local coder from one Qwen3.8-27B base

> The runtime stack (GPU engine + repo memory) and full credits — **ROCmFPX by Carlo
> (charlie12345)** × **kortex by Cyber Ifrit** — are documented in **[STACK.md](STACK.md)**.

**Goal:** a fast, local coding model on the RX 9060 XT (16GB) that *learns from frontier
teachers* — not a frontier model itself (a ~3B-active model imitates frontier coding, it does
not become frontier), but a genuinely strong fast coder distilled from real frontier output.

## ▶ CURRENT DIRECTION (2026-08, supersedes the older sections below)

**One base — `Qwen/Qwen3.8-27B` (dense) — two products from it:**

1. **Quantized dense 27B ≤16GB** — the daily driver. Best today: GDN-aware
   `Qwen3.8-27B-Ridge-3.7bpw` (11.7GB, coherent, vision), ~9 tok/s (bandwidth wall).
2. **Own MoE MTP A3B "escha"** — sparse-upcycle the dense 27B → ~3B-active MoE
   (`node/05_upcycle.py`), escha-class speed, then specialize on a frontier-distilled corpus.

**Corpus (frontier-distilled, all free via Qwen ambassador API):** a ~5.2k **Qwen3.8-Max**
frontier seed + bulk from **Qwen3.7-Plus** (`node/02b_teacher_api.py`, quota-guarded, gentle).

**Speed track (the real 150-200 tok/s lever):** run the finished A3B on native **ROCm/HIP**
(gfx1200) instead of Vulkan — a ~2-3× bandwidth-efficiency jump. See
[`docs/ROCM_HIP_RDNA4.md`](docs/ROCM_HIP_RDNA4.md) and `local/build_hip_rdna4_windows.ps1`.

**Read these, in order:**
[`RUNBOOK.md`](RUNBOOK.md) (end-to-end current plan) ·
[`MOE_UPCYCLE_PLAN.md`](MOE_UPCYCLE_PLAN.md) (dense→MoE + the bandwidth physics) ·
[`docs/ROCM_HIP_RDNA4.md`](docs/ROCM_HIP_RDNA4.md) (the speed backend).

*The sections below describe the earlier distill-into-3.6-A3B design and are kept for history.*

---

<details><summary>Earlier design (historical): distill an abliterated Qwen3.8-27B into 3.6-A3B</summary>

**Goal:** one local model on the RX 9060 XT (16GB) that codes like Qwen3.8-27B (which beats
Opus 4.6 on SWE-bench Pro / QwenSWEBench / IFBench), runs at MoE speed (~40-120 tok/s),
reasons in short chains, and is **abliterated** for authorized security work.

**How:** you cannot quantize a dense model into a fast MoE (that's sparse-upcycling — a full
retrain, confirmed by Qwen Discord). Instead we **distill** Qwen3.8-27B's coding ability
*into* Qwen3.6-35B-A3B, which is *already* an MoE (256 experts / 8 active). The teacher is an
**already-abliterated** Qwen3.8-27B, so the student inherits the abliteration through
distillation — no separate abliteration step. Training runs on a rented MI300X; final
quantization runs at home on the RX 9060 XT.

Read **`PLAN.md`** before spending a cent: goal, 12-risk register, phased $5/$25/$40 gates,
cost model, and the fallbacks for every ROCm unknown.

---

## Why the abliterated-GGUF teacher is the clean design

The teacher is **`Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF`** (GGUF only, no safetensors).
That single choice solves three problems at once:

1. **No vLLM, no torch-clobber.** A GGUF teacher runs in **llama.cpp** (ROCmFPX/HIP) as a
   separate process; the student trains in Unsloth. They never share a Python/torch env, so
   the "`pip install vllm` pulls a CUDA wheel and breaks ROCm torch" failure (which killed
   the first smoke run) simply cannot happen.
2. **Abliteration is baked in.** The teacher's completions are uncensored, so SFT-distilling
   them teaches the student to comply — no separate local abliteration of a 70GB model
   (removes the old R10 offload headache). See the honest R12 caveat in PLAN.md.
3. **Higher-quality teacher, free.** Someone already abliterated the full dense 3.8; we
   distill *that*, not a base model we'd have to abliterate ourselves.

```
                    RENTED MI300X (Unsloth Studio ROCm image)          THIS MACHINE (RX 9060 XT, 16GB)
  node/                                                     local/
   01_build_corpus.py   agentic-coding prompts               convert_quant.sh
   build_rocmfpx.sh     build llama.cpp (HIP/gfx942)            -> Q2_0_ROCMFPX GGUF, fits 16GB,
   serve_teacher.sh     download + serve abliterated 3.8 GGUF     fork-native fast 2-bit
   02_teacher_gguf.py   generate targets via llama-server API   (abliterate.py kept ONLY for
   03_unsloth_sft.py    SFT-distill -> Qwen3.6-35B-A3B            the TEACHER_MODE=hf path)
   merge_lora.py        adapter -> full bf16
   eval.py              GO/NO-GO gate vs base
   run_node.sh          orchestrates all of the above
```

---

## End-to-end flow

1. **PLAN.md + RENTAL.md** — pre-flight, droplet OFF, free. HF token, zip transcripts.
2. **`node/run_node.sh --smoke`** (~$5) — builds llama.cpp, serves the abliterated teacher,
   generates 50 targets, SFT-distills, evals. Every risk surfaces cheap.
3. **`node/run_node.sh --poc`** (~$25) — real PoC + eval gate. GO only if distilled >= base.
4. **`node/run_node.sh --full`** (~$40) — ship run. Produces `student-merged/` (bf16),
   already abliteration-transferred.
5. `scp` the merged model home. **STOP THE DROPLET.**
6. **`local/convert_quant.sh ../student-merged`** — → `Q2_0_ROCMFPX` GGUF. (No local
   abliteration needed — it's baked in.) Serve with ROCmFPX llama-server.

### Two teacher modes
- **`gguf` (default):** abliterated GGUF teacher → student is abliteration-transferred.
  No separate abliteration. Simpler, ROCm-clean.
- **`TEACHER_MODE=hf`:** base (non-abliterated) 3.8 via batched HF → then abliterate the
  merged model LOCALLY with `local/abliterate.sh`. Use only if you want direct, thorough
  abliteration rather than transferred.

---

## Honest scorecard

| goal | outcome |
|---|---|
| A3B, Opus-competitive agentic coding | **yes**, on the coding/tools distribution distilled |
| abliterated | **yes (transferred)** via the abliterated teacher — see R12 caveat |
| short reasoning | **yes**, distilled from the teacher's thinking |
| fits 16GB, MoE speed | **yes**, ~9-12GB Q2_0_ROCMFPX, ~40-120 tok/s |
| sustained 300+ tok/s | **no** — physics (218 GB/s). 100-120 burst is the ceiling. |
| general Opus clone | **no** — student capacity ceiling is 3.6-A3B; we push it on coding |

### The free alternative (if you don't need MoE speed)
The abliterated 3.8 also ships a **Q2_K (10.7GB)** that fits 16GB, and a **Q8_0 (28.6GB)**
you can requant locally to `Q2_0_ROCMFPX` — giving abliterated **full-3.8** quality at dense
~25-64 tok/s for **$0**, no training. The distillation is only worth it for the MoE speed
jump (25-64 → 40-120). Decide before spending.

## Reference
- `reference/reference-MTP-and-cliff-findings.md` — measured ROCmFPX MTP/quant findings and
  the hardware laws (why only uniform 2-bit is fast on the 9060 XT).
- ROCmFPX is a pinned submodule (`charlie12345/ROCmFPX @ b2f5829`); models are gitignored.

</details>
