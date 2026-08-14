# finetuneqwen3.827b — distill Qwen3.8-27B into a fast, local, abliterated A3B

**Goal:** one local model on the RX 9060 XT (16GB) that codes like Qwen3.8-27B (which beats
Opus 4.6 on SWE-bench Pro / QwenSWEBench / IFBench), runs at MoE speed (~40-60 tok/s real,
120+ burst), reasons in short chains, and is abliterated for authorized security work.

**How:** you cannot quantize a dense model into a fast MoE (confirmed by Qwen Discord —
that's sparse-upcycling, a full retrain). Instead we **distill** Qwen3.8-27B's coding
ability *into* Qwen3.6-35B-A3B, which is *already* an MoE (256 experts / 8 active — escha's
base). Training runs on a rented MI300X; abliteration + final quant run at home.

Read **`PLAN.md`** before spending a cent. It's the pre-flight: goal, 9-risk register,
phased $5/$25/$40 gates, cost model, and the fallbacks for every ROCm unknown.

---

## The split: what runs where, and why

```
                    RENTED MI300X (AMD Dev Cloud, 192GB, ROCm)          THIS MACHINE (RX 9060 XT, 16GB)
                    ── the CLEAN, compute-heavy half ──                 ── the sensitive + local half ──
  node/                                                     local/
   01_build_corpus.py   agentic-coding prompts               abliterate.py / abliterate.sh
   02_teacher_generate  Qwen3.8-27B answers (vLLM, thinking)    remove refusals (kept OFF the cloud:
   03_distill_train     KL-distill -> A3B (LoRA, guarded)       ToS-clean + private). Streams the
   merge_lora.py        adapter -> full bf16                    70GB model through NVMe offload.
   eval.py              GO/NO-GO gate vs base                 convert_quant.sh
   build_rocmfpx.sh     (optional) HIP build for on-node        -> Q2_0_ROCMFPX GGUF, fits 16GB,
                        quant/inference validation                fork-native fast 2-bit
   run_node.sh          orchestrates the above                serve via ROCmFPX llama-server
```

**Why abliteration is local:** removing refusals on a corporate GPU cloud invites ToS
trouble and leaves traces. The distillation (legitimate model training) is clean to run on
AMD's cloud; the abliteration is not their business. Clean separation, and it's how your
existing v2 was already made — locally, with Abliterix.

**ROCmFPX on the MI300X (`node/build_rocmfpx.sh`, optional):** the MI300X supports the real
HIP/ROCm backend that your local Vulkan-only build can't. Building it there lets you
sanity-quant + inference-test the distilled model on the node before downloading 70GB. Skip
it on smoke/PoC runs to save money — the *shipped* quant happens locally after abliteration.

---

## End-to-end flow

1. **PLAN.md + RENTAL.md** — pre-flight, droplet OFF, free. Get HF token, zip transcripts.
2. **`node/run_node.sh --smoke`** (~$5) — full pipeline on 50 prompts. Every risk surfaces cheap.
3. **`node/run_node.sh --poc`** (~$25) — real PoC + eval gate. GO only if distilled >= base.
4. **`node/run_node.sh --full`** (~$40) — ship run. Produces `student-merged/` (bf16).
5. `scp` the merged model home. **STOP THE DROPLET.**
6. **`local/abliterate.sh ../student-merged`** — abliterate (NVMe offload or Abliterix).
7. **`local/convert_quant.sh <abliterated>`** — → `Q2_0_ROCMFPX` GGUF.
8. Serve via ROCmFPX llama-server (see local/convert_quant.sh for the exact command); bench tool calls before trusting it.

---

## Honest scorecard (so there are no surprises)

| goal | outcome |
|---|---|
| A3B, Opus-competitive agentic coding | **yes**, on the coding/tools distribution distilled |
| short reasoning (not 32k-token stalls) | **yes**, distilled from the teacher's thinking |
| fits 16GB, MoE speed | **yes**, ~9-12GB Q2_0_ROCMFPX, ~40-60 tok/s real / 120+ burst |
| abliterated, fully local | **yes**, kept off the cloud |
| sustained 300 tok/s | **no** — physics (218 GB/s). 300 is a bigger-card burst number. |
| general Opus clone | **no** — student capacity ceiling is 3.6-A3B; we push it on coding |

## Reference
- `reference/reference-MTP-and-cliff-findings.md` — measured ROCmFPX MTP/quant findings,
  the hardware laws, and why only uniform 2-bit is fast on the 9060 XT.
