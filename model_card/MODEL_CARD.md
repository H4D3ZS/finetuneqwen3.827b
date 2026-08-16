---
license: apache-2.0
base_model: Qwen/Qwen3.6-35B-A3B
tags:
- qwen3
- moe
- a3b
- mtp
- gguf
- rocmfpx
- code
pipeline_tag: text-generation
---

# Qwen3.8-Distill-35B-A3B-Coder-Abliterated (Q2 ROCmFPX, PoC)

A 2-bit **ROCmFPX** GGUF of a distilled Qwen3.6-35B-A3B (MoE, 256 experts / ~3B active), sized to
run on a **16GB consumer GPU**. Ships with the MTP (`nextn`) head and build instructions for the
matching runtime.

> **Honest status:** this is a proof-of-concept. On the internal 10-task smoke eval the distilled
> model **tied its base** (6/10 vs 6/10) — no regression, no measurable gain yet — and it is now
> quantized to 2-bit, which trades quality for fit. Publishing it as a reproducible artifact of the
> pipeline (distill → graft MTP → ROCmFPX 2-bit GGUF), **not** as a benchmark-winning coder.
> The quality fix is a larger, tool-calling-heavy corpus — a separate follow-up run.

## What this is

- **Base / architecture:** `Qwen/Qwen3.6-35B-A3B` (`Qwen3_5MoeForCausalLM`, 256 experts, ~3B
  active). The "3.8" in the name refers to the **teacher**, not the base.
- **Teacher:** abliterated `Qwen3.8-27B` (GGUF Q8_0) via llama.cpp — sequence-level reasoning
  distillation (teacher `<think>` chains as SFT targets).
- **Method:** Unsloth 4-bit QLoRA, `completion_only_loss`, 1 epoch / 850 teacher completions,
  merged to bf16, MTP head grafted back from base, converted + quantized with ROCmFPX.
- **Quant (the interesting part):** a hand-built **role-aware mix** — 2-bit experts
  (`Q2_0_ROCMFPX`, the ~90% bulk) + **Q6 attention / embeddings / shared-experts / output**
  (`Q6_0_ROCMFPX`, the coherence-critical ~10%), norms in F32. **12GB total**, fits a 16GB card
  with ~4GB left for KV/context. This is the llama.cpp/ROCmFPX analogue of the eschamoe/OTQ
  role-aware idea: pure 2-bit-everywhere collapses the model; keeping *attention* precise while
  2-bit'ing the experts preserves coherence. See the exact `--tensor-type` recipe in
  [BUILD.md](BUILD.md).
- **"Abliterated":** transferred over the training corpus (teacher was abliterated) — corpus-scoped,
  NOT a globally abliterated model.

## Run it

You need a `llama-server` built from the pinned **ROCmFPX** source — see **[BUILD.md](BUILD.md)**.

```bash
llama-server -m *-Q2_ROCMFPX.gguf --host 127.0.0.1 --port 8080 \
  -ngl 99 -c 16384 -fa on --jinja --alias qwen38-distill-a3b
# OpenAI-compatible API at http://127.0.0.1:8080/v1
```

16GB card: context and concurrency share one KV pool — pick single-stream long context
(`-c 32768 -np 1`) **or** many short sessions (`-c 8192 -np 8`).

## Known limitations (measured)

- No accuracy gain over base yet; 2-bit lowers quality further.
- Weak on tool-calling/agentic tasks (thin PoC corpus) — the first thing the next run must fix.
- MTP `nextn` tensors are present but speculative decoding depends on your runtime's support
  (see BUILD.md). Text-only; no vision.

## Files

- `*-Q2_ROCMFPX.gguf` — the model (~16GB-card fit)
- `BUILD.md` — build the ROCmFPX runtime (pinned commit `b2f5829`)
- `build_rocmfpx.sh` — exact build script used

Apache-2.0, inheriting the base model's terms.
