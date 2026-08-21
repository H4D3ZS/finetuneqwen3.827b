# Kortex Latent Core (KLC) — the missing core, made into a system

**Goal:** turn kortex from *retrieve-text-and-inject* (RAG with a stable prefix cache) into
*inject-the-gist-as-latent-cognition* — the model reasons directly from the 6 KB HRR gist,
with **zero file text** in the prompt. This is the piece the Neural VFS paper describes but the
current `aim-proxy` cannot do against black-box APIs (they only accept tokens, not vectors).

It only works on a model **we own at the embedding/KV level** — our local Qwen3.8-27B on
llama.cpp/ROCmFPX. That is the whole reason the finetune and kortex are one project.

## The honest grounding (this is a known, proven technique — not sci-fi)
- **Gisting** (Mu, Li, Goodman 2023, arXiv:2304.08467): a model is trained so that a prompt is
  compressed into a few "gist" activations that the model can condition on later, ~zero extra
  tokens. Proven to retain most task performance. **KLC = Gisting where the gist is kortex's HRR
  vector instead of a learned token.**
- **Soft prompts / prefix-tuning** (Lester 2021; Li & Liang 2021): frozen model + trained
  continuous "soft tokens" prepended in embedding space. This is the injection mechanism.
- **Context distillation** (Snell 2022): teacher = model **with** the full file text in context;
  student = same model reading **only** the projected gist. Train student to match teacher.

## The hard truth about what "production-grade complete" requires
Three of the six components below are buildable in pure Python **now**. Two need a **GPU training
run** (hours). One needs an **engine hook** in ROCmFPX (llama.cpp) to feed soft-token embeddings at
serve time — llama.cpp *has* an embedding-input path in its C API, but `llama-server`'s HTTP API may
not expose it, so this likely needs a small C++ change (a conversation with Carlo, who owns ROCmFPX).
Anyone who claims this is a weekend script is wrong. We build it in stages and **measure at each one**.

**And the honest risk:** the *current* text-injection kortex already gets 91% token reduction and a
99.97% cache hit. KLC is a research **upgrade bet** — it may or may not beat that on real work. We
build it so we can *measure* whether latent beats text; we do not assume it.

## System architecture (6 components)

```
   ┌─────────────── kortex (Rust, exists) ───────────────┐
   │ HRR path-key superposition  →  per-file gist vectors │   ← your paper's core
   └───────────────┬─────────────────────────────────────┘
                   │ (1) gist-export: emit gist + cleanup codebook as .klc binary
                   ▼
   ┌─────────── KLC (Python + our model) ────────────┐
   │ (2) projector   gist[1536] → k soft-tokens[5120] │  ← trained MLP
   │ (3) distill     teacher(full text) → student(gist)│  ← the training run
   │ (4) eval        zero-text file-QA vs text baseline │  ← the proof
   └───────────────┬─────────────────────────────────┘
                   │ (5) engine hook: llama.cpp accepts soft-token embeddings at serve
                   ▼
   ┌─────────── aim-proxy v2 (Rust) ─────────────────┐
   │ (6) route: send gist → server projects+injects   │  ← zero text on the wire
   └──────────────────────────────────────────────────┘
```

### (1) Gist export — Rust, in `kortex/libaim`
Add `aim-index export-gist <repo> --out repo.klc`. Emits: the global superposition gist, each
file's spherical path-key, and the per-file content embedding (the "cleanup codebook" so retrieval
can snap noisy unbinds to exact vectors). Binary format: `[header][d][k][gist f32×d][ (path_key f32×d, content f32×d) × k ]`.
Status: **buildable now** (kortex already computes all of these; this just serializes them).

### (2) Projector — Python/PyTorch (`projector.py`, built in stage 1)
`GistProjector(d_gist=1536, d_model=5120, n_soft=8)`: LayerNorm → MLP → reshape to `n_soft × d_model`.
Maps one gist to `n_soft` soft-token embeddings that live in Qwen's input space. ~40M params, tiny.
Status: **built (this commit)**.

### (3) Context-distillation training — Python (`distill_train.py`, stage 2, needs GPU)
For each (file, question, answer) triple:
- **Teacher pass:** Qwen3.8-27B reads `[full file text] + question` → target logits/answer.
- **Student pass:** Qwen reads `[projector(gist) as n_soft embeddings] + question`.
- **Loss:** KL(student‖teacher) on answer tokens (context distillation) + CE on the gold answer.
- Train the **projector** (+ optional small LoRA on the model) with the base frozen.
Data comes from our own repos + corpus. Status: **skeleton built; run needs the GPU + hours.**

### (4) Eval — Python (`eval_latent.py`, stage 2)
Held-out file-QA: accuracy with **zero text** (gist only) vs the **text-injection baseline** vs
**no-context** floor. This is the number that decides if KLC is real. Status: **skeleton built.**

### (5) Engine hook — ROCmFPX/llama.cpp (stage 3)
**UPDATE (2026 lit review):** the state of the art (Latent Context Compilation, arXiv:2602.21221;
Context Distillation as Latent Memory Management, arXiv:2605.28889; "Cartridges"/C3) does NOT inject
soft-token *embeddings*. It compiles the context into a **portable KV-cache** ("Buffer Tokens retained
as a standard KV cache") via a *disposable LoRA compiler*, then the frozen model runs on that KV with
zero extra params. This is a BETTER fit for llama.cpp because **llama.cpp already supports KV-cache
save/restore** (server slot save/restore, `llama_state_seq_save_file`/`load`). So the path becomes:
- **A (preferred, likely NO deep engine change):** offline, distill repo -> a prefix **KV-cache blob**
  (disposable-LoRA method, component 3). At serve, **load it as the prefix slot state** with llama.cpp's
  existing state-save/restore. The model reasons over the repo with zero text tokens AND the KV is
  prefix-cached for free. Confirm ROCmFPX exposes slot save/restore over HTTP (or via the C API harness).
- **B:** soft-token embeddings via `llama_batch.embd` (original plan) — needs a server field; llama.cpp
  server does NOT expose input `prompt_embeds` today (only vLLM/NIM do), so this is the C++ change.
- **C (fallback, still text):** decode gist -> short text summary (better RAG). Safety net.
Status: **A may need only glue + a slot-restore endpoint check. `QUESTIONS-FOR-CARLO.md` updated.**

### (6) aim-proxy v2 — Rust (stage 4)
When the target is our local server (not a black-box API), send the **gist** (not text); the server
projects + injects. Falls back to text-injection (current behavior) for API targets. Status: **stage 4.**

## Build order (each stage ends in a measurement, not a promise)
1. **Projector + data contract** (Python, now) — no GPU. ✅ this commit.
2. **Distill + eval** (GPU run) — produces THE number: does gist-only beat no-context and approach
   text-injection? If no → we learned the idea's ceiling cheaply. If yes → continue.
3. **Engine hook** (Carlo/C++) — serve soft-tokens for real.
4. **aim-proxy v2** — wire it into the live Claude Code path, zero text on the wire.

Nothing here is faked. Stage 1 is real code below. Stage 2 is the honest gate.
