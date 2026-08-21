# KLC engine hook — questions for Carlo (ROCmFPX owner)

KLC needs the served model to accept **soft-token embeddings** (continuous vectors, not token ids)
prepended before the prompt tokens. llama.cpp's C API supports this (`llama_batch.embd`), but
`llama-server`'s HTTP API doesn't expose it. We need one of these.

**Q0 (NEW, now the preferred path) — KV-cache slot save/restore.** 2026 SOTA (Latent Context
Compilation / Cartridges) compiles context into a portable **KV-cache blob**, not soft-token embeddings.
Does ROCmFPX's `llama-server` expose upstream llama.cpp's **slot state save/restore**
(`/slots/{id}?action=save|restore`, `llama_state_seq_save_file`/`load`)? If yes, we can inject a
pre-compiled repo KV as the prefix slot and skip an input-embeddings API entirely. Please confirm the
endpoint + whether restored KV works as a cached prefix under `--spec-type draft-mtp`.

**Q1 — Does ROCmFPX's `llama-server` already accept input embeddings on any endpoint?**
(e.g. a `prompt_embeddings` / `input_embeddings` field, or the `/embeddings` infra reused for input.)
If yes, point me at the field + shape it expects.

**Q2 — If not, would you add a `soft_tokens` field to `/v1/chat/completions`?**
Shape: `soft_tokens: float[n_soft][n_embd]` (e.g. 8 × 5120), decoded via `llama_batch` with
`embd` set, at positions 0..n_soft-1, BEFORE the text tokens, with `pos` continuing normally.
KV for these positions is computed once and prefix-cached like any prefix. This is the whole hook.

**Q3 — Interaction with MTP speculative decode:** do injected `embd` positions play nicely with
`--spec-type draft-mtp`? (The draft head reads hidden states; soft-token positions should be fine as
context but confirm the draft doesn't assume token ids exist for every position.)

**Q4 — Fallback:** if a server change is too invasive, can you expose a minimal C-API harness
binary (`llama-embd-infer`) that takes `[soft_tokens.bin] + prompt` and streams completion? We'd
drive KLC sessions through that instead of the HTTP server.

Context: this is for the Neural VFS / kortex gist-injection — feeding the 6 KB HRR gist as latent
soft-tokens so the model reasons over a repo with zero file text in the prompt. Target model is our
own Q2_0_ROCMFPX Qwen3.8-27B (n_embd=5120, has the MTP head at blk.64).
