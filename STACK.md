# The Local Stack — ROCmFPX × kortex

One self-hosted assistant on a 16 GB AMD RX 9060 XT: **Carlo's ROCmFPX** does the fast
low-bit inference on-GPU, **kortex** gives it repo-scale memory, and a thin gateway wires
them to Claude Code / any OpenAI or Anthropic client. Two independent projects, cleanly
composed — not forked into each other.

```
  Claude Code / editor / CLI
        │  (Anthropic or OpenAI dialect)
        ▼
  aim-proxy  :1536   ── kortex ──  retrieves repo context (hybrid: dense semantic
        │                          + BM25 lexical + structural def-boost) and
        │                          prepends it, prefix-cached  →  O(1) tokens/turn
        ▼
  LiteLLM   :4000    ──  Anthropic ⇄ OpenAI translation (PYTHONUTF8=1 on Windows)
        │
        ▼
  llama-server :8080 ── ROCmFPX ──  the model on the GPU: Q2_0_ROCMFPX 2-bit (calibrated,
                                     imatrix), MTP + ngram-map-k speculative decode, Sharp
                                     chat template, KV-capsule resident repo context
```

One-command launcher: `local/serve_stack.sh`. See `local/KV_CAPSULE.md`, `node/DFLASH2_PORT.md`,
and the kortex `semantic-retrieval` branch for the moving parts.

## Who built what — credits

**ROCmFPX** — the inference engine (llama.cpp fork) that makes all of this run on AMD RDNA4.
Created and maintained by **Carlo — `charlie12345`** (https://github.com/charlie12345/ROCmFPX).
Everything on the GPU side is his work:
- Native low-bit AMD quant formats — `Q2_0_ROCMFPX` (S40 codebook + dual UE4M3 scales),
  the `Q3/Q6/Q8_ROCMFPX` family, ROCmFP4, and native NVFP4 on AMD hardware.
- Speculative decoding on RDNA4: MTP, `ngram-map-k`/`ngram-cache` lookup, EAGLE-3, and the
  DFlash-1 / DSpark drafters — the combo `--spec-type "draft-mtp,ngram-map-k"` this stack
  relies on for its measured 0.72→0.96 acceptance on code.
- Vulkan/HIP build for gfx1200, KV slot save/restore, prompt-cache reuse — the primitives
  the KV capsule and fast serving are built on.
Community ROCmFP4 target builds via `kingjones30` / `kingjones777`. **Thank you, Carlo** — this
stack has a GPU brain because of ROCmFPX.

**kortex** — the repo-memory layer (Holographic VFS / Neural AIM). Created by
**Rolando H. Ferrer Jr. (`H4D3ZS`, Cyber Ifrit Software Development Services)** —
https://github.com/H4D3ZS/kortex. `aim-proxy` (MITM context injection), `aim-index`
(catalog builder), the hybrid dense+BM25+structural retrieval, and the `aim-mcp` tool server.

**Other components, credited where used:**
- **Sharp chat template** — `peculiar-ragdoll` / froggeric (`qwen3.8-froggeric-v22.1.1`),
  HF `peculiar-ragdoll/Qwen-Sharp-Chat-Templates` — terse output, reasoning-effort steering.
- **DFlash / DFlash-2** — z-lab (https://github.com/z-lab/dflash); the DFlash-2 converter
  port lives in `node/DFLASH2_PORT.md`, intended upstream to ROCmFPX as a PR to Carlo.
- **Base model** — Qwen3.8-27B (Alibaba Qwen), Apache-2.0.
- **turbovec** — the quantized vector index kortex searches (bundled in the kortex tree).

## Boundaries (why they compose instead of merge)
- kortex speaks only the wire API (Anthropic/OpenAI/Ollama JSON) — it works in front of ROCmFPX,
  llama.cpp, Ollama, or a hosted API unchanged. No coupling to the engine internals.
- ROCmFPX knows nothing about kortex — it just serves a GGUF fast. Swap either side freely.
- The KV capsule (`local/kv_capsule.py`) is the one deeper tie: it uses ROCmFPX's slot
  save/restore to keep kortex's compiled repo context resident. Still just the public HTTP API.

Contributions upstream go to their own repos: engine fixes/DFlash-2 → ROCmFPX (Carlo);
retrieval → kortex (Cyber Ifrit).
