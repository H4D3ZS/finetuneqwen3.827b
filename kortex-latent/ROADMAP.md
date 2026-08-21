# Master Integration Roadmap — "integrate all of it"

Unifies decode-speed + context tracks into one program. **Key discovery (2026-08-21):** most engine
features are ALREADY in the ROCmFPX `llama-server` binary — this collapses the "needs-Carlo" column.

Dependency legend:  🟢 buildable/turn-on NOW · 🟡 needs a draft/training artifact (GPU) · 🔴 needs Carlo/new runtime

## Track A — Decode speed
| Item | Status in ROCmFPX | Dep | Action |
|---|---|---|---|
| MTP speculative | ✅ working (`--spec-type draft-mtp`), 37-57 t/s | 🟢 | current default |
| **n-gram / prompt-lookup** | ✅ built (`ngram-simple/map-k/map-k4v/mod/cache`, `--lookup-cache-static/dynamic`) | 🟢 | **A/B vs MTP on code; free tokens on copy-heavy edits** |
| **EAGLE-3** | ✅ built (`--spec-type draft-eagle3`) | 🟡 | get/train an EAGLE-3 draft for Qwen3.8-27B → SOTA 2-3× |
| DFlash / DSpark | ✅ built | 🟡 | needs the z-lab draft (download + convert + quantize) — see [[dflash-on-rocmfpx-resume]] |
| Activation sparsity (PowerInfer/Deja Vu) | ❌ not in this engine | 🔴 | different runtime; highest ceiling, last |

## Track B — Context (kortex latent core = LCC recipe)
| Item | Status | Dep | Action |
|---|---|---|---|
| Gist export (kortex → .klc) | to build | 🟢 | Rust `aim-index export-gist` |
| Projector | ✅ `projector.py` runs | 🟢 | built |
| **Disposable-LoRA distiller** (LCC / Cartridges) | to build | 🟡 | `distill_train.py`; teacher=full-text, student=gist→**KV blob**; GPU rental (27B won't train on 16GB) |
| Eval gate (gist-only vs text-inject) | to build | 🟡 | `eval_latent.py` — THE go/no-go number |
| **KV-blob injection at serve** | ✅ `--slot-save-path` + `/slots/{id}?action=save\|restore` exist | 🟢 | **no C++ change** — load compiled repo KV as prefix slot |
| aim-proxy v2 (route gist not text) | to build | 🟢 | after eval passes |

## Track C — Context prompt-cache wins (turn on now)
| Item | Status | Dep | Action |
|---|---|---|---|
| `--cache-reuse N` | ✅ | 🟢 | KV-shift reuse when prefix partially changes |
| `--cache-disk` + `--cache-disk-limit` | ✅ | 🟢 | SSD-backed prompt cache across restarts |
| KV eviction (SnapKV/StreamingLLM) | partial (`--keep`, SWA) | 🟡/🔴 | `--swa-full`/`--keep` now; full SnapKV = engine work |

## Build order (each ends in a MEASUREMENT)
1. ✅ **DONE (this session)** — `--cache-reuse`, `--cache-disk`, `--slot-save-path` on; and **combined
   `--spec-type "draft-mtp,ngram-map-k"` is now the serve_stack default** (verified: acceptance
   0.72→0.96 on copy-heavy edits, ngram spans 100% accepted). MTP=novel tokens, ngram=repeated code free.
2. 🟡 **EAGLE-3 draft** — biggest proven decode jump; source/train a Qwen3.8-27B EAGLE-3 draft.
3. 🟡 **LCC distiller + eval** — build `distill_train.py`/`eval_latent.py`; run on rented GPU; **gate**: does gist-only beat no-context and approach text-injection? Inject via existing slot-restore.
4. 🟢 **aim-proxy v2** — wire winning path into the live Claude Code stack.
5. 🔴 **Activation sparsity** — the moonshot; separate runtime, do last.

## Honest framing
Two proven wins are ~free (ngram-lookup, cache-reuse) and one SOTA win (EAGLE-3) needs only a draft, not
engine surgery. The research core (LCC-kortex) still needs a GPU training run + the eval gate before we
trust it over the text-injection kortex that already gives 91% token savings. We integrate in this order
so every step is measured, not assumed.
