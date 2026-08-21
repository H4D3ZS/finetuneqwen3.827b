# KV Capsule — resident repo context, no re-processing (local, no rental)

The non-band-aid intermediate toward latent context (Tier 3). Instead of paying
for the repo context in prompt tokens every turn, we process the repo's stable
structure ONCE, save its KV to disk, and `restore` it in ~100ms at session
start. The model then carries the repo as resident KV — never re-tokenized.

This is the SAME llama.cpp slot save/restore path the learned latent core (LCC)
will use; the capsule is the STATIC precursor (hand-compiled structure). The
LEARNED version (a distilled gist instead of hand-picked text) needs the
finetune — see `../node/PREFLIGHT.md`.

## Measured on this box (W2 27B, ROCmFPX)
| step | result |
|---|---|
| process capsule cold | ~4600 tokens, ~11.5 s (once, ever) |
| save KV → disk | ~300 MB, ~230 ms |
| **restore in a new session** | **~107 ms** |
| query after restore | **~4100 tokens CACHED (repo context re-processed: 0)** |

Plain prompt-cache already reuses an identical prefix (measured: 1888→4 tokens
processed on repeat). The capsule adds what prompt-cache can't: it **survives
restarts**, **loads instantly with no warm pass**, and is built to stay
**byte-identical as you edit files** (it holds the file tree + definition
SIGNATURES, not the churning bodies) — dodging the 95% cache-invalidation
problem. Volatile file *contents* still come through the live hybrid retrieval
(kortex); only the stable structural spine is capsuled.

## Use
```bash
# server must run with --slot-save-path <dir> (serve_stack.sh already sets it)
cd local
python kv_capsule.py compile --repo ..                  # -> capsule.txt (deterministic)
python kv_capsule.py bake --capsule capsule.txt --slot-file repo-capsule.bin   # process once + save KV
# then, at the start of any session:
python kv_capsule.py restore --slot-file repo-capsule.bin                       # ~100ms, repo resident
```

## Honest caveats (real limits, not bugs)
- **Tied to the exact model + exact capsule bytes.** Re-quant / re-finetune, or a
  capsule change, means re-bake (`compile` then `bake`). Restoring a stale KV
  against a different model is undefined — don't. (Not auto-wired into
  serve_stack for this reason; it's an explicit step.)
- **~300 MB per capsule** (q8 KV × 64 layers). Disk-cheap; not RAM-resident until
  restored into a slot.
- **Structural, not complete.** The capsule is the MAP (tree + signatures), not
  full file bodies — it makes the model *oriented*, not omniscient. Full content
  is the live-retrieval path's job. The LEARNED capsule (LCC) that compresses
  actual content into KV is the endgame and needs the owned/finetuned model.
- **The capsule must be a PREFIX of the real request** for its KV to be reused
  (llama.cpp matches by token prefix). The current flow bakes it as a leading
  user turn; keep real queries appended after the capsule text, or (cleaner, TODO)
  bake it as a fixed system block the proxy always prepends.
