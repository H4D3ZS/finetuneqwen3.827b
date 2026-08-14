# MTP decode collapses at exactly `-c 65536` (Vulkan, Qwen3.6-35B-A3B ROCmFP2)

Native MTP sustains ~95 tok/s up to `-c 61440` and drops to ~34 tok/s at `-c 65536` —
a 2.7x cliff at exactly 64K, with **identical draft acceptance and essentially identical
VRAM on both sides**. That rules out memory pressure and points at a buffer sizing or
indexing threshold in the MTP verify path.

Everything else about MTP works well here: at `-c 61440` this model does **80.8 tok/s
decode with 0.10 s warm prefill** on an append-only conversation, versus 20.8 tok/s
without `--spec-type`. The cliff is the only thing capping usable context.

## Environment

| | |
|---|---|
| build | `charlie12345/ROCmFPX` `main` @ `b2f5829` (descendant of `d3ca537`, +12) |
| backend | **Vulkan0** — `AMD Radeon RX 9060 XT` (16304 MiB), Windows 10 |
| note | this build is Vulkan-only: `GGML_HIP:BOOL=OFF`, no ROCm SDK present |
| model | `cafonez/Escha-W2-35B-A3B-ROCmFP2` → `Qwen3.6-35B-A3B-Escha-W2-ROCmFP2.gguf` |
| arch | `qwen35moe`, 41 layers, 256 experts / 8 active, native MTP: 1 predict layer |
| size | 13,099,217,088 bytes (12.2 GiB), fully resident (`-ngl 99`) |

## Repro

```
llama-server -m Qwen3.6-35B-A3B-Escha-W2-ROCmFP2.gguf --alias escha-w2 \
  -c <CTX> -fa on -ctk q8_0 -ctv q8_0 -ngl 99 -rea off -np 1 --no-mmap \
  --spec-type draft-mtp --spec-draft-ngl all \
  --spec-draft-n-max 3 --spec-draft-p-min 0.1 --port 8080
```

Vary only `-c`. Fixed 1,210-token prompt, `n_predict 200`, `temperature 0`,
`cache_prompt false`, 3 repetitions per setting.

## Observed

| `-c` | decode tok/s (3 reps) | mean | VRAM |
|---:|---|---:|---:|
| 32,768 | 94.02, 94.36, 94.25 | **94.21** | — |
| 40,960 | 94.04, 95.54, 94.47 | **94.68** | — |
| 49,152 | 94.38, 95.51, 95.71 | **95.20** | — |
| 53,248 | 91.97, 94.21, 94.70 | **93.63** | 13.15 GiB |
| 57,344 | 93.98, 95.35, 95.51 | **94.95** | 13.20 GiB |
| 61,440 | 93.32, 94.92, 95.39 | **94.54** | 13.25 GiB |
| **65,536** | 34.41, 34.65, 34.14 | **34.40** | — |
| 81,920 | 34.51, 34.41, 34.40 | **34.44** | — |
| 131,072 | — | **33.82** | 13.27 GiB |

## Why this does not look like memory pressure

- **Draft acceptance is identical either side of the cliff**: `mean acc len = 2.88`,
  `acc rate/pos = (0.783, 0.623, 0.478)` at every setting. The drafts are equally good;
  only the verify/decode throughput changes.
- **VRAM barely moves**: 13.15 → 13.27 GiB across the entire 53k–131k range, on a
  16 GiB card. The model stays fully resident; there is no spill at any point.
- **`-fit off` changes nothing** (33.94 tok/s at `-c 98304` with fitting disabled), so
  it is not the fitter silently moving layers.
- **Without `--spec-type` decode is flat** at ~33–36 tok/s across the same range. The
  cliff exists only on the MTP path.
- Run-to-run variance within a load is ~2%, so a 2.7x step is far outside noise.
- The boundary is an exact power of two, which suggests a threshold rather than a
  gradual resource effect.

## Impact

`-c 61440` is the practical ceiling for MTP. After a coding agent's fixed system-prompt
and tool-schema overhead (~36.7k tokens measured for Claude Code), that leaves roughly
20k tokens of usable conversation before compaction. Raising the ceiling to 128k would
roughly quintuple usable context at the same 95 tok/s.

## Ruled out

None of these move the cliff:

- `--no-kv-unified` (default logs `kv_unified = true`; the flag made no difference)
- `-ctxcp 128` / `--ctx-checkpoints`, and `-cpent -1`
- `--cache-ram 15000`
- `-ctk f16 -ctv f16 -ctkd f16 -ctvd f16` (the promoted profile from
  `ROCmFPX-EXPERIMENT.md`, since `DEEPSEEK-V4-ROCMFP4-PORT-STATUS.md:285` flags
  quantized KV under MTP)
- `--no-spec-draft-backend-sampling`
- `-b 2048 -ub 512`
- `--spec-draft-n-max` 1, 2, 3, 4 × `--spec-draft-p-min` 0.1, 0.2, 0.75
- `-fit off`
- **context-checkpoint pool size**: `-ctxcp 2`, `-ctxcp 4`, and `-cpent -1` (checkpoints
  disabled entirely) all still measure ~34 tok/s at `-c 131072`. Checkpoints are *not*
  the cause, though they are a visible symptom — past the cliff, checkpoint creation
  slows ~5.7x (the two startup checkpoints are created 0.35 s apart at `-c 61440` and
  1.98 s apart at `-c 65536`), and the per-task cycle goes 3.05 s → 10.7 s. That ratio
  tracks the decode drop, so whatever slows down is shared by both paths rather than
  being checkpoint-specific.

Separately worth knowing: **`--cache-disk` is a hard regression here** — cold prefill
26.9 s → 114.3 s and decode 88 → 34.8 tok/s on an otherwise identical `-c 61440` MTP
run. It may deserve its own look.

## Suggested regression coverage

A decode measurement at `-c 65536` compared against `-c 61440`, asserting they stay
within ~10% of each other, would catch this. The existing MTP suite sweeps context but
appears not to compare across the 64K boundary.

---

## Minor: the full-prefix-extension requirement is easy to trip when benchmarking

Not a bug — `tools/server/server-context.cpp:2791-2803` is clearly intentional and well
commented — but it cost hours to diagnose, and the current benchmark suite cannot
surface it because every MTP benchmark runs with the prompt cache disabled
(`ROCmFPX-EXPERIMENT.md:416`, `ROCmFPX-SERVING.md:213` `STRICT_BENCH=1`).

```cpp
if (common_speculative_state_required(spec.get()) &&
    n_past > 0 && n_past < slot.prompt.n_tokens()) {
    // "prompt cache cold fallback: reason=spec-boundary-mismatch"
    n_past = 0;
```

The slot holds `previous prompt + previous completion`. A multi-turn benchmark that
appends to the *prompt* while omitting the model's own generated output diverges
mid-prefix, trips this, and re-prefills the whole conversation — which reads exactly
like "MTP destroys the prefix cache" (26.7 s warm prefill instead of 0.10 s). With a
true full-prefix extension the cache hits normally and MTP is a ~4x win.

A line in the MTP docs stating that cache reuse requires a full-prefix extension, and
that multi-turn benchmarks must replay the prior completion verbatim, would save the
next person the same detour.

---

## FEATURE: make MTP reuse the KV cache on partial-prefix turns (attempted, precisely located, not landed)

The single highest-value MTP change for agentic use. Currently MTP cold-prefills the
WHOLE conversation on any turn whose prefix is a partial match (system-reminder or
tool-result shifts the middle) - measured 91s re-prefill of 28k tokens, every such turn.
Fixing it lifts real-session decode from ~25 tok/s (no MTP) to ~40 tok/s (MTP) on ALL
turns, not just clean appends.

### What was tried (reverted to stock - do not ship the patches, they crash)
1. `server-task.cpp` ~3007: let the live SLOT's partial prefix survive selection
   (don't force f_keep=-1 on partial under spec). WORKS - selection now reuses lcp.
2. `server-context.cpp` ~2796: on partial match, keep n_past (reuse KV), clear the
   per-slot spec buffers (spec_draft/spec_i_batch/spec_ckpt) + spec state, let the tail
   re-warm. Selection + KV reuse now correct (log: "spec partial-prefix: reuse N cached").

### Why it crashes (the exact remaining gap)
`common/speculative.cpp:1606` aborts with "missing MTP boundary for seq_id=0 pos=P
(current=-1/0 previous=-1/0)". Root cause:
- MTP's draft() needs the target hidden state `pending_h[seq]` at pos = first-drafted-1.
- `pending_h` is ONLY stashed by the MTP-specific `process()` (speculative.cpp:1680-1701),
  which runs during the speculative decode loop.
- After a partial KV reuse, the divergent tail is prefilled through NORMAL decode, which
  never calls MTP process(), so `pending_h` is never stashed at the resume position.
- First draft() then finds no boundary at pos_needed -> abort.

### The fix (for whoever lands it)
Before the first draft() after a partial-reuse turn, run one MTP `process()` pass over
the last prefilled token so it stashes `pending_h` at the tail's final position - exactly
what the full-prefill path does implicitly. Mirror the full-match "n_past--" one-token
replay (server-context.cpp:2966-2970), but at the partial boundary. Risk: position
tracking across ubatches (the deferred-boundary bridge at speculative.cpp:690-760) must
stay paired, or it re-crashes. Speculative decoding is lossless (target verifies every
token), so a mis-seed can only lower acceptance, never corrupt output - the only failure
mode is the abort, which a correct boundary stash removes.

Build: cmake --build build --config Release --target llama-server (VS2026, ~40s incremental).

### CORRECTION after full trace: this is an architectural limit, not a shallow bug

Deeper reading of `speculative.cpp:1588-1611` (the boundary lookup at the START of every
process()) shows why cold-reprocess is the CORRECT choice, not an oversight:

- To process a batch beginning at position P, MTP needs the target hidden state stashed
  at P-1 (`pending_h` at pos_needed = P-1). Only P=0 (BOS) needs none.
- `pending_h` is the pre-norm hidden state - stored SEPARATELY from the KV cache and only
  produced by a forward pass. Reused KV carries K/V projections, NOT this hidden state.
- Therefore after reusing cached KV to position lcp, the boundary at lcp-1 does not exist
  anywhere and cannot be reconstructed without a forward pass ending at lcp-1 - which,
  chained back, means reprocessing from the last stored boundary (usually BOS).

A real fix needs EITHER:
  (a) cache hidden-state boundaries at every position: ~n_embd*4 bytes/token
      = 2048*4 = 8KB/token, ~224MB at 28k ctx - a genuine design change to the MTP
      draft state + its RAM/disk serialization, or
  (b) recompute the boundary from the nearest stored one - which for an arbitrary partial
      point degenerates to a full reprocess anyway.

So MTP + partial-prefix reuse is NOT a small server-glue patch. The stock cold-reprocess
is a reasonable engineering choice. Practical guidance stands: MTP wins on append-heavy
workloads (clean full-prefix extension, cache hits), loses on partial-heavy ones
(mid-conversation prefix shifts, cold reprocess). Pick per workload; do not ship the
partial-reuse patches (they crash on the missing boundary).
