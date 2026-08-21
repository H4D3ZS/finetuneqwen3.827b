# Porting DFlash-2 (candidate selector) into ROCmFPX — scoped plan

ROCmFPX is **DFlash-1 only** (grep: no `candidate_selector`/`selector` in runtime or converter;
latest main 0a59add still DFlash-1). The z-lab Qwen3.8-27B draft is **DFlash-2**. This is the plan to
add DFlash-2 support. References (algorithm is documented — this is a PORT, not research):
- Official: https://github.com/z-lab/dflash  (cloned to ~/Desktop/dflash-ref; selector in `dflash/model.py`)
- Production port: SGLang PR #35371 "DFlash2: local convolution + candidate selector"
- Other backend: https://github.com/ARahim3/mlx-dspark (MLX port)

## The algorithm (from dflash-ref/dflash/model.py, CandidateSelector, ~60 lines)
3 weights: `predecessor_codebook` Emb[vocab, rank=256], `successor_codebook` Emb[vocab, 256],
`hidden_projection` Linear[hidden=5120, 256, bias=False]. Config: selector_rank=256, selector_top_k=16.

```
select(hidden, logits, anchor_ids, temperature):
  unary, candidates = topk(logits, 16)          # top-16 target-head candidates per block slot
  hp = hidden_projection(hidden)                # [B, block, 256]
  predecessor = anchor_ids                      # the verified anchor token id
  for pos in range(block):
    score[k] = unary[pos,k] + dot( predecessor_codebook(predecessor) * hp[pos],
                                    successor_codebook(candidates[pos,k]) )   # bilinear lattice
    idx = argmax_k score            (or sample at T>0)
    predecessor = candidates[pos, idx]
    path.append(predecessor)
  return path                        # the chosen coherent block path
```
Net effect: +1.1–1.4 accepted tokens/step over DFlash-1, same verify width.

## SMART PORT STRATEGY (low-risk — keeps the selector OUT of llama.cpp's rigid graph)
The selector is a **post-drafting rerank**, not part of the model forward graph. So DON'T modify
`llama-model.cpp`'s per-arch graph builder (the risky part). Instead:
1. **Converter** (`convert_hf_to_gguf.py` DFlashModel + `gguf-py/.../tensor_mapping.py`):
   emit the 3 selector tensors under new names (e.g. `dflash.selector.pred_codebook`, `.succ_codebook`,
   `.hidden_proj`) and metadata `dflash.selector_rank`, `dflash.selector_top_k`. (Currently the
   converter has NO mapping for `model.candidate_selector.*` → that's the convert failure.)
2. **speculative.cpp** (`common_speculative_impl_draft_dflash`): after DFlash-1 produces block draft
   logits + hidden states (already available via `llama_set_embeddings_layer_inp`), run the bilinear
   walk on CPU (block_size × 16 × 256 dot products = trivial). Load the 3 tensors as raw arrays from
   the draft gguf (they don't need to be in the compute graph). Replaces the current top_k=1 pick.
3. Metadata already plumbed: block_size, mask_token_id, target_layer_ids all load today — mirror that.

Why low-risk: no new arch tensors in the model graph loader; the selector is pure post-step CPU math
over data llama.cpp already exposes. Codebooks are big (vocab 248320 × 256 × 2 ≈ 254M params ≈ 500MB
fp16) — load once, mmap; keep in RAM, not VRAM (CPU rerank).

## STATUS (2026-08)
**Stage 1 DONE** — ROCmFPX `convert_hf_to_gguf.py` DFlashModel now maps ALL DFlash-2 tensors: selector
(`dflash.selector.{pred_codebook,succ_codebook,hidden_proj}.weight`) + per-layer dynamic convs
(`blk.N.dflash.{attn,mlp}_conv.{base,proj.weight}`) + metadata (selector_rank/top_k, conv_kernel/group_size).
Conversion SUCCEEDS → valid 81-tensor GGUF at `~/Desktop/Qwen3.8-27B-DFlash2-draft-f16.gguf` (3.6GB).
Changes UNCOMMITTED in the ROCmFPX checkout.

**Stage 2 boundary CHARACTERIZED (the real work):** loading the draft fails
`done_getting_tensors: wrong number of tensors; expected 81, got 58` — the DFlash-1 loader
(`llama_model_dflash` in src/llama-model.cpp) recognizes only 58 of the 81 tensors; our 23 DFlash-2
tensors (selector+conv) are unknown to it. AND the reference shows the dynamic conv is INSIDE each
layer's forward (attention_conv.prepare/finish wrapping attention, mlp_conv wrapping MLP) — DFlash-2
**restructured the backbone**, it is NOT a pure post-step. So Stage 2 = teach llama_model_dflash the
DFlash-2 tensor set AND add the grouped-dynamic-causal-conv into the per-layer graph
(GroupedDynamicCausalConv: `_grouped_dynamic_convolve` in dflash-ref/dflash/model.py L478-495) — this IS
real llama.cpp model-graph surgery. The selector CAN still be a CPU post-step (L2b), but the conv cannot.
Best as a ROCmFPX PR with Carlo (he owns the DFlash-1 graph/RoPE). This is where solo tinkering hits the
"needs the engine owner" wall honestly.

## Stages (each compiles + tests)
1. ✅ DONE — Converter emits selector+conv tensors+meta → conversion SUCCEEDS. Verified 81 tensors + 4 meta keys.
2. speculative.cpp loads the 3 tensors (log shapes). Test: server starts with `--spec-type draft-dflash`.
3. Implement the bilinear walk; wire into the draft pick. Test: correctness (draft matches ref on a
   fixed input) + acceptance vs DFlash-1/MTP.
4. Tune n_max (block_size=8), p_min for the 9060 XT; benchmark vs MTP+ngram (current: 0.72–0.96 accept).

## Coordinate with Carlo (charlie12345/ROCmFPX)
He owns the DFlash-1 runtime. This is ideal as a PR to ROCmFPX (his review de-risks the graph/RoPE
interactions; benefits all his users). Share this doc + the reference links. Draft kept at
~/models/dflash-src/zlab-draft-full/model.safetensors (3.6GB).
