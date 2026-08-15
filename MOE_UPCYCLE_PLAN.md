# Dense→MoE upcycling plan — Qwen3.8-27B-ABLITERATED

Supersedes the decision in `PLAN.md` §1 ("distillation INTO an existing MoE, NOT dense->MoE
upcycling"). That call was correct for a $70 budget. This document is what the reversal
actually costs, what it can and cannot produce, and how to do it if you go ahead.

Read §1 and §2 before spending anything. They change the goal.

---

## 1. The headline: "A3B" is arithmetically impossible from this backbone

In any MoE, attention and the output head are **always active** — sparsity only applies to
the FFN. For `Qwen3.8-27B` (`hidden_size` 5120, 64 layers, `intermediate_size` 17408,
`vocab_size` 248320, untied embeddings):

| component | params | active per token |
|---|---:|---|
| FFN (all 64 layers) | 17.11 B | sparsifiable |
| attention (16 full + 48 linear/DeltaNet) | ~4.35 B | **always** |
| output head (248320 × 5120) | 1.27 B | **always** |
| input embedding | 1.27 B | lookup only |
| vision tower (27 layers) | ~0.5 B | droppable |

**Always-active floor ≈ 5.62 B.** You cannot reach 3B active from this model at any
expert count or top-k. The floor alone is nearly twice the target.

What you *can* reach by splitting the FFN:

| split | active FFN | **active total** | speedup vs dense |
|---|---:|---:|---:|
| 32 experts, top-4 | 2.14 B | 7.76 B | 2.9× |
| 64 experts, top-8 | 2.14 B | 7.76 B | 2.9× |
| 128 experts, top-8 | 1.07 B | 6.69 B | 3.4× |

**The achievable target is `Qwen3.8-27B-A7.8B`** — 27B total, ~7.8B active, ~3× faster
than the dense 27B. That is a real and useful result. It is not an A3B.

### Why the existing A3B is a different shape entirely

`Qwen3.6-35B-A3B` reaches 3B active by having a **much smaller backbone**, not by better
sparsity: `hidden_size` 2048 (vs 5120), 40 layers (vs 64), 2 KV heads (vs 4),
`moe_intermediate_size` 512 × 256 experts. Upcycling preserves `hidden_size` and layer
count — it cannot transform a 5120-wide, 64-layer model into a 2048-wide, 40-layer one.
Reaching A3B from the 27B would mean *also* distilling into a narrower backbone, which is
the pipeline you already have.

### 1a. Why this is the *only* route to escha-class speed on the RX 9060 XT

Local decode speed is **memory-bandwidth bound**, not compute bound: every token, the GPU
reads the active weights from VRAM, so `tok/s ≈ bandwidth ÷ bytes-read-per-token`. The card
has ~320 GB/s (128-bit GDDR6); real sustained decode is ~65% of peak.

| model | active | reads/token @ Q2 | real tok/s |
|---|---:|---:|---:|
| dense 27B (vision dropped) | 24 B | 7.5 GB | **~28** |
| dense 27B @ Q3 | 24 B | 10.5 GB | ~20 |
| **upcycled 27B-A7.8B** | 7.8 B | 2.4 GB | **~85** |
| escha A3B (existing) | 3 B | 0.9 GB | ~220 ceiling |

**Escha is fast because it is A3B, not because it is well-tuned.** It reads 0.9 GB/token;
the dense 27B reads 7.5 GB/token — 8× more — and hits a hard ~28 tok/s wall. Sustaining
100 tok/s requires reading ≤2 GB/token; dense 27B @ Q2 overshoots that by **3.6×**. No
quant, kernel, or driver work closes a 3.6× bandwidth gap (it would need ~1-bit weights).

The only lever that cuts bytes-per-token while keeping the 27B's knowledge is **cutting
active params** — i.e. this MoE conversion. "Make the dense 27B fast" and "upcycle it to
MoE" are therefore the same task. At A7.8B the read drops to 2.4 GB/token → **~85 tok/s
sustained, bursting past 100 with the MTP head.** That is as close to escha's speed as the
27B's weight footprint physically allows; a hard 100+ floor only exists in the ~3B-active
class (the existing A3B distill), at its lower quality ceiling.

The one dense-only engineering lever — **speculative decoding** (a small draft model the
27B verifies in batches, ~2–2.5× on predictable code) — is blocked here: the draft must
share the target tokenizer, Qwen3.8-27B's vocab is 248320, and the local `Qwen3-0.6B` is
the old 151k vocab. No Qwen3.8-vocab tiny model is published, so it would have to be
distilled first — a separate project, and still short of sustained 100 against the 8× gap.

---

## 2. The paper does the opposite of what you want

`arXiv:2212.05055` (Komatsuzaki et al., *Sparse Upcycling*) copies each dense MLP into
N **full-size** experts and routes top-k. Consequences:

- **Total params multiply** — 27B + (N−1) × 17.11B. At N=8 that is ~147B.
- **Active params go up, not down** — top-2 of full-size experts = 2× the dense FFN cost.
- The payoff is quality-per-training-token, **not inference speed.**

The paper's own headline number is that upcycled models beat dense using **~50% of the
original dense pretraining sunk cost.** Qwen3.8-27B's pretraining is on the order of tens
of trillions of tokens; 50% of that is not a number you can buy.

**What you actually want is expert *splitting*** (MoEfication / LLaMA-MoE lineage): partition
each 17408-wide FFN into N narrow experts of 17408/N, route top-k. Total params stay ~27B,
active params drop, inference gets faster. This is the technique in the table above.

Splitting is the *harder* of the two to recover from: upcycling starts with every expert
a perfect copy of a trained FFN, while splitting starts with every expert a **fragment** of
one. Each token now sees only k/N of the computation the dense model was trained to use.
Quality drops sharply on init and recovery training is doing real work, not fine-tuning.

This is what your Qwen Discord answer meant by "recalibrate the entire model." It is correct.

---

## 3. Compute budget — the actual blocker

Assumptions, all stated so you can check them: 7.76B active params, `6 × N_active × tokens`,
MI300X bf16 peak 1307 TFLOPS at **38% MFU ≈ 497 TFLOPS/GPU** (optimistic for a hybrid
DeltaNet MoE with expert-parallel routing on ROCm), $1.99/GPU-hr.

| tokens | 1× MI300X | 8× MI300X | cost | what you get |
|---:|---:|---:|---:|---|
| 1 B | 1.1 days | 3 hrs | **$52** | router settles; still badly degraded |
| 10 B | 10.9 days | 1.4 days | **$518** | partial recovery, clearly below dense 27B |
| 100 B | 108 days | 13.6 days | **$5,182** | plausibly near-dense on coding |
| 300 B | 326 days | 41 days | **$15,546** | approaching "worth it" |

Your remaining credit is ~$70. **The minimum viable recovery run is ~2 orders of magnitude
past your budget**, and the realistic one is ~$5K. There is no configuration of this plan
that fits the current budget — that is the finding, not a pessimistic framing of it.

Cost scales with *active* params, so the 3.4× split (6.69B active) is ~14% cheaper to
train and faster at inference — but every expert is narrower, so it starts further from
the dense model and needs more tokens. These roughly cancel. Don't optimize here.

---

## 4. The second blocker: you do not have the data

Recovery training is **pretraining-shaped**, not SFT-shaped: broad, diverse, next-token
prediction over raw text. It is not instruction pairs.

- `01_build_corpus.py` produces 8K–30K *prompts* ≈ **~10–50M tokens**.
- The 10B-token floor needs **200–1000× more data than the entire pipeline produces.**

You would be pulling FineWeb-Edu, DCLM-baseline, The Stack v2, and Nemotron-CC and building
a mixture — a data-engineering project in its own right, plus multi-TB of storage and
egress on a billed instance. Distillation targets from the dense teacher can supplement the
tail of the run to bias it toward your coding/agentic distribution, but they cannot carry it.

---

## 5. If you proceed: phased plan with kill-gates

Every phase ends in a gate. Blowing a gate means stop, not "push through."

### Phase 0 — offline, free (droplet down)
Nothing here needs a GPU. Do all of it before relaunching anything.

1. **Get the trainable weights.** `Qwen/Qwen3.8-27B` (base, safetensors, ~54GB) — the
   pristine pretrained checkpoint, gated (needs `HF_TOKEN`). Train from this, **not** the
   abliterated variant: abliteration is a directional weight edit that recovery training
   would partly undo anyway (R-U4), so applying it before training wastes the edit. The
   GGUF repos cannot be trained from at all — GGUF is quantized, fused, gradient-free.
   `Blackfrost-AI/…-ABLITERATED-GGUF` stays a **behavioral reference** (target: 11/450
   residual refusals) — its method is undocumented and not reproducible, so we match its
   *result* with our own tool, not its steps.
2. **Write the conversion script** — **DONE: `node/05_upcycle.py`.** Emits the Qwen3.8 MoE
   config (class id `qwen3_5_moe_text` / `Qwen3_5MoeForCausalLM` — Qwen's own code line for
   the 3.8 family; the base config is stamped `qwen3_5_text`, so this string must stay to
   load) with `hidden_size` 5120, `num_hidden_layers` 64, `num_experts` 64,
   `moe_intermediate_size` 272, `num_experts_per_tok` 8. Weight mapping:
   - attention, norms, embeddings, output head → **copied verbatim**
   - each `gate_proj`/`up_proj`/`down_proj` → **fused into experts** `experts.gate_up_proj`
     `[E,2m,H]` + `experts.down_proj` `[E,H,m]`, sliced along the intermediate axis
     (contiguous by default — see §6 R-U2; `--cluster coactivation` for the real run)
   - router `mlp.gate.weight` + shared expert (`shared_expert.*`, `shared_expert_gate`) →
     **random/zero init**, the parts that must be learned
   - `mtp` layer → **random init** at Phase 3
   - vision tower → dropped (`--drop-vision`, targeting a text coder)
   The script **streams shard-by-shard**, so conversion runs on this 40GB-RAM box despite
   the 54GB checkpoint. It asserts **bit-exact FFN tiling** before writing (R-U5).
3. **Gate 0a — local, free:** `python node/05_upcycle.py --verify --src … --out …` asserts
   every FFN reconstructs bit-exactly from its expert shards. Runs here; needs no GPU and no
   full model load. This is the mapping-correctness gate.
   **Gate 0b — on the node:** `--forward-check` instantiates the model and runs one forward
   pass (finite logits, correct shape). This needs GPU + >54GB RAM, so it runs on the droplet
   at the *start* of Phase 1, not locally — the 27B won't fit in 40GB RAM for a forward pass.
4. **Build the data mixture spec** — sources, weights, token counts, dedup plan. On paper.
5. **Baseline the dense 27B** on your eval so "recovered" has a number attached.

### Phase 1 — sanity run (~$50, 1 GPU, ~1 day)
Convert, then train on ~1B tokens.
- Watch: loss must fall steeply from the post-split spike; router entropy must drop from
  uniform without collapsing to one expert; expert load balance within ~3× of even.
- **Gate 1:** loss trending to within ~15% of the dense model's on held-out text, and no
  expert receiving <1% or >20% of tokens. If the router has collapsed, the split strategy is
  wrong (§6 R2) — fix it here, where it costs $50, not at $5K.

### Phase 2 — recovery run ($500 → $5,000)
Scale tokens. Checkpoint every ~500 steps; these runs die.
- **Gate 2 at 10B tokens:** if it is not clearly closing on the dense baseline, stop. The
  remaining $4.5K will not rescue a bad init.

### Phase 3 — MTP head
`mtp_num_hidden_layers: 1` — one extra layer (~340M params at this width) predicting t+2,
**randomly initialized**. Train it *after* the backbone stabilizes, with the backbone frozen
or at low LR; training it during recovery adds a noisy auxiliary loss to an already unstable
optimization. This is the cheapest phase and the only one that behaves like normal training.

### Phase 4a — abliterate (on the node, cheap — forward passes + rank-1 edit)
Run `local/abliterate.py` on the recovered MoE checkpoint. This is the *first* abliteration
applied to these weights (base was pristine), so nothing was wasted upstream. Gate against
`eval.py`'s security-compliance set; target Blackfrost's 11/450 residual-refusal result.

### Phase 4b — local (free)
`convert_hf_to_gguf.py` → `llama-quantize`. At 27B total, Q2_0_ROCMFPX lands ~9-10GB and
fits 16GB comfortably. **Risk:** llama.cpp/ROCmFPX must handle the Qwen3.8 MoE config (class
id `qwen3_5_moe_text`) with
non-standard dims. Dims come from GGUF metadata, so it *should* — verify at Phase 1 with a
throwaway conversion of the 1B-token checkpoint, not at the end.

*(Phase-letter note: abliteration is 4a because it edits weights and must precede the GGUF
conversion in 4b; both are downstream of a recovered backbone.)*

---

## 6. Risk register (additions to `PLAN.md` §3)

**R-U1 — budget is off by 100×. [RUN-ENDING]**
$70 remaining vs a $518 floor and a $5,182 realistic run. No mitigation exists inside the
current budget. This is a funding decision, not an engineering one.

**R-U2 — contiguous FFN slicing produces a broken router. [HIGH]**
Neurons in a dense FFN are not grouped by function, so slicing `[0:272], [272:544], …`
makes 64 experts that are each an arbitrary fragment — none individually useful, so the
router has no signal to learn from. Mitigation: cluster neurons by co-activation on a
sample of your corpus (the MoEfication approach) and group correlated neurons into the
same expert. Costs a few GPU-hours; it is the difference between Gate 1 passing and failing.

**R-U3 — hybrid DeltaNet attention is uncharted for this. [HIGH]**
Every upcycling/splitting result in the literature is on standard dense transformers. This
model is 48/64 layers linear attention with gated DeltaNet. Whether recovery dynamics hold
is genuinely unknown. Phase 1 is the only way to find out.

**R-U4 — abliteration ordering. [RESOLVED by training from the base]**
Abliteration is a weight-level directional edit that general-corpus recovery training can
partially restore. Resolved by training from the *pristine* `Qwen/Qwen3.8-27B` and
abliterating **last** (Phase 4a, `local/abliterate.py`), so no edit is spent before the
training that would undo it — the same "abliterate after training" rule as `PLAN.md`.
Gate: `eval.py`'s security-compliance checks must pass post-abliteration, benchmarked
against Blackfrost's 11/450 residual-refusal result as the behavioral target.

**R-U5 — no reference implementation. [MEDIUM]**
There is no published dense→MoE conversion script for the Qwen3.8 (`qwen3_5`-class) line.
`05_upcycle.py` is new
code operating on 54GB of weights, and a silent mapping error looks exactly like "the model
needs more training." Mitigation: Gate 0 asserts per-tensor that concatenating the 64 expert
shards reproduces the original FFN weight bit-exactly.

**R-U6 — single-GPU wall-clock. [MEDIUM]**
108 days on one MI300X is not a viable schedule; a multi-GPU node is the same total cost but
needs expert-parallel sharding to work on ROCm. Budget setup time for that.

---

## 7. Recommendation

Do **Phase 0 and Gate 0** now. They are free, they run on this machine, and they produce
the one artifact that makes every later decision concrete: a converted checkpoint whose
shapes are proven correct. If `05_upcycle.py` can't produce a model that does a clean
forward pass, nothing downstream matters.

Then decide on Phase 1's $50 with real information.

Be clear-eyed that Phases 2+ are a ~$5K project with a genuinely uncertain outcome on an
architecture nobody has published results for. The honest comparison:

| path | cost | outcome |
|---|---:|---|
| **Run the dense 27B as-is** (Q3_K_M, 13.3GB) | $0 | full 27B quality, ~30-50 tok/s, vision works |
| Fix the existing A3B distill (corpus rebuild) | ~$30 | fast MoE, capacity capped at 3.6-A3B |
| **This plan** | ~$5,000 | 27B-A7.8B at ~3× dense speed, *if* recovery works |

The dense 27B already runs on your card today and gives you the quality you were chasing.
Upcycling buys inference speed — roughly 3× — for four figures and a research risk. That
trade can be worth it. It should be made deliberately.
