# Distillation run — comprehensive plan & risk register

The goal is to spend the $70 MI300X credit ONCE and get a usable model, not to debug a
$197 mistake at 3am on a billed instance. This is the pre-flight. Read it fully before
starting the droplet.

---

## 1. The goal, stated precisely (so we know if we hit it)

Produce **`qwen38-distilled-a3b-Q2_0_ROCMFPX.gguf`**: a 2-bit MoE GGUF that
- runs on the RX 9060 XT (16GB) at escha-class speed (~40-60 tok/s real, 120+ burst),
- codes measurably better than the base Qwen3.6-35B-A3B (escha) on agentic tasks,
- reasons in SHORT chains (distilled from the 27B's thinking, not 32k-token stalls),
- is abliterated (complies on authorized security work).

**This is distillation INTO an existing MoE, NOT dense->MoE upcycling.** Confirmed correct
by Qwen Discord: upcycling is a full retrain; distilling into Qwen3.6-35B-A3B (already MoE)
is not. Student capacity ceiling = 3.6-A3B; we push it toward 3.8 ON the coding distribution.

### Execution stack (decided)
- **Base image:** AMD Developer Cloud "Unsloth Studio" or "PyTorch 2.10 (ROCm)" — ships ROCm
  torch pre-built, so we install NOTHING heavy. Removes the torch/ROCm install gamble.
- **Teacher gen:** vLLM (in image, or the "vLLM 0.27.1" image) — 67x cheaper than HF (R1).
- **Training:** **Unsloth SFT-distillation** (sequence-level) is PRIMARY — `03_unsloth_sft.py`.
  The custom logit-KL trainer (`03_distill_train.py`) is the fallback, run with `KL=1`.
  Unsloth is lower-risk: it auto-handles MoE LoRA targets + 4-bit on ROCm, and the teacher
  is NOT loaded during training (frees VRAM). See how it softens R2/R3/R4/R7 below.

## 2. Success criteria / definition of "not fucked up"

GO to full run only if the PoC clears ALL of these:
- [ ] teacher generation finished under budget (vLLM, not HF sequential — see Risk 1)
- [ ] distilled student loads and generates COHERENT code (not repetition/garbage)
- [ ] it calls tools with valid JSON (the thing that makes it agentic)
- [ ] it beats base 3.6-A3B on the 10-task held-out eval (Risk 6), even slightly
- [ ] total PoC spend < $30

If any fail: stop, fix, do NOT spend the remaining credit on a broken pipeline.

---

## 3. RISK REGISTER (ranked by how badly it wastes money/the run)

### R1 [BUDGET-ENDING] Teacher generation must be vLLM-batched, not HF sequential
- **Impact:** HF `generate()` one-prompt-at-a-time = 98 hr = $197 for the PoC alone. 2.8x
  over the entire budget. This was in the first draft of 02_teacher_generate.py.
- **Fix:** 02 now uses vLLM (ROCm build) for batched generation -> 1.5 hr = $3. HF path
  kept only as a tiny-N fallback with a loud cost warning.
- **Verify on node:** `pip show vllm` succeeds; the script prints "using vLLM" not "HF fallback".

### R2 [SOFTENED by Unsloth SFT] Tokenizer alignment
- With Unsloth SFT (primary) there is NO logit-KL, so exact tokenizer id-match is not
  load-bearing — the student just learns text. Same tokenizer family is still wanted.
- Only relevant if you switch to the logit-KL trainer (`KL=1`), which keeps the probe-string
  id-match guard that aborts in seconds on mismatch.

### R3 [SOFTENED by Unsloth] LoRA targets on the MoE
- Unsloth auto-selects the correct LoRA target modules for the Qwen MoE arch. The fallback
  trainer auto-detects them too and aborts if 0 trainable params.
- **Verify:** "trainable params: X (Y%)" with Y ~0.1-2%. If 0% -> stop.

### R4 [SOFTENED by Unsloth/image] 4-bit on ROCm/gfx942
- Unsloth's 4-bit path is tuned for ROCm and ships in the AMD Unsloth image. The fallback
  trainer tries bnb 4-bit and auto-drops to bf16 LoRA (192GB has room) if it fails.
- **Verify:** training starts and loss decreases; if you see "bf16 LoRA", that's fine.

### R5 [LOSE-BILLED-HOURS] No resume on crash
- **Impact:** OOM, disconnect, or preemption mid-train wastes every hour spent so far.
- **Fix:** 03 saves stepN checkpoints AND accepts --resume to continue from the latest.
  Run inside `tmux` so an SSH drop doesn't kill it.
- **Verify:** checkpoints appear in student-distilled/stepN during training.

### R6 [CANT-JUDGE] Weak validation
- **Impact:** "it generated something" is not proof it's better. Spending the full run
  blind is the classic waste.
- **Fix:** eval.py runs a 10-task held-out coding/tool eval on BOTH the distilled
  student and the base 3.6-A3B, prints a side-by-side. GO only if distilled >= base.

### R7 [REMOVED by Unsloth SFT] Full-vocab KL memory
- Gone on the primary path: SFT has no teacher logits and no teacher in memory during
  training. Only applies if you run `KL=1` (then keep batch=1, grad_accum=16, seq_len<=4096).

### R8 [GARBAGE-IN] Corpus quality
- **Impact:** the student is only as good as what the teacher was asked. Padded/synthetic
  prompts -> a student good at nothing.
- **Fix:** weight the corpus toward REAL tasks: SWE-bench statements + your own uploaded
  coding/security session transcripts. 8k good prompts beat 30k padded. Inspect prompts.jsonl
  before step 2.

### R9 [WON'T-CONVERT] Trained student -> GGUF
- **Impact:** if LoRA, the adapter must be MERGED into the base before convert_hf_to_gguf,
  or the GGUF is just the base model (no distillation).
- **Fix:** `node/merge_lora.py` merges on the node (192GB, trivial) BEFORE download, so what
  comes home is a real model. `local/convert_quant.sh` converts the merged+abliterated bf16.

### R10 [CANT-ABLITERATE-LOCALLY] 70GB model on a 56GB machine
- **Impact:** the merged bf16 A3B is ~54-70GB. This machine has 16GB VRAM + 40GB RAM = 56GB.
  A naive load OOMs; abliteration can't run.
- **Fix:** `local/abliterate.sh` streams the model through the NVMe (`--offload_folder`), or
  uses Abliterix (which made your v2 and handles big models). Slow (hours, overnight) but
  works. Needs ~80GB free on a fast drive (E: has it). Verify compliance before quantizing.

### R11 [DOWNLOAD-COST] pulling 70GB home
- **Impact:** downloading the merged bf16 over the rental's network takes time you're billed
  for if you do it while the droplet runs interactively.
- **Fix:** `scp` runs from YOUR machine pulling FROM the node - the node just serves the file.
  Still, stop the droplet the instant scp completes. ~70GB at typical speeds ≈ 20-40 min.

---

## 4. Phased execution with GO/NO-GO gates

Training + eval on the RENTED MI300X (`node/`). Abliteration + final quant at HOME (`local/`).

```
PHASE 0  (free, droplet OFF)   pre-flight checklist below
   |  gate: HF token ready, corpus inspected, this plan read
PHASE 1  (~$5, MI300X ON)      node/run_node.sh --smoke     (n=50, full pipeline)
   |  gate: vLLM works, tokenizer id-match, LoRA params > 0, 1 train step, no OOM
PHASE 2  (~$25)                node/run_node.sh --poc       (8k, 1 epoch) + eval gate
   |  gate: distilled >= base on eval, coherent code, tools work, spend < $30
PHASE 3  (~$40)                node/run_node.sh --full      (30k, 2 epoch) -> merged bf16
   |  gate: eval improved further
PHASE 4  (free, home)          scp merged bf16 home; STOP THE DROPLET
PHASE 5  (free/overnight, local) local/abliterate.sh -> local/convert_quant.sh -> serve
```

**The Phase-1 smoke test (--smoke, ~$5) is the single most important cost-saver.** It runs
the ENTIRE node pipeline end-to-end on 50 prompts in ~15 min. Every R1-R9 failure surfaces
here for $5 instead of at scale for $50. NEVER skip it.

**Abliteration is Phase 5, LOCAL, off the clock.** It runs on this machine (Abliterix or the
bundled abliterate.py with NVMe offload — a 70GB model does not fit 56GB, so it streams).
Kept off AMD's cloud on purpose: ToS-clean, private, and it's how your v2 was already made.

## 5. Cost model (honest)

| phase | what | ~time | ~cost |
|---|---|---|---|
| 1 smoke | full pipeline, n=50 | 15 min | $5 |
| 2 PoC | 8k prompts, 1 epoch, eval | 8-12 hr | $20-25 |
| 3 full | 30k prompts, 2 epoch, abliterate | 15-20 hr | $30-40 |
| idle waste | forgetting to stop the droplet | — | $1.99/hr, AVOID |
| **total** | | | **~$55-70** — fits the credit |

Buffer: if the full run risks overrunning, stop after Phase 2 with a shippable PoC model
and do the full run later. A PoC-quality distilled model is already a real deliverable.

## 6. Pre-flight checklist (do with droplet OFF, free)

- [ ] `HF_TOKEN` created, has access to Qwen3.8-27B and Qwen3.6-35B-A3B (accept licenses)
- [ ] transcripts zipped: `tar czf my_transcripts.tgz -C ~/.claude/projects .`
- [ ] read RENTAL.md, understand stop-the-droplet billing
- [ ] decide student: default Qwen3.6-35B-A3B (escha base, proven fast). Don't change without reason.
- [ ] `tmux` habit confirmed (run survives SSH drop)
- [ ] this risk register understood — especially R1 (vLLM) and the Phase-1 smoke gate

## 7. Known unknowns (only the node resolves these)

- vLLM ROCm wheel availability for this exact image (fallback: HF with small N + warning)
- bnb 4-bit on gfx942 (fallback: bf16 LoRA, fits 192GB)
- exact teacher throughput (affects Phase-2 cost; the smoke test measures it before you commit)

None of these are blockers — each has a fallback. The smoke test surfaces all three for $5.
