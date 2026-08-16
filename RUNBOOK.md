# RUNBOOK — build your own escha-class frontier-seeded coder

The current, actual plan (supersedes the teacher-gen parts of `RENTAL.md`). The big change:
**teacher generation moved OFF the MI300X onto free ModelScope APIs**, so the droplet is
rented only for the ~1-hour training step. Ambassador API + local RX 9060 XT do the rest.

## The architecture

| stage | where | teacher / tool | cost | status |
|---|---|---|---|---|
| 1. corpus (100k prompts) | local | 4 code-instruct datasets | $0 | ✅ done |
| 2a. frontier seed (5.2k) | ModelScope API | **Qwen3.8-Max** (5,500/mo cap) | $0 | ⏳ generating |
| 2b. bulk (94.8k) | ModelScope API | **Qwen3-235B-A22B** (no monthly cap) | $0 | ⏳ generating |
| 3. distill (LoRA SFT) | **MI300X** | base Qwen3.6-35B-A3B student | ~$5-25 | ⬜ after gen |
| 4a. abliterate | local | `local/abliterate.py` | $0 | ⬜ |
| 4b. quantize (2-bit) | local | `local/convert_quant.sh` (ROCmFPX) | $0 | ⬜ |

Teacher choice rationale: Qwen3.8-Max is frontier but quota-capped (5,500 req/**month**), so
it seeds only the ~5.2k HARDEST prompts. Qwen3-235B-A22B has no monthly cap and is
near-frontier — it carries the 94.8k bulk. Both ≫ the local 27B; neither needs a GPU.

## Monitor / resume the running generation (safe to close the terminal)

Both runs are **resumable** — rerun the same command and it skips finished prompts.

```bash
# progress
wc -l node/teacher_seed_max/*.jsonl   # seed, target 5200
wc -l node/teacher_bulk_235b/*.jsonl  # bulk, target 94800

# resume if interrupted (export the key first; NEVER commit it)
export MODELSCOPE_API_KEY=ms-xxxx
python node/02b_teacher_api.py --format openai \
  --base-url https://api-inference.modelscope.ai/v1 --api-key-env MODELSCOPE_API_KEY \
  --model Qwen/Qwen3-235B-A22B-Instruct-2507 \
  --prompts node/prompts_bulk.jsonl --out node/teacher_bulk_235b --concurrency 10
# seed: same, with --model Qwen-Ambassador/Qwen3.8-Max --prompts node/prompts_seed.jsonl
#       --out node/teacher_seed_max --max-requests 5250   (respects the monthly cap)
```

Bulk ETA ≈ 2-3.5 days at concurrency 10 (~19/min measured; raise concurrency if no 429s).
Seed ETA ≈ 3 hours.

## Step 3 — train on the MI300X (only when 2a+2b are done)

Rent the droplet ONLY now (generation is finished). See `RENTAL.md` for SSH basics.

```bash
# workstation: ship code + the generated data (NOT the whole HF cache)
tar czf teacher_data.tgz -C node teacher_seed_max teacher_bulk_235b
NODE=user@129.x.x.x
scp -r ~/Desktop/finetuneqwen3.827b "$NODE":~/finetuneqwen3.827b
scp teacher_data.tgz "$NODE":~/finetuneqwen3.827b/node/
# node:
ssh "$NODE"
cd ~/finetuneqwen3.827b/node && tar xzf teacher_data.tgz
tmux new -s train        # survive disconnect
bash train_only.sh       # ~1-2 hr wall (compute is minutes; setup dominates)
```

Training time scales with corpus size but stays cheap: ~100k pairs ≈ ~1.5-2 hr ≈ **~$4**.
Gate: `eval.py` must show it beats base Qwen3.6-35B-A3B on coding. If not, stop and inspect
the corpus before quantizing.

## Step 4 — finish locally (free), then STOP THE DROPLET

```bash
scp -r "$NODE":/scratch/distill/student-distilled-merged ~/Desktop/student-final
# then in the AMD console: STOP/DESTROY the droplet. Billing ends on stop, not disconnect.

python local/abliterate.py --model ~/Desktop/student-final --out ~/Desktop/student-ablit
ROCMFPX_DIR=~/Desktop/ROCmFPX ./local/convert_quant.sh ~/Desktop/student-ablit
# -> qwen38-distilled-a3b-abliterated-Q2_0_ROCMFPX.gguf ; serve like local/serve_distilled.sh
```

Result: **your own escha-class A3B** — ~3B active, escha-speed on the 16GB card, distilled
from a Qwen3.8-Max frontier seed + Qwen3-235B bulk, abliterated. Point Claude Code at it via
the LiteLLM gateway (`qwen38-distill-a3b-coder`).

## Honest ceilings (so the result isn't a surprise)

- A 3B-active student **imitates** frontier coding on the covered distribution; it does not
  *become* frontier. Expect: clearly better than base escha, near-frontier on common coding,
  not Opus-parity. That is the physics of 3B active (see `MOE_UPCYCLE_PLAN.md` §1a).
- Your real transcript tasks are **not** in this corpus — they are agentic (need file/repo
  context) and a tool-less teacher just refuses them. Personalizing on them needs SWE-bench-
  style trajectory reconstruction: a separate future pass.
