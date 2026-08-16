# Smoke test — run this FIRST on the MI300X

Goal: prove the whole abliterated-teacher distillation pipeline runs end-to-end on **50
prompts / 1 epoch** (~$5, ~20–30 min) *before* spending real money on `--poc` / `--full`.
It validates every seam — corpus → teacher serving → generation → SFT → merge → eval — at
toy scale. If it passes, the only thing that changes at scale is `N` and `EPOCHS`.

## One command

```bash
cd ~/finetuneqwen3.827b/node
bash run_node.sh --smoke        # N=50, EPOCHS=1, CONC=4, TEACHER_MODE=gguf, USE_EMPERO=1
```

That's it. It builds the ROCmFPX fork (HIP/gfx942), serves the abliterated teacher, generates
targets, **also** runs the Empero teacher on the general slice, distills the merged data into
the A3B student, merges, and evals.

## Two-teacher blend (why there are two teacher passes)

The student learns from **two** teachers, split by prompt sensitivity (tagged `route=` by
`01_build_corpus.py`):

- **Abliterated Qwen3.8-27B** (`serve_teacher.sh`, :8081) → the **sensitive** slice
  (offensive-security / agentic / cybersec). Uncensored — this is the differentiator.
- **Empero Qwen3.8-9B** (`serve_empero.sh`, :8082) → the **general** slice (math/code/reasoning).
  Apache-2.0, distilled from the 2.4T-A95B frontier teacher. It is **censored**, so it *only*
  ever sees `general` prompts — never an offensive one, or its refusals would re-poison the
  compliance we distil from the abliterated teacher.

Both teacher sets land under `teacher_data/` (Empero in `teacher_data/empero/`) and the SFT step
merges them automatically. **Turn Empero off** with `USE_EMPERO=0 bash run_node.sh --smoke` —
then the abliterated teacher covers `--route all` so the general half still gets targets.

> **First run is heavier than the $5 tag.** With `USE_EMPERO=1`, the *first* run builds **two**
> llama.cpp trees (the fork for the 27B, **plus stock master** for Empero — its Qwen3.5 Gated
> DeltaNet arch will NOT load on the fork) and downloads **two** GGUFs (~29GB + ~10GB). Budget
> extra time/credit for that once; subsequent runs reuse both. Want a fast first shakedown?
> `USE_EMPERO=0 bash run_node.sh --smoke` validates the core path, then flip Empero on for `--poc`.

## What must be true before you start

- **On the MI300X**, in the Unsloth Studio ROCm image (ROCm 7.2.4). `python3` has torch+ROCm.
- `torch.cuda.is_available()` is True. `run_node.sh` guards this and re-checks after every
  `pip install` — **never** `pip install vllm` (its CUDA wheel clobbers ROCm torch; that
  killed smoke run 1). The pipeline needs no vLLM: the teacher is a llama.cpp server process.
- `hf` (huggingface_hub CLI) works and you're logged in (`hf auth whoami`). The teacher
  download falls back to `huggingface-cli` automatically if `hf` is absent.
- `/scratch` has ~40 GB free (teacher Q8_0 ≈ 29 GB + student shards).

## The seams it exercises — and the bugs already fixed (2026-08-16)

These were fixed in the repo before handing you this brief; listed so you know what to
re-verify if a step fails:

1. **Port + arg alignment.** `serve_teacher.sh` serves on **:8081**; `run_node.sh` now polls
   `:8081/health` and calls `02_teacher_gguf.py --url http://127.0.0.1:8081/v1/chat/completions`
   (was `--base-url …:8080`, which would crash + hang the health wait 25 min).
2. **Teacher binary path.** `run_node.sh` now `export LCPP="$PWD/ROCmFPX"` so `serve_teacher.sh`
   finds the fork it just built (its default was `/root/llama.cpp`).
3. **Teacher weights.** `run_node.sh` now downloads `Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF`
   (`*Q8_0*.gguf`) into `/scratch/teacher/` if missing — `serve_teacher.sh` only *checks*, never
   downloads. If the repo's filename doesn't match `*Q8_0*`, adjust the `--include` glob.

## Watch these while it runs

- **`$SCRATCH/teacher-serve.log`** — the teacher must reach `server is listening` and its
  `/health` must return 200. If it never comes up, the run exits with a pointer to this log.
- **Step 2b throughput** — `02_teacher_gguf.py` prints `N/50  rate/s  eta`. With `--parallel 16`
  on the teacher and `--concurrency 4` here, 50 short completions should finish in a couple min.
  `stream.jsonl` is fsync'd per completion, so a kill mid-run loses nothing and re-runs resume.
- **Step 3 SFT** — expect either `student: Unsloth FastLanguageModel (4-bit QLoRA)` or the
  `transformers+peft fallback` line. Either is fine; the LoRA adapter is identical. Loss should
  descend across the ~a-dozen steps. On 50 prompts this is seconds-to-minutes, not hours.
- **Step 4/5 merge + eval** — `eval.py` runs the 10-task GO/NO-GO gate (incl. the security
  tasks). On 50 prompts don't expect a *pass* — expect it to **run without erroring** and print
  scores. Quality is a `--poc`/`--full` question.

## Pass / fail

- **PASS** = the script prints `== SMOKE PASSED end-to-end on 50 prompts` and exits 0. Every
  seam works; scale up with `bash run_node.sh --poc` (N=8000, ~$25) then `--full` (N=30000, ~$40).
- **FAIL** = report per CLAUDE.md rules: which step, the exact error, and the relevant log tail
  (`teacher-serve.log` for serving, stderr for python steps). Don't scale up until smoke is green.

## After a green `--full`

The student is **already abliteration-transferred** (the teacher was abliterated), so on
`TEACHER_MODE=gguf` you **skip** the local abliterate step. Pull `student-merged/` home,
quantize to `Q2_0_ROCMFPX` with the fork's `llama-quantize`, and serve via the RX 9060 XT
launcher (`lemonade-claude.sh -m <id>`). See the tail of `run_node.sh` for the exact commands.
