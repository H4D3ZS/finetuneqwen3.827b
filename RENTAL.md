# MI300X rental run — exact SSH workflow

AMD Developer Cloud, single MI300X (192GB, ROCm), $1.99/hr, ~$70 credit ≈ 35 hr.
You are billed **while the instance runs**, credit or not — stop it the moment you are done.

## Golden rule
**Every minute the droplet is ON costs money.** Do prep (reading this, editing configs)
with the droplet OFF. Only start it when you are ready to run, and destroy/stop it the
second `scp` of the result finishes.

---

## 0. Before you start the droplet (free, do it now)
- Have your `HF_TOKEN` ready (huggingface.co/settings/tokens) — Qwen repos are gated.
- Decide the run: `--poc` first (~$25, validates quality) then `--full` (~$60). Never blind-spend the full run.
- (Optional but high-value) zip your own transcripts to enrich the corpus:
      # on the workstation - your own coding/security session transcripts
      tar czf my_transcripts.tgz -C "C:/Users/HADES/.claude/projects" .

## 1. Start the droplet, note its SSH address
From the AMD Developer Cloud console: launch the MI300X droplet, copy the `ssh` command
it gives you (e.g. `ssh user@129.x.x.x`). Clock is now running.

## 2. Get the pipeline + your data onto the node
From the workstation (Git Bash):
```bash
NODE=user@129.x.x.x            # from the console
scp -r ~/Desktop/finetuneqwen3.827b "$NODE":~/finetuneqwen3.827b
scp my_transcripts.tgz "$NODE":/scratch/                    # optional, enriches corpus
```

## 3. SSH in and run
```bash
ssh "$NODE"
mkdir -p /scratch/distill/my_transcripts
tar xzf /scratch/my_transcripts.tgz -C /scratch/distill/my_transcripts   # if uploaded
export HF_TOKEN=hf_xxxxxxxx
cd ~/finetuneqwen3.827b
tmux new -s run          # so the run survives an SSH disconnect - IMPORTANT
bash node/run_node.sh --poc    # ~10-15 hr, ~$20-30. Validate FIRST.
# detach from tmux with Ctrl-b then d; reconnect later with `tmux attach -t run`
```

## 4. Validate the PoC before spending more
When `--poc` finishes, sanity-check the distilled student on the node:
```bash
python 04_abliterate.py --model /scratch/distill/student-distilled --out /tmp/ab || true
# quick generation test - does it write real code + call tools?
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer; import torch
m="/scratch/distill/student-distilled"
tok=AutoTokenizer.from_pretrained(m); model=AutoModelForCausalLM.from_pretrained(m,torch_dtype=torch.bfloat16,device_map="auto")
ids=tok.apply_chat_template([{"role":"user","content":"Write a Python prime check with a docstring, then a bash one-liner to find SUID binaries."}],return_tensors="pt",add_generation_prompt=True).to(model.device)
print(tok.decode(model.generate(ids,max_new_tokens=300)[0][ids.shape[1]:],skip_special_tokens=True))
PY
```
If it writes clean code and complies with the security prompt → the thesis holds, run `--full`.
If it is worse than the base student → stop, we adjust the corpus/hyperparams before spending more.

## 5. Pull the result home, then STOP THE DROPLET
```bash
# from the workstation
scp -r "$NODE":/scratch/distill/student-abliterated ~/Desktop/student-final
```
Then in the AMD console: **stop / destroy the droplet immediately.** Confirm it shows stopped.
Billing ends when it stops, not when you disconnect SSH.

## 6. Finish on the workstation (free)
```bash
cd ~/Desktop/finetuneqwen3.827b
./local/convert_quant.sh ~/Desktop/student-final
# -> qwen38-distilled-a3b-Q2_0_ROCMFPX.gguf in ROCmFPX/
```
Serve with the ROCmFPX llama-server (the exact command is printed by
and a `model_tuning` entry mirroring escha's MTP flags), then:
```bash
local/convert_quant.sh when it finishes).
```

## Cost control cheatsheet
| action | ~cost |
|---|---|
| idle droplet, 1 hr | $1.99 (stop it!) |
| PoC run (8k, 1 epoch) | $20-30 |
| full run (30k, 2 epoch) | $50-80 |
| teacher generation (step 2) | the long pole — most of the cost |

To cut cost if the teacher step is slow: lower `--n`, or in `02_teacher_generate.py` raise
batch/lower `--max_new`. The corpus quality matters more than its size — 8k good prompts
beats 30k padded ones.
