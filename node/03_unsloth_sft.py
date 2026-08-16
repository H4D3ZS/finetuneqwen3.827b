#!/usr/bin/env python3
"""
Step 3 (PRIMARY, Unsloth path): sequence-level distillation via SFT.

The teacher (step 2, vLLM) already produced {prompt, completion} pairs WITH thinking. Here
the student simply learns to reproduce them - "sequence-level knowledge distillation". This
is the low-risk path vs the custom logit-KL trainer (03_distill_train.py):
  - no teacher in memory during training (freed VRAM; kills risk R7)
  - no logit alignment, so tokenizer id-match is not load-bearing (softens R2)
  - Unsloth auto-handles LoRA targets on the Qwen MoE + 4-bit on ROCm (softens R3, R4)
  - runs on AMD's supported Unsloth/PyTorch ROCm image - no custom ROCm gymnastics

It prefers Unsloth's FastLanguageModel (fast + memory-light); if Unsloth doesn't support the
exact MoE arch on this image, it falls back to plain transformers+peft+TRL SFT, which is
architecturally identical, just slower. Either way the OUTPUT is the same LoRA adapter.

    python 03_unsloth_sft.py --student Qwen/Qwen3.6-35B-A3B \
        --data teacher_data/ --out student-distilled/ --resume

Then: merge_lora.py -> eval.py -> (home) abliterate -> quant.
"""
import argparse, os, json, glob

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--student", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--data", required=True, help="dir of {prompt, completion} jsonl from step 2")
    p.add_argument("--out", default="student-distilled")
    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)   # SFT LoRA likes a higher lr than KL
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no_unsloth", action="store_true", help="force the transformers+peft fallback")
    return p.parse_args()

def load_pairs(data_dir):
    # Recursive so --data can point at a parent (e.g. node/) and gather every teacher_* dir's
    # shards at once (teacher_seed_max/ + teacher_bulk_235b/). Extra fields like {id} are fine.
    rows, files = [], sorted(set(
        glob.glob(os.path.join(data_dir, "*.jsonl")) +
        glob.glob(os.path.join(data_dir, "**", "*.jsonl"), recursive=True)))
    for fn in files:
        with open(fn, encoding="utf-8") as f:
            for line in f:
                try: r = json.loads(line)
                except Exception: continue
                if r.get("prompt") and r.get("completion"):
                    rows.append(r)
    if not rows:
        raise SystemExit(f"no {{prompt,completion}} rows under {data_dir}. Run step 2/2b first.")
    print(f"{len(rows)} training pairs from {len(files)} shard files")
    return rows

def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    rows = load_pairs(a.data)
    from datasets import Dataset
    # Keep prompt/completion SEPARATE so trl (>=0.20) can mask the prompt tokens itself via
    # SFTConfig(completion_only_loss=True). The old DataCollatorForCompletionOnlyLM was removed
    # in trl 0.24, and its marker approach would mask nothing here anyway (no chat markers in
    # the raw text). Loss is computed only on the teacher's reasoning+answer -> what we distill.
    ds = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in rows])

    model = tok = None
    used = "unsloth"
    if not a.no_unsloth:
        try:
            from unsloth import FastLanguageModel
            model, tok = FastLanguageModel.from_pretrained(
                a.student, max_seq_length=a.seq_len, load_in_4bit=True, dtype=None)
            model = FastLanguageModel.get_peft_model(
                model, r=32, lora_alpha=64, lora_dropout=0.0, bias="none",
                use_gradient_checkpointing="unsloth", random_state=0,
                # Unsloth auto-picks the right target modules for the arch; this list is a hint.
                target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
            print("student: Unsloth FastLanguageModel (4-bit QLoRA)")
        except Exception as e:
            print(f"Unsloth path unavailable ({type(e).__name__}: {e}) -> transformers+peft fallback")
            used = "fallback"

    if used == "fallback":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        tok = AutoTokenizer.from_pretrained(a.student)
        try:
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                     bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
            model = AutoModelForCausalLM.from_pretrained(a.student, quantization_config=bnb,
                device_map="auto", trust_remote_code=True,
                attn_implementation=os.environ.get("ATTN_IMPL","sdpa"))
            model = prepare_model_for_kbit_training(model)
        except Exception as e:
            print(f"bnb 4-bit failed ({e}) -> bf16 LoRA")
            model = AutoModelForCausalLM.from_pretrained(a.student, torch_dtype=torch.bfloat16,
                device_map="auto", trust_remote_code=True,
                attn_implementation=os.environ.get("ATTN_IMPL","sdpa"))
            model.gradient_checkpointing_enable()
        import torch.nn as nn
        targets = sorted({n.split(".")[-1] for n,m in model.named_modules()
                          if isinstance(m, nn.Linear) and any(k in n for k in
                          ("q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"))
                          and "router" not in n})
        if not targets: raise SystemExit("no LoRA targets detected - inspect the model (R3).")
        print(f"LoRA targets: {targets}")
        model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM", target_modules=targets))
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    from trl import SFTTrainer, SFTConfig
    cfg = SFTConfig(
        output_dir=a.out, per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum, warmup_steps=a.warmup,
        num_train_epochs=a.epochs, learning_rate=a.lr, logging_steps=1,
        save_steps=a.save_every, save_total_limit=3, bf16=True,
        optim="adamw_8bit", lr_scheduler_type="cosine", seed=0,
        max_seq_length=a.seq_len, completion_only_loss=True, report_to="none")
    trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=ds, args=cfg)
    print("training on responses only (completion_only_loss=True; prompt tokens masked)")

    trainer.train(resume_from_checkpoint=a.resume)
    # save the merged-ready adapter
    if used == "unsloth":
        model.save_pretrained(a.out); tok.save_pretrained(a.out)
    else:
        trainer.save_model(a.out); tok.save_pretrained(a.out)
    print(f"done ({used}) -> {a.out}. Next: merge_lora.py, eval.py.")

if __name__ == "__main__":
    main()
