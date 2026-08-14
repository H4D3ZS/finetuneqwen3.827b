#!/usr/bin/env python3
"""
Distill Qwen3.8-27B (teacher) into Qwen3.6-35B-A3B (student MoE).

Runs on a rented multi-GPU node (4-8x H100 80GB). NOT on the workstation - a 35B in
bf16 with grads + Adam is ~280GB. This is where the training lives; inference comes home.

Loss = alpha * KL(student || teacher soft targets) + (1-alpha) * CE(student, hard labels).
Soft-target distillation transfers the teacher's *distribution* (its "how it reasons"),
not just its argmax - that is what carries agentic-coding behavior into the faster student.

Usage (on the rented node, after steps 01-02):
    accelerate launch --multi_gpu 03_distill_train.py \
        --teacher Qwen/Qwen3.8-27B \
        --student Qwen/Qwen3.6-35B-A3B \
        --data teacher_data/ \
        --out student-distilled/ \
        --lora            # drop for full-parameter (needs more VRAM, cleaner result)

Focused-distribution note: keep the corpus (step 01) narrow - agentic coding, tool calls,
repo-level edits. A 3B-active student absorbs a narrow teacher distribution far better
than a broad one. Distill the job you actually run, not "everything".
"""
import argparse, os, json, math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="Qwen/Qwen3.8-27B")
    p.add_argument("--student", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--data", required=True, help="dir of *.jsonl with {prompt, completion}")
    p.add_argument("--out", default="student-distilled")
    p.add_argument("--lora", action="store_true", help="QLoRA: ~4x cheaper, small quality hit")
    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--alpha", type=float, default=0.9, help="weight on soft (KL) vs hard (CE)")
    p.add_argument("--temp", type=float, default=2.0, help="distillation temperature")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--resume", action="store_true", help="continue from newest step checkpoint")
    return p.parse_args()

class DistillData(Dataset):
    """Each row: {prompt, completion}. The teacher's soft targets are computed on the fly
    in the training loop (teacher kept in eval/no-grad), so step 02 only needs the text
    pairs - not pre-dumped logits, which are huge. If you DID pre-dump logits in step 02,
    swap this to read them and skip the teacher forward pass (faster, more disk)."""
    def __init__(self, data_dir, tok, seq_len):
        self.rows = []
        for fn in os.listdir(data_dir):
            if fn.endswith(".jsonl"):
                with open(os.path.join(data_dir, fn), encoding="utf-8") as f:
                    for line in f:
                        self.rows.append(json.loads(line))
        self.tok, self.seq_len = tok, seq_len
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        text = r["prompt"] + r["completion"]
        enc = self.tok(text, truncation=True, max_length=self.seq_len,
                       return_tensors="pt", padding="max_length")
        ids = enc.input_ids[0]
        # mask the prompt tokens out of the hard-label CE (distill only on the completion)
        p_len = len(self.tok(r["prompt"], truncation=True, max_length=self.seq_len).input_ids)
        labels = ids.clone()
        labels[:p_len] = -100
        labels[enc.attention_mask[0] == 0] = -100
        return {"input_ids": ids, "attention_mask": enc.attention_mask[0], "labels": labels}

def student_cfg_vocab(model_id):
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    return getattr(c, "vocab_size", None) or getattr(getattr(c, "text_config", c), "vocab_size", None)

def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.student)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    dtype = torch.bfloat16
    common = dict(torch_dtype=dtype, trust_remote_code=True, attn_implementation=os.environ.get("ATTN_IMPL","flash_attention_2"))

    print("loading teacher (frozen)...")
    teacher = AutoModelForCausalLM.from_pretrained(a.teacher, device_map="auto", **common)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)

    # R2: KL distillation aligns teacher/student logits BY INDEX. Same vocab size is
    # necessary but NOT sufficient - the two tokenizers must map identical text to identical
    # IDs, or the KL is noise and the whole billed run is wasted with no error. Check both.
    tv, sv = teacher.config.vocab_size, student_cfg_vocab(a.student)
    if tv != sv:
        raise SystemExit(f"VOCAB SIZE MISMATCH: teacher {tv} vs student {sv}. Need a shared vocab.")
    from transformers import AutoTokenizer as _AT
    t_tok = _AT.from_pretrained(a.teacher)
    probe = "def f(x):\n    return x**2  # tool_call: read_file(path='a.py')  面白い"
    if t_tok(probe).input_ids != tok(probe).input_ids:
        raise SystemExit("TOKENIZER ID MISMATCH: same vocab size but different token IDs. "
                         "KL would be garbage. Use teacher/student from the same tokenizer family.")
    print(f"tokenizer id-match ok (vocab {tv})")

    print("loading student...")
    if a.lora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        # R4: try bnb 4-bit (QLoRA); on ROCm/gfx942 it may not load -> auto bf16 LoRA.
        try:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                                     bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
            student = AutoModelForCausalLM.from_pretrained(a.student, quantization_config=bnb,
                                                           device_map="auto", **common)
            student = prepare_model_for_kbit_training(student)
            print("student: bnb 4-bit (QLoRA)")
        except Exception as e:
            print(f"bnb 4-bit failed ({type(e).__name__}: {e}) -> bf16 LoRA (192GB has room)")
            student = AutoModelForCausalLM.from_pretrained(a.student, device_map="auto", **common)
            student.gradient_checkpointing_enable()
        # R3: auto-detect the real Linear module names on THIS (MoE) architecture instead of
        # hardcoding gate/up/down - Qwen MoE names experts differently, and a wrong target
        # list attaches LoRA to nothing (0 trainable params = a paid no-op).
        import torch.nn as nn
        names = set()
        for n, m in student.named_modules():
            if isinstance(m, nn.Linear):
                leaf = n.split(".")[-1]
                if leaf not in ("lm_head",) and "router" not in n and "gate" != leaf:
                    names.add(leaf)
        # keep the projections that matter; drop the router gate (train it separately if needed)
        targets = sorted(n for n in names if any(k in n for k in
                    ("q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj","qkv","wqkv")))
        if not targets:
            raise SystemExit("R3: no LoRA target modules detected on the student. Aborting "
                             "before wasting compute - inspect the model's module names.")
        print(f"LoRA targets ({len(targets)}): {targets}")
        lcfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
                          task_type="CAUSAL_LM", target_modules=targets)
        student = get_peft_model(student, lcfg)
        student.print_trainable_parameters()
        tp = sum(p.numel() for p in student.parameters() if p.requires_grad)
        if tp == 0:
            raise SystemExit("R3: 0 trainable params after LoRA attach. Targets wrong - stop.")
    else:
        student = AutoModelForCausalLM.from_pretrained(a.student, device_map="auto", **common)
        student.gradient_checkpointing_enable()

    # R5: resume from the newest checkpoint if --resume and one exists
    start_step = 0
    if a.resume and os.path.isdir(a.out):
        ckpts = sorted((d for d in os.listdir(a.out) if d.startswith("step")),
                       key=lambda d: int(d[4:]) if d[4:].isdigit() else -1)
        if ckpts:
            from peft import PeftModel
            latest = os.path.join(a.out, ckpts[-1])
            print(f"R5: resuming from {latest}")
            if a.lora:
                student = PeftModel.from_pretrained(student, latest, is_trainable=True)
            start_step = int(ckpts[-1][4:])

    ds = DistillData(a.data, tok, a.seq_len)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True)
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=a.lr)
    total_steps = math.ceil(len(dl) / a.grad_accum) * a.epochs
    sched = get_cosine_schedule_with_warmup(opt, a.warmup, total_steps)

    T = a.temp
    step = 0
    for epoch in range(a.epochs):
        for i, b in enumerate(dl):
            b = {k: v.to(student.device) for k, v in b.items()}
            with torch.no_grad():
                t_logits = teacher(input_ids=b["input_ids"],
                                   attention_mask=b["attention_mask"]).logits
            s_out = student(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
            s_logits = s_out.logits

            # soft distillation: KL on temperature-softened distributions, over completion tokens
            mask = (b["labels"] != -100).unsqueeze(-1)
            s_lp = F.log_softmax(s_logits / T, dim=-1)
            t_p  = F.softmax(t_logits / T, dim=-1)
            kl = F.kl_div(s_lp, t_p, reduction="none").sum(-1, keepdim=True)
            kl = (kl * mask).sum() / mask.sum().clamp(min=1) * (T * T)

            # hard-label CE on the real completion tokens
            ce = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)),
                                 b["labels"].view(-1), ignore_index=-100)

            loss = (a.alpha * kl + (1 - a.alpha) * ce) / a.grad_accum
            loss.backward()

            if (i + 1) % a.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in student.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                step += 1
                if step % 10 == 0:
                    print(f"epoch {epoch} step {step}/{total_steps} "
                          f"loss {loss.item()*a.grad_accum:.4f} kl {kl.item():.4f} ce {ce.item():.4f}")
                if step % a.save_every == 0:
                    student.save_pretrained(f"{a.out}/step{step}")

    student.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"done -> {a.out}. Next: 05_convert_quant.sh to GGUF + Q2_0_ROCMFPX, then serve locally.")

if __name__ == "__main__":
    main()
