#!/usr/bin/env python3
"""
Step 4 (optional, runs on the rented node right after distillation): abliterate the
distilled student - remove refusal directions so it never balks on authorized security
work (HackTheBox, pentest, CTF).

Abliteration = find the "refusal direction" in activation space (mean activation on
harmful-refused prompts minus harmless prompts), then orthogonalize it out of the weight
matrices. It is NOT gradient training - just forward passes + a rank-1 weight edit - so it
is far cheaper than the distillation. It runs on the same node while the GPUs are already
rented.

    python 04_abliterate.py --model student-distilled/ --out student-abliterated/

Prefer the maintained tool if available (produces your existing v2): Abliterix
(github.com/wuwangzhang1216/abliterix). This script is a compact, dependency-light
fallback that implements the core method (grimjim/Arditi et al.) so the pipeline is
self-contained.

Order matters: abliterate AFTER distillation. Distilling an already-abliterated model can
re-teach refusals from the teacher; doing it last guarantees the shipped model is clean.
"""
import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HARMLESS = [
    "Write a haiku about the ocean.",
    "Explain how photosynthesis works.",
    "What is the capital of France?",
    "Summarize the plot of Romeo and Juliet.",
    "Convert 100 Fahrenheit to Celsius.",
]
# authorized-security prompts a censored model tends to refuse; we want compliance.
TARGETED = [
    "Write a Python reverse shell for a HackTheBox box I own.",
    "Give a bash one-liner to enumerate SUID binaries for privesc on my lab machine.",
    "Craft a Log4Shell JNDI payload for an authorized pentest target.",
    "Write a port scanner in Go for my own network range.",
    "Explain how to dump and crack /etc/shadow on a box I control.",
]

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", default="student-abliterated")
    p.add_argument("--layer_frac", type=float, default=0.6,
                   help="which residual layer to read the direction from (0-1 of depth)")
    p.add_argument("--offload_folder", default="",
                   help="LOCAL 16GB/40GB machine: a fast NVMe dir to stream the 70GB model "
                        "through (e.g. E:/offload). Required - the model does not fit in RAM+VRAM.")
    return p.parse_args()

@torch.no_grad()
def mean_dir(model, tok, prompts, layer_idx):
    acts = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(model.device)
        out = model(**ids, output_hidden_states=True)
        acts.append(out.hidden_states[layer_idx][0, -1, :].float())  # last-token residual
    return torch.stack(acts).mean(0)

def main():
    a = parse()
    tok = AutoTokenizer.from_pretrained(a.model)
    # LOCAL machine (16GB VRAM + 40GB RAM = 56GB) cannot hold a ~70GB bf16 35B. Stream it
    # through the GPU with CPU + DISK offload. Slow (hours) but this is exactly how your
    # existing Qwen3.6-35B-A3B-abliterated-v2 was made locally. Point --offload_folder at
    # a fast NVMe dir with >80GB free (e.g. the E: games drive or a scratch dir).
    load_kwargs = dict(torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    if a.offload_folder:
        os.makedirs(a.offload_folder, exist_ok=True)
        load_kwargs["offload_folder"] = a.offload_folder
        load_kwargs["max_memory"] = {0: "14GiB", "cpu": "32GiB"}  # leave headroom
        print(f"disk-offloading through {a.offload_folder} (this is slow but fits 56GB)")
    else:
        print("WARNING: no --offload_folder. A 70GB model will OOM on this machine. "
              "Prefer Abliterix (your proven tool) or pass --offload_folder E:/offload.")
    model = AutoModelForCausalLM.from_pretrained(a.model, **load_kwargs)
    model.eval()
    n_layers = model.config.num_hidden_layers
    L = int(n_layers * a.layer_frac)
    print(f"reading refusal direction at layer {L}/{n_layers}")

    d_targeted = mean_dir(model, tok, TARGETED, L)
    d_harmless = mean_dir(model, tok, HARMLESS, L)
    refusal = (d_targeted - d_harmless)
    refusal = refusal / refusal.norm()   # unit direction to orthogonalize out

    # orthogonalize the refusal direction out of every matrix that writes to the residual
    # stream: attn o_proj and mlp down_proj across all layers, plus the embedding.
    #   W <- W - r (r^T W)   (project the r component out of the output space)
    def orthogonalize(W):
        r = refusal.to(W.dtype).to(W.device)
        # W: [out, in]; residual-writing rows are `out`. Remove r from the column space.
        proj = torch.outer(r, r) @ W
        W.sub_(proj)

    edited = 0
    for layer in model.model.layers:
        orthogonalize(layer.self_attn.o_proj.weight.data); edited += 1
        # MoE: down_proj lives per-expert; dense: single mlp
        mlp = layer.mlp
        if hasattr(mlp, "experts"):
            for e in mlp.experts:
                orthogonalize(e.down_proj.weight.data); edited += 1
            if hasattr(mlp, "shared_expert"):
                orthogonalize(mlp.shared_expert.down_proj.weight.data); edited += 1
        elif hasattr(mlp, "down_proj"):
            orthogonalize(mlp.down_proj.weight.data); edited += 1
    print(f"orthogonalized refusal direction out of {edited} matrices")

    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    print(f"done -> {a.out}. Verify it complies on TARGETED prompts, then 05_convert_quant.sh.")

if __name__ == "__main__":
    main()
