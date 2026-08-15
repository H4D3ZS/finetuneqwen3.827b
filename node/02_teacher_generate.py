#!/usr/bin/env python3
"""
Step 2: teacher (Qwen3.8-27B) generates the distillation targets.

*** COST-CRITICAL (Risk R1 in PLAN.md) ***
Uses vLLM for BATCHED generation. Doing this one-prompt-at-a-time with HF generate() leaves
the MI300X ~idle and costs ~67x more: 8k prompts x 2k tokens is 98 hr / $197 on HF vs
1.5 hr / $3 on vLLM. HF is kept ONLY as a small-N fallback and prints a loud cost warning.

REASONING DISTILLATION: generate WITH thinking on (--think). The completion keeps the
<think>...</think> block so the student learns to reason like the 27B, only faster+shorter
(the DeepSeek-R1-Distill effect). This is what makes the distilled A3B Opus-competitive.

    # on the node:
    python 02_teacher_generate.py --teacher Qwen/Qwen3.8-27B \
        --prompts prompts.jsonl --out teacher_data/ --n 8000 --think

Output: teacher_data/*.jsonl of {prompt, completion}. Step 03 distills from these.
"""
import argparse, os, json, sys

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="Qwen/Qwen3.8-27B")
    p.add_argument("--prompts", required=True)
    p.add_argument("--out", default="teacher_data")
    p.add_argument("--n", type=int, default=8000)
    p.add_argument("--max_new", type=int, default=4096, help="thinking eats budget; keep >=4096")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--shard", type=int, default=2000)
    p.add_argument("--think", action="store_true", help="Qwen thinking mode (recommended ON)")
    p.add_argument("--tp", type=int, default=1, help="tensor-parallel size (1 for single MI300X)")
    p.add_argument("--force_hf", action="store_true", help="skip vLLM even if present (use batched HF)")
    p.add_argument("--batch", type=int, default=16, help="HF generation batch size (raise if VRAM allows)")
    p.add_argument("--seq_len", type=int, default=4096, help="max prompt length for HF batching")
    return p.parse_args()

def load_prompts(path, n):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line)["prompt"])
            if len(out) >= n: break
    return out

def build_texts(tok, prompts, think):
    kw = {}
    try:
        tok.apply_chat_template([{"role":"user","content":"x"}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=think)
        kw["enable_thinking"] = think
    except TypeError:
        print("  (no enable_thinking kwarg; template default applies)")
    return [tok.apply_chat_template([{"role":"user","content":p}], tokenize=False,
                                    add_generation_prompt=True, **kw) for p in prompts]

def write_shards(pairs, out_dir, shard_size):
    os.makedirs(out_dir, exist_ok=True)
    shard = 0
    for i in range(0, len(pairs), shard_size):
        chunk = pairs[i:i+shard_size]
        with open(os.path.join(out_dir, f"shard_{shard:04d}.jsonl"), "w", encoding="utf-8") as f:
            for pr, co in chunk:
                f.write(json.dumps({"prompt": pr, "completion": co}) + "\n")
        shard += 1
    print(f"wrote {len(pairs)} pairs across {shard} shards -> {out_dir}/")

def run_vllm(a, prompts):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.teacher)
    texts = build_texts(tok, prompts, a.think)
    print(f"vLLM: loading {a.teacher} (tp={a.tp}) ...")
    llm = LLM(model=a.teacher, tensor_parallel_size=a.tp, dtype="bfloat16",
              trust_remote_code=True, gpu_memory_utilization=0.90, max_model_len=8192)
    sp = SamplingParams(temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_new)
    print(f"vLLM: generating {len(texts)} completions (batched) ...")
    outs = llm.generate(texts, sp)
    # keep completions in prompt order
    pairs = [(texts[i], outs[i].outputs[0].text) for i in range(len(texts))]
    return pairs

def run_hf(a, prompts):
    print(f"batched HF generation: {len(prompts)} prompts, batch={a.batch}")
    import torch
    # BATCHED HF generation - the robust ROCm path. ~15x faster than one-at-a-time and it
    # runs on the ROCm torch already in the image (NO vLLM, no CUDA wheel to clobber torch).
    # 8k prompts ~= $11 on the MI300X vs $197 sequential. This is the default now.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.teacher, padding_side="left")  # left-pad for decode-only
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.teacher, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
        attn_implementation=os.environ.get("ATTN_IMPL","sdpa"))
    model.eval()
    texts = build_texts(tok, prompts, a.think)
    bs = a.batch
    pairs = []
    for i in range(0, len(texts), bs):
        chunk = texts[i:i+bs]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=a.seq_len).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=True,
                                  temperature=a.temperature, top_p=a.top_p,
                                  pad_token_id=tok.pad_token_id)
        gen = out[:, enc.input_ids.shape[1]:]  # strip the prompt (left-padded, so uniform)
        for j, text in enumerate(chunk):
            pairs.append((text, tok.decode(gen[j], skip_special_tokens=not a.think)))
        print(f"  {min(i+bs, len(texts))}/{len(texts)}")
    return pairs

def main():
    a = parse()
    prompts = load_prompts(a.prompts, a.n)
    print(f"{len(prompts)} prompts, think={a.think}")
    # Prefer vLLM ONLY if a real ROCm/CUDA build is present (import + a device). Otherwise use
    # batched HF - which is what works out-of-the-box on the AMD Unsloth image. NEVER let a
    # plain `pip install vllm` clobber torch (it pulls a CUDA wheel); run_node.sh no longer does.
    use_vllm = False
    if not a.force_hf:
        try:
            import vllm, torch  # noqa
            if torch.cuda.is_available():
                use_vllm = True
        except Exception:
            use_vllm = False
    if not use_vllm:
        print("using BATCHED HF generation (ROCm-native, no vLLM). batch=%d" % a.batch)
    pairs = run_vllm(a, prompts) if use_vllm else run_hf(a, prompts)
    write_shards(pairs, a.out, a.shard)
    print("next: 03_unsloth_sft.py --data", a.out)

if __name__ == "__main__":
    main()
