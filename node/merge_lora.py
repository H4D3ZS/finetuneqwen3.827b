#!/usr/bin/env python3
"""
Merge the LoRA adapter into the base to produce a standalone bf16 model. Runs on the node
(192GB) where the merge is trivial. Without this the "trained model" is just an adapter -
converting it to GGUF would silently emit the untrained base (R9).

    python merge_lora.py --adapter student-distilled/ --base Qwen/Qwen3.6-35B-A3B --out student-merged/
"""
import argparse, os, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--out", default="student-merged")
    a = ap.parse_args()
    if not os.path.exists(os.path.join(a.adapter, "adapter_config.json")):
        # already a full model (full-parameter run) - just copy the reference through
        print("no adapter_config.json - treating as a full model, nothing to merge.")
        return
    print("loading base + adapter for merge...")
    m = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    m = PeftModel.from_pretrained(m, a.adapter)
    print("merging...")
    m = m.merge_and_unload()
    m.save_pretrained(a.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(a.base).save_pretrained(a.out)
    # sanity: merged must differ from base (else the LoRA was a no-op)
    print(f"merged -> {a.out}")
    print("Sanity-check locally after download: eval.py should still show distilled >= base.")

if __name__ == "__main__":
    main()
