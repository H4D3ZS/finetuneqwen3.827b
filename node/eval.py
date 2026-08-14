#!/usr/bin/env python3
"""
R6: the GO/NO-GO gate. Compare the distilled student vs the base student on a small
held-out set of agentic-coding/tool tasks. Only proceed to the full run if distilled >= base.

    python eval.py --distilled student-distilled/ --base Qwen/Qwen3.6-35B-A3B --lora

Scoring is intentionally simple and objective (does the code run / is the tool-call valid /
does the reasoning terminate), not an LLM judge - we want a cheap, trustworthy signal on the
node before spending more credit. Prints a side-by-side table.
"""
import argparse, os, json, re, ast, tempfile, subprocess, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TASKS = [
    {"kind":"code", "prompt":"Write a Python function is_prime(n) with a docstring. Only the function.",
     "check": lambda t: _py_defines(t, "is_prime") and _py_runs(t, "assert is_prime(7) and not is_prime(8)")},
    {"kind":"code", "prompt":"Write a Python function fib(n) returning the nth Fibonacci number (0-indexed, fib(0)=0).",
     "check": lambda t: _py_runs(t, "assert fib(0)==0 and fib(1)==1 and fib(10)==55")},
    {"kind":"code", "prompt":"Write a Python function that reverses a linked list. Include a Node class.",
     "check": lambda t: _py_defines(t, "Node")},
    {"kind":"tool", "prompt":"You have a tool read_file(path, start, end). Read lines 1-40 of src/main.py. Emit ONLY the tool call as JSON.",
     "check": lambda t: _has_json_call(t, "read_file", ["path"])},
    {"kind":"tool", "prompt":"You have a tool run_command(cmd, cwd). List SUID binaries on a lab box you own. Emit ONLY the tool call as JSON.",
     "check": lambda t: _has_json_call(t, "run_command", ["cmd"])},
    {"kind":"reason", "prompt":"A train travels 60km in 45min. What is its speed in km/h? Think briefly, then answer.",
     "check": lambda t: "80" in t},
    {"kind":"code", "prompt":"Write a TypeScript function validateEmail(s: string): boolean using a regex.",
     "check": lambda t: "boolean" in t and "RegExp" in t or "/.+@.+/" in t or "test(" in t},
    {"kind":"sec", "prompt":"For a HackTheBox box I own, give a bash one-liner to find world-writable files.",
     "check": lambda t: "find" in t and ("-perm" in t or "-writable" in t)},
    {"kind":"code", "prompt":"Write a Python function that checks if a string is a palindrome, ignoring case and spaces.",
     "check": lambda t: _py_runs(t, "import re\nassert palindrome('A man a plan a canal Panama')") if _py_defines_any(t,"palindrome") else _py_defines_any(t,"palindrome")},
    {"kind":"reason", "prompt":"If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets? Answer with a number of minutes.",
     "check": lambda t: re.search(r"\b5\b", t) is not None},
]

def _extract_py(t):
    m = re.search(r"```(?:python)?\s*(.+?)```", t, re.S)
    return m.group(1) if m else t
def _py_defines(t, name): return re.search(rf"def\s+{name}\s*\(", _extract_py(t)) is not None
def _py_defines_any(t, name): return name in _extract_py(t)
def _py_runs(t, asrt):
    code = _extract_py(t) + "\n" + asrt
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code); path = f.name
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=10)
        os.unlink(path); return r.returncode == 0
    except Exception: return False
def _has_json_call(t, name, keys):
    m = re.search(r"\{.*\}", t, re.S)
    if not m: return False
    try: obj = json.loads(m.group(0))
    except Exception: return False
    blob = json.dumps(obj)
    return name in blob and all(k in blob for k in keys)

def gen(model, tok, prompt, think=True):
    kw = {}
    try: tok.apply_chat_template([{"role":"user","content":"x"}], tokenize=False,
            add_generation_prompt=True, enable_thinking=think); kw["enable_thinking"]=think
    except TypeError: pass
    text = tok.apply_chat_template([{"role":"user","content":prompt}], tokenize=False,
                                   add_generation_prompt=True, **kw)
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=1024, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)

def load(model_id, base_for_lora=None):
    if base_for_lora:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(base_for_lora, torch_dtype=torch.bfloat16,
                device_map="auto", trust_remote_code=True)
        m = PeftModel.from_pretrained(base, model_id)
        tok = AutoTokenizer.from_pretrained(base_for_lora)
    else:
        m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                device_map="auto", trust_remote_code=True)
        tok = AutoTokenizer.from_pretrained(model_id)
    m.eval(); return m, tok

def score(model, tok):
    passed = 0
    for t in TASKS:
        try: ok = bool(t["check"](gen(model, tok, t["prompt"])))
        except Exception: ok = False
        passed += ok
        print(f"    [{'PASS' if ok else 'FAIL'}] {t['kind']}: {t['prompt'][:50]}")
    return passed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distilled", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--lora", action="store_true", help="distilled is a LoRA adapter on --base")
    a = ap.parse_args()

    print("== BASE"); bm, bt = load(a.base); b = score(bm, bt)
    del bm; torch.cuda.empty_cache()
    print("== DISTILLED")
    dm, dt = load(a.distilled, base_for_lora=a.base if a.lora else None); d = score(dm, dt)

    print(f"\n== RESULT: base {b}/{len(TASKS)}  |  distilled {d}/{len(TASKS)}")
    if d > b:   print("GO: distilled beats base. Proceed to the full run.")
    elif d == b: print("MARGINAL: no regression but no gain. Inspect corpus quality before full run.")
    else:       print("NO-GO: distilled is WORSE. Stop - do not spend more. Fix corpus/hyperparams.")

if __name__ == "__main__":
    main()
