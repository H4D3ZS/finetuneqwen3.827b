#!/usr/bin/env python3
"""
Step 1: build the distillation corpus - the PROMPTS the teacher will answer.

Distill NARROW. A 3B-active student absorbs a focused distribution far better than a broad
one, so the corpus is agentic coding + tool use + repo edits - the exact jobs where
Qwen3.8-27B beats Opus (SWE-bench Pro, QwenSWEBench, IFBench). Do NOT pad it with general
chat; that dilutes the capacity you have.

Sources (mix to taste with --weights):
  - real GitHub issues/PRs (repo-level edit tasks)   -> needs a dump or the GH API
  - SWE-bench / SWE-gym style task statements
  - tool-call scenarios (function-calling traces)
  - your OWN transcripts: HackTheBox sessions, your own project's coding tasks

This script emits prompts.jsonl of {prompt}. Step 02 has the teacher answer them.

    python 01_build_corpus.py --out prompts.jsonl --n 20000 \
        --sources swebench,toolcalls,local_transcripts \
        --local_glob "C:/Users/HADES/.claude/projects/**/*.jsonl"

Start with 10-20k focused prompts for a LoRA run. Scale to 100k+ only for full-parameter.
"""
import argparse, os, json, glob, random, re

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="prompts.jsonl")
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--sources", default="swebench,toolcalls,local_transcripts",
                   help="comma list: swebench,toolcalls,local_transcripts,seed")
    p.add_argument("--local_glob", default="", help="glob of your own *.jsonl transcripts")
    p.add_argument("--hf_swebench", default="princeton-nlp/SWE-bench_Lite",
                   help="HF dataset id for task statements (needs `datasets`)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def from_swebench(hf_id, limit):
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [swebench] pip install datasets to use this source; skipping"); return []
    out = []
    try:
        ds = load_dataset(hf_id, split="test")
    except Exception as e:
        print(f"  [swebench] load failed: {e}; skipping"); return []
    for r in ds:
        problem = r.get("problem_statement") or r.get("text") or ""
        if not problem: continue
        out.append("You are an agentic coding assistant working in a real repository. "
                   "Read the issue, locate the relevant files, and implement the fix.\n\n"
                   f"Issue:\n{problem}\n\nProduce the patch and explain each change.")
        if len(out) >= limit: break
    print(f"  [swebench] {len(out)} prompts"); return out

def from_toolcalls(limit):
    # synthetic-but-realistic tool-call scenarios: the model must emit a tool call with
    # correctly-typed nested args (the thing unsloth flags as improved in 3.8).
    tools = [
        ("run_command", "execute a shell command", '{"cmd": "...", "cwd": "..."}'),
        ("read_file", "read a file range", '{"path": "...", "start": 1, "end": 40}'),
        ("edit_file", "apply a search/replace edit", '{"path": "...", "search": "...", "replace": "..."}'),
        ("http_request", "make an HTTP call", '{"method": "GET", "url": "...", "headers": {}}'),
        ("query_db", "run a parameterized SQL query", '{"sql": "...", "params": []}'),
    ]
    tasks = [
        "Find all SUID binaries on the target and report likely privesc paths.",
        "Enumerate the web app's API routes and flag any missing auth middleware.",
        "Add a rate limiter to the /login route and write the test first.",
        "Given a Log4Shell-vulnerable lab box, craft the JNDI payload and the listener.",
        "Refactor the payment gateway to route on card BIN, then update the tests.",
        "Reconcile yesterday's Stripe payouts against the orders table; report mismatches.",
    ]
    out = []
    for _ in range(limit):
        t = random.choice(tasks); name, desc, schema = random.choice(tools)
        out.append(f"You have tools available. Task: {t}\n"
                   f"Use the `{name}` tool ({desc}, schema {schema}) when needed. "
                   "Call tools with correctly-typed nested arguments.")
    print(f"  [toolcalls] {len(out)} prompts"); return out

def from_local(glob_pat, limit):
    if not glob_pat: return []
    out = []
    for fn in glob.glob(glob_pat, recursive=True):
        try:
            with open(fn, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    try: obj = json.loads(line)
                    except Exception: continue
                    # pull user-turn text out of Claude Code transcript rows
                    msg = obj.get("message", obj)
                    content = msg.get("content") if isinstance(msg, dict) else None
                    if isinstance(content, str) and len(content) > 40 and msg.get("role") == "user":
                        out.append(content)
                    elif isinstance(content, list):
                        for b in content:
                            if isinstance(b, dict) and b.get("type") == "text" and len(b.get("text","")) > 40:
                                out.append(b["text"])
                    if len(out) >= limit: break
        except Exception: continue
        if len(out) >= limit: break
    print(f"  [local] {len(out)} prompts from your transcripts"); return out

def main():
    a = parse(); random.seed(a.seed)
    want = a.sources.split(",")
    per = max(1, a.n // max(1, len(want)))
    pool = []
    if "swebench" in want:         pool += from_swebench(a.hf_swebench, per)
    if "toolcalls" in want:        pool += from_toolcalls(per)
    if "local_transcripts" in want: pool += from_local(a.local_glob, per)
    if "seed" in want or not pool:
        pool += ["Implement a thread-safe LRU cache in Rust with unit tests.",
                 "Write a Python function to parse a nested JSON config with validation."] * per
    random.shuffle(pool)
    pool = pool[:a.n]
    with open(a.out, "w", encoding="utf-8") as f:
        for p in pool:
            f.write(json.dumps({"prompt": p}) + "\n")
    print(f"wrote {len(pool)} prompts -> {a.out}")
    print("next: 02_teacher_generate.py --prompts", a.out)

if __name__ == "__main__":
    main()
