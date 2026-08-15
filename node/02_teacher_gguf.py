#!/usr/bin/env python3
"""
Step 2 (GGUF-teacher path): generate distillation targets from an ALREADY-ABLITERATED
Qwen3.8-27B GGUF served by llama.cpp/ROCmFPX.

Why this instead of the HF/vLLM path:
  - The abliterated 3.8 (Blackfrost-AI) ships as GGUF only - no safetensors - so it can't
    be loaded by transformers/vLLM. llama.cpp serves it natively.
  - This ENTIRELY avoids the vLLM-clobbers-ROCm-torch failure: the teacher runs in
    llama-server (a separate process), the student trains in Unsloth. No shared torch.
  - Bonus: the teacher is abliterated, so its completions are uncensored -> the student
    learns to comply on authorized security work. Abliteration is transferred through
    distillation; no separate abliteration step needed (see the honest caveat in PLAN.md R12).

Prereq: a llama-server is already running the abliterated GGUF, e.g. via serve_teacher.sh:
    llama-server -m Qwen3.8-27B-ABLITERATED-Q8_0.gguf -ngl 99 -fa on -np 8 --port 8080

Then:
    python3 02_teacher_gguf.py --base-url http://127.0.0.1:8080 \
        --prompts prompts.jsonl --out teacher_data/ --n 8000 --concurrency 8

Concurrency drives llama-server's parallel slots (-np). 8 parallel requests on an MI300X
serving Q8_0 is fast and cheap - this is the batched-equivalent for a GGUF teacher.
"""
import argparse, os, json, sys, time, threading, queue, urllib.request

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8080")
    p.add_argument("--prompts", required=True)
    p.add_argument("--out", default="teacher_data")
    p.add_argument("--n", type=int, default=8000)
    p.add_argument("--max_new", type=int, default=4096, help="thinking eats budget; keep >=4096")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--concurrency", type=int, default=8, help="parallel requests = server -np")
    p.add_argument("--shard", type=int, default=2000)
    p.add_argument("--think", action="store_true",
                   help="request thinking. DEFAULT OFF: this Qwen3.8-27B is a thinking model "
                        "that otherwise burns the whole token budget in <think> and returns "
                        "EMPTY content (validated: 30/40 empty). Off -> short clean code answers, "
                        "which is also what a fast A3B student wants (short chains, no 32k stalls).")
    return p.parse_args()

def load_prompts(path, n):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line)["prompt"]);
            if len(out) >= n: break
    return out

def one_request(base_url, prompt, a):
    # OpenAI-compatible chat endpoint that llama-server exposes.
    body = {
        "model": "teacher", "messages": [{"role": "user", "content": prompt}],
        "temperature": a.temperature, "top_p": a.top_p, "max_tokens": a.max_new,
        "stream": False,
        # Qwen3.8 is a thinking model. Unless --think, disable thinking via the template so the
        # model goes straight to the answer. WITHOUT this it returns empty `content` (all output
        # lands in `reasoning_content` and it never emits </think> within budget). Validated.
        "chat_template_kwargs": {"enable_thinking": bool(a.think)},
    }
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        d = json.load(r)
    m = d["choices"][0]["message"]
    # Prefer content; fall back to reasoning_content so --think runs still capture the trace.
    out = (m.get("content") or "").strip()
    if not out and a.think:
        rc = (m.get("reasoning_content") or "").strip()
        out = f"<think>\n{rc}\n</think>" if rc else ""
    return out

def worker(base_url, a, in_q, out_list, lock, prog):
    while True:
        try: idx, prompt = in_q.get_nowait()
        except queue.Empty: return
        for attempt in range(3):
            try:
                comp = one_request(base_url, prompt, a)
                # Drop junk: empty, too short, or a degenerate </think> repetition loop.
                if len(comp) < 40 or comp.count("</think>") > 3:
                    with lock:
                        prog[1] += 1
                        if attempt == 2: print(f"  [drop idx {idx}] empty/degenerate", file=sys.stderr, flush=True)
                    if attempt < 2: time.sleep(1); continue
                    break
                with lock:
                    out_list.append((prompt, comp))
                    prog[0] += 1
                    if prog[0] % 25 == 0: print(f"  {prog[0]} kept ({prog[1]} dropped)", flush=True)
                break
            except Exception as e:
                if attempt == 2:
                    with lock: print(f"  [skip idx {idx}] {type(e).__name__}: {e}", file=sys.stderr)
                else: time.sleep(2)
        in_q.task_done()

def write_shards(pairs, out_dir, shard_size):
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for s, i in enumerate(range(0, len(pairs), shard_size)):
        with open(os.path.join(out_dir, f"shard_{s:04d}.jsonl"), "w", encoding="utf-8") as f:
            for pr, co in pairs[i:i+shard_size]:
                f.write(json.dumps({"prompt": pr, "completion": co}) + "\n")
        n = s + 1
    print(f"wrote {len(pairs)} pairs across {n} shards -> {out_dir}/")

def main():
    a = parse()
    prompts = load_prompts(a.prompts, a.n)
    print(f"{len(prompts)} prompts -> teacher at {a.base_url} (concurrency {a.concurrency})")
    # sanity: is the teacher up?
    try:
        urllib.request.urlopen(a.base_url.rstrip("/") + "/health", timeout=10)
    except Exception as e:
        sys.exit(f"teacher not reachable at {a.base_url} ({e}). Start it with serve_teacher.sh first.")
    in_q = queue.Queue()
    for i, p in enumerate(prompts): in_q.put((i, p))
    out_list, lock, prog = [], threading.Lock(), [0, 0]  # [kept, dropped]
    threads = [threading.Thread(target=worker, args=(a.base_url, a, in_q, out_list, lock, prog))
               for _ in range(a.concurrency)]
    t0 = time.time()
    for t in threads: t.start()
    for t in threads: t.join()
    print(f"generated {len(out_list)} completions in {time.time()-t0:.0f}s")
    write_shards(out_list, a.out, a.shard)
    print("next: 03_unsloth_sft.py --data", a.out)

if __name__ == "__main__":
    main()
