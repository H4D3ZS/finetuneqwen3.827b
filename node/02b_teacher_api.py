#!/usr/bin/env python3
"""
Step 2b (API-teacher path): generate distillation targets from a HOSTED teacher via API.
Multi-teacher: run it once per teacher and merge the shards. Supports two wire formats:

  - openai   : OpenAI-compatible /v1/chat/completions (ModelScope Qwen3.8-Max, local
               llama-server, most hosts). Auth: Bearer token.
  - anthropic: Anthropic /v1/messages (Claude Opus 5 / Fable 5). Auth: x-api-key.

Why a separate script from 02 (local llama-server): API teachers need three things a local
run doesn't — a QUOTA GUARD (Qwen3.8-Max ambassador cap is 5,500 req/MONTH), RESUME (a
frontier seed can span days; a crash must not re-spend quota/$), and INCREMENTAL shard
writes (never hold thousands of completions in memory to lose on exit).

    # Qwen3.8-Max frontier seed (ModelScope, OpenAI-compatible)
    python node/02b_teacher_api.py --format openai \
      --base-url https://api-inference.modelscope.ai/v1 \
      --model Qwen-Ambassador/Qwen3.8-Max --api-key-env MODELSCOPE_API_KEY \
      --prompts node/prompts_seed.jsonl --out node/teacher_seed_max --max-requests 5000

    # Claude seed (Anthropic). Pick fable (cheap) or opus (best).
    python node/02b_teacher_api.py --format anthropic \
      --base-url https://api.anthropic.com --model claude-fable-5 --api-key-env ANTHROPIC_API_KEY \
      --prompts node/prompts_seed.jsonl --out node/teacher_seed_claude --max-requests 2000

NEVER pass the key on the CLI (it lands in shell history / process list). Use --api-key-env
and export the key, or a .env. Keys must never be committed (see .gitignore).
"""
import argparse, os, json, sys, time, threading, queue, hashlib, urllib.request, urllib.error

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--format", choices=["openai", "anthropic"], required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-key-env", default="API_KEY", help="ENV VAR holding the key (not the key)")
    p.add_argument("--prompts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=10_000_000)
    p.add_argument("--max_new", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--concurrency", type=int, default=4, help="keep low for hosted APIs (rate limits)")
    p.add_argument("--shard", type=int, default=500, help="rows per shard file")
    p.add_argument("--max-requests", type=int, default=0, help="QUOTA GUARD: stop after N new "
                   "successful completions (0 = unlimited). Set to your monthly cap minus margin.")
    p.add_argument("--think", action="store_true", help="request thinking (default off: short code)")
    return p.parse_args()

def pid(prompt):                       # stable id so resume can skip finished prompts
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]

def load_done(out_dir):
    done = set()
    if os.path.isdir(out_dir):
        for fn in os.listdir(out_dir):
            if fn.endswith(".jsonl"):
                for line in open(os.path.join(out_dir, fn), encoding="utf-8"):
                    try: done.add(json.loads(line)["id"])
                    except Exception: pass
    return done

def build_request(a, key, prompt):
    url = a.base_url.rstrip("/")
    if a.format == "anthropic":
        # Anthropic base is bare (https://api.anthropic.com); append the full path.
        url += "/v1/messages"
        body = {"model": a.model, "max_tokens": a.max_new, "temperature": a.temperature,
                "messages": [{"role": "user", "content": prompt}]}
        headers = {"Content-Type": "application/json", "x-api-key": key,
                   "anthropic-version": "2023-06-01"}
    else:
        # OpenAI convention: base_url already ends in /v1 -> append only /chat/completions.
        url += "" if url.endswith("/chat/completions") else "/chat/completions"
        body = {"model": a.model, "max_tokens": a.max_new, "temperature": a.temperature,
                "top_p": a.top_p, "messages": [{"role": "user", "content": prompt}],
                "chat_template_kwargs": {"enable_thinking": bool(a.think)}}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    return urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)

def extract(a, d):
    if a.format == "anthropic":
        return "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text").strip()
    return (d["choices"][0]["message"].get("content") or "").strip()

def worker(a, key, in_q, writer, counter, stop):
    while not stop.is_set():
        try: idx, prompt = in_q.get_nowait()
        except queue.Empty: return
        for attempt in range(4):
            if stop.is_set(): return
            try:
                with urllib.request.urlopen(build_request(a, key, prompt), timeout=600) as r:
                    d = json.load(r)
                comp = extract(a, d)
                if len(comp) < 40 or comp.count("</think>") > 3:
                    break                                   # drop junk, don't retry-spend
                writer.write(pid(prompt), prompt, comp)
                with counter["lock"]:
                    counter["n"] += 1
                    if counter["max"] and counter["n"] >= counter["max"]:
                        print(f"  quota guard: hit --max-requests {counter['max']}, stopping", flush=True)
                        stop.set()
                    if counter["n"] % 25 == 0: print(f"  {counter['n']} done", flush=True)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:                            # rate limited -> exponential backoff
                    time.sleep(min(60, 2 ** attempt * 5)); continue
                if attempt == 3: print(f"  [skip {idx}] HTTP {e.code}: {e.read().decode()[:120]}", file=sys.stderr, flush=True)
            except Exception as e:
                if attempt == 3: print(f"  [skip {idx}] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                else: time.sleep(2)
        in_q.task_done()

class ShardWriter:
    """Thread-safe incremental sharded writer. Appends as completions arrive; resumable."""
    def __init__(self, out_dir, shard_size):
        os.makedirs(out_dir, exist_ok=True)
        self.dir, self.size, self.lock = out_dir, shard_size, threading.Lock()
        existing = [f for f in os.listdir(out_dir) if f.endswith(".jsonl")]
        self.shard = len(existing); self.count = 0; self.fh = None
    def _roll(self):
        if self.fh is None or self.count >= self.size:
            if self.fh: self.fh.close()
            self.fh = open(os.path.join(self.dir, f"shard_{self.shard:04d}.jsonl"), "a", encoding="utf-8")
            self.shard += 1; self.count = 0
    def write(self, id_, prompt, completion):
        with self.lock:
            self._roll()
            self.fh.write(json.dumps({"id": id_, "prompt": prompt, "completion": completion}) + "\n")
            self.fh.flush(); self.count += 1
    def close(self):
        if self.fh: self.fh.close()

def main():
    a = parse()
    key = os.environ.get(a.api_key_env)
    if not key:
        sys.exit(f"set the key: export {a.api_key_env}=... (never pass it on the CLI)")
    prompts = [json.loads(l)["prompt"] for l in open(a.prompts, encoding="utf-8")][:a.n]
    done = load_done(a.out)
    todo = [(i, p) for i, p in enumerate(prompts) if pid(p) not in done]
    print(f"{len(prompts)} prompts | {len(done)} already done (resume) | {len(todo)} to generate")
    print(f"teacher: {a.model} ({a.format}) | quota guard: {a.max_requests or 'off'} | concurrency {a.concurrency}")
    if not todo: print("nothing to do."); return

    in_q = queue.Queue()
    for item in todo: in_q.put(item)
    writer = ShardWriter(a.out, a.shard)
    counter = {"n": 0, "max": a.max_requests, "lock": threading.Lock()}
    stop = threading.Event()
    threads = [threading.Thread(target=worker, args=(a, key, in_q, writer, counter, stop))
               for _ in range(a.concurrency)]
    t0 = time.time()
    for t in threads: t.start()
    try:
        for t in threads: t.join()
    except KeyboardInterrupt:
        print("\ninterrupted — shards are safe on disk, rerun to resume."); stop.set()
    writer.close()
    print(f"generated {counter['n']} new completions in {time.time()-t0:.0f}s -> {a.out}/")
    print("merge all teacher_* dirs at train time: 03_unsloth_sft.py --data <dir> (repeatable)")

if __name__ == "__main__":
    main()
