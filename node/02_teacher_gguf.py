#!/usr/bin/env python3
"""
Step 2 (GGUF teacher): generate distillation targets by hitting the abliterated teacher's
llama-server API concurrently.

WHY THIS EXISTS (R1 + the smoke-run-1 failure):
  - The teacher is a separate PROCESS, not a python import. No vLLM, no HF model load, and
    critically no pip install that can swap ROCm torch for a CUDA wheel. The env that killed
    smoke run 1 cannot break this path.
  - llama-server batches across --parallel slots, so N concurrent requests are served
    together. That is what makes this cheap rather than 67x expensive.
  - The teacher is ABLITERATED, so its answers carry no refusals -> the student inherits
    compliance on the corpus distribution (see R12: this is TRANSFERRED abliteration, which
    covers what the corpus covers, not a global guarantee).

    bash serve_teacher.sh &
    python3 02_teacher_gguf.py --prompts prompts.jsonl --out teacher_data/ --n 8000 --think
"""
import argparse, os, json, sys, time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request, urllib.error


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8081/v1/chat/completions")
    p.add_argument("--prompts", required=True)
    p.add_argument("--out", default="teacher_data")
    p.add_argument("--n", type=int, default=8000)
    p.add_argument("--max_new", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--shard", type=int, default=2000)
    p.add_argument("--think", action="store_true", help="keep the teacher's reasoning block")
    p.add_argument("--concurrency", type=int, default=16,
                   help="match serve_teacher.sh PARALLEL")
    p.add_argument("--timeout", type=int, default=900)
    return p.parse_args()


def load_prompts(path, n):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line)["prompt"])
            if len(out) >= n:
                break
    return out


def wait_for_server(url, tries=60):
    health = url.rsplit("/v1/", 1)[0] + "/health"
    for i in range(tries):
        try:
            with urllib.request.urlopen(health, timeout=5) as r:
                if r.status == 200:
                    print(f"teacher up after {i}s")
                    return True
        except Exception:
            time.sleep(1)
    return False


def one(a, prompt):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": a.max_new,
        "temperature": a.temperature,
        "top_p": a.top_p,
        "stream": False,
    }
    if not a.think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(
        a.url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=a.timeout) as r:
        obj = json.loads(r.read().decode())
    msg = obj["choices"][0]["message"]
    text = msg.get("content") or ""
    # llama.cpp surfaces the thinking block separately on reasoning models; keep it so the
    # student learns to reason, which is the whole point of reasoning distillation.
    if a.think and msg.get("reasoning_content"):
        text = f"<think>\n{msg['reasoning_content']}\n</think>\n\n{text}"
    return text


def load_done_prompts(out_dir):
    """Resume support: prompts already completed (so a re-run skips them)."""
    done = set()
    stream = os.path.join(out_dir, "stream.jsonl")
    if os.path.exists(stream):
        with open(stream, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["prompt"])
                except Exception:
                    pass
    return done


def main():
    a = parse()
    prompts = load_prompts(a.prompts, a.n)
    os.makedirs(a.out, exist_ok=True)
    # RESUME + CRASH-SAFETY: skip prompts already in stream.jsonl, append new ones as they land.
    # Killing this process at ANY point keeps every completion written so far (fixes the
    # "all-in-memory, lost on kill" flaw).
    done_prompts = load_done_prompts(a.out)
    todo = [p for p in prompts if p not in done_prompts]
    print(f"{len(prompts)} prompts ({len(done_prompts)} already done, {len(todo)} to do), "
          f"think={a.think}, concurrency={a.concurrency}")
    if not todo:
        print("nothing to do; all prompts already generated.")
        print("next: 03_unsloth_sft.py --data", a.out); return
    if not wait_for_server(a.url):
        sys.exit("teacher server never came up - start serve_teacher.sh first")

    stream_path = os.path.join(a.out, "stream.jsonl")
    lock = threading.Lock()
    done = failed = 0
    t0 = time.time()
    fout = open(stream_path, "a", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futs = {ex.submit(one, a, p): p for p in todo}
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                co = fut.result()
                with lock:
                    fout.write(json.dumps({"prompt": p, "completion": co}) + "\n")
                    fout.flush(); os.fsync(fout.fileno())   # durable per completion
            except Exception as e:
                failed += 1
                print(f"  [warn] prompt failed: {type(e).__name__}: {e}", file=sys.stderr)
            done += 1
            if done % 25 == 0 or done == len(todo):
                el = time.time() - t0
                rate = done / el if el else 0
                eta = (len(todo) - done) / rate if rate else 0
                print(f"  {done}/{len(todo)}  {rate:.2f}/s  eta {eta/60:.1f}m  failed={failed}",
                      flush=True)
    fout.close()
    total = len(load_done_prompts(a.out))
    print(f"{done} new ({failed} failed), {total} total saved -> {stream_path}", flush=True)
    print("next: 03_unsloth_sft.py --data", a.out)


if __name__ == "__main__":
    main()
