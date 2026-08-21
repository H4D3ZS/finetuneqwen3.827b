#!/usr/bin/env python3
"""
Compile a STABLE "repo context capsule" and bake it into a llama.cpp KV slot.

This is the local, no-rental precursor to the learned latent core (LCC): the
repo's *structure* (file tree + key definitions) rarely changes, so we process
it ONCE, save the KV to disk, and restore it in ~70ms at the start of every
session. The model then carries the repo context as resident KV — never
re-tokenized, never re-processed — instead of paying for it in prompt tokens
each turn. Volatile file *contents* still flow through the live hybrid
retrieval path (kortex); only the stable structural spine is capsuled.

Measured on this box (W2 27B): a 1890-token capsule = 4.7s to process cold,
then 70ms to restore forever after, 0 tokens re-processed.

Why a stable capsule beats plain prompt-caching: prompt-cache invalidates the
instant any byte of the prefix changes (the paper's 95% invalidation problem).
The structural capsule is built to be byte-identical across turns even as you
edit files — it holds the tree + definition SIGNATURES, not the churning bodies.

Usage:
    # 1) compile the capsule text (deterministic, sorted -> stable bytes)
    python kv_capsule.py compile --repo .. --out capsule.txt
    # 2) warm it into slot 0 and save the KV (server must run with --slot-save-path)
    python kv_capsule.py bake --capsule capsule.txt --server http://127.0.0.1:8080 \
        --slot-file repo-capsule.bin
    # 3) at serve start, restore it (or let serve_stack do it):
    python kv_capsule.py restore --server http://127.0.0.1:8080 --slot-file repo-capsule.bin
"""
import argparse, json, os, re, sys, urllib.request

# Language-agnostic definition patterns: capture the SIGNATURE line, not the body.
DEF_PATTERNS = [
    re.compile(r'^\s*(?:pub\s+)?(?:async\s+)?fn\s+\w+.*'),           # rust
    re.compile(r'^\s*(?:export\s+)?(?:async\s+)?function\s+\w+.*'),  # js/ts
    re.compile(r'^\s*def\s+\w+.*:'),                                 # python
    re.compile(r'^\s*(?:pub\s+)?(?:struct|enum|trait|impl|type)\s+\w+.*'),
    re.compile(r'^\s*class\s+\w+.*'),
    re.compile(r'^\s*(?:export\s+)?(?:const|let|var)\s+[A-Z_][A-Z0-9_]*\s*='),  # CONSTS
    re.compile(r'^\s*[A-Z_][A-Z0-9_]{2,}\s*='),                      # shell/env CONSTS
]
CODE_EXT = {".rs", ".py", ".js", ".ts", ".sh", ".go", ".c", ".h", ".cpp", ".toml", ".jinja"}
IGNORE_DIRS = {".git", "target", "node_modules", "ROCmFPX", ".aim", ".kv-slots",
               ".kv-cache", ".stack-logs", "__pycache__", "teacher_bulk_235b",
               "teacher_bulk_37plus", "teacher_seed_max", "teacher_data_val"}


# Generated artifacts must never be indexed, or the capsule includes its own
# output and stops being byte-stable across runs.
IGNORE_FILE_RE = re.compile(r'^(capsule\d*\.txt|.*\.bin)$')


def compile_capsule(repo: str) -> str:
    repo = os.path.abspath(repo)
    tree, defs = [], []
    for root, dirs, files in os.walk(repo):
        dirs[:] = sorted(d for d in dirs if d not in IGNORE_DIRS and not d.startswith("."))
        rel_root = os.path.relpath(root, repo)
        for fn in sorted(files):
            if IGNORE_FILE_RE.match(fn):
                continue
            ext = os.path.splitext(fn)[1]
            rel = os.path.normpath(os.path.join(rel_root, fn)).replace("\\", "/")
            tree.append(rel)
            if ext not in CODE_EXT:
                continue
            try:
                with open(os.path.join(root, fn), encoding="utf-8", errors="ignore") as f:
                    lines = f.read().splitlines()
            except OSError:
                continue
            sigs = [ln.rstrip() for ln in lines
                    if any(p.match(ln) for p in DEF_PATTERNS)]
            if sigs:
                defs.append((rel, sigs[:40]))  # cap per file to bound size
    out = ["REPO STRUCTURE CAPSULE (stable; definitions, not bodies).", "", "## Files"]
    out += sorted(tree)
    out += ["", "## Definitions"]
    for rel, sigs in sorted(defs):
        out.append(f"### {rel}")
        out += sigs
    # ascii-safe so the bytes are stable regardless of console codepage
    return "\n".join(out).encode("ascii", "ignore").decode()


def _post(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def _meta_path(slot_file: str) -> str:
    return slot_file + ".meta"


def _capsule_fingerprint(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def bake(capsule_path: str, server: str, slot_file: str, slot: int) -> None:
    text = open(capsule_path, encoding="utf-8").read()
    print(f"capsule: {len(text)} chars (~{len(text)//4} tokens) -> warming slot {slot}")
    d = _post(f"{server}/v1/chat/completions", {
        "model": "capsule", "messages": [{"role": "user", "content": text}],
        "max_tokens": 1, "cache_prompt": True})
    t = d.get("timings", {})
    print(f"  processed prompt_n={t.get('prompt_n')} in {round(t.get('prompt_ms',0))}ms")
    d = _post(f"{server}/slots/{slot}?action=save", {"filename": slot_file})
    print(f"  saved KV: n_saved={d.get('n_saved')} bytes={d.get('n_written')} "
          f"in {round(d.get('timings',{}).get('save_ms',0))}ms -> {slot_file}")
    # Guard sidecar: restoring stale KV against a changed model/capsule is
    # undefined, so record what this KV was baked against.
    meta = {"capsule_sha": _capsule_fingerprint(text),
            "model": os.environ.get("CAPSULE_MODEL", ""),
            "n_saved": d.get("n_saved")}
    open(_meta_path(slot_file), "w", encoding="utf-8").write(json.dumps(meta))


def restore(server: str, slot_file: str, slot: int, capsule_path: str = "") -> int:
    # If we can see the current capsule + the bake sidecar, refuse a stale
    # restore (guards against loading KV baked for a different model/capsule).
    mp = _meta_path(slot_file)
    if capsule_path and os.path.exists(capsule_path) and os.path.exists(mp):
        want = _capsule_fingerprint(open(capsule_path, encoding="utf-8").read())
        meta = json.loads(open(mp, encoding="utf-8").read())
        cur = os.environ.get("CAPSULE_MODEL", meta.get("model", ""))
        if meta.get("capsule_sha") != want or (cur and cur != meta.get("model", cur)):
            print("capsule stale (capsule text or model changed) — skipping restore; re-bake.")
            return 1
    d = _post(f"{server}/slots/{slot}?action=restore", {"filename": slot_file})
    print(f"restored KV: n_restored={d.get('n_restored')} "
          f"in {round(d.get('timings',{}).get('restore_ms',0))}ms (repo context now resident)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compile"); c.add_argument("--repo", default=".."); c.add_argument("--out", default="capsule.txt")
    b = sub.add_parser("bake"); b.add_argument("--capsule", default="capsule.txt")
    b.add_argument("--server", default="http://127.0.0.1:8080"); b.add_argument("--slot-file", default="repo-capsule.bin"); b.add_argument("--slot", type=int, default=0)
    r = sub.add_parser("restore"); r.add_argument("--server", default="http://127.0.0.1:8080")
    r.add_argument("--slot-file", default="repo-capsule.bin"); r.add_argument("--slot", type=int, default=0)
    r.add_argument("--capsule", default="capsule.txt", help="staleness guard against this capsule text")
    a = ap.parse_args()
    if a.cmd == "compile":
        text = compile_capsule(a.repo)
        open(a.out, "w", encoding="utf-8").write(text)
        print(f"compiled {a.out}: {len(text)} chars (~{len(text)//4} tokens)")
    elif a.cmd == "bake":
        bake(a.capsule, a.server, a.slot_file, a.slot)
    elif a.cmd == "restore":
        return restore(a.server, a.slot_file, a.slot, a.capsule)
    return 0


if __name__ == "__main__":
    sys.exit(main())
