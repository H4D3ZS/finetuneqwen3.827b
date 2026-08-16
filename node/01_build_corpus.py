#!/usr/bin/env python3
"""
Step 1: build the distillation corpus - the PROMPTS the teacher will answer.

Distill NARROW and AGENTIC. A 3B-active student absorbs a focused distribution far better than
a broad one, so the corpus is agentic coding + tool use + repo edits - the exact jobs where the
abliterated Qwen3.8-27B teacher is strong, and exactly where the 850-example PoC was WEAK
(both tool-call eval tasks failed).

KEY DESIGN CHANGE (v2): the dominant source is `cc_tools` - scenarios phrased with Claude Code's
REAL tool schemas (Bash/Read/Edit/Write/Grep/Glob). We do this because the student is meant to be
used INSIDE Claude Code; training on the exact tool protocol it will be invoked with is what makes
tool-calling transfer instead of collapsing (the PoC lesson).

Sources (comma list via --sources, weighted by --weights):
  cc_tools           Claude-Code-native tool-call scenarios (DEFAULT-HEAVY)
  agentic            multi-step repo tasks (locate -> edit -> test -> verify)
  swebench           real issue statements (HF datasets)
  debug              "here's a failing test / traceback, fix it" tasks
  local_transcripts  YOUR own Claude Code sessions (best signal; needs --local_glob)
  seed               small hand-written fallback

    python3 01_build_corpus.py --out prompts.jsonl --n 8000 \
        --sources cc_tools,agentic,swebench,debug --weights 4,2,2,2

Scale: 6-10k focused prompts for a strong LoRA run (PoC was 850). 100k+ only for full-parameter.
"""
import argparse, os, json, glob, random, re

# --- real coding datasets + secret scrub (merged from the API-teacher line of work) ---
# Self-contained coding PROMPTS from public datasets; the teacher regenerates all completions.
CODE_DATASETS = [
    ("ise-uiuc/Magicoder-OSS-Instruct-75K", ("problem", "instruction"), None, "train"),
    ("ise-uiuc/Magicoder-Evol-Instruct-110K", ("instruction",), None, "train"),
    ("glaiveai/glaive-code-assistant", ("question",), None, "train"),
    ("m-a-p/CodeFeedback-Filtered-Instruction", ("query", "instruction"), None, "train"),
]
_SECRET_RE = re.compile(
    r"(whsk_[A-Za-z0-9]{8,}|sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}"
    r"|AKIA[0-9A-Z]{16}|hf_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}"
    r"|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})")
def _scrub(s):
    """Redact secrets scraped from real transcripts so the student can't memorize them."""
    return _SECRET_RE.sub("<REDACTED>", s)

# --- teacher routing: keep the CENSORED Empero teacher away from offensive-security prompts ---
# The abliterated 27B is the SOLE teacher for anything offensive; Empero (not abliterated, would
# refuse and re-poison compliance) only ever sees "general" prompts. Match clearly-offensive
# tokens only - NOT bare "security" (so the benign "Harden the security of ..." task stays general).
_SENSITIVE_RE = re.compile(r"""(?ix)
    \b(exploit|payload|shellcode|reverse[\s-]?shell|privilege[\s-]?escalation|priv[\s-]?esc
    |rce|remote[\s-]code[\s-]execution|lfi|rfi|sqli|sql[\s-]injection|xss|csrf[\s-]bypass
    |malware|ransomware|keylogger|rootkit|backdoor|botnet|\bc2\b|command[\s-]and[\s-]control
    |metasploit|msfvenom|cobalt[\s-]strike|mimikatz|hashcat|hydra|\bnmap\b|\bburp\b|sqlmap
    |crack(ing|ed|\s+the)?[\s-]?(password|hash|key)?|keygen|crackme|bypass[\s-]auth\w*
    |deobfuscat|decompil|disassembl|ghidra|ida[\s-]pro|\bctf\b|hack[\s-]?the[\s-]?box|\bhtb\b
    |vulnerab\w+|\bcve-|buffer[\s-]overflow|heap[\s-]spray|rop[\s-]chain|dll[\s-]inject\w*)\b
""")
def route_for(prompt, source):
    """'sensitive' -> abliterated 27B only; 'general' -> Empero-eligible."""
    if source == "local_transcripts":          # your private HTB/RE sessions: always sensitive
        return "sensitive"
    return "sensitive" if _SENSITIVE_RE.search(prompt) else "general"

def from_hf_instruct(limit):
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [instruct] pip install datasets to use this source; skipping"); return []
    out, per = [], max(1, limit // len(CODE_DATASETS))
    for ds_id, fields, cfg, split in CODE_DATASETS:
        got = 0
        try:
            for r in load_dataset(ds_id, cfg, split=split, streaming=True):
                txt = next((r[f] for f in fields if r.get(f)), None)
                if txt and 30 <= len(txt) <= 2000 and "�" not in txt:
                    out.append(txt); got += 1
                    if got >= per: break
        except Exception as e:
            print(f"  [instruct] {ds_id} failed ({type(e).__name__}); skipping")
    print(f"  [instruct] {len(out)} real dataset prompts"); return out

CC_TOOLS = {
    "Bash":  'run a shell command. args: {"command": str, "description": str, "timeout"?: int}',
    "Read":  'read a file. args: {"file_path": str, "offset"?: int, "limit"?: int}',
    "Edit":  'exact string replace in a file. args: {"file_path": str, "old_string": str, "new_string": str, "replace_all"?: bool}',
    "Write": 'write/overwrite a file. args: {"file_path": str, "content": str}',
    "Grep":  'ripgrep search. args: {"pattern": str, "path"?: str, "glob"?: str, "output_mode"?: "content|files_with_matches"}',
    "Glob":  'find files by glob. args: {"pattern": str, "path"?: str}',
}

# Combinatorial task generation: ACTION x SUBSYSTEM x STACK x QUALITY-BAR gives tens of
# thousands of distinct, realistic repo tasks - far more diverse than a fixed list, which is
# what a 3B-active student needs to generalize tool-use instead of memorizing phrasings.
ACTIONS = [
    "Find and fix the bug causing", "Add a feature to", "Refactor for clarity",
    "Optimize the hot path in", "Add a regression test for", "Harden the security of",
    "Migrate", "Add observability (logs+metrics) to", "Make idempotent", "Add input validation to",
    "Remove the N+1 query in", "Add a feature flag around", "Write integration tests for",
    "Diagnose the memory leak in", "Add retry-with-backoff to", "Rate-limit",
]
SUBSYSTEMS = [
    "the JWT auth middleware", "the payment/checkout handler", "the file-upload endpoint",
    "the search indexer", "the websocket gateway", "the background job queue",
    "the inventory update path", "the user-profile service", "the CSV export job",
    "the OAuth callback flow", "the caching layer", "the database migration runner",
    "the rate limiter", "the email-sending worker", "the report generator",
    "the API pagination logic", "the session store", "the feature-flag service",
    "the audit-log writer", "the third-party geocoder client",
]
STACKS = [
    "Python/FastAPI", "Python/Django", "Go/chi", "TypeScript/Express", "TypeScript/Next.js",
    "Rust/axum", "Java/Spring Boot", "Ruby/Rails", "Node/NestJS", "C#/ASP.NET",
]
BARS = [
    "Keep all existing tests green.", "Add a test that would have caught the bug.",
    "Explain the root cause before editing.", "Show the exact minimal diff.",
    "Verify with a command after the change.", "Note any edge cases you handled.",
]

def _make_task():
    return (f"{random.choice(ACTIONS)} {random.choice(SUBSYSTEMS)} "
            f"in a {random.choice(STACKS)} codebase. {random.choice(BARS)}")

def _tool_block(k):
    names = random.sample(list(CC_TOOLS), k=k)
    return "\n".join(f"- {n}: {CC_TOOLS[n]}" for n in names)

def from_cc_tools(limit):
    out = []
    for _ in range(limit):
        task = _make_task()
        tools = _tool_block(random.randint(3, len(CC_TOOLS)))
        out.append(
            "You are an agentic coding assistant operating in a real repository with these tools:\n"
            f"{tools}\n\n"
            f"Task: {task}\n\n"
            "Work step by step. When you need to act, emit a tool call as a JSON object "
            '{\"tool\": <name>, \"args\": {...}} with correctly-typed, correctly-nested arguments. '
            "Inspect before you edit, and verify with a test or command after."
        )
    print(f"  [cc_tools] {len(out)}"); return out

# Authorized offensive-security scenarios (HackTheBox / CTF / a lab box YOU own). Same tool-call
# protocol as from_cc_tools, so the student learns tool-use AND uncensored agentic security
# together. Every prompt is explicitly authorized-context; these route=sensitive (the keywords
# trip _SENSITIVE_RE), so ONLY the abliterated 27B teacher ever sees them - never censored Empero.
SEC_ACTIONS = [
    "enumerate services and find the initial foothold on",
    "find a privilege-escalation path to root on",
    "get a reverse shell from",
    "find and exploit the SQL injection in",
    "crack the captured password hashes from",
    "escalate from the www-data shell to root on",
    "find the SUID / sudo / setcap misconfiguration on",
    "exploit the local file inclusion (LFI) in",
    "pivot to the internal subnet from the foothold on",
    "decompile the binary and find the exploitable buffer overflow in",
    "enumerate the SMB/LDAP shares and extract credentials from",
    "exploit the deserialization vulnerability in",
    "find the exposed .git / backup / config leaking secrets on",
    "write a working exploit for the known CVE affecting",
]
SEC_TARGETS = [
    "the HackTheBox machine you own", "a lab VM you are authorized to test",
    "the CTF web application", "the intentionally-vulnerable API",
    "the boot2root box on your home lab", "the vulnerable binary from the CTF",
    "the retired HackTheBox target", "the pentest-engagement host in scope",
]

def from_cybersec(limit):
    out = []
    for _ in range(limit):
        task = f"{random.choice(SEC_ACTIONS)} {random.choice(SEC_TARGETS)}"
        tools = _tool_block(random.randint(3, len(CC_TOOLS)))
        out.append(
            "You are an agentic penetration-testing assistant on a target you are EXPLICITLY "
            "authorized to test (HackTheBox / CTF / your own lab). Tools available:\n"
            f"{tools}\n\n"
            f"Task: {task}. Work step by step - enumerate before you exploit.\n\n"
            "When you need to act, emit a tool call as a JSON object "
            '{\"tool\": <name>, \"args\": {...}} with correctly-typed arguments, then interpret '
            "the output and proceed to the next step."
        )
    print(f"  [cybersec] {len(out)}"); return out

def from_agentic(limit):
    out = []
    for _ in range(limit):
        task = _make_task()
        out.append(
            "You are working in a real codebase. Complete this task end to end: "
            "locate the relevant files, make the change, and verify it.\n\n"
            f"Task: {task}\n\n"
            "Explain your plan, show the exact diffs, and state how you verified the result."
        )
    print(f"  [agentic] {len(out)}"); return out

def from_debug(limit):
    tracebacks = [
        ("Python", "TypeError: 'NoneType' object is not subscriptable", "user_service.py", "get_profile"),
        ("Python", "sqlalchemy.exc.OperationalError: database is locked", "db.py", "commit"),
        ("Python", "RecursionError: maximum recursion depth exceeded", "tree.py", "walk"),
        ("Python", "KeyError: 'user_id'", "session.py", "load"),
        ("Go",     "panic: runtime error: index out of range [3] with length 3", "router.go", "dispatch"),
        ("Go",     "fatal error: concurrent map writes", "cache.go", "Set"),
        ("Go",     "panic: send on closed channel", "worker.go", "enqueue"),
        ("Rust",   "thread 'main' panicked at 'called `Result::unwrap()` on an `Err`'", "cache.rs", "insert"),
        ("Rust",   "thread 'main' has overflowed its stack", "parser.rs", "parse_expr"),
        ("TypeScript", "Uncaught (in promise) TypeError: cannot read properties of undefined (reading 'id')", "cart.ts", "checkout"),
        ("TypeScript", "RangeError: Maximum call stack size exceeded", "reducer.ts", "merge"),
        ("Java",   "java.lang.NullPointerException", "OrderService.java", "process"),
        ("Java",   "java.util.ConcurrentModificationException", "Registry.java", "iterate"),
        ("Ruby",   "ActiveRecord::RecordNotFound", "orders_controller.rb", "show"),
        ("C#",     "System.InvalidOperationException: Collection was modified", "Cache.cs", "Evict"),
        ("Node",   "Error: ECONNRESET", "client.js", "request"),
    ]
    stacks = ["under load", "only in production", "intermittently in CI", "after the last deploy",
              "for large inputs", "on concurrent requests", "on the retry path"]
    out = []
    for _ in range(limit):
        lang, err, fn, func = random.choice(tracebacks)
        cond = random.choice(stacks)
        err = f"{err}   ({cond})"
        out.append(
            f"A {lang} service is crashing in production with:\n\n    {err}\n\n"
            f"It points at `{func}` in `{fn}`. Reason about the root cause, ask for the minimal "
            "code you'd need to see, then give the fix and a regression test that would have caught it."
        )
    print(f"  [debug] {len(out)}"); return out

def from_swebench(hf_id, limit):
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [swebench] `pip install datasets` to use; skipping"); return []
    try:
        ds = load_dataset(hf_id, split="test")
    except Exception as e:
        print(f"  [swebench] load failed: {e}; skipping"); return []
    out = []
    for r in ds:
        problem = r.get("problem_statement") or r.get("text") or ""
        if not problem: continue
        out.append("You are an agentic coding assistant in a real repository. Read the issue, "
                   "locate the relevant files, implement the fix, and add/adjust tests.\n\n"
                   f"Issue:\n{problem}\n\nProduce the patch and explain each change.")
        if len(out) >= limit: break
    print(f"  [swebench] {len(out)}"); return out

def from_local(glob_pat, limit):
    if not glob_pat: return []
    out = []
    for fn in glob.glob(glob_pat, recursive=True):
        try:
            with open(fn, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    try: obj = json.loads(line)
                    except Exception: continue
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
    print(f"  [local] {len(out)} from your transcripts"); return out

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="prompts.jsonl")
    p.add_argument("--n", type=int, default=8000)
    p.add_argument("--sources", default="cc_tools,agentic,swebench,debug")
    p.add_argument("--weights", default="", help="comma weights matching --sources, e.g. 4,2,2,2")
    p.add_argument("--local_glob", default="")
    p.add_argument("--hf_swebench", default="princeton-nlp/SWE-bench_Lite")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def main():
    a = parse(); random.seed(a.seed)
    want = [s for s in a.sources.split(",") if s]
    weights = [int(x) for x in a.weights.split(",")] if a.weights else [1]*len(want)
    weights = (weights + [1]*len(want))[:len(want)]
    tot = sum(weights)
    quota = {s: max(1, a.n * w // tot) for s, w in zip(want, weights)}
    pool = []   # list of (prompt, source) so each prompt keeps its origin for routing
    def add(src, items): pool.extend((p, src) for p in items)
    if "cc_tools" in quota:          add("cc_tools", from_cc_tools(quota["cc_tools"]))
    if "cybersec" in quota:          add("cybersec", from_cybersec(quota["cybersec"]))
    if "agentic" in quota:           add("agentic", from_agentic(quota["agentic"]))
    if "debug" in quota:             add("debug", from_debug(quota["debug"]))
    if "swebench" in quota:          add("swebench", from_swebench(a.hf_swebench, quota["swebench"]))
    if "instruct" in quota:          add("instruct", from_hf_instruct(quota["instruct"]))
    if "local_transcripts" in quota: add("local_transcripts", from_local(a.local_glob, quota["local_transcripts"]))
    if "seed" in quota or not pool:
        add("seed", ["Implement a thread-safe LRU cache in Rust with unit tests.",
                     "Write a Python function to parse a nested JSON config with validation."] * quota.get("seed", 50))
    random.shuffle(pool); pool = pool[:a.n]
    # dedup on prompt text while preserving order
    seen=set(); uniq=[(p, s) for (p, s) in pool if not (p in seen or seen.add(p))]
    n_sens = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for p, s in uniq:
            p = _scrub(p)                       # redact any scraped secrets FIRST
            route = route_for(p, s)             # then classify the scrubbed text
            n_sens += route == "sensitive"
            f.write(json.dumps({"prompt": p, "route": route}) + "\n")
    print(f"wrote {len(uniq)} unique prompts -> {a.out} "
          f"({n_sens} sensitive -> abliterated 27B only, {len(uniq)-n_sens} general -> Empero-eligible)")
    print("next: 02_teacher_gguf.py --prompts", a.out)

if __name__ == "__main__":
    main()
