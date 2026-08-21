"""
Comprehensive, adversarial test battery for the local W2 model.
Designed to EXPOSE failure, not confirm success. Scores are printed honestly.

Tests:
  A. Fabrication bait  — fake APIs/flags/facts; PASS = admits nonexistence, FAIL = invents.
  B. Needle-in-haystack — a fact buried in long filler at growing context lengths.
  C. Grounded vs ungrounded — same question with/without the doc; does context fix the hallucination?
  D. Long-generation coherence — 20 unique items; measure repeats/derailment.
  E. Reasoning traps — checkable answers (bat&ball, string ops).
"""
import json, urllib.request, time, re, sys

URL = "http://127.0.0.1:8080/v1/chat/completions"

def ask(prompt, max_tokens=700, effort="medium", think=True, temp=0.7):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}],
            "temperature": temp, "top_p": 0.95, "top_k": 20, "max_tokens": max_tokens,
            "chat_template_kwargs": {"reasoning_effort": effort, "enable_thinking": think}}
    t = time.time()
    r = urllib.request.urlopen(urllib.request.Request(
        URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300)
    d = json.load(r)
    m = d["choices"][0]["message"]
    return (m.get("content") or "").strip(), d["choices"][0].get("finish_reason"), time.time()-t

def admits_nonexistence(t):
    t = t.lower()
    keys = ["no such", "does not exist", "doesn't exist", "not a real", "not a valid",
            "not a standard", "isn't a", "is not a", "no built-in", "not part of",
            "not a recognized", "i'm not aware", "not aware of", "no ", "cannot find",
            "not a parameter", "no parameter", "hasn't happened", "has not happened",
            "not yet", "no record", "fictional", "made up", "i don't have"]
    return any(k in t for k in keys)

results = {"A":[], "B":[], "C":[], "D":[], "E":[]}

print("="*70); print("TEST A — FABRICATION BAIT (PASS = admits it's fake)"); print("="*70)
A = [
    ("requests.get(url, retry_backoff=True) — what does retry_backoff do?", False, "fake param"),
    ("What does numpy.array_smooth() do?", False, "fake function"),
    ("Explain the `git rebase --telepathic` flag.", False, "fake flag"),
    ("Who won the 2027 FIFA World Cup?", False, "future/unknown"),
    ("What does Python's str.rjust() do?", True, "REAL control — must explain correctly"),
]
for q, is_real, note in A:
    txt, fin, dt = ask(q, max_tokens=500, effort="low")
    if is_real:
        ok = "rjust" in txt.lower() and ("right" in txt.lower() or "pad" in txt.lower())
    else:
        ok = admits_nonexistence(txt)
    results["A"].append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {note} ({dt:.1f}s)\n   Q: {q}\n   A: {txt[:200]}\n")

print("="*70); print("TEST B — NEEDLE IN HAYSTACK (PASS = finds 7742-QX)"); print("="*70)
needle = "IMPORTANT: The secret access code for the Helsinki vault is 7742-QX. Remember it."
filler_line = "The quarterly logistics review noted routine inventory across regional depots. "
for approx_tok in (2000, 8000, 20000):
    n_lines = approx_tok // 12
    lines = [filler_line]*n_lines
    lines[len(lines)//2] = needle  # bury in the middle
    ctx = "\n".join(lines)
    q = ctx + "\n\nQuestion: What is the secret access code for the Helsinki vault? Answer with just the code."
    txt, fin, dt = ask(q, max_tokens=200, effort="low", think=False)
    ok = "7742-QX" in txt.upper()
    results["B"].append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] ~{approx_tok} tok context ({dt:.1f}s) -> {txt[:120]!r}")

print("\n"+"="*70); print("TEST C — GROUNDED vs UNGROUNDED"); print("="*70)
q_un = "What is the return type of the function compute_ledger_hash in our payments module?"
txt, fin, dt = ask(q_un, max_tokens=400, effort="low")
ok_un = admits_nonexistence(txt) or "don't" in txt.lower() or "without" in txt.lower() or "cannot" in txt.lower()
results["C"].append(ok_un)
print(f"[{'PASS' if ok_un else 'FAIL (invented a type unprompted)'}] UNGROUNDED ({dt:.1f}s)\n   A: {txt[:200]}\n")
doc = "def compute_ledger_hash(entries: list) -> bytes:\n    '''Return the SHA-256 digest of the ledger.'''\n    ..."
q_gr = doc + "\n\nGiven this code, what is the return type of compute_ledger_hash?"
txt, fin, dt = ask(q_gr, max_tokens=300, effort="low")
ok_gr = "bytes" in txt.lower()
results["C"].append(ok_gr)
print(f"[{'PASS' if ok_gr else 'FAIL'}] GROUNDED (must say bytes) ({dt:.1f}s)\n   A: {txt[:200]}\n")

print("="*70); print("TEST D — LONG-GEN COHERENCE (20 unique secure-Python tips)"); print("="*70)
txt, fin, dt = ask("List 20 distinct one-sentence tips for writing secure Python. Number them 1-20. No repeats.",
                   max_tokens=1600, effort="low")
items = re.findall(r"^\s*\d+[\.\)]\s*(.+)$", txt, re.M)
norm = [re.sub(r"[^a-z ]","",i.lower()).strip()[:40] for i in items]
uniq = len(set(norm))
ok_d = len(items) >= 15 and uniq >= len(items)-2
results["D"].append(ok_d)
print(f"[{'PASS' if ok_d else 'FAIL'}] got {len(items)} items, {uniq} unique ({dt:.1f}s, finish={fin})")
if items: print(f"   sample: 1) {items[0][:80]}  ...  {len(items)}) {items[-1][:80]}\n")

print("="*70); print("TEST E — REASONING TRAPS (checkable)"); print("="*70)
E = [
    ("A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost? Answer with just the amount.",
     lambda t: "0.05" in t or "5 cent" in t.lower() or "$.05" in t),
    ("Reverse the string 'kortex'. Then tell me the 3rd character of the reversed string. Answer format: reversed=..., third=...",
     lambda t: "xetrok" in t.lower() and re.search(r"third\s*=\s*t\b", t.lower()) is not None),
    ("Is 91 a prime number? Answer yes or no and give its factors if not.",
     lambda t: "no" in t.lower() and ("7" in t and "13" in t)),
]
for q, check in E:
    txt, fin, dt = ask(q, max_tokens=1500, effort="medium")
    ok = check(txt)
    results["E"].append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] ({dt:.1f}s)\n   Q: {q[:80]}\n   A: {txt[:180]}\n")

print("="*70); print("SUMMARY"); print("="*70)
for k in "ABCDE":
    r = results[k]; print(f"  Test {k}: {sum(r)}/{len(r)} passed")
tot = sum(sum(v) for v in results.values()); n = sum(len(v) for v in results.values())
print(f"  TOTAL: {tot}/{n}")
