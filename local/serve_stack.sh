#!/usr/bin/env bash
# The local Claude Code stack — ROCmFPX × kortex.  Full architecture + credits: ../STACK.md
#   GPU inference engine: ROCmFPX by Carlo (charlie12345, github.com/charlie12345/ROCmFPX)
#   Repo memory / retrieval: kortex by Rolando Ferrer (H4D3ZS, Cyber Ifrit)
#
# Bring up the FULL local Claude Code stack in one command, with the two fixes that
# otherwise make it silently fail on Windows baked in:
#
#   Claude Code ─(Anthropic /v1/messages)─▶ aim-proxy :1536  ── kortex repo memory (retrieval)
#                                              │
#                                              ▼ (Anthropic, augmented)
#                                           LiteLLM :4000     ── Anthropic➜OpenAI translation
#                                              │
#                                              ▼ (OpenAI /v1)
#                                           llama-server :8080 ── the model on your GPU (ROCmFPX)
#
# WHY THIS SCRIPT EXISTS (two non-obvious Windows failures it fixes):
#   1. LiteLLM crashes at startup printing its banner: Windows console is cp1252 and the banner
#      has emoji/box-drawing chars -> UnicodeEncodeError, port never binds. Fix: PYTHONUTF8=1.
#   2. aim-proxy's retrieval budget defaults to 100 ms, but retrieval here takes ~300 ms, so it
#      forwards UN-AUGMENTED (memory silently off). Fix: KORTEX_RETRIEVAL_BUDGET_MS=1500.
#
# Usage:
#   bash local/serve_stack.sh            # serve the current MODEL (default: Escha A3B test model)
#   MODEL=/path/to/dense.gguf ALIAS=qwen38-27b bash local/serve_stack.sh
#
# Then point Claude Code at the stack (prints these at the end too):
#   export ANTHROPIC_BASE_URL=http://127.0.0.1:1536
#   export ANTHROPIC_AUTH_TOKEN=sk-local
#   export ANTHROPIC_MODEL=$ALIAS
#   export ANTHROPIC_SMALL_FAST_MODEL=$ALIAS
#   claude
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KORTEX="${KORTEX_DIR:-/c/Users/HADES/Documents/vscodium-rust/kortex}"
SERVER="${SERVER:-$HOME/Desktop/ROCmFPX/build/bin/Release/llama-server.exe}"
PROXY="${AIM_PROXY:-$KORTEX/target/release/aim-proxy.exe}"

# Default model = OUR OWN calibrated W2 (Q2_0_ROCMFPX 2.79bpw, 8.9GB) of dense Qwen3.8-27B, built
# locally WITH an imatrix (local/imatrix-calib.txt) + the MTP head (blk.64) forced to q8_0 so the
# imatrix check doesn't bail. Coherent, fits 40K ctx, 37-57 t/s w/ MTP. NOTE: naive 2-bit (no imatrix)
# BROKE (incoherent code); plain Q3_0_ROCMFPX is 14.5GB (too big, 8 t/s); Q3_0_ROCMFPX_AGENT is 18.6GB
# (needs 24GB). See [[local-quant-pipeline]]. Escha A3B fallback: MODEL=.../Qwen3.6-35B-A3B-Escha-W2-ROCmFP2.gguf.
MODEL="${MODEL:-/c/Users/HADES/Desktop/Qwen3.8-27B-W2imat-ROCmFPX.gguf}"
ALIAS="${ALIAS:-qwen38-27b-w2i}"
# 16GB budget: ~11.4GB model + compute buffers leaves room for KV. q4 KV keeps ~40K context resident.
CTX="${CTX:-40960}"   # under the 64K MTP cliff (reference/reference-MTP-and-cliff-findings.md)
KVTYPE="${KVTYPE:-q4_0}"   # q4_0 KV to fit long context beside the 11.4GB model; q8_0 if you drop CTX
# Reasoning effort: stock Qwen3.5 template defaults to 'xhigh' (overthinks). Force 'medium'. The Sharp
# template (froggeric) is used by default for terse, low-filler output; it also defaults to medium.
REASONING="${REASONING:-medium}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$REPO/local/qwen-sharp-chat_template.jinja}"
LOGDIR="${LOGDIR:-$REPO/.stack-logs}"; mkdir -p "$LOGDIR"
# Prompt-cache wins (all already in the ROCmFPX binary — see kortex-latent/ROADMAP.md):
#  cache-reuse  = reuse KV via shifting even when the prefix partially changes (huge for agentic loops)
#  cache-disk   = SSD-backed prompt cache that survives restarts
#  slot-save    = enables /slots/{id}?action=save|restore — the KV-injection path for the kortex LCC core
CACHEDIR="${CACHEDIR:-$REPO/.kv-cache}"; mkdir -p "$CACHEDIR"
SLOTDIR="${SLOTDIR:-$REPO/.kv-slots}"; mkdir -p "$SLOTDIR"
# Combined drafters: MTP handles novel tokens, ngram-map-k makes repeated/copied code near-free
# (verified: acceptance 0.72->0.96 on copy-heavy edits). Comma-combine works on ROCmFPX. Swap to
# "draft-mtp,ngram-map-k,draft-eagle3" once we have an EAGLE-3 draft. Solo options still valid.
SPECTYPE="${SPECTYPE:-draft-mtp,ngram-map-k}"

for f in "$SERVER" "$PROXY" "$MODEL"; do
  [ -e "$f" ] || { echo "MISSING: $f" >&2; exit 1; }
done

free_port() { local p="$1"; local pid
  pid="$(netstat -ano 2>/dev/null | grep ":$p " | grep LISTEN | awk '{print $NF}' | head -1)"
  [ -n "${pid:-}" ] && { echo "  freeing :$p (pid $pid)"; taskkill //PID "$pid" //F >/dev/null 2>&1 || true; sleep 1; }
}
wait_port() { local p="$1" n="${2:-30}" i
  for i in $(seq 1 "$n"); do netstat -ano 2>/dev/null | grep ":$p " | grep -q LISTEN && return 0; sleep 1; done; return 1; }

echo "== clearing ports 8080 / 4000 / 1536"; free_port 8080; free_port 4000; free_port 1536

echo "== 1/3 llama-server ($ALIAS) on :8080"
# MTP drives speculative decode (the Q3 carries the MTP head). Sharp template (--chat-template-file)
# gives terse, low-filler output; --chat-template-kwargs forces reasoning_effort=medium so thinking
# stays moderate, not xhigh runaway. q4 KV (-ctk/-ctv) keeps ~40K context resident beside the model.
"$SERVER" -m "$MODEL" --alias "$ALIAS" --host 127.0.0.1 --port 8080 \
  -c "$CTX" -fa on -ctk "$KVTYPE" -ctv "$KVTYPE" -ngl 99 -np 1 --no-mmap --jinja \
  --chat-template-file "$CHAT_TEMPLATE" \
  --chat-template-kwargs "{\"reasoning_effort\":\"$REASONING\"}" \
  --cache-reuse 256 --cache-disk "$CACHEDIR" --cache-disk-limit 8192 --slot-save-path "$SLOTDIR" \
  --spec-type "$SPECTYPE" --spec-draft-ngl all --spec-draft-n-max 3 --spec-draft-p-min 0.1 \
  > "$LOGDIR/llama.log" 2>&1 &
wait_port 8080 60 && echo "   up" || { echo "   llama-server failed; see $LOGDIR/llama.log"; exit 1; }

# Optional: restore the repo KV capsule so repo context is resident from turn 1
# (see local/KV_CAPSULE.md). Off by default — the guard skips a stale capsule.
if [ "${KV_CAPSULE:-0}" = "1" ]; then
  echo "== KV capsule restore (repo context resident)"
  ( cd "$REPO/local" && CAPSULE_MODEL="$MODEL" PYTHONUTF8=1 python kv_capsule.py restore \
      --slot-file "$SLOTDIR/repo-capsule.bin" --capsule capsule.txt || true )
fi

echo "== 2/3 LiteLLM gateway on :4000 (PYTHONUTF8=1 fixes the banner crash)"
( cd "$REPO/local" && PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  litellm --config litellm_config.yaml --port 4000 > "$LOGDIR/litellm.log" 2>&1 & )
wait_port 4000 40 && echo "   up" || { echo "   litellm failed; see $LOGDIR/litellm.log"; exit 1; }

echo "== 3/3 aim-proxy (kortex memory) on :1536 (budget 1500ms so retrieval actually fires)"
KORTEX_AIM_CATALOG="$REPO/.aim" KORTEX_WORKSPACE="$REPO" \
KORTEX_UPSTREAM_ANTHROPIC="http://127.0.0.1:4000" \
KORTEX_RETRIEVAL_BUDGET_MS="${KORTEX_RETRIEVAL_BUDGET_MS:-1500}" \
"$PROXY" > "$LOGDIR/aim-proxy.log" 2>&1 &
wait_port 1536 20 && echo "   up" || { echo "   aim-proxy failed; see $LOGDIR/aim-proxy.log"; exit 1; }

cat <<EOF

== STACK UP.  Point Claude Code at it:

  export ANTHROPIC_BASE_URL=http://127.0.0.1:1536
  export ANTHROPIC_AUTH_TOKEN=sk-local
  export ANTHROPIC_MODEL=$ALIAS
  export ANTHROPIC_SMALL_FAST_MODEL=$ALIAS
  claude

Logs: $LOGDIR/{llama,litellm,aim-proxy}.log
Rebuild the memory catalog after big doc changes:
  # SEMANTIC retrieval (needs Lemonade on :13305 serving Qwen3-Embedding-0.6B). Drop --model/--backend
  # to fall back to the lexical hash embedder. See kortex 'semantic-retrieval' branch.
  $KORTEX/target/release/aim-index.exe build "$REPO" --out "$REPO/.aim" --model Qwen3-Embedding-0.6B-GGUF --backend lemonade --ignore ROCmFPX --ignore teacher_bulk_235b --ignore teacher_bulk_37plus --ignore teacher_seed_max --ignore teacher_data_val --ignore node --ignore .stack-logs
EOF
