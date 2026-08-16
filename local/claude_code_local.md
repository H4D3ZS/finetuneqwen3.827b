# Local Claude Code backend — abliterated Qwen3.8-27B via ROCmFPX

Use your own GPU as a Claude Code backend when you hit your Anthropic usage limit. Three parts:

```
Claude Code ──(Anthropic /v1/messages)──▶ LiteLLM gateway ──(OpenAI /v1)──▶ llama-server (ROCmFPX)
   :env                                      :4000                              :8080
```

**Why the gateway:** Claude Code speaks the Anthropic Messages API; llama-server (and ROCmFPX)
speak the OpenAI API. LiteLLM translates between them, including streaming. It is the
maintained, correct path — do not hand-roll an SSE proxy.

---

## 1. Serve the model (terminal 1)

```bash
# one-time: grab a quant that fits 16GB (Q3_K_M leaves ~2.5GB for context)
huggingface-cli download Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF \
  Qwen3.8-27B-ABLITERATED-Q3_K_M.gguf --local-dir ~/models

MODEL=~/models/Qwen3.8-27B-ABLITERATED-Q3_K_M.gguf ./serve_coder.sh
# -> OpenAI API on http://127.0.0.1:8080/v1
```

## 2. Run the LiteLLM gateway (terminal 2)

```bash
pip install "litellm[proxy]"
litellm --config litellm_config.yaml     # listens on :4000, exposes /v1/messages
```

`litellm_config.yaml` (in this folder) points a model name at the local llama-server.

## 3. Point Claude Code at it (terminal 3)

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:4000
export ANTHROPIC_AUTH_TOKEN=sk-local            # any non-empty string; llama-server ignores it
export ANTHROPIC_MODEL=qwen38-coder-abliterated
export ANTHROPIC_SMALL_FAST_MODEL=qwen38-coder-abliterated   # so background calls use local too
claude
```

Unset those three env vars (or open a fresh shell) to switch back to the real Anthropic API.

---

## Honest caveats

- **This 27B is DENSE — no MoE, no MTP.** It is a capable fallback but SLOWER than your
  distilled A3B. The A3B is the 100–200 tok/s path; this is the "I'm rate-limited, use my GPU"
  path. Different jobs.
- **Q3_K_M on 16GB** leaves a small context window. Keep `CTX` modest (16k) or it won't fit.
  This is exactly the context-capacity problem **kortex** solves — run kortex in front and the
  small window stops mattering.
- **Tool-calling / agentic quality** on a local 27B-Q3 is below hosted Claude. Fine for
  edits, completions, and offline work; verify before trusting it on complex multi-step tasks.
- **It's abliterated**, so it won't refuse authorized security work — the point of this build.
- The ROCmFPX `llama-server` you built on the node is HIP/gfx942 (MI300X). On your RX 9060 XT,
  use your LOCAL Vulkan ROCmFPX build — point `ROCMFPX_DIR` at it in `serve_coder.sh`.
