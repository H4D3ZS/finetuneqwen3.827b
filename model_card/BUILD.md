# Building the runtime (ROCmFPX / llama.cpp) yourself

This GGUF uses the **ROCmFPX** quant family (`Q2_0_ROCMFPX`), which needs a matching
`llama-server` / `llama-quantize`. Build it from the pinned source so your binary understands
the 2-bit ROCmFP codebook and the `nextn` (MTP) tensors.

## Pinned source

- Repo:   https://github.com/charlie12345/ROCmFPX.git
- Commit: `b2f5829db8beefc22b49481247d180a48b06793a` (b2f5829)

Building any other commit is not guaranteed to read `Q2_0_ROCMFPX`.

## Quick build (AMD, HIP — MI300X / gfx942, adapt `AMDGPU_TARGETS` for your card)

```bash
git clone https://github.com/charlie12345/ROCmFPX.git
cd ROCmFPX && git checkout b2f5829

export ROCM_PATH=/opt/rocm            # e.g. /opt/rocm-7.2.4 on some installs
cmake -S . -B build \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx942 \           # RX 9060 XT = gfx1200 (RDNA4); MI300X = gfx942
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$ROCM_PATH"
cmake --build build -j --target llama-server llama-quantize llama-cli
# -> build/bin/llama-server, build/bin/llama-quantize
```

For a consumer RDNA4 card (RX 9060 XT) you can also build the **Vulkan** backend
(`-DGGML_VULKAN=ON` instead of `-DGGML_HIP=ON`) if you prefer it over ROCm/HIP.

The `build_rocmfpx.sh` in this repo is the exact script used to build the binaries that
produced this GGUF (HIP, gfx942). Edit `AMDGPU_TARGETS` for your GPU.

## The quant recipe (how this 12GB build was made)

Pure 2-bit-everywhere collapses this model; pure 3.5-bit is coherent but 19GB (won't fit 16GB).
The fix is **role-aware**: 2-bit experts (the bulk) + Q6 attention/embeddings/shared-experts.
Reproduce from an f16 GGUF (`convert_hf_to_gguf.py ... --outtype f16`) with:

```bash
./build/bin/llama-quantize \
  --token-embedding-type Q6_0_ROCMFPX \
  --output-tensor-type  Q6_0_ROCMFPX \
  --tensor-type attn_q=Q6_0_ROCMFPX  --tensor-type attn_k=Q6_0_ROCMFPX \
  --tensor-type attn_v=Q6_0_ROCMFPX  --tensor-type attn_qkv=Q6_0_ROCMFPX \
  --tensor-type attn_output=Q6_0_ROCMFPX --tensor-type attn_gate=Q6_0_ROCMFPX \
  --tensor-type ffn_gate_shexp=Q6_0_ROCMFPX --tensor-type ffn_up_shexp=Q6_0_ROCMFPX \
  --tensor-type ffn_down_shexp=Q6_0_ROCMFPX \
  model-f16.gguf model-Q2KXL_ROCMFPX.gguf Q2_0_ROCMFPX
# base Q2_0_ROCMFPX -> the ffn_*_exps (256-expert) tensors go 2-bit; overrides keep the rest at Q6.
```

## Serve it (OpenAI-compatible API on :8080)

```bash
./build/bin/llama-server \
  -m Qwen3.8-Distill-35B-A3B-Coder-Abliterated-Q2_ROCMFPX.gguf \
  --host 127.0.0.1 --port 8080 \
  -ngl 99 -c 16384 -fa on --jinja \
  --alias qwen38-distill-a3b
```

On a 16GB card, context and concurrency share one KV pool — pick one:
single-stream long context (`-c 32768 -np 1`) **or** many short sessions (`-c 8192 -np 8`).

## Note on MTP

The `nextn` (MTP) tensors are present in this GGUF (block 40). Whether they are used for
speculative decoding depends on your `llama-server` build's runtime support — check its startup
log for a draft/nextn line. If not auto-used, the model still runs correctly as a standard
single-token decoder.
