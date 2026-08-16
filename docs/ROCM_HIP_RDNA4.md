# ROCm/HIP backend for RDNA4 (RX 9060 XT) — the speed track

Goal: take the finished A3B from ~71 tok/s (Vulkan) toward **~150-200 tok/s** by running it on
native ROCm/HIP kernels instead of Vulkan. This is the *deployment-speed* track — separate
from training. Training does not change inference speed; the backend does.

## Why this exists

Decode is memory-bandwidth bound: `tok/s = (bandwidth × efficiency) ÷ bytes-per-token`.
Measured on the RX 9060 XT (Vulkan backend):

| model | active | tok/s | bandwidth efficiency |
|---|---:|---:|---:|
| A3B (MoE, Q2) | 3B | 71 | **~20%** (MoE routing hurts Vulkan coalescing) |
| dense 27B (Q2) | 24B | 28 | ~66% |

The MoE leaves ~3× on the table on Vulkan. ROCm/HIP MoE kernels recover most of it → the
2-3× that lands the 150-200 range. **No `HSA_OVERRIDE` fallback** — gfx1200 is natively
supported by the ROCm 7 runtime already on this machine.

## What's already here vs. what's needed

| piece | status |
|---|---|
| HIP **runtime** (`amdhip64_7.dll`, `amd_comgr_3.dll`) | ✅ present (ships with Adrenalin = ROCm 7, native gfx1200) |
| ROCMFPX **gfx1200 HIP kernels** (`ggml/rocmfp4`, `ggml/rocmfpx`) | ✅ in the fork; `scripts/build-rdna4.sh` targets gfx1200 |
| HIP **SDK** (compiler `clang++`/`hipcc`, device libs, rocWMMA) | ⬜ **install from GitHub** (AMD HIP SDK for Windows matching ROCm 7, or TheRock) |
| Windows HIP **build** of ROCmFPX | ⬜ `local/build_hip_rdna4_windows.ps1` (this repo) |

The card is Navi 44 = **`gfx1200`**. Do NOT build `gfx1201` (that's Navi 48 / RX 9070) — a
mismatched build loads a model then **segfaults** (documented in the fork's
`docs/BUILD-AMD-ARCHITECTURES.md`).

## Steps

1. **Get the HIP SDK** (the one missing piece) from GitHub — the AMD HIP SDK for Windows
   matching the installed ROCm 7 runtime, or TheRock/community gfx1200 device-lib bundle.
   It must provide `bin\clang++.exe` (or `amdclang++`), the gfx1200 device libraries, and
   rocWMMA headers.
2. **Build** (PowerShell):
   ```powershell
   $env:HIP_SDK = "C:\Program Files\AMD\ROCm\7.0"   # your HIP SDK root
   .\local\build_hip_rdna4_windows.ps1
   ```
3. **Validate the kernels** (never trust them un-tested):
   ```powershell
   .\ROCmFPX\build-rdna4\bin\test-backend-ops.exe -b ROCm0
   .\ROCmFPX\build-rdna4\bin\test-quantize-fns.exe
   ```
4. **Benchmark** vs the Vulkan baseline (71 tok/s):
   ```powershell
   .\ROCmFPX\build-rdna4\bin\llama-bench.exe -m <a3b>.gguf -dev ROCm0 -ngl 999
   ```
5. **Serve** with `-dev ROCm0` (point `local/serve_distilled.sh` at the `build-rdna4` binary).

## Phases

- **Phase 1 — functional HIP build = the 2-3×.** Steps above. The gfx1200 kernels exist; this
  is a build + benchmark, not kernel authoring. Expect ~71 → **~150-200 tok/s**.
- **Phase 2 — peak, the "make our own" work.** The fork's README notes native **rocWMMA FP4
  tensor-core** paths are "future optimization work." Writing the FP4 WMMA tiles for RDNA4's
  matrix cores (wave32, gfx12 WMMA intrinsics) is the real kernel engineering to push past
  200 — a genuine open contribution.

## Honest ceiling

Even a perfect HIP build won't hit 300 on this card sustained — that needs both peak kernel
efficiency AND working speculative decoding (a trained same-vocab draft model; the native MTP
gave no speedup on Vulkan). 150-200 is the realistic HIP target; 300 is a stretch that stacks
Phase-2 kernels + spec-decode. See [`MOE_UPCYCLE_PLAN.md`](../MOE_UPCYCLE_PLAN.md) §1a.
