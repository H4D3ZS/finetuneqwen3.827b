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
| HIP **SDK** (compiler `clang++`/`hipcc`, device libs, rocWMMA) | ⬜ build via **TheRock** — `local/build_rocm_therock_windows.ps1` |
| Windows HIP **build** of ROCmFPX | ⬜ `local/build_hip_rdna4_windows.ps1` (this repo) |

### Getting the HIP SDK: TheRock (confirmed gfx1200 support)

The `develop` branch of [ROCm/ROCm](https://github.com/ROCm/ROCm/tree/develop) points to
[**TheRock**](https://github.com/ROCm/TheRock) — ROCm's unified CMake build with **Windows
support**. Its `cmake/therock_amdgpu_targets.cmake` lists our card as a first-class target
(not a fallback):

```cmake
therock_add_amdgpu_target(gfx1200 "AMD RX 9060 / XT" FAMILY dgpu-all gfx120X-all
```

Two ways to get the SDK:
- **From source (bleeding edge):** `local/build_rocm_therock_windows.ps1` — clones TheRock,
  fetches sources, and builds with `-DTHEROCK_AMDGPU_TARGETS=gfx1200`. Long (hours) but latest.
- **Prebuilt Windows dist (faster, still real):** if a release tarball covers gfx120X, grab it
  from [TheRock releases](https://github.com/ROCm/TheRock/releases) and point `$env:HIP_SDK`
  at its `rocm/` dir — no multi-hour build.

Prereqs either way: Visual Studio 2022 (C++), Python 3.10+, CMake 3.25+, Ninja, git.

The card is Navi 44 = **`gfx1200`**. Do NOT build `gfx1201` (that's Navi 48 / RX 9070) — a
mismatched build loads a model then **segfaults** (documented in the fork's
`docs/BUILD-AMD-ARCHITECTURES.md`).

## Steps

1. **Get the HIP SDK** (the one missing piece) — build it for gfx1200 via TheRock:
   ```powershell
   .\local\build_rocm_therock_windows.ps1 -Dest C:\src\TheRock
   ```
   (or grab a prebuilt gfx120X dist from TheRock releases). It provides `bin\clang++.exe`,
   the gfx1200 device libraries, and rocWMMA headers.
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

## Build status & the Windows blocker (as of this session)

**Solved (all captured in `cmake/rocm_hip_xplatform.cmake` + the glue):**
- Direct-download prebuilt gfx1200 ROCm SDK via `pip install rocm[devel,libraries]` from
  `https://rocm.nightlies.amd.com/v2/gfx120X-all/` (no multi-hour source build).
- The SDK omits CMake `find_package` configs → **hand-built glue**: import libs generated
  from the runtime DLLs (`rocblas.lib`/`hipblas.lib` via `dumpbin`+`lib`), headers fetched
  from `ROCm/rocm-libraries` (+ `hipblas-common`), generated `*-export/-version.h`, and
  `hip/rocblas/hipblas` config packages. `find_package(hip/rocblas/hipblas)` → **configure OK**.
- On Windows the fork forces `CXX_IS_HIPCC` (compiles `.cu` as LANGUAGE CXX), so the HIP flags
  must ride on `hip::device`: `-x hip --offload-arch=gfx1200 -fhip-new-launch-api
  -include __clang_hip_runtime_wrapper.h -isystem <clang>/cuda_wrappers --hip-device-lib-path`.
  With these, **the gfx1200 device code compiles** (device builtins resolve).

**PATCHED (Windows now builds):** the last blocker was a clang HIP + MSVC math-header conflict —
`__clang_hip_cmath.h` / `__clang_cuda_math_forward_declares.h` redeclare `isgreater`/`isless`/`isunordered`
as `__device__`, conflicting with MSVC UCRT's `__host__ __device__` versions (Linux is immune —
glibc uses macros). **Fix:** `local/build_hip_glue.sh` step 4 guards *only* the comparison
functions behind `#if !defined(_MSC_VER)` (MSVC's host+device versions serve device code; the
classifiers `isnan`/`isinf`/… are left intact — `__clang_cuda_complex_builtins.h` needs them).
With that patch the full gfx1200 HIP backend **compiles clean on Windows** (clang 20, ROCm 7.9.0rc).

Use a stable **rc** SDK (7.9.0rc) not the 7.14.0a alpha. Linux is unaffected either way and needs
no patch. `cmake/rocm_hip_xplatform.cmake` drives both platforms.

## Honest ceiling

Even a perfect HIP build won't hit 300 on this card sustained — that needs both peak kernel
efficiency AND working speculative decoding (a trained same-vocab draft model; the native MTP
gave no speedup on Vulkan). 150-200 is the realistic HIP target; 300 is a stretch that stacks
Phase-2 kernels + spec-decode. See [`MOE_UPCYCLE_PLAN.md`](../MOE_UPCYCLE_PLAN.md) §1a.
