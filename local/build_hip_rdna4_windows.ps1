<#
  Build the ROCmFPX HIP backend on WINDOWS for the RX 9060 XT (Navi 44 = gfx1200, RDNA4).

  WHY: the current local ROCmFPX build is VULKAN-ONLY, which runs the A3B at ~20% of memory
  bandwidth (~71 tok/s). The native ROCm/HIP kernels run MoE at ~50-66% efficiency -> the
  ~150-200 tok/s target. This is a BUILD, not a kernel rewrite: the fork already ships gfx1200
  HIP kernels (ggml/rocmfp4, ggml/rocmfpx) and a build-rdna4.sh; this is its Windows twin.

  PREREQS (the part you get from GitHub):
    - HIP runtime: PRESENT already (amdhip64_7.dll from your Adrenalin driver = ROCm 7,
      native gfx1200 support, no HSA_OVERRIDE needed).
    - HIP SDK (compiler + gfx1200 device libs + rocWMMA): the MISSING piece to compile.
      Install the AMD HIP SDK for Windows that matches ROCm 7 (or TheRock / community
      gfx1200 device-lib bundle). Set $env:HIP_SDK to its root (the dir with bin\, lib\,
      include\, and bin\clang++.exe / hipcc).

  USAGE (PowerShell):
    $env:HIP_SDK = "C:\Program Files\AMD\ROCm\7.0"     # <-- your HIP SDK root
    .\local\build_hip_rdna4_windows.ps1
    # then benchmark against the 71 tok/s Vulkan baseline:
    .\ROCmFPX\build-rdna4\bin\llama-bench.exe -m <a3b>.gguf -dev ROCm0 -ngl 999
#>
param(
  [string]$RocmFpxDir = "C:\Users\HADES\Desktop\ROCmFPX",
  [string]$GfxArch    = "gfx1200",            # RX 9060 XT = Navi 44 = gfx1200 (NOT gfx1201!)
  [int]$Jobs          = [Environment]::ProcessorCount
)
$ErrorActionPreference = "Stop"

if (-not $env:HIP_SDK) { throw "Set `$env:HIP_SDK to your HIP SDK root (contains bin\clang++.exe)." }
$clang = Join-Path $env:HIP_SDK "bin\clang++.exe"
if (-not (Test-Path $clang)) { $clang = Join-Path $env:HIP_SDK "bin\amdclang++.exe" }
if (-not (Test-Path $clang)) { throw "No clang++/amdclang++ under $env:HIP_SDK\bin. Wrong HIP SDK path?" }

Write-Host "== HIP SDK: $env:HIP_SDK"
Write-Host "== compiler: $clang"
Write-Host "== target:   $GfxArch (RDNA4). gfx1200!=gfx1201 -- a mismatch loads then segfaults."

$build = Join-Path $RocmFpxDir "build-rdna4"
$env:HIP_PATH = $env:HIP_SDK
$env:PATH     = "$($env:HIP_SDK)\bin;$env:PATH"

# Configure. GGML_HIP_FORCE_MMQ=ON keeps the quantized matmul on the integer path (good for
# the ROCMFPX codebook types). Keep Vulkan ON as a fallback device in the same binary.
cmake -S $RocmFpxDir -B $build `
  -G "Ninja" `
  -DGGML_HIP=ON `
  -DGGML_HIP_FORCE_MMQ=ON `
  -DGGML_VULKAN=ON `
  -DGGML_CUDA=OFF `
  -DGPU_TARGETS="$GfxArch" `
  -DAMDGPU_TARGETS="$GfxArch" `
  -DCMAKE_HIP_ARCHITECTURES="$GfxArch" `
  -DCMAKE_HIP_COMPILER="$clang" `
  -DCMAKE_CXX_COMPILER="$clang" `
  -DCMAKE_BUILD_TYPE=Release `
  -DLLAMA_CURL=OFF

cmake --build $build --config Release -j $Jobs --target llama-server llama-bench llama-quantize

Write-Host ""
Write-Host "== built -> $build\bin"
Write-Host "== VALIDATE the kernels before trusting them:"
Write-Host "   $build\bin\test-backend-ops.exe -b ROCm0"
Write-Host "   $build\bin\test-quantize-fns.exe"
Write-Host "== BENCH the A3B (compare vs 71 tok/s Vulkan):"
Write-Host "   $build\bin\llama-bench.exe -m <a3b>.gguf -dev ROCm0 -ngl 999"
Write-Host "== SERVE with HIP: llama-server.exe ... -dev ROCm0   (see local/serve_distilled.sh)"
