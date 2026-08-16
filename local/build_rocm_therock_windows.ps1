<#
  Build the ROCm/HIP SDK for the RX 9060 XT (gfx1200, RDNA4) on WINDOWS from source, via
  TheRock (ROCm's unified CMake build). This produces the compiler + device libs + rocWMMA
  that `build_hip_rdna4_windows.ps1` then needs to build the ROCmFPX HIP backend.

  CONFIRMED SUPPORTED: TheRock's cmake/therock_amdgpu_targets.cmake lists
    therock_add_amdgpu_target(gfx1200 "AMD RX 9060 / XT" ...)
  so this is a first-class target, not a fallback/override.

  PREREQS (install first):
    - Visual Studio 2022 (Desktop C++ workload) — provides the MSVC toolchain
    - Python 3.10+  ·  CMake 3.25+  ·  Ninja  ·  git
    - ~200GB free disk and time: a full-from-source ROCm build is long (hours).

  FASTER ALTERNATIVE (still "real", just precompiled): TheRock publishes prebuilt Windows
  dist tarballs. If one covers gfx1200/gfx120X, download+extract it and SKIP this build —
  point $env:HIP_SDK straight at its rocm/ dir. See:  https://github.com/ROCm/TheRock/releases

  USAGE (PowerShell, from wherever you want TheRock cloned):
    .\local\build_rocm_therock_windows.ps1 -Dest C:\src\TheRock
    # then set the HIP SDK to the produced tree and build ROCmFPX:
    $env:HIP_SDK = "C:\src\TheRock\build\dist\rocm"
    .\local\build_hip_rdna4_windows.ps1
#>
param(
  [string]$Dest      = "C:\src\TheRock",
  [string]$GfxTarget = "gfx1200",            # RX 9060 XT (Navi 44). Use gfx120X-all for 1200+1201.
  [switch]$FamiliesMode                       # -FamiliesMode => build the whole gfx120X family
)
$ErrorActionPreference = "Stop"

foreach ($t in @("git","cmake","ninja","python")) {
  if (-not (Get-Command $t -ErrorAction SilentlyContinue)) { throw "$t not on PATH. Install prereqs (VS2022, Python, CMake, Ninja, git) first." }
}

if (-not (Test-Path $Dest)) {
  git clone https://github.com/ROCm/TheRock.git $Dest
}
Set-Location $Dest

# Bleeding edge = develop; TheRock's default branch already tracks it. Pull latest.
git fetch origin; git checkout develop 2>$null; git pull --ff-only 2>$null

Write-Host "== python venv + deps"
python -m venv .venv
& ".\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host "== fetch ROCm component sources (clr/HIP, amd-llvm, device-libs, rocWMMA, ...)"
python .\build_tools\fetch_sources.py

$targetFlag = if ($FamiliesMode) { "-DTHEROCK_AMDGPU_FAMILIES=gfx120X-all" } else { "-DTHEROCK_AMDGPU_TARGETS=$GfxTarget" }
Write-Host "== configure for $targetFlag  (RX 9060 XT = gfx1200)"
cmake -B build -G Ninja . $targetFlag

Write-Host "== build (this is the long part; -j is implicit with Ninja)"
cmake --build build

Write-Host ""
Write-Host "== ROCm/HIP SDK built. The dist tree (compiler + gfx1200 device libs + rocWMMA) is under:"
Write-Host "   $Dest\build\dist\rocm   (verify bin\clang++.exe / hipcc exist there)"
Write-Host "== next: point ROCmFPX's HIP build at it —"
Write-Host "   `$env:HIP_SDK = `"$Dest\build\dist\rocm`""
Write-Host "   .\local\build_hip_rdna4_windows.ps1"
