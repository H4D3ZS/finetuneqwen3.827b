# rocm_hip_xplatform.cmake — cross-platform (Windows + Linux) HIP setup for building the
# ROCmFPX / llama.cpp HIP backend against a TheRock ROCm SDK, without needing a system ROCm
# install. Use as a CMake toolchain-style include:
#
#   cmake -S <ROCmFPX> -B build-rdna4 -G Ninja \
#     -DCMAKE_PROJECT_INCLUDE=<this>/rocm_hip_xplatform.cmake \
#     -DGGML_HIP=ON -DGGML_HIP_FORCE_MMQ=ON -DGPU_TARGETS=gfx1200
#
# or point the ROCmFPX build scripts (local/build_hip_rdna4_windows.ps1 / build-rdna4.sh) at it.
#
# WHY THIS EXISTS: TheRock's pip SDK (rocm[devel,libraries]) ships the compiler + gfx1200
# device libs + runtime DLLs, but NOT the CMake find_package(hip/rocblas/hipblas) configs, and
# on Windows the .cu files are compiled LANGUAGE CXX and rely on hip::device to carry the HIP
# flags. This file provides all of that, on both platforms, from one configurable knob set.
#
# KNOBS (cache vars):
#   ROCM_HIP_ARCH   default gfx1200 (RX 9060 XT). RX 9070/XT = gfx1201.
#   ROCM_SDK_ROOT   ROCm root. Auto-detected: $ENV{ROCM_PATH}, else `rocm-sdk path --root`
#                   (TheRock pip), else /opt/rocm (Linux system).
#   ROCM_GLUE_DIR   dir holding the hand-built find_package glue (configs+headers+import libs)
#                   for SDKs that omit them (Windows pip). Optional on Linux system ROCm.

set(ROCM_HIP_ARCH "gfx1200" CACHE STRING "AMDGPU target (gfx1200=RX9060XT, gfx1201=RX9070)")

# --- locate the ROCm SDK root, cross-platform ---
if(NOT ROCM_SDK_ROOT)
  if(DEFINED ENV{ROCM_PATH})
    set(ROCM_SDK_ROOT "$ENV{ROCM_PATH}")
  else()
    find_program(_rocm_sdk NAMES rocm-sdk rocm-sdk.exe)
    if(_rocm_sdk)
      execute_process(COMMAND ${_rocm_sdk} path --root OUTPUT_VARIABLE ROCM_SDK_ROOT
                      OUTPUT_STRIP_TRAILING_WHITESPACE)
      # the compiler/runtime live in the sibling _rocm_sdk_core; devel is the reported root
      get_filename_component(_sp "${ROCM_SDK_ROOT}" DIRECTORY)
      if(EXISTS "${_sp}/_rocm_sdk_core")
        set(ROCM_CORE "${_sp}/_rocm_sdk_core")
      endif()
    elseif(EXISTS "/opt/rocm")
      set(ROCM_SDK_ROOT "/opt/rocm")
    endif()
  endif()
endif()
if(NOT ROCM_CORE)
  set(ROCM_CORE "${ROCM_SDK_ROOT}")   # system ROCm: everything under one root
endif()
if(NOT ROCM_SDK_ROOT)
  message(FATAL_ERROR "ROCm SDK not found. Set -DROCM_SDK_ROOT=... or ROCM_PATH, or `pip install rocm[devel,libraries]`.")
endif()
message(STATUS "ROCm SDK: ${ROCM_SDK_ROOT} (core: ${ROCM_CORE}), arch ${ROCM_HIP_ARCH}")

# --- the HIP compiler (clang++) ---
find_program(HIP_CLANGXX NAMES clang++ clang++.exe
             PATHS "${ROCM_CORE}/lib/llvm/bin" "${ROCM_SDK_ROOT}/llvm/bin" "${ROCM_SDK_ROOT}/bin" NO_DEFAULT_PATH)
if(HIP_CLANGXX AND NOT CMAKE_HIP_COMPILER)
  set(CMAKE_HIP_COMPILER "${HIP_CLANGXX}" CACHE FILEPATH "" FORCE)
endif()

# --- point find_package(hip/rocblas/hipblas) at the glue when the SDK omits configs ---
if(ROCM_GLUE_DIR AND EXISTS "${ROCM_GLUE_DIR}/cmake/hip/hip-config.cmake")
  set(hip_DIR     "${ROCM_GLUE_DIR}/cmake/hip"     CACHE PATH "" FORCE)
  set(rocblas_DIR "${ROCM_GLUE_DIR}/cmake/rocblas" CACHE PATH "" FORCE)
  set(hipblas_DIR "${ROCM_GLUE_DIR}/cmake/hipblas" CACHE PATH "" FORCE)
  list(PREPEND CMAKE_PREFIX_PATH "${ROCM_GLUE_DIR}/cmake")
else()
  # system ROCm / SDK that DOES ship configs
  list(PREPEND CMAKE_PREFIX_PATH "${ROCM_SDK_ROOT}" "${ROCM_SDK_ROOT}/lib/cmake")
endif()

# --- the HIP compile flags carried by hip::device (Windows: .cu compiled as CXX) ---
# On Windows these must be applied to hip::device (see cmake/hip/hip-config.cmake in the glue).
# On Linux with enable_language(HIP) they come from the toolchain; harmless to also set.
set(ROCM_HIP_DEVICE_FLAGS
  "-x hip --offload-arch=${ROCM_HIP_ARCH} -fhip-new-launch-api"
  CACHE STRING "device compile flags for hip::device")

message(STATUS "HIP compiler: ${CMAKE_HIP_COMPILER}")
message(STATUS "hip::device flags: ${ROCM_HIP_DEVICE_FLAGS}")

# KNOWN ISSUE (documented for reproducibility): ROCm 7.14 *alpha* Windows wheels have a
# clang-23 math-header regression (`__clang_hip_cmath.h` redeclares isgreater/isless as
# __device__, conflicting with MSVC's __host__ __device__ versions). Use a STABLE rc/release
# SDK (e.g. 7.9.0rc) on Windows until fixed upstream. Linux is unaffected.
