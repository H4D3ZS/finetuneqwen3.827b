#!/usr/bin/env bash
# Regenerate the CMake find_package glue for a TheRock pip ROCm SDK that ships DLLs+compiler
# but NOT the hip/rocblas/hipblas cmake configs (the Windows case). Produces, under $GLUE:
#   lib/{rocblas,hipblas}.lib  (import libs generated from the runtime DLLs)
#   include/{rocblas,hipblas,hipblas-common}/*  (headers from ROCm/rocm-libraries + generated)
#   cmake/{hip,rocblas,hipblas}/*-config.cmake   (imported-target config packages)
# Point the build at it with -DROCM_GLUE_DIR=$GLUE (see cmake/rocm_hip_xplatform.cmake).
#
#   SP=<site-packages> GLUE=/c/src/rocm-glue VSMSVC="<VS>/VC/Tools/MSVC/<ver>" bash local/build_hip_glue.sh
set -euo pipefail
: "${SP:?set SP to the rocm-sdk site-packages dir (has _rocm_sdk_core/devel/libraries_*)}"
GLUE="${GLUE:-/c/src/rocm-glue}"
CORE="$SP/_rocm_sdk_core"; LIBS="$SP/_rocm_sdk_libraries_gfx120X_all"
mkdir -p "$GLUE/lib" "$GLUE/include" "$GLUE/cmake/hip" "$GLUE/cmake/rocblas" "$GLUE/cmake/hipblas"

echo "== 1. import libs from the runtime DLLs (needs MSVC dumpbin+lib)"
DUMPBIN=$(find "${VSMSVC:?set VSMSVC to your MSVC toolchain root}" -iname dumpbin.exe -path "*Hostx64/x64*" | head -1)
LIB=$(find "$VSMSVC" -iname lib.exe -path "*Hostx64/x64*" | head -1)
for name in rocblas hipblas; do
  "$DUMPBIN" //exports "$LIBS/bin/${name}.dll" | awk 'NR>1 && /^[ ]+[0-9]+/ && $4 {print $4}' > "/tmp/${name}.exp"
  { echo "EXPORTS"; cat "/tmp/${name}.exp"; } > "$GLUE/lib/${name}.def"
  "$LIB" "-def:$GLUE/lib/${name}.def" "-out:$GLUE/lib/${name}.lib" "-machine:x64" >/dev/null
done

echo "== 2. headers from ROCm/rocm-libraries (sparse) + hipblas-common"
RL="${RL:-/c/src/rocm-libraries}"
if [ ! -d "$RL/.git" ]; then
  git clone --filter=blob:none --no-checkout --depth 1 https://github.com/ROCm/rocm-libraries.git "$RL"
  ( cd "$RL"; git sparse-checkout init --cone
    git sparse-checkout set projects/rocblas/library/include projects/hipblas/library/include projects/hipblas-common
    git checkout )
fi
INC="$GLUE/include"; mkdir -p "$INC/rocblas/internal" "$INC/hipblas" "$INC/hipblas-common"
cp "$RL/projects/rocblas/library/include/rocblas.h" "$INC/rocblas/"
cp "$RL/projects/rocblas/library/include/internal/"* "$INC/rocblas/internal/"
cp "$RL/projects/hipblas/library/include/hipblas.h" "$INC/hipblas/"
cp "$RL/projects/hipblas-common/library/include/hipblas-common/hipblas-common.h" "$INC/hipblas-common/"
# generated export/version headers (empty export macros; versions are cosmetic)
printf '#pragma once\n#define ROCBLAS_EXPORT\n#define ROCBLAS_DEPRECATED_MSG(m)\n#define ROCBLAS_CLANG_STATIC\n' > "$INC/rocblas/internal/rocblas-export.h"
printf '#pragma once\n#define ROCBLAS_VERSION_MAJOR 4\n#define ROCBLAS_VERSION_MINOR 5\n#define ROCBLAS_VERSION_PATCH 0\n#define ROCBLAS_VERSION_TWEAK 0\n#define ROCBLAS_TENSILE_COMMIT_ID "u","u"\n' > "$INC/rocblas/internal/rocblas-version.h"
printf '#pragma once\n#define HIPBLAS_EXPORT\n#define HIPBLAS_DEPRECATED_MSG(m)\n' > "$INC/hipblas/hipblas-export.h"
printf '#pragma once\n#define hipblasVersionMajor 2\n#define hipblaseVersionMinor 4\n#define hipblasVersionMinor 4\n#define hipblasVersionPatch 0\n#define hipblasVersionTweak 0\n#define hipblasVersionK 100\n' > "$INC/hipblas/hipblas-version.h"

echo "== 3. cmake config packages (imported targets)"
BITCODE="$CORE/lib/llvm/amdgcn/bitcode"
CW=$(dirname "$(find "$CORE/lib/llvm/lib/clang" -iname cuda_wrappers -type d | head -1)")/cuda_wrappers
ARCH="${ROCM_HIP_ARCH:-gfx1200}"
cat > "$GLUE/cmake/hip/hip-config.cmake" <<EOF
set(hip_VERSION "7.0.0")
set(hip_FOUND TRUE)
foreach(t hip::host hip::device)
  if(NOT TARGET \${t})
    add_library(\${t} INTERFACE IMPORTED)
    set_target_properties(\${t} PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "$CORE/include"
      INTERFACE_COMPILE_DEFINITIONS "__HIP_PLATFORM_AMD__=1;__HIP_PLATFORM_HCC__=1"
      INTERFACE_LINK_LIBRARIES "$CORE/lib/amdhip64.lib")
  endif()
endforeach()
# device-only HIP compile flags (Windows compiles .cu as CXX -> must ride hip::device)
set_property(TARGET hip::device APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS "__HIP__=1;__HIPCC__=1")
set_property(TARGET hip::device PROPERTY INTERFACE_COMPILE_OPTIONS
  "-isystem;$CW;-x;hip;--offload-arch=$ARCH;-fhip-new-launch-api;-include;__clang_hip_runtime_wrapper.h;--hip-device-lib-path=$BITCODE")
EOF
for lib in rocblas hipblas; do
  V=$([ "$lib" = rocblas ] && echo 4.5.0 || echo 2.4.0)
  cat > "$GLUE/cmake/$lib/$lib-config.cmake" <<EOF
set(${lib}_VERSION "$V")
set(${lib}_FOUND TRUE)
if(NOT TARGET roc::$lib)
  add_library(roc::$lib INTERFACE IMPORTED)
  set_target_properties(roc::$lib PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "$GLUE/include"
    INTERFACE_LINK_LIBRARIES "$GLUE/lib/$lib.lib")
endif()
EOF
done
echo "== glue ready at $GLUE  (pass -DROCM_GLUE_DIR=$GLUE)"
echo "NOTE: Windows clang HIP + MSVC math-header conflict (isgreater/isless) is still open;"
echo "      see docs/ROCM_HIP_RDNA4.md. Linux builds do not hit it."
