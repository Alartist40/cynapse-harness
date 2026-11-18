#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="${ROOT_DIR}/engine/leafcutter_core/rust/llama.cpp"
BUILD_DIR="${LLAMA_DIR}/build"
BIN_DIR="${ROOT_DIR}/bin"

if [ ! -d "${LLAMA_DIR}" ]; then
    echo "Error: vendored llama.cpp not found at ${LLAMA_DIR}" >&2
    exit 1
fi

ARCH="$(uname -m)"
EXTRA_FLAGS=()
if [ "${ARCH}" = "aarch64" ]; then
    # GCC 12 hits an ICE in KleidiAI bf16 neon pack; disable KleidiAI on GCC <= 12
    EXTRA_FLAGS+=("-DGGML_CPU_KLEIDIAI=OFF" "-DCMAKE_C_FLAGS=-march=armv9-a+dotprod+fp16+i8mm" "-DCMAKE_CXX_FLAGS=-march=armv9-a+dotprod+fp16+i8mm")
fi

cmake -S "${LLAMA_DIR}" -B "${BUILD_DIR}" \
    -DGGML_VULKAN=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    "${EXTRA_FLAGS[@]}"

cmake --build "${BUILD_DIR}" --target llama-server llama-cli llama-quantize -j4

mkdir -p "${BIN_DIR}" "${HOME}/.cynapse/bin"
cp --remove-destination "${BUILD_DIR}/bin/llama-server" "${BIN_DIR}/llama-server"
cp --remove-destination "${BUILD_DIR}/bin/llama-cli" "${BIN_DIR}/llama-cli"
cp --remove-destination "${BUILD_DIR}/bin/llama-quantize" "${BIN_DIR}/llama-quantize"
cp --remove-destination "${BUILD_DIR}/bin/llama-server" "${HOME}/.cynapse/bin/llama-server"

echo "Built and installed llama-server, llama-cli, and llama-quantize into ${BIN_DIR}"
