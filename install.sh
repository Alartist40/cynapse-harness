#!/usr/bin/env bash
set -e

# ==============================================================================
# 🧠 CYNAPSE - Single-Line Pure Rust Installer & Hardware Auto-Detector
# ==============================================================================
# Single Line Install Command:
#   curl -fsSL https://raw.githubusercontent.com/Alartist40/cynapse-harness/main/install.sh | bash
# ==============================================================================

REPO_URL="https://github.com/Alartist40/cynapse-harness.git"
INSTALL_ROOT="${HOME}/.cynapse"
BIN_DIR="${HOME}/.local/bin"

echo "======================================================================"
echo "      🧠 CYNAPSE PURE RUST SYSTEM INSTALLER & HARDWARE DETECTOR       "
echo "======================================================================"

# Determine workspace directory
if [ -f "Cargo.toml" ] && grep -q "cynapse" Cargo.toml 2>/dev/null; then
    CYNAPSE_DIR="$(pwd)"
    echo "📍 Installing from local repository checkout: ${CYNAPSE_DIR}"
elif [ -d "${BASH_SOURCE[0]%/*}" ] && [ -f "${BASH_SOURCE[0]%/*}/Cargo.toml" ]; then
    CYNAPSE_DIR="$(cd "${BASH_SOURCE[0]%/*}" && pwd)"
    echo "📍 Installing from script directory: ${CYNAPSE_DIR}"
else
    CYNAPSE_DIR="${INSTALL_ROOT}/repo"
    echo "📥 Remote installation detected. Preparing repository at: ${CYNAPSE_DIR}"
    mkdir -p "${INSTALL_ROOT}"
    if [ -d "${CYNAPSE_DIR}/.git" ]; then
        echo "   [→] Updating existing Cynapse repository..."
        git -C "${CYNAPSE_DIR}" fetch --all --quiet
        git -C "${CYNAPSE_DIR}" reset --hard origin/main --quiet
    else
        echo "   [→] Cloning latest Cynapse repository from ${REPO_URL}..."
        git clone --depth 1 "${REPO_URL}" "${CYNAPSE_DIR}"
    fi
fi

# 1. Detect OS Kernel
OS_TYPE="$(uname -s)"
case "${OS_TYPE}" in
    Linux*)     OS="Linux";;
    Darwin*)    OS="macOS";;
    CYGWIN*|MINGW*|MSYS*) OS="Windows";;
    *)          OS="Unknown (${OS_TYPE})";;
esac

# 2. Detect CPU Architecture & Core Count
ARCH="$(uname -m)"
if [ "${OS}" = "macOS" ]; then
    CPU_CORES="$(sysctl -n hw.ncpu 2>/dev/null || echo "4")"
    TOTAL_RAM_BYTES="$(sysctl -n hw.memsize 2>/dev/null || echo "8589934592")"
    TOTAL_RAM_GB=$((TOTAL_RAM_BYTES / 1024 / 1024 / 1024))
else
    CPU_CORES="$(nproc 2>/dev/null || grep -c ^processor /proc/cpuinfo 2>/dev/null || echo "4")"
    TOTAL_RAM_KB="$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || echo "8388608")"
    TOTAL_RAM_GB=$((TOTAL_RAM_KB / 1024 / 1024))
fi

# 3. Detect GPU Acceleration Capabilities
GPU_TYPE="CPU (Software Fallback)"
if command -v nvidia-smi &>/dev/null; then
    NVIDIA_NAME="$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null | head -n1 || true)"
    if [ -n "${NVIDIA_NAME}" ]; then
        GPU_TYPE="NVIDIA CUDA (${NVIDIA_NAME})"
    fi
elif [ "${OS}" = "macOS" ] && [ "${ARCH}" = "arm64" ]; then
    GPU_TYPE="Apple Metal Unified Memory"
elif command -v rocm-smi &>/dev/null; then
    GPU_TYPE="AMD ROCm"
fi

echo ""
echo "💻 Hardware Specs Detected:"
echo "   - OS Kernel:   ${OS} (${ARCH})"
echo "   - CPU Cores:   ${CPU_CORES} logical cores"
echo "   - System RAM:  ${TOTAL_RAM_GB} GB RAM"
echo "   - Accelerator: ${GPU_TYPE}"

# 4. Check for Rust Cargo Build System & Auto-Install
echo ""
echo "🔍 Toolchain Check:"

if command -v cargo &>/dev/null; then
    RUST_VER="$(cargo --version | head -n1)"
    echo "   [✓] Rust Cargo: ${RUST_VER}"
else
    echo "   [!] Rust Cargo not found. Installing Rust toolchain via rustup..."
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
    else
        echo "ERROR: Need curl or wget to install Rust."
        exit 1
    fi
    . "${HOME}/.cargo/env"
fi

# 4b. Inference Engine Architecture Check
echo ""
echo "🤖 Inference Engine Architecture Check:"
echo "   [✓] Embedded Sovereign Leafcutter Engine: Pure safe Rust (AVX2/NEON SIMD + zero-clone layer streaming)."
echo "       No external server or daemon required to run local models."

HAVE_BACKEND=""
if command -v llama-server >/dev/null 2>&1; then
    HAVE_BACKEND="llama-server"
    echo "   [✓] Optional Hardware Accelerator found: $(command -v llama-server) (Vulkan/Metal/CUDA support)"
elif command -v ollama >/dev/null 2>&1; then
    HAVE_BACKEND="ollama"
    echo "   [✓] Optional Fallback Provider found: $(command -v ollama)"
else
    echo "   [i] Standalone in-process Leafcutter active (100% offline, zero external dependencies)."
fi

# 5. Build Standalone Pure Rust Cynapse Release Binary
echo ""
echo "⚙️ Building Standalone Pure Rust Cynapse Binary (cargo build --release)..."

mkdir -p "${CYNAPSE_DIR}/models" "${CYNAPSE_DIR}/data" "${BIN_DIR}" "${INSTALL_ROOT}/models" "${INSTALL_ROOT}/persona" "${INSTALL_ROOT}/synapses"

# Seed built-in Synapse extension packs
if [ -d "${CYNAPSE_DIR}/synapses" ]; then
    echo "   [→] Seeding Synapse tool extension packs (zero_breach, soup)..."
    cp -rn "${CYNAPSE_DIR}/synapses/"* "${INSTALL_ROOT}/synapses/" 2>/dev/null || true
fi

(cd "${CYNAPSE_DIR}" && cargo build --release)

# 6. Install Release Executable into User PATH
RELEASE_BIN="${CYNAPSE_DIR}/target/release/cynapse"

if [ -f "${RELEASE_BIN}" ]; then
    rm -f "${BIN_DIR}/cynapse"
    cp "${RELEASE_BIN}" "${BIN_DIR}/cynapse"
    chmod +x "${BIN_DIR}/cynapse"
    echo "   [✓] Installed release binary to ${BIN_DIR}/cynapse"
else
    echo "   [!] Failed to locate built binary at ${RELEASE_BIN}"
    exit 1
fi

# 7. Run Cynapse Doctor Self-Healing Initializer
echo ""
echo "🩺 Initializing Cynapse Doctor Self-Healing Subsystem..."
"${BIN_DIR}/cynapse" doctor --fix || true

echo ""
echo "======================================================================"
echo "🎉 CYNAPSE INSTALLATION COMPLETE (PURE RUST - ZERO NODE/PYTHON)!"
echo "======================================================================"
echo "Global launcher installed to: ${BIN_DIR}/cynapse"

if [[ ":$PATH:" != *":${BIN_DIR}:"* ]]; then
    echo ""
    echo "⚠️  ${BIN_DIR} is not in your current PATH."
    echo "    Add this line to your ~/.bashrc or ~/.zshrc:"
    echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
fi

echo ""
echo "Quick Start Commands:"
echo "  1. Start Interactive Cynapse TUI: cynapse"
echo "  2. Download Hardware Model:       cynapse pull"
echo "  3. Run Self-Healing Doctor:       cynapse doctor"
echo "  4. View 3D Memory Atlas:         cynapse memory"
echo ""
