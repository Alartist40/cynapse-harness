# Orange Pi 6 Plus (CIX P1 / CD8180) — Cynapse GPU & ARMv9 Setup Guide

## 1. Why It Ran at `0.6 tok/s` (Hardware & Software Reality)

1. **NPU Cannot Run `.gguf` Models**:
   The Orange Pi 6 Plus uses the **CIX P1 (CD8180)** SoC with an **Arm China Zhouyi Z3 NPU** (~28.8 TOPS; "45 TOPS" is combined CPU+GPU+NPU marketing). That NPU is a fixed-function INT8 vision/CNN accelerator (YOLO, CLIP, static ONNX graphs via CIX NDE SDK). **No `.gguf` LLM engine (`llama.cpp`, Ollama, or Leafcutter) can execute autoregressive GGUF models on the Zhouyi NPU.**
2. **Why `cynapse` Dropped to `0.6 tok/s`**:
   The repository's `./bin/llama-server` symlink points to an x86_64 Ollama binary. On ARM64 (`aarch64`), without a native ARM64 `llama-server` binary in `~/.cynapse/bin/llama-server`, `cynapse` fell back to pure-Rust `NativeLeafcutter`, which lacks ARMv9 `i8mm`/`dotprod` KleidiAI kernels and GPU offloading.
3. **What *Does* Accelerate `.gguf` on OPi 6 Plus**:
   - **Arm Immortalis-G720 MC10 GPU (Vulkan)**: Offloads model layers (`-ngl 99`) over your unified **LPDDR5** memory via `-DGGML_VULKAN=ON`.
   - **8x Cortex-A720 Big Cores (`ARMv9` + `KleidiAI`)**: Compiling with `-march=armv9-a+dotprod+fp16+i8mm` and `-DGGML_CPU_KLEIDIAI=ON` avoids falling back to scalar code or stalling on the 4 slow Cortex-A520 efficiency cores.

---

## 2. Step-by-Step Setup on Orange Pi 6 Plus (~5 minutes)

### Step 1 — Pull Latest Code Safely (Preserving Any Local OPi Commits)
If you made local commits on your OPi, rebase them cleanly on top of `origin/main`:
```bash
cd ~/cynapse-harness   # (or your repo directory on OPi)
git fetch origin
git pull --rebase origin main
```
*(If you have uncommitted local edits, run `git stash && git pull --rebase origin main && git stash pop`.)*

### Step 2 — Install Vulkan & Build Dependencies
```bash
sudo apt update && sudo apt install -y \
  build-essential cmake git libvulkan-dev glslc vulkan-tools
```

### Step 3 — Verify Vulkan GPU Visibility
Confirm your Immortalis-G720 / Mali-G720 GPU is visible to Linux Vulkan:
```bash
vulkaninfo --summary
```
Look for `Immortalis-G720` or `Mali-G720` under `GPU0`. *(If missing, ensure Mesa Panfrost/PanVK or Arm Mali Vulkan ICD is installed in `/usr/share/vulkan/icd.d/`.)*

### Step 4 — Build Native ARMv9 + Vulkan `llama-server` (~3–4 min)
```bash
rm -rf /tmp/llama.cpp
git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
cmake -S /tmp/llama.cpp -B /tmp/llama.cpp/build \
  -DGGML_VULKAN=ON \
  -DGGML_CPU_KLEIDIAI=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_C_FLAGS="-march=armv9-a+dotprod+fp16+i8mm" \
  -DCMAKE_CXX_FLAGS="-march=armv9-a+dotprod+fp16+i8mm"
cmake --build /tmp/llama.cpp/build --target llama-server -j8
mkdir -p ~/.cynapse/bin
cp /tmp/llama.cpp/build/bin/llama-server ~/.cynapse/bin/llama-server
```
*(Note: `-DBUILD_SHARED_LIBS=OFF` bundles the Vulkan backend directly inside `~/.cynapse/bin/llama-server` so you never get missing `.so` errors.)*

### Step 5 — Verify GPU Device Detection
```bash
~/.cynapse/bin/llama-server --list-devices
```
Expected output: `Vulkan0: ...` (must **not** be empty).

### Step 6 — Rebuild & Install `cynapse` Release Binary (~2 min)
```bash
cargo build --release -p cynapse
mkdir -p ~/.local/bin
cp --remove-destination target/release/cynapse ~/.local/bin/cynapse
cynapse doctor
```

### Step 7 — Run & Verify GPU Offload
Launch `cynapse`, send a prompt, and inspect `~/.cynapse/logs/llama-server.log` in another terminal:
```bash
tail -n 30 ~/.cynapse/logs/llama-server.log
```
Confirm you see:
- `load_tensors: offloaded ... layers to GPU`
- `prompt eval time` & `eval time` running on `Vulkan0` instead of `0.6 tok/s` fallback.

---

## 3. Performance Tuning Tips on OPi 6 Plus

1. **Pin to the 8 Big Cortex-A720 Cores (If CPU-bound)**:
   CIX P1 has 8 big A720 cores and 4 small A520 cores. If running CPU-heavy ops, launch with `taskset -c 0-7 cynapse` (or verify big-core IDs via `lscpu -e`) so threads never wait on A520 efficiency cores.
2. **Quantized KV Cache (`q8_0`)**:
   Already enabled automatically by `cynapse` (`--cache-type-k q8_0 --cache-type-v q8_0`), cutting LPDDR5 KV bandwidth in half.
3. **Draft Speculative Model**:
   Keep `qwen2.5-0.5b-instruct-q4_k_m.gguf` in `./models/` alongside `Ornith-1.5-9B-Q4_K_M.gguf` so `llama-server` auto-enables `--spec-draft-model` drafting.
