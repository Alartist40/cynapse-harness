//! LeafcutterLLM — Memory-safe LLM inference engine

pub mod api;
pub mod async_io;
pub mod backend;
pub mod bridge;
pub mod cache;
pub mod config;
pub mod detect;

/// Deterministic-mode controls (kimi-k3-in-c "determinism contract"):
/// `LEAFCUTTER_DETERMINISTIC=1` forces serial, f64-accumulated dot products
/// and disables the numeric-regime switches (Q8_K integer dot, AVX2 dual-
/// accumulator FMA splits) so two runs on the same model produce
/// bit-identical logits regardless of machine or thread count.
pub mod deterministic;

/// Background safety monitor — observes CPU temp/RSS and prints warnings
/// to stderr.  Never throttles execution; pure advisory.
pub mod cpu_monitor;
pub mod gguf_provider;

pub mod init;

pub mod inference;
pub mod kernels;

#[cfg(feature = "llama-ffi")]
pub mod llama_ffi;

#[cfg(not(feature = "llama-ffi"))]
pub mod llama_ffi {
    //! Stub module when llama.cpp FFI is not available.
    pub fn backend_init() {}
    pub fn backend_free() {}
    pub struct LlamaModel;
    pub struct LlamaContext;
    pub struct LlamaBatch;

    /// Check whether the real llama.cpp backend is available.
    pub const fn is_available() -> bool { false }

    impl LlamaModel {
        pub fn load(_path: &std::path::Path, _n_gpu_layers: i32) -> Result<Self, String> {
            Err("llama.cpp FFI not available. Build with --features llama-ffi".into())
        }
        pub fn n_vocab(&self) -> i32 { 0 }
        pub fn n_embd(&self) -> i32 { 0 }
        pub fn n_layer(&self) -> i32 { 0 }
        pub fn n_ctx_train(&self) -> i32 { 0 }
        pub fn add_bos_token(&self) -> bool { false }
        pub fn bos_token(&self) -> i32 { 0 }
        pub fn eos_token(&self) -> i32 { 2 }
    }
    impl LlamaContext {
        pub fn new(_model: &LlamaModel, _n_ctx: u32, _n_threads: i32) -> Result<Self, String> {
            Err("llama.cpp FFI not available. Build with --features llama-ffi".into())
        }
        pub fn tokenize(&self, _text: &str, _add_special: bool, _parse_special: bool) -> Vec<i32> {
            vec![]
        }
        pub fn token_to_piece(&self, _token: i32) -> String {
            String::new()
        }
        pub fn forward(&mut self, _tokens: &[i32]) -> Result<Vec<f32>, String> {
            Err("llama.cpp FFI not available. Build with --features llama-ffi".into())
        }
        pub fn generate(&mut self, _prompt: &[i32], _max_tokens: usize, _temperature: f32, _eos_token: i32) -> Vec<i32> {
            vec![]
        }
        pub fn generate_stream<F>(
            &mut self,
            _prompt_tokens: &[i32],
            _max_tokens: usize,
            _temperature: f32,
            _eos_token: i32,
            _on_token: F,
        ) -> Vec<i32>
        where
            F: FnMut(&str),
        {
            vec![]
        }
        pub fn sample_greedy(&self) -> i32 { 0 }
        pub fn sample_temperature(&self, _t: f32) -> i32 { 0 }
    }

    pub struct LlamaInProcessEngine;

    impl LlamaInProcessEngine {
        pub fn load(_path: &std::path::Path, _n_gpu_layers: i32, _n_ctx: u32) -> Result<Self, String> {
            Err("llama.cpp FFI not available. Build with --features llama-ffi".into())
        }
        pub fn generate_stream<F>(
            &self,
            _prompt: &str,
            _max_tokens: usize,
            _temperature: f32,
            _on_token: F,
        ) -> Result<Vec<i32>, String>
        where
            F: FnMut(&str),
        {
            Err("llama.cpp FFI not available. Build with --features llama-ffi".into())
        }
    }
}

pub mod model;
pub mod ollama_backend;
pub mod ornith_config;
pub mod ornith_kernels;
pub mod streaming_ornith;
pub mod profiles;
pub mod bpe_tokenizer;
pub mod safetensor_backend;
pub mod safetensors_loader;
pub mod shard;
pub mod tokenizer;
pub mod launch;

/// Pin the global Rayon thread pool to performance cores on heterogeneous
/// big.LITTLE ARM64 SoCs (e.g. Orange Pi 6 Plus CIX Sky1: 8× Cortex-A720
/// @ 2.2–2.6 GHz + 4× Cortex-A520 @ 1.8 GHz) so parallel GEMV barriers
/// never stall on slow efficiency cores.
pub fn pin_rayon_to_fast_cores() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        {
            let mut freqs: Vec<(usize, u64)> = Vec::new();
            if let Ok(entries) = std::fs::read_dir("/sys/devices/system/cpu") {
                for entry in entries.flatten() {
                    let name = entry.file_name();
                    let s = name.to_string_lossy();
                    if let Some(num_str) = s.strip_prefix("cpu") {
                        if let Ok(cpu_id) = num_str.parse::<usize>() {
                            let freq_path = entry.path().join("cpufreq").join("cpuinfo_max_freq");
                            if let Ok(raw) = std::fs::read_to_string(&freq_path) {
                                if let Ok(khz) = raw.trim().parse::<u64>() {
                                    freqs.push((cpu_id, khz));
                                }
                            }
                        }
                    }
                }
            }
            if freqs.len() >= 6 {
                let min_khz = freqs.iter().map(|&(_, f)| f).min().unwrap_or(0);
                let max_khz = freqs.iter().map(|&(_, f)| f).max().unwrap_or(0);
                if max_khz > min_khz {
                    let mut fast_cpus: Vec<usize> = freqs
                        .into_iter()
                        .filter(|&(_, f)| f > min_khz)
                        .map(|(id, _)| id)
                        .collect();
                    fast_cpus.sort_unstable();
                    if fast_cpus.len() >= 4 {
                        let cpus = fast_cpus.clone();
                        let _ = rayon::ThreadPoolBuilder::new()
                            .num_threads(fast_cpus.len())
                            .start_handler(move |idx| {
                                let cpu = cpus[idx % cpus.len()];
                                unsafe {
                                    let mut set: libc::cpu_set_t = std::mem::zeroed();
                                    libc::CPU_ZERO(&mut set);
                                    libc::CPU_SET(cpu, &mut set);
                                    libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
                                }
                            })
                            .build_global();
                    }
                }
            }
        }
    });
}
