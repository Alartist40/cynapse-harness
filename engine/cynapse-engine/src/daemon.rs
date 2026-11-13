//! Managed llama-server child daemon lifecycle.
//!
//! Supervises an in-process child process running `llama-server` on `127.0.0.1:38265`.
//! Automatically locates `llama-server` binary, tunes hardware flags (`--parallel 2`,
//! `--flash-attn auto`, `--cache-type-k q8_0`, `-ngl -1`), monitors health probes,
//! and ensures clean process termination on drop.

use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use anyhow::{bail, Result};

pub const DEFAULT_DAEMON_PORT: u16 = 38265;

pub struct LlamaServerDaemon {
    child: Option<Child>,
    pub port: u16,
    pub model_path: PathBuf,
}

impl LlamaServerDaemon {
    /// Discovers `llama-server` executable in local workspace bin/, system PATH, or fallback locations
    pub fn find_binary() -> Option<PathBuf> {
        // 1. Workspace local bin directory
        if let Ok(cwd) = std::env::current_dir() {
            let local_bin = cwd.join("bin").join("llama-server");
            if local_bin.is_file() {
                return Some(local_bin);
            }
        }
        for local_rel in ["bin/llama-server", "./bin/llama-server", "../bin/llama-server"] {
            let p = PathBuf::from(local_rel);
            if p.is_file() {
                return Some(p);
            }
        }

        if let Ok(home) = std::env::var("HOME") {
            let user_bin = PathBuf::from(home).join(".cynapse").join("bin").join("llama-server");
            if user_bin.is_file() {
                return Some(user_bin);
            }
        }

        if let Ok(path_env) = std::env::var("PATH") {
            for entry in std::env::split_paths(&path_env) {
                let candidate = entry.join("llama-server");
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }

        let fallback_paths = [
            "/usr/local/lib/ollama/llama-server",
            "/usr/local/bin/llama-server",
            "/usr/bin/llama-server",
            "/opt/homebrew/bin/llama-server",
        ];
        for path in fallback_paths {
            let p = PathBuf::from(path);
            if p.is_file() {
                return Some(p);
            }
        }

        None
    }

    /// Discovers `libggml-vulkan.so` for GPU offloading via `GGML_BACKEND_PATH`.
    pub fn find_vulkan_backend() -> Option<PathBuf> {
        let mut candidates = Vec::new();
        if let Ok(cwd) = std::env::current_dir() {
            candidates.push(cwd.join("lib").join("libggml-vulkan.so"));
        }
        if let Ok(home) = std::env::var("HOME") {
            candidates.push(PathBuf::from(&home).join(".cynapse").join("lib").join("libggml-vulkan.so"));
        }
        candidates.push(PathBuf::from("/usr/local/lib/ollama/vulkan/libggml-vulkan.so"));
        candidates.push(PathBuf::from("/usr/local/lib/libggml-vulkan.so"));
        candidates.into_iter().find(|p| p.is_file())
    }

    /// Checks if a daemon is actively ready (HTTP /health returns 200 OK, not 503 Loading).
    pub fn is_healthy(port: u16) -> bool {
        use std::io::{Read, Write};
        let addr_str = format!("127.0.0.1:{}", port);
        let Ok(addr) = addr_str.parse() else { return false; };
        let Ok(mut stream) = TcpStream::connect_timeout(&addr, Duration::from_millis(250)) else {
            return false;
        };
        let _ = stream.set_read_timeout(Some(Duration::from_millis(250)));
        let _ = stream.set_write_timeout(Some(Duration::from_millis(250)));
        if stream.write_all(b"GET /health HTTP/1.0\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n").is_err() {
            return false;
        }
        let mut buf = [0u8; 64];
        match stream.read(&mut buf) {
            Ok(n) if n > 0 => {
                let head = String::from_utf8_lossy(&buf[..n]);
                head.starts_with("HTTP/1.1 200") || head.starts_with("HTTP/1.0 200")
            }
            _ => false,
        }
    }

    /// Spawns a supervised `llama-server` child process.
    pub fn spawn(model_path: &Path, port: u16) -> Result<Self> {
        let bin = match Self::find_binary() {
            Some(b) => b,
            None => bail!("`llama-server` binary not found in ~/.cynapse/bin/ or PATH"),
        };

        if !model_path.exists() {
            bail!("Model file not found: {}", model_path.display());
        }

        let log_dir = if let Ok(home) = std::env::var("HOME") {
            PathBuf::from(home).join(".cynapse").join("logs")
        } else {
            PathBuf::from("./logs")
        };
        let _ = std::fs::create_dir_all(&log_dir);
        let log_path = log_dir.join("llama-server.log");
        let (out_stdio, err_stdio) = if let Ok(f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
        {
            let err_f = f.try_clone().map(Stdio::from).unwrap_or_else(|_| Stdio::null());
            (Stdio::from(f), err_f)
        } else {
            (Stdio::null(), Stdio::null())
        };

        let vulkan_backend = Self::find_vulkan_backend();
        let ngl = if let Some(custom_ngl) = crate::gpu_layers() {
            custom_ngl.to_string()
        } else if vulkan_backend.is_some() {
            // AMD/Intel/Unified Vulkan GPU detected via libggml-vulkan.so
            "99".to_string()
        } else {
            match crate::detect_vram_free_mb() {
                Some(vram) if vram > 6000 => "99".to_string(),
                Some(vram) if vram > 3000 => "33".to_string(),
                Some(vram) if vram > 1000 => "16".to_string(),
                _ => "0".to_string(),
            }
        };
        let threads = crate::physical_core_count().to_string();
        // Context size: uses configured engine context or default ENGINE_CTX_SIZE
        let ctx_size = crate::engine_ctx_size().to_string();

        let kv_dir = if let Ok(home) = std::env::var("HOME") {
            PathBuf::from(home).join(".cynapse").join("kv_snapshots")
        } else {
            PathBuf::from("./data/kv_snapshots")
        };
        let _ = std::fs::create_dir_all(&kv_dir);

        let mut cmd = Command::new(&bin);
        cmd.arg("-m")
            .arg(model_path)
            .arg("--port")
            .arg(port.to_string())
            .arg("--host")
            .arg("127.0.0.1")
            .arg("-ngl")
            .arg(&ngl)
            .arg("-t")
            .arg(&threads)
            .arg("--flash-attn")
            .arg("auto")
            .arg("--cache-type-k")
            .arg("q8_0")
            .arg("--cache-type-v")
            .arg("q8_0")
            .arg("--slot-save-path")
            .arg(&kv_dir)
            .arg("--parallel")
            .arg("1")
            .arg("--ctx-size")
            .arg(&ctx_size)
            .arg("--no-webui")
            .arg("--jinja")
            .stdout(out_stdio)
            .stderr(err_stdio);

        // Set LD_LIBRARY_PATH to ensure Cynapse libraries and acceleration backends resolve
        let mut ld_dirs = Vec::new();
        if let Ok(cwd) = std::env::current_dir() {
            ld_dirs.push(cwd.join("lib").to_string_lossy().to_string());
        }
        ld_dirs.push("./lib".to_string());
        if let Ok(home) = std::env::var("HOME") {
            ld_dirs.push(format!("{}/.cynapse/lib", home));
        }
        ld_dirs.push("/usr/local/lib/ollama".to_string());
        ld_dirs.push("/usr/local/lib/ollama/vulkan".to_string());
        if let Ok(existing_ld) = std::env::var("LD_LIBRARY_PATH") {
            if !existing_ld.is_empty() {
                ld_dirs.push(existing_ld);
            }
        }
        cmd.env("LD_LIBRARY_PATH", ld_dirs.join(":"));
        if let Some(ref vk_path) = vulkan_backend {
            cmd.env("GGML_BACKEND_PATH", vk_path);
        }

        // Check for speculative decoding draft model in parent directory or ~/.cynapse/models
        let mut draft_found = false;
        if let Some(parent) = model_path.parent() {
            if let Ok(entries) = std::fs::read_dir(parent) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("").to_lowercase();
                    if (name.contains("needle") || name.contains("draft") || name.contains("qwen2.5-0.5b"))
                        && name.ends_with(".gguf")
                        && p != model_path
                    {
                        cmd.arg("--spec-draft-model").arg(&p);
                        cmd.arg("--spec-draft-n-max").arg("4");
                        draft_found = true;
                        break;
                    }
                }
            }
        }
        if !draft_found {
            // Enable prompt ngram speculative decoding for immediate zero-RAM acceleration
            cmd.arg("--spec-type").arg("ngram-simple");
        }

        let child = cmd.spawn()?;

        let mut daemon = Self {
            child: Some(child),
            port,
            model_path: model_path.to_path_buf(),
        };

        // Poll HTTP /health == 200 OK for up to 30 seconds (9B GGUF takes ~10-20s to load)
        let start = Instant::now();
        while start.elapsed() < Duration::from_secs(30) {
            if let Some(ref mut c) = daemon.child {
                if let Ok(Some(status)) = c.try_wait() {
                    bail!("`llama-server` exited prematurely with status {}. Check logs at {}", status, log_path.display());
                }
            }
            if Self::is_healthy(port) {
                return Ok(daemon);
            }
            std::thread::sleep(Duration::from_millis(250));
        }

        daemon.shutdown();
        bail!("`llama-server` did not become healthy within 30 seconds. Check logs at {}", log_path.display());
    }

    /// Singleton daemon spawner: checks health, respects failure cooldown, and spawns if binary and model exist.
    pub fn get_or_spawn_daemon(model_path: &Path, port: u16) -> bool {
        static ACTIVE_DAEMON: std::sync::OnceLock<std::sync::Mutex<Option<LlamaServerDaemon>>> = std::sync::OnceLock::new();
        static SPAWN_COOLDOWN: std::sync::OnceLock<std::sync::Mutex<(usize, Option<Instant>)>> = std::sync::OnceLock::new();

        if Self::is_healthy(port) {
            return true;
        }

        let cd_lock = SPAWN_COOLDOWN.get_or_init(|| std::sync::Mutex::new((0, None)));
        if let Ok(cd) = cd_lock.lock() {
            if let Some(until) = cd.1 {
                if Instant::now() < until {
                    return false;
                }
            }
        }

        let mutex = ACTIVE_DAEMON.get_or_init(|| std::sync::Mutex::new(None));
        let mut guard = match mutex.lock() {
            Ok(g) => g,
            Err(_) => return false,
        };

        if Self::is_healthy(port) {
            return true;
        }

        match Self::spawn(model_path, port) {
            Ok(d) => {
                if let Ok(mut cd) = cd_lock.lock() {
                    *cd = (0, None);
                }
                *guard = Some(d);
                true
            }
            Err(_) => {
                if let Ok(mut cd) = cd_lock.lock() {
                    cd.0 += 1;
                    let backoff_secs = match cd.0 {
                        1 => 30,
                        2 => 60,
                        _ => 300,
                    };
                    cd.1 = Some(Instant::now() + Duration::from_secs(backoff_secs));
                }
                false
            }
        }
    }

    /// Graceful shutdown of the child process.
    pub fn shutdown(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for LlamaServerDaemon {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_daemon_health_check_offline() {
        // High random unassigned port should report not healthy
        assert!(!LlamaServerDaemon::is_healthy(59991));
    }
}
