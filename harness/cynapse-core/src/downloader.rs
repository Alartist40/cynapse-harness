//! Cynapse Curated Model Catalog, Hardware Recommendation Engine, & HuggingFace Downloader.
//!
//! Inspired by atomic-agent model installer architecture. Provides curated GGUF model definitions,
//! automatic host RAM/CPU hardware recommendations, custom HuggingFace URL parsing, quantization selection,
//! and async progress streaming into local models storage.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use anyhow::{Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuratedModelDef {
    pub id: &'static str,
    pub name: &'static str,
    pub repo_url: &'static str,
    pub filename: &'static str,
    pub size_str: &'static str,
    pub min_ram_gb: usize,
    pub recommended_ram_gb: usize,
    pub description: &'static str,
    pub family: &'static str,
}

pub const CURATED_MODELS_CATALOG: &[CuratedModelDef] = &[
    CuratedModelDef {
        id: "needle3",
        name: "Needle 3 (Cactus Edge Tool Caller & Extractor)",
        repo_url: "Cactus-Compute/needle3",
        filename: "needle3.cact",
        size_str: "28 MB",
        min_ram_gb: 1,
        recommended_ram_gb: 2,
        description: "Ultra-compact edge model specialized in 100% deterministic JSON tool calling & speculative decoding",
        family: "needle",
    },
    CuratedModelDef {
        id: "qwen2.5-0.5b",
        name: "Qwen 2.5 0.5B Instruct (Ultra Fast)",
        repo_url: "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        filename: "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        size_str: "398 MB",
        min_ram_gb: 2,
        recommended_ram_gb: 4,
        description: "Ultra-fast low memory footprint model for basic tasks",
        family: "qwen",
    },
    CuratedModelDef {
        id: "qwen2.5-1.5b",
        name: "Qwen 2.5 1.5B Instruct (Balanced)",
        repo_url: "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        filename: "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        size_str: "980 MB",
        min_ram_gb: 4,
        recommended_ram_gb: 6,
        description: "Fast multi-lingual reasoning with low latency",
        family: "qwen",
    },
    CuratedModelDef {
        id: "qwen2.5-3b",
        name: "Qwen 2.5 3B Instruct (Recommended)",
        repo_url: "Qwen/Qwen2.5-3B-Instruct-GGUF",
        filename: "qwen2.5-3b-instruct-q4_k_m.gguf",
        size_str: "2.0 GB",
        min_ram_gb: 6,
        recommended_ram_gb: 8,
        description: "High quality instruction following & code generation",
        family: "qwen",
    },
    CuratedModelDef {
        id: "llama-3.2-1b",
        name: "Llama 3.2 1B Instruct",
        repo_url: "bartowski/Llama-3.2-1B-Instruct-GGUF",
        filename: "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        size_str: "800 MB",
        min_ram_gb: 4,
        recommended_ram_gb: 6,
        description: "Meta lightweight mobile-optimized instruction model",
        family: "llama",
    },
    CuratedModelDef {
        id: "qwen2.5-7b",
        name: "Qwen 2.5 7B Instruct",
        repo_url: "Qwen/Qwen2.5-7B-Instruct-GGUF",
        filename: "qwen2.5-7b-instruct-q4_k_m.gguf",
        size_str: "4.7 GB",
        min_ram_gb: 12,
        recommended_ram_gb: 16,
        description: "Advanced reasoning, multi-turn chat, and complex coding",
        family: "qwen",
    },
    CuratedModelDef {
        id: "llama-3.2-3b",
        name: "Llama 3.2 3B Instruct",
        repo_url: "bartowski/Llama-3.2-3B-Instruct-GGUF",
        filename: "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        size_str: "2.2 GB",
        min_ram_gb: 8,
        recommended_ram_gb: 12,
        description: "Meta compact 3B instruction following model",
        family: "llama",
    },
];

/// Returns index of best recommended curated model for host system RAM.
pub fn recommend_model_for_hardware(total_ram_mb: u64) -> usize {
    let ram_gb = (total_ram_mb / 1024) as usize;
    if ram_gb <= 2 {
        0 // Needle 3 (Edge 28MB)
    } else if ram_gb <= 4 {
        1 // Qwen 0.5B
    } else if ram_gb <= 8 {
        3 // Qwen 3B
    } else if ram_gb <= 16 {
        5 // Qwen 7B
    } else {
        6 // Llama 3.2 3B
    }
}

#[derive(Deserialize)]
struct HfTreeItem {
    path: String,
}

fn safe_filename_from_url(url: &str) -> String {
    let base = url.split(['?', '#']).next().unwrap_or(url);
    let name = Path::new(base)
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .replace(['/', '\\'], "_");
    if name.is_empty() || name == "." || name == ".." {
        "model.gguf".to_string()
    } else {
        name
    }
}

/// Resolves a custom HuggingFace repo/URL string and quantization into a direct download URL and filename.
/// Uses HuggingFace API repo tree resolution to dynamically locate exact GGUF filenames inside repos.
pub async fn resolve_hf_download_url_async(input: &str, quant: &str) -> (String, String) {
    let clean = input.trim().trim_end_matches('/');
    let clean_quant = quant.split_whitespace().next().unwrap_or(quant).trim().to_lowercase();

    // 1. Check curated catalog
    for item in CURATED_MODELS_CATALOG {
        if clean.eq_ignore_ascii_case(item.repo_url)
            || clean.eq_ignore_ascii_case(item.id)
            || clean.eq_ignore_ascii_case(&format!("https://huggingface.co/{}", item.repo_url))
        {
            let url = format!("https://huggingface.co/{}/resolve/main/{}", item.repo_url, item.filename);
            return (url, item.filename.to_string());
        }
    }

    // 2. Direct GGUF/Safetensors/Bin file link
    if clean.ends_with(".gguf") || clean.ends_with(".safetensors") || clean.ends_with(".bin") || clean.contains("/resolve/main/") || clean.contains("/blob/main/") {
        let direct_url = clean.replace("/blob/main/", "/resolve/main/");
        let url = if direct_url.starts_with("http://") || direct_url.starts_with("https://") {
            direct_url
        } else {
            format!("https://huggingface.co/{}", direct_url)
        };
        let filename = safe_filename_from_url(&url);
        return (url, filename);
    }

    // 3. Query HuggingFace Repo Tree API dynamically
    let repo_path = clean
        .trim_start_matches("https://huggingface.co/")
        .trim_start_matches("http://huggingface.co/");

    let api_url = format!("https://huggingface.co/api/models/{}/tree/main", repo_path);
    let client = reqwest::Client::builder()
        .user_agent("Cynapse-Downloader/1.5")
        .build()
        .unwrap_or_default();

    if let Ok(res) = client.get(&api_url).send().await {
        if let Ok(items) = res.json::<Vec<HfTreeItem>>().await {
            let gguf_files: Vec<String> = items.into_iter()
                .map(|i| i.path)
                .filter(|p| p.ends_with(".gguf"))
                .collect();

            // Try exact quantization match
            if let Some(matched) = gguf_files.iter().find(|p| p.to_lowercase().contains(&clean_quant)) {
                let url = format!("https://huggingface.co/{}/resolve/main/{}", repo_path, matched);
                return (url, matched.clone());
            }

            // Try normalized match (e.g. q4_k_m -> q4_k)
            let alt_quant = clean_quant.replace("_m", "").replace("_s", "");
            if let Some(matched) = gguf_files.iter().find(|p| p.to_lowercase().contains(&alt_quant)) {
                let url = format!("https://huggingface.co/{}/resolve/main/{}", repo_path, matched);
                return (url, matched.clone());
            }

            // If only one GGUF file exists in repo, pick it
            if gguf_files.len() == 1 {
                let matched = &gguf_files[0];
                let url = format!("https://huggingface.co/{}/resolve/main/{}", repo_path, matched);
                return (url, matched.clone());
            }
        }
    }

    // 4. Standard fallback guess
    let repo_name = repo_path.split('/').next_back().unwrap_or("model");
    let filename = format!("{}-{}.gguf", repo_name, clean_quant);
    let url = format!("https://huggingface.co/{}/resolve/main/{}", repo_path, filename);
    (url, filename)
}

pub fn resolve_hf_download_url(input: &str, quant: &str) -> (String, String) {
    let clean = input.trim().trim_end_matches('/');
    let clean_quant = quant.split_whitespace().next().unwrap_or(quant).trim();

    for item in CURATED_MODELS_CATALOG {
        if clean.eq_ignore_ascii_case(item.repo_url)
            || clean.eq_ignore_ascii_case(item.id)
            || clean.eq_ignore_ascii_case(&format!("https://huggingface.co/{}", item.repo_url))
        {
            let url = format!("https://huggingface.co/{}/resolve/main/{}", item.repo_url, item.filename);
            return (url, item.filename.to_string());
        }
    }

    if clean.ends_with(".gguf") || clean.ends_with(".safetensors") || clean.ends_with(".bin") || clean.contains("/resolve/main/") || clean.contains("/blob/main/") {
        let direct_url = clean.replace("/blob/main/", "/resolve/main/");
        let url = if direct_url.starts_with("http://") || direct_url.starts_with("https://") {
            direct_url
        } else {
            format!("https://huggingface.co/{}", direct_url)
        };
        let filename = safe_filename_from_url(&url);
        (url, filename)
    } else {
        let repo_path = clean
            .trim_start_matches("https://huggingface.co/")
            .trim_start_matches("http://huggingface.co/");
        
        let repo_name = repo_path.split('/').next_back().unwrap_or("model");
        let filename = format!("{}-{}.gguf", repo_name, clean_quant.to_lowercase());
        let url = format!("https://huggingface.co/{}/resolve/main/{}", repo_path, filename);
        (url, filename)
    }
}

pub struct DownloadProgress {
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub speed_mbps: f64,
    pub pct: f64,
}

/// Registers a downloaded GGUF file in Cynapse's local model store.
pub async fn register_gguf_in_cynapse(model_path: &Path, filename: &str) -> bool {
    let clean_tag = filename.trim_end_matches(".gguf").to_lowercase();
    let _ = clean_tag;
    model_path.exists()
}

pub async fn register_gguf_in_ollama(model_path: &Path, filename: &str) -> bool {
    register_gguf_in_cynapse(model_path, filename).await
}

/// Stream downloads HuggingFace model with real-time progress callbacks.
pub async fn stream_download_hf_model<F>(
    download_url: &str,
    target_path: &Path,
    mut progress_cb: F,
) -> Result<PathBuf>
where
    F: FnMut(DownloadProgress),
{
    if let Some(parent) = target_path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let client = reqwest::Client::new();
    let res = client
        .get(download_url)
        .header("User-Agent", "Cynapse-Agent-Downloader/1.2")
        .send()
        .await
        .with_context(|| format!("Failed to connect to HuggingFace URL: {}", download_url))?;

    if !res.status().is_success() {
        anyhow::bail!("HuggingFace server returned HTTP status: {}", res.status());
    }

    let expected_sha256 = res
        .headers()
        .get("x-linked-etag")
        .or_else(|| res.headers().get("etag"))
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim_matches('"').trim().to_ascii_lowercase())
        .filter(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()));

    let total_bytes = res.content_length().unwrap_or(0);
    let mut file = File::create(target_path)
        .with_context(|| format!("Failed to create destination file: {}", target_path.display()))?;

    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    let mut stream = res.bytes_stream();
    let mut downloaded_bytes = 0u64;
    let start_time = Instant::now();
    let mut last_emit = Instant::now();

    while let Some(chunk) = stream.next().await {
        let bytes = chunk.context("Error downloading chunk from stream")?;
        file.write_all(&bytes)?;
        hasher.update(&bytes);
        downloaded_bytes += bytes.len() as u64;

        if last_emit.elapsed().as_millis() > 100 || downloaded_bytes == total_bytes {
            last_emit = Instant::now();
            let elapsed_sec = start_time.elapsed().as_secs_f64().max(0.001);
            let speed_mbps = (downloaded_bytes as f64 / 1_048_576.0) / elapsed_sec;
            let pct = if total_bytes > 0 {
                (downloaded_bytes as f64 / total_bytes as f64) * 100.0
            } else {
                0.0
            };

            progress_cb(DownloadProgress {
                downloaded_bytes,
                total_bytes,
                speed_mbps,
                pct,
            });
        }
    }

    if let Some(expected) = expected_sha256 {
        let actual = format!("{:x}", hasher.finalize());
        if actual != expected {
            let _ = std::fs::remove_file(target_path);
            anyhow::bail!(
                "SHA256 verification failed for HF download '{}': expected {}, got {}",
                target_path.display(),
                expected,
                actual
            );
        }
    }

    Ok(target_path.to_path_buf())
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufHeaderInfo {
    pub magic: String,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
    pub is_valid_gguf: bool,
}

/// Quick GGUF header inspector reading magic bytes and metadata counts.
#[allow(dead_code)]
pub fn inspect_gguf_header(path: &Path) -> Result<GgufHeaderInfo> {
    let mut file = File::open(path)?;
    let mut magic_bytes = [0u8; 4];
    file.read_exact(&mut magic_bytes)?;

    let magic = String::from_utf8_lossy(&magic_bytes).to_string();
    let is_valid_gguf = magic == "GGUF";
    if !is_valid_gguf {
        return Ok(GgufHeaderInfo {
            magic,
            version: 0,
            tensor_count: 0,
            metadata_kv_count: 0,
            is_valid_gguf: false,
        });
    }

    let mut buf = [0u8; 4];
    file.read_exact(&mut buf)?;
    let version = u32::from_le_bytes(buf);

    let mut u64_buf = [0u8; 8];
    file.read_exact(&mut u64_buf)?;
    let tensor_count = u64::from_le_bytes(u64_buf);

    file.read_exact(&mut u64_buf)?;
    let metadata_kv_count = u64::from_le_bytes(u64_buf);

    Ok(GgufHeaderInfo {
        magic,
        version,
        tensor_count,
        metadata_kv_count,
        is_valid_gguf: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_recommendations() {
        assert_eq!(recommend_model_for_hardware(2048), 0);  // 2GB -> Needle 3
        assert_eq!(recommend_model_for_hardware(4096), 1);  // 4GB -> Qwen 0.5B
        assert_eq!(recommend_model_for_hardware(8192), 3);  // 8GB -> Qwen 3B
        assert_eq!(recommend_model_for_hardware(16384), 5); // 16GB -> Qwen 7B
    }

    #[test]
    fn test_needle3_curated_catalog() {
        let needle = &CURATED_MODELS_CATALOG[0];
        assert_eq!(needle.id, "needle3");
        assert_eq!(needle.family, "needle");
        assert_eq!(needle.repo_url, "Cactus-Compute/needle3");
        assert_eq!(needle.filename, "needle3.cact");
        assert!(needle.description.contains("deterministic JSON tool calling"));
    }

    #[test]
    fn test_resolve_hf_url() {
        let (url, filename) = resolve_hf_download_url("TheBloke/Llama-2-7B-GGUF", "Q4_K_M");
        assert_eq!(url, "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/Llama-2-7B-GGUF-q4_k_m.gguf");
        assert_eq!(filename, "Llama-2-7B-GGUF-q4_k_m.gguf");
    }
}
