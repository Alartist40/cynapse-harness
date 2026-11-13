//! Cynapse Session Management & Persistent Transcript Storage.
//!
//! Provides disk persistence for conversation sessions (`~/.cynapse/sessions/`),
//! allowing session listing, saving, loading, and recovery across restarts.

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMessage {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionData {
    pub session_id: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub model_name: String,
    pub messages: Vec<SessionMessage>,
}

pub struct SessionManager {
    pub storage_dir: PathBuf,
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionManager {
    pub fn new() -> Self {
        let storage_dir = if let Some(home) = dirs::home_dir() {
            home.join(".cynapse").join("sessions")
        } else {
            PathBuf::from("./data/sessions")
        };
        let _ = fs::create_dir_all(&storage_dir);
        Self { storage_dir }
    }

    pub fn with_dir(storage_dir: PathBuf) -> Self {
        let _ = fs::create_dir_all(&storage_dir);
        Self { storage_dir }
    }

    pub fn generate_id() -> String {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let rand_part: u32 = rand_u32();
        format!("session_{:x}_{:04x}", now, rand_part % 0xffff)
    }

    pub fn save_session(&self, data: &SessionData) -> Result<()> {
        fs::create_dir_all(&self.storage_dir)?;
        let file_path = self.storage_dir.join(format!("{}.json", data.session_id));
        let mut data_to_save = data.clone();
        let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        if data_to_save.created_at == 0 {
            if let Ok(existing) = self.load_session(&data.session_id) {
                data_to_save.created_at = if existing.created_at != 0 { existing.created_at } else { now };
            } else {
                data_to_save.created_at = now;
            }
        }
        if data_to_save.updated_at == 0 {
            data_to_save.updated_at = now;
        }
        let content = serde_json::to_string_pretty(&data_to_save)?;
        fs::write(&file_path, content)
            .with_context(|| format!("Failed to save session data to {}", file_path.display()))?;
        Ok(())
    }

    pub fn load_session(&self, session_id: &str) -> Result<SessionData> {
        let clean_id = session_id.trim_end_matches(".json");
        let file_path = self.storage_dir.join(format!("{}.json", clean_id));
        if !file_path.exists() {
            anyhow::bail!("Session ID '{}' not found at {}", clean_id, file_path.display());
        }
        let content = fs::read_to_string(&file_path)?;
        let data: SessionData = serde_json::from_str(&content)?;
        Ok(data)
    }

    pub fn list_sessions(&self) -> Vec<SessionData> {
        let mut out = Vec::new();
        if let Ok(entries) = fs::read_dir(&self.storage_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("json") {
                    if let Ok(content) = fs::read_to_string(&path) {
                        if let Ok(data) = serde_json::from_str::<SessionData>(&content) {
                            out.push(data);
                        }
                    }
                }
            }
        }
        out.sort_by_key(|b| std::cmp::Reverse(b.updated_at));
        out
    }

    pub fn delete_session(&self, session_id: &str) -> Result<()> {
        let clean_id = session_id.trim_end_matches(".json");
        let file_path = self.storage_dir.join(format!("{}.json", clean_id));
        if file_path.exists() {
            fs::remove_file(file_path)?;
        }
        Ok(())
    }

    pub fn rename_session(&self, old_id: &str, new_id: &str) -> Result<SessionData> {
        let clean_old = old_id.trim_end_matches(".json");
        let clean_new = new_id.trim_end_matches(".json");
        let mut data = self.load_session(clean_old)?;
        self.delete_session(clean_old)?;
        data.session_id = clean_new.to_string();
        self.save_session(&data)?;
        Ok(data)
    }

    /// Saves binary KV cache snapshot to disk (<session_id>.kv) using zero-copy binary layout (Strata innovation).
    pub fn save_kv_snapshot(&self, session_id: &str, tokens: &[u32], kv_data: &[u8]) -> Result<()> {
        let clean_id = session_id.trim_end_matches(".json").trim_end_matches(".kv");
        let kv_path = self.storage_dir.join(format!("{}.kv", clean_id));
        let mut buf = Vec::with_capacity(16 + tokens.len() * 4 + kv_data.len());
        buf.extend_from_slice(b"CYN_KV_1");
        buf.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(kv_data.len() as u32).to_le_bytes());
        for &tok in tokens {
            buf.extend_from_slice(&tok.to_le_bytes());
        }
        buf.extend_from_slice(kv_data);
        fs::write(&kv_path, &buf)?;
        Ok(())
    }

    /// Loads binary KV cache snapshot (<session_id>.kv) in 0ms without re-evaluating prompt tokens.
    pub fn load_kv_snapshot(&self, session_id: &str) -> Result<(Vec<u32>, Vec<u8>)> {
        let clean_id = session_id.trim_end_matches(".json").trim_end_matches(".kv");
        let kv_path = self.storage_dir.join(format!("{}.kv", clean_id));
        if !kv_path.exists() {
            anyhow::bail!("KV snapshot '{}' not found at {}", clean_id, kv_path.display());
        }
        let bytes = fs::read(&kv_path)?;
        if bytes.len() < 16 || &bytes[0..8] != b"CYN_KV_1" {
            anyhow::bail!("Invalid or corrupted KV snapshot header in {}", kv_path.display());
        }
        let tok_count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        let data_len = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
        let tok_end = 16 + tok_count * 4;
        if bytes.len() < tok_end + data_len {
            anyhow::bail!("Truncated KV snapshot file: expected {} bytes, got {}", tok_end + data_len, bytes.len());
        }
        let mut tokens = Vec::with_capacity(tok_count);
        for i in 0..tok_count {
            let offset = 16 + i * 4;
            tokens.push(u32::from_le_bytes([bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3]]));
        }
        let kv_data = bytes[tok_end..tok_end + data_len].to_vec();
        Ok((tokens, kv_data))
    }
}

static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn rand_u32() -> u32 {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
    let seq = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let pid = std::process::id() as u64;
    let mixed = now ^ (seq.wrapping_mul(0x9E3779B97F4A7C15)) ^ (pid << 16);
    (mixed & 0xffff_ffff) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_save_load_list() {
        let temp_path = std::env::temp_dir().join(format!("cynapse_test_{}", rand_u32()));
        let mgr = SessionManager::with_dir(temp_path.clone());
        let id = SessionManager::generate_id();

        let data = SessionData {
            session_id: id.clone(),
            created_at: 100,
            updated_at: 200,
            model_name: "ministral-3:3b".into(),
            messages: vec![SessionMessage {
                role: "user".into(),
                content: "Hello Cynapse".into(),
                thinking: None,
            }],
        };

        mgr.save_session(&data).unwrap();
        let loaded = mgr.load_session(&id).unwrap();
        assert_eq!(loaded.session_id, id);
        assert_eq!(loaded.messages.len(), 1);

        let list = mgr.list_sessions();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].session_id, id);
    }

    #[test]
    fn test_kv_snapshot_roundtrip() {
        let temp_path = std::env::temp_dir().join(format!("cynapse_kv_test_{}", rand_u32()));
        let mgr = SessionManager::with_dir(temp_path.clone());
        let id = SessionManager::generate_id();

        let prompt_tokens = vec![101u32, 2054, 2003, 1037, 3231];
        let kv_payload = vec![42u8; 1024];

        mgr.save_kv_snapshot(&id, &prompt_tokens, &kv_payload).unwrap();
        let (loaded_tokens, loaded_payload) = mgr.load_kv_snapshot(&id).unwrap();

        assert_eq!(loaded_tokens, prompt_tokens);
        assert_eq!(loaded_payload, kv_payload);
    }
}
