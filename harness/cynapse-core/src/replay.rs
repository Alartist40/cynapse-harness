//! Offline Session Replay & Hardware Fit-Test (inspired by atomic-agent).

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayTurn {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFitScore {
    pub model: String,
    pub ttft_ms: u64,
    pub tok_per_sec: f64,
    pub tool_json_ok: bool,
    pub score: u8,
}

pub fn replay_dir() -> PathBuf {
    dirs::home_dir()
        .map(|h| h.join(".cynapse").join("replays"))
        .unwrap_or_else(|| PathBuf::from("./data/replays"))
}

pub fn save_replay(session_id: &str, turns: &[ReplayTurn]) -> Result<PathBuf> {
    let dir = replay_dir();
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{}.json", session_id));
    std::fs::write(&path, serde_json::to_string_pretty(turns)?)?;
    Ok(path)
}

pub fn score_model_fit(model: &str, ttft_ms: u64, tok_per_sec: f64, tool_json_ok: bool) -> ModelFitScore {
    let speed_pts = ((tok_per_sec * 4.0).min(40.0)) as u8;
    let ttft_pts = if ttft_ms <= 1500 { 30 } else if ttft_ms <= 5000 { 20 } else { 10 };
    let tool_pts = if tool_json_ok { 30 } else { 0 };
    ModelFitScore {
        model: model.to_string(),
        ttft_ms,
        tok_per_sec,
        tool_json_ok,
        score: (speed_pts + ttft_pts + tool_pts).min(100),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_model_fit_bounds() {
        let s = score_model_fit("LFM3-1.7B-GGUF", 1200, 18.0, true);
        assert_eq!(s.score, 100);
        let s2 = score_model_fit("slow.gguf", 9000, 1.0, false);
        assert_eq!(s2.score, 14);
    }
}
