//! Tool Output Compressor for Cynapse Agent System.
//!
//! Inspired by Atomic Agent's `result-compressor.ts` & DeepSeek Harness `compaction-tool-result-pruner`:
//! Shrinks verbose tool output (cargo build logs, test runs, grep results, file reads)
//! into a compact summary that preserves error signatures and the trailing execution tail
//! without exhausting small model context windows.

use serde::{Deserialize, Serialize};

use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompressedResult {
    pub summary: String,
    pub signature: Option<String>,
    pub truncated: bool,
    pub original_len: usize,
    pub spill_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressorOptions {
    pub max_summary_length: usize,
    pub max_tail_lines: usize,
    pub spill_threshold: usize,
    /// Head budget for tool-result pruner (borrowed from DeepSeek Harness).
    pub head_chars: usize,
    /// Tail budget for tool-result pruner.
    pub tail_chars: usize,
}

impl Default for CompressorOptions {
    fn default() -> Self {
        Self {
            max_summary_length: 1200,
            max_tail_lines: 20,
            spill_threshold: 1200,
            head_chars: 400,
            tail_chars: 400,
        }
    }
}

pub const PRUNE_MARKER: &str = "\n\n[... tool output middle pruned for small-model context ...]\n\n";

/// Replay-safe, deterministic tool-output middle pruner (DeepSeek Harness paradigm).
/// Keeps head and tail while eliding redundant middle output if over threshold.
pub fn prune_tool_output(raw: &str, threshold_chars: usize, head_chars: usize, tail_chars: usize) -> Option<String> {
    let char_count = raw.chars().count();
    if char_count <= threshold_chars {
        return None;
    }
    if head_chars + tail_chars >= char_count {
        return None;
    }

    let head: String = raw.chars().take(head_chars).collect();
    let tail: String = raw.chars().skip(char_count.saturating_sub(tail_chars)).collect();

    Some(format!("{}{}{}", head, PRUNE_MARKER, tail))
}

/// Spills large raw outputs to `~/.cynapse/spill/<hash>.log` to keep model context windows lean.
pub fn spill_to_disk(raw: &str) -> Option<PathBuf> {
    let spill_dir = if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".cynapse").join("spill")
    } else {
        PathBuf::from(".cynapse").join("spill")
    };

    if fs::create_dir_all(&spill_dir).is_err() {
        return None;
    }

    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    raw.hash(&mut hasher);
    let hash_val = hasher.finish();

    let file_path = spill_dir.join(format!("{:016x}.log", hash_val));
    if fs::write(&file_path, raw).is_ok() {
        Some(file_path)
    } else {
        None
    }
}

/// Enforces the 5-item reader limit for lists and multi-line grep/directory search results.
pub fn cap_list_items(text: &str, max_items: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() <= max_items {
        return text.to_string();
    }
    let shown = &lines[..max_items];
    let omitted = lines.len() - max_items;
    format!("{}\n[{} items shown, {} more omitted]", shown.join("\n"), max_items, omitted)
}

const ERROR_MARKERS: &[&str] = &[
    "error:",
    "error[",
    "failed:",
    "panic:",
    "fatal:",
    "exception:",
    "traceback",
    "assertionerror",
    "cannot find",
    "undefined",
    "syntax error",
];

/// Extracts key diagnostic signature lines if output indicates error or failure.
fn extract_signature(text: &str, is_error: bool) -> Option<String> {
    if !is_error {
        return None;
    }
    let mut sigs = Vec::new();
    for line in text.lines() {
        let lower = line.to_lowercase();
        for marker in ERROR_MARKERS {
            if lower.contains(marker) {
                let trimmed = line.trim();
                let sig = if trimmed.chars().count() > 250 {
                    trimmed.chars().take(250).collect::<String>() + "…"
                } else {
                    trimmed.to_string()
                };
                if !sigs.contains(&sig) {
                    sigs.push(sig);
                }
                break;
            }
        }
        if sigs.len() >= 3 {
            break;
        }
    }
    if sigs.is_empty() {
        None
    } else {
        Some(sigs.iter().map(|s| format!("key: {}", s)).collect::<Vec<_>>().join("\n"))
    }
}

/// Extracts trailing lines while skipping massive intermediate logs.
fn extract_tail(text: &str, max_lines: usize) -> (String, bool) {
    if text.is_empty() {
        return (String::new(), false);
    }
    let non_empty_lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    if non_empty_lines.len() <= max_lines {
        return (non_empty_lines.join("\n"), false);
    }
    let omitted = non_empty_lines.len() - max_lines;
    let tail_slice = &non_empty_lines[omitted..];
    let result = format!("… [omitted {} lines]\n{}", omitted, tail_slice.join("\n"));
    (result, true)
}

/// Compresses a raw tool output string into a compact representation.
pub fn compress_tool_result(raw: &str, is_error: bool, options: Option<CompressorOptions>) -> CompressedResult {
    let opts = options.unwrap_or_default();
    let original_len = raw.len();
    let normalised = raw.replace("\r\n", "\n").trim_end().to_string();

    // Stage 1: Deterministic head/tail middle-pruning (DeepSeek Harness style) if output exceeds threshold
    let (staged_text, middle_pruned) = if let Some(pruned) = prune_tool_output(
        &normalised,
        opts.spill_threshold,
        opts.head_chars,
        opts.tail_chars,
    ) {
        (pruned, true)
    } else {
        (normalised, false)
    };

    let (tail, tail_truncated) = extract_tail(&staged_text, opts.max_tail_lines);
    let signature = extract_signature(&staged_text, is_error);

    let mut parts = Vec::new();
    if let Some(ref sig) = signature {
        parts.push(sig.clone());
    }
    if !tail.is_empty() {
        parts.push(tail);
    }

    let joined = parts.join("\n");
    let over_length = joined.chars().count() > opts.max_summary_length;

    let spill_path = if original_len > opts.spill_threshold {
        spill_to_disk(raw).map(|p| p.to_string_lossy().to_string())
    } else {
        None
    };

    let mut summary = if over_length {
        let mut truncated_text: String = joined.chars().take(opts.max_summary_length.saturating_sub(15)).collect();
        truncated_text.push_str("\n… [truncated]");
        truncated_text
    } else {
        joined
    };

    if let Some(ref spill) = spill_path {
        summary.push_str(&format!("\n[Full output ({} bytes) spilled to {}; use grep or read_file to inspect]", original_len, spill));
    }

    CompressedResult {
        summary,
        signature,
        truncated: middle_pruned || tail_truncated || over_length || spill_path.is_some(),
        original_len,
        spill_path,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_short_output() {
        let text = "Hello world\nSecond line";
        let res = compress_tool_result(text, false, None);
        assert_eq!(res.summary, text);
        assert!(!res.truncated);
        assert_eq!(res.signature, None);
    }

    #[test]
    fn test_prune_tool_output_middle() {
        let text = format!("START_DATA {} END_DATA", "x".repeat(2000));
        let pruned = prune_tool_output(&text, 500, 20, 20);
        assert!(pruned.is_some());
        let val = pruned.unwrap();
        assert!(val.starts_with("START_DATA"));
        assert!(val.ends_with("END_DATA"));
        assert!(val.contains(PRUNE_MARKER));
    }

    #[test]
    fn test_compress_error_signature_extracted() {
        let text = "Compiling cynapse...\nerror: unresolved import `super::foo`\n   --> src/lib.rs:10\nBuild failed.";
        let res = compress_tool_result(text, true, None);
        assert!(res.signature.is_some());
        assert!(res.signature.as_ref().unwrap().contains("key: error: unresolved import"));
    }

    #[test]
    fn test_compress_multi_error_signatures() {
        let text = "Compiling...\nerror[E0425]: cannot find value `foo`\nerror[E0425]: cannot find value `bar`\nfatal: aborting";
        let res = compress_tool_result(text, true, None);
        assert!(res.signature.is_some());
        let sig = res.signature.unwrap();
        assert!(sig.contains("key: error[E0425]: cannot find value `foo`"));
        assert!(sig.contains("key: error[E0425]: cannot find value `bar`"));
    }

    #[test]
    fn test_compress_tail_omission() {
        let mut lines = Vec::new();
        for i in 1..=50 {
            lines.push(format!("Log line {}", i));
        }
        let text = lines.join("\n");
        let res = compress_tool_result(&text, false, Some(CompressorOptions {
            max_summary_length: 1000,
            max_tail_lines: 5,
            spill_threshold: 1200,
            head_chars: 200,
            tail_chars: 200,
        }));
        assert!(res.truncated);
        assert!(res.summary.contains("… [omitted 45 lines]"));
        assert!(res.summary.contains("Log line 50"));
    }

    #[test]
    fn test_cap_list_items() {
        let items = "item 1\nitem 2\nitem 3\nitem 4\nitem 5\nitem 6\nitem 7";
        let capped = cap_list_items(items, 5);
        assert!(capped.contains("item 1"));
        assert!(capped.contains("item 5"));
        assert!(capped.contains("[5 items shown, 2 more omitted]"));
        assert!(!capped.contains("item 6"));
    }

    #[test]
    fn test_spill_to_disk() {
        let massive = "a".repeat(2000);
        let res = compress_tool_result(&massive, false, Some(CompressorOptions {
            max_summary_length: 500,
            max_tail_lines: 5,
            spill_threshold: 500,
            head_chars: 50,
            tail_chars: 50,
        }));
        assert!(res.spill_path.is_some());
        assert!(res.summary.contains("spilled to"));
        assert!(res.truncated);
    }
}
