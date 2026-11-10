//! Atomic-Agent Inspired Offline Execution Core.
//!
//! Provides strict GBNF JSON tool grammar validation, stable KV-cache prompt prefixing,
//! parallel read-only tool execution batching, and circular loop-guard protection for offline LLM engines.

use std::collections::{HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use serde::{Deserialize, Serialize};

// ─── Two-Phase Tool Loop Tracker ─────────────────────────────────────────────
// Inspired by Atomic-Agent's `ToolLoopTracker` (loop-detector.ts).
// Two-phase history: check() → record_call() → record_outcome().
// Two counters: args-only repeat (warn), args+result no-progress streak (critical).
// Wandering detector for distinct-args tools. Breaker for consecutive vetoes.

/// Verdict level from the loop tracker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopVerdict {
    /// No loop detected.
    Ok,
    /// Args-only repeat crossing warning threshold — advisory notice.
    Warn,
    /// No-progress streak (same args + same result) crossing critical threshold.
    Critical,
    /// Distinct-args spread on a wandering-prone tool.
    Wandering,
}

/// Classification of which detector fired.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopDetector {
    GenericRepeat,
    NoProgress,
    Wandering,
}

/// Result of a loop check.
#[derive(Debug, Clone)]
pub struct LoopCheckResult {
    pub level: LoopVerdict,
    pub detector: LoopDetector,
    /// Repeat count (warn), streak length (critical), or distinct-args spread (wandering).
    pub count: usize,
}

#[derive(Debug, Clone)]
struct HistoryEntry {
    tool: String,
    args_hash: u64,
    result_hash: Option<u64>,
    vetoed: bool,
}

/// Options for the tool loop tracker.
#[derive(Debug, Clone)]
pub struct ToolLoopTrackerOptions {
    /// Args-only repeat count that fires a warn. Min 2. Default 3.
    pub warning_threshold: usize,
    /// No-progress streak (args+result) that fires a critical veto. Default 5.
    pub critical_threshold: usize,
    /// Consecutive vetoes that trip the breaker. Default 3.
    pub breaker_veto_streak: usize,
    /// Sliding window size. Default 30.
    pub history_size: usize,
    /// Distinct-args spread that fires a wandering warn. Default 6.
    pub wandering_threshold: usize,
    /// Distinct-args spread that escalates to breaker. Default 12.
    pub wandering_escalation: usize,
}

impl Default for ToolLoopTrackerOptions {
    fn default() -> Self {
        Self {
            warning_threshold: 3,
            critical_threshold: 5,
            breaker_veto_streak: 3,
            history_size: 30,
            wandering_threshold: 6,
            wandering_escalation: 12,
        }
    }
}

/// Two-phase per-turn tool loop tracker.
///
/// Phase 1: `check(tool, args)` — read-only verdict against history-so-far.
/// Phase 2a: `record_call(tool, args)` — commit the call entry.
/// Phase 2b: `record_outcome(tool, args, result_hash)` — patch the result.
#[derive(Debug, Clone)]
pub struct ToolLoopTracker {
    opts: ToolLoopTrackerOptions,
    history: VecDeque<HistoryEntry>,
    consecutive_veto_tool: Option<String>,
    consecutive_veto_count: usize,
}

impl ToolLoopTracker {
    pub fn new(opts: ToolLoopTrackerOptions) -> Self {
        let history_size = opts.history_size;
        Self {
            opts,
            history: VecDeque::with_capacity(history_size.min(64)),
            consecutive_veto_tool: None,
            consecutive_veto_count: 0,
        }
    }

    /// Check a prospective call against history-so-far. Call BEFORE `record_call`.
    pub fn check(&self, tool: &str, args: &serde_json::Value) -> LoopCheckResult {
        let args_hash = hash_tool_call(tool, args);

        // 1. No-progress streak (args+result identical)
        let no_progress = self.get_no_progress_streak(tool, args_hash);
        if no_progress + 1 >= self.opts.critical_threshold {
            return LoopCheckResult {
                level: LoopVerdict::Critical,
                detector: LoopDetector::NoProgress,
                count: no_progress,
            };
        }

        // 2. Wandering detector (distinct args on same tool)
        if is_wandering_prone(tool) {
            let spread = self.effective_spread(tool, args_hash);
            let already_repeats = self.get_repeat_count(tool, args_hash) > 0;
            if spread >= self.opts.wandering_threshold && !already_repeats {
                return LoopCheckResult {
                    level: LoopVerdict::Wandering,
                    detector: LoopDetector::Wandering,
                    count: spread,
                };
            }
        }

        // 3. Args-only repeat
        let repeat_count = self.get_repeat_count(tool, args_hash);
        if repeat_count + 1 >= self.opts.warning_threshold {
            return LoopCheckResult {
                level: LoopVerdict::Warn,
                detector: LoopDetector::GenericRepeat,
                count: repeat_count,
            };
        }

        LoopCheckResult {
            level: LoopVerdict::Ok,
            detector: LoopDetector::GenericRepeat,
            count: 0,
        }
    }

    /// Push a pending history entry. Call after `check()`, before tool execution.
    pub fn record_call(&mut self, tool: &str, args: &serde_json::Value) {
        let args_hash = hash_tool_call(tool, args);
        self.history.push_back(HistoryEntry {
            tool: tool.to_string(),
            args_hash,
            result_hash: None,
            vetoed: false,
        });
        self.trim_history();
    }

    /// Patch the latest pending entry with a result hash. Call after tool execution.
    pub fn record_outcome(&mut self, tool: &str, args: &serde_json::Value, result_hash: Option<u64>) {
        let args_hash = hash_tool_call(tool, args);
        // Find latest pending entry for (tool, args)
        for entry in self.history.iter_mut().rev() {
            if entry.tool == tool && entry.args_hash == args_hash && entry.result_hash.is_none() && !entry.vetoed {
                entry.result_hash = result_hash;
                return;
            }
        }
        // No pending entry: append a finished entry
        self.history.push_back(HistoryEntry {
            tool: tool.to_string(),
            args_hash,
            result_hash,
            vetoed: false,
        });
        self.trim_history();
    }

    /// Record a veto (no-progress loop). The vetoed entry is excluded from the streak.
    pub fn record_veto(&mut self, tool: &str, args: &serde_json::Value) {
        let args_hash = hash_tool_call(tool, args);
        // Patch latest pending as vetoed
        for entry in self.history.iter_mut().rev() {
            if entry.tool == tool && entry.args_hash == args_hash && entry.result_hash.is_none() && !entry.vetoed {
                entry.vetoed = true;
                break;
            }
        }
        // Track consecutive vetoes
        if self.consecutive_veto_tool.as_deref() == Some(tool) {
            self.consecutive_veto_count += 1;
        } else {
            self.consecutive_veto_tool = Some(tool.to_string());
            self.consecutive_veto_count = 1;
        }
    }

    /// Whether the breaker has tripped (too many consecutive vetoes on same tool).
    pub fn is_breaker_tripped(&self, tool: &str) -> bool {
        self.consecutive_veto_tool.as_deref() == Some(tool)
            && self.consecutive_veto_count >= self.opts.breaker_veto_streak
    }

    /// Whether the wandering spread has crossed the escalation threshold.
    pub fn is_wandering_escalated(&self, tool: &str, args: &serde_json::Value) -> bool {
        if !is_wandering_prone(tool) {
            return false;
        }
        let args_hash = hash_tool_call(tool, args);
        self.effective_spread(tool, args_hash) >= self.opts.wandering_escalation
    }

    pub fn clear(&mut self) {
        self.history.clear();
        self.consecutive_veto_tool = None;
        self.consecutive_veto_count = 0;
    }

    // ── Internals ──

    fn trim_history(&mut self) {
        while self.history.len() > self.opts.history_size {
            self.history.pop_front();
        }
    }

    /// Count matching (tool, argsHash) entries in window.
    fn get_repeat_count(&self, tool: &str, args_hash: u64) -> usize {
        self.history.iter()
            .filter(|e| e.tool == tool && e.args_hash == args_hash)
            .count()
    }

    /// Walk backwards counting consecutive (tool, argsHash) entries with same resultHash.
    fn get_no_progress_streak(&self, tool: &str, args_hash: u64) -> usize {
        let mut streak = 0;
        let mut latest_result: Option<u64> = None;
        for entry in self.history.iter().rev() {
            if entry.vetoed || entry.result_hash.is_none() {
                continue;
            }
            if entry.tool != tool || entry.args_hash != args_hash {
                break;
            }
            let rh = entry.result_hash.unwrap();
            if latest_result.is_none() {
                latest_result = Some(rh);
                streak = 1;
                continue;
            }
            if rh != latest_result.unwrap() {
                break;
            }
            streak += 1;
        }
        streak
    }

    /// Distinct completed argsHashes for tool in the window, plus one if current is new.
    fn effective_spread(&self, tool: &str, current_args_hash: u64) -> usize {
        let mut seen = HashSet::new();
        for entry in &self.history {
            if entry.tool != tool {
                continue;
            }
            if entry.result_hash.is_none() || entry.vetoed {
                continue;
            }
            seen.insert(entry.args_hash);
        }
        if seen.contains(&current_args_hash) {
            seen.len()
        } else {
            seen.len() + 1
        }
    }
}

/// Tools whose repeated invocation with changing args is a "wandering" loop.
pub fn is_wandering_prone(tool: &str) -> bool {
    tool == "execute_command"
}

/// Stable signature for a (tool, args) pair.
pub fn hash_tool_call(tool: &str, args: &serde_json::Value) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    tool.hash(&mut hasher);
    args.to_string().hash(&mut hasher);
    hasher.finish()
}

/// Hash a tool result for no-progress comparison.
pub fn hash_tool_result(tool: &str, args: &serde_json::Value, result: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    tool.hash(&mut hasher);
    args.to_string().hash(&mut hasher);
    result.hash(&mut hasher);
    hasher.finish()
}

/// Computes the Levenshtein edit distance between two strings.
pub fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let len_a = a_chars.len();
    let len_b = b_chars.len();

    if len_a == 0 { return len_b; }
    if len_b == 0 { return len_a; }

    let mut dp = vec![vec![0; len_b + 1]; len_a + 1];
    for i in 0..=len_a { dp[i][0] = i; }
    for j in 0..=len_b { dp[0][j] = j; }

    for i in 1..=len_a {
        for j in 1..=len_b {
            let cost = if a_chars[i - 1] == b_chars[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }
    dp[len_a][len_b]
}

/// Canonical expected parameter names for standard tools.
pub fn tool_canonical_params(tool_name: &str) -> &'static [&'static str] {
    match tool_name {
        "read_file" => &["path"],
        "write_file" => &["path", "content"],
        "patch_file" => &["path", "target_content", "replacement_content", "allow_multiple"],
        "run_gates" => &["path"],
        "grep" => &["pattern", "path"],
        "execute_command" => &["command"],
        "spawn_subagent" => &["instruction", "role"],
        "memory_search" => &["query", "limit"],
        "memory_timeline" => &["node_id", "limit"],
        "memory_inspect" => &["node_id"],
        "security_forensic_scan" => &["scan_type", "include_browser"],
        _ => &["arg1", "arg2"],
    }
}

/// Resolves common model parameter aliases into canonical names.
pub fn canonicalize_param_key(tool_name: &str, raw_key: &str) -> String {
    let k = raw_key.trim().to_lowercase();
    match k.as_str() {
        // Path aliases
        "filepath" | "file" | "filename" | "target_file" | "file_path" | "p" => "path".to_string(),
        "dir" | "directory" | "dir_path" | "folder" => {
            if tool_name == "grep" { "path".to_string() } else { "path".to_string() }
        }
        // Content aliases
        "text" | "body" | "code" | "data" | "file_content" | "c" => "content".to_string(),
        // Command aliases
        "cmd" | "exec" | "run" | "bash" | "script" => "command".to_string(),
        // Pattern / Query aliases
        "query" | "regex" | "search" | "term" | "q" => {
            if tool_name == "memory_search" { "query".to_string() } else { "pattern".to_string() }
        }
        // Memory Node ID aliases
        "node_id" | "id" | "node" | "key" | "target_id" => "node_id".to_string(),
        // Target content aliases (for surgical patching)
        "target" | "find" | "search_content" | "match_text" | "old_content" | "old_text" => "target_content".to_string(),
        // Replacement content aliases
        "replacement" | "replace_with" | "new_content" | "new_text" | "replace" => "replacement_content".to_string(),
        // Allow multiple aliases
        "multiple" | "replace_all" | "all" => "allow_multiple".to_string(),
        // Subagent aliases
        "prompt" | "task" | "goal" => "instruction".to_string(),
        "agent_role" | "bot" | "profile" => "role".to_string(),
        _ => raw_key.to_string(),
    }
}

/// Coerces loosely typed or aliased arguments into strict canonical schemas.
pub fn coerce_tool_arguments(tool_name: &str, args: &serde_json::Value) -> serde_json::Value {
    match args {
        serde_json::Value::Object(map) => {
            let mut coerced = serde_json::Map::new();
            for (key, val) in map {
                let canonical_key = canonicalize_param_key(tool_name, key);

                // Type coercion for booleans
                let coerced_val = if canonical_key == "allow_multiple" {
                    match val {
                        serde_json::Value::Bool(b) => serde_json::Value::Bool(*b),
                        serde_json::Value::String(s) => {
                            let s_low = s.trim().to_lowercase();
                            serde_json::Value::Bool(s_low == "true" || s_low == "1" || s_low == "yes")
                        }
                        serde_json::Value::Number(n) => {
                            serde_json::Value::Bool(n.as_i64().map(|v| v != 0).unwrap_or(false))
                        }
                        other => other.clone(),
                    }
                } else if let serde_json::Value::String(s) = val {
                    // If string is an inner JSON object, attempt unwrapping
                    if s.trim_start().starts_with('{') && s.trim_end().ends_with('}') {
                        if let Ok(inner) = serde_json::from_str::<serde_json::Value>(s) {
                            inner
                        } else {
                            val.clone()
                        }
                    } else {
                        val.clone()
                    }
                } else if let serde_json::Value::Number(n) = val {
                    // If a number was passed for path or pattern, convert to string
                    serde_json::Value::String(n.to_string())
                } else {
                    val.clone()
                };

                coerced.insert(canonical_key.clone(), coerced_val.clone());
                if canonical_key != *key {
                    coerced.insert(key.clone(), coerced_val);
                }
            }
            serde_json::Value::Object(coerced)
        }
        serde_json::Value::String(s) => {
            let mut trimmed = s.trim().to_string();
            // Tiny Model repair: Fix markdown JSON blocks or missing trailing braces emitted by 0.5B-1.5B models
            if trimmed.starts_with("```json") {
                trimmed = trimmed.trim_start_matches("```json").trim_end_matches("```").trim().to_string();
            } else if trimmed.starts_with("```") {
                trimmed = trimmed.trim_start_matches("```").trim_end_matches("```").trim().to_string();
            }
            if trimmed.starts_with('{') && !trimmed.ends_with('}') {
                trimmed.push('}');
            }
            if trimmed.starts_with('{') && trimmed.ends_with('}') {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&trimmed) {
                    return coerce_tool_arguments(tool_name, &val);
                }
            }
            // Direct single string parameter mapping
            let mut map = serde_json::Map::new();
            match tool_name {
                "read_file" | "run_gates" => {
                    map.insert("path".to_string(), serde_json::Value::String(s.clone()));
                }
                "execute_command" => {
                    map.insert("command".to_string(), serde_json::Value::String(s.clone()));
                }
                "grep" => {
                    map.insert("pattern".to_string(), serde_json::Value::String(s.clone()));
                }
                "memory_search" => {
                    map.insert("query".to_string(), serde_json::Value::String(s.clone()));
                }
                "memory_timeline" | "memory_inspect" => {
                    map.insert("node_id".to_string(), serde_json::Value::String(s.clone()));
                }
                "security_forensic_scan" => {
                    map.insert("scan_type".to_string(), serde_json::Value::String(s.clone()));
                }
                _ => {
                    map.insert("arg1".to_string(), serde_json::Value::String(s.clone()));
                }
            }
            serde_json::Value::Object(map)
        }
        other => other.clone(),
    }
}

/// Computes an actionable error message with Levenshtein nearest-key hints
/// if unknown arguments are encountered.
pub fn describe_argument_error(tool_name: &str, unknown_key: &str) -> Option<String> {
    let canonical = tool_canonical_params(tool_name);
    let mut best_match: Option<(&'static str, usize)> = None;

    for &param in canonical {
        let dist = levenshtein_distance(unknown_key, param);
        if dist <= 2 {
            if let Some((_, best_dist)) = best_match {
                if dist < best_dist {
                    best_match = Some((param, dist));
                }
            } else {
                best_match = Some((param, dist));
            }
        }
    }

    best_match.map(|(suggested, _)| {
        format!("Unknown argument '{}' for tool '{}'. Did you mean '{}'?", unknown_key, tool_name, suggested)
    })
}

/// Canonical unpacked arguments for file and execution tools.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CanonicalArgs {
    ReadFile { path: String },
    WriteFile { path: String, content: String },
    PatchFile { path: String, target_content: String, replacement_content: String, allow_multiple: bool },
    RunGates { path: Option<String> },
    Grep { pattern: String, path: Option<String> },
    ExecuteCommand { command: String },
    SpawnSubagent { instruction: String, role: Option<String> },
    MemorySearch { query: String, limit: Option<usize> },
    MemoryTimeline { node_id: String, limit: Option<usize> },
    MemoryInspect { node_id: String },
    Generic { arg1: String, arg2: Option<String> },
}

/// Standard parsed tool invocation structure.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

impl ToolCall {
    pub fn unpack_canonical(&self) -> CanonicalArgs {
        let args = &self.arguments;
        match self.name.as_str() {
            "read_file" => {
                let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("").to_string();
                CanonicalArgs::ReadFile { path }
            }
            "write_file" => {
                let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string();
                CanonicalArgs::WriteFile { path, content }
            }
            "patch_file" => {
                let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let target_content = args.get("target_content").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let replacement_content = args.get("replacement_content").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let allow_multiple = args.get("allow_multiple").and_then(|v| v.as_bool()).unwrap_or(false);
                CanonicalArgs::PatchFile { path, target_content, replacement_content, allow_multiple }
            }
            "run_gates" => {
                let path = args.get("path").and_then(|v| v.as_str()).map(|s| s.to_string());
                CanonicalArgs::RunGates { path }
            }
            "grep" => {
                let pattern = args.get("pattern").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let path = args.get("path").or_else(|| args.get("dir")).and_then(|v| v.as_str()).map(|s| s.to_string());
                CanonicalArgs::Grep { pattern, path }
            }
            "execute_command" => {
                let command = args.get("command").and_then(|v| v.as_str()).unwrap_or("").to_string();
                CanonicalArgs::ExecuteCommand { command }
            }
            "spawn_subagent" => {
                let instruction = args.get("instruction").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let role = args.get("role").and_then(|v| v.as_str()).map(|s| s.to_string());
                CanonicalArgs::SpawnSubagent { instruction, role }
            }
            "memory_search" => {
                let query = args.get("query").or_else(|| args.get("q")).and_then(|v| v.as_str()).unwrap_or("").to_string();
                let limit = args.get("limit").and_then(|v| v.as_u64()).map(|u| u as usize);
                CanonicalArgs::MemorySearch { query, limit }
            }
            "memory_timeline" => {
                let node_id = args.get("node_id").or_else(|| args.get("id")).and_then(|v| v.as_str()).unwrap_or("").to_string();
                let limit = args.get("limit").and_then(|v| v.as_u64()).map(|u| u as usize);
                CanonicalArgs::MemoryTimeline { node_id, limit }
            }
            "memory_inspect" => {
                let node_id = args.get("node_id").or_else(|| args.get("id")).and_then(|v| v.as_str()).unwrap_or("").to_string();
                CanonicalArgs::MemoryInspect { node_id }
            }
            _ => {
                let a1 = args.get("arg1").or_else(|| args.get("path")).or_else(|| args.get("command")).and_then(|v| v.as_str()).unwrap_or("").to_string();
                let a2 = args.get("arg2").or_else(|| args.get("content")).and_then(|v| v.as_str()).map(|s| s.to_string());
                CanonicalArgs::Generic { arg1: a1, arg2: a2 }
            }
        }
    }

    pub fn to_arg_pair(&self) -> (String, Option<String>) {
        match self.unpack_canonical() {
            CanonicalArgs::ReadFile { path } => (path, None),
            CanonicalArgs::WriteFile { path, content } => (path, Some(content)),
            CanonicalArgs::PatchFile { path, target_content, replacement_content, allow_multiple } => {
                let payload = serde_json::json!({
                    "target_content": target_content,
                    "replacement_content": replacement_content,
                    "allow_multiple": allow_multiple,
                }).to_string();
                (path, Some(payload))
            }
            CanonicalArgs::RunGates { path } => (path.unwrap_or_else(|| "GATES.md".to_string()), None),
            CanonicalArgs::Grep { pattern, path } => (pattern, path),
            CanonicalArgs::ExecuteCommand { command } => (command, None),
            CanonicalArgs::SpawnSubagent { instruction, role } => (instruction, role),
            CanonicalArgs::MemorySearch { query, limit } => (query, limit.map(|l| l.to_string())),
            CanonicalArgs::MemoryTimeline { node_id, limit } => (node_id, limit.map(|l| l.to_string())),
            CanonicalArgs::MemoryInspect { node_id } => (node_id, None),
            CanonicalArgs::Generic { arg1, arg2 } => (arg1, arg2),
        }
    }
}

/// Helper to parse a single JSON value into a ToolCall
fn parse_single_tool_call(v: &serde_json::Value) -> Option<ToolCall> {
    let name = v.get("name")
        .or_else(|| v.get("tool"))
        .or_else(|| v.get("function").and_then(|f| f.get("name")))
        .and_then(|s| s.as_str())?;

    let arguments = v.get("arguments")
        .or_else(|| v.get("args"))
        .or_else(|| v.get("parameters"))
        .or_else(|| v.get("function").and_then(|f| f.get("arguments")))
        .cloned()
        .unwrap_or_else(|| {
            // Flattened arguments fallback: gather non-metadata keys
            let mut map = serde_json::Map::new();
            if let Some(obj) = v.as_object() {
                for (k, val) in obj {
                    if k != "name" && k != "tool" && k != "type" && k != "function" {
                        map.insert(k.clone(), val.clone());
                    }
                }
            }
            serde_json::Value::Object(map)
        });

    let parsed_args = if let Some(s) = arguments.as_str() {
        serde_json::from_str(s).unwrap_or(arguments)
    } else {
        arguments
    };

    let coerced_args = coerce_tool_arguments(name, &parsed_args);

    Some(ToolCall {
        name: name.to_string(),
        arguments: coerced_args,
    })
}

/// Isolates and strips reasoning scratchpad blocks (<thinking>...</thinking>, <thought>...</thought>,
/// or untagged monologue preambles like "The user is... Let me check...") from raw model outputs.
pub fn isolate_thinking_scratchpad(output: &str) -> (String, Option<String>) {
    let mut cleaned = output.to_string();
    let mut thinking_parts = Vec::new();

    let patterns = [
        ("<think>", "</think>"),
        ("<thinking>", "</thinking>"),
        ("<thought>", "</thought>"),
    ];

    for (start_tag, end_tag) in patterns {
        while let Some(s_idx) = cleaned.find(start_tag) {
            let content_start = s_idx + start_tag.len();
            if let Some(e_idx) = cleaned[content_start..].find(end_tag) {
                let actual_end = content_start + e_idx;
                let thought_text = cleaned[content_start..actual_end].trim().to_string();
                if !thought_text.is_empty() {
                    thinking_parts.push(thought_text);
                }
                cleaned = format!("{}{}", &cleaned[..s_idx], &cleaned[actual_end + end_tag.len()..]);
            } else {
                break;
            }
        }
    }

    // Detect leading untagged Chain-of-Thought monologue (e.g. "The user is...", "Let me check...")
    let trimmed = cleaned.trim();
    let monologue_prefixes = [
        "The user is",
        "The user wants",
        "The user asks",
        "The user requested",
        "Let me check",
        "Let me see",
        "Thinking Process:",
        "Thought:",
        "I will now",
        "I should lead with",
        "First, I need to",
        "First, I will",
    ];
    let starts_with_monologue = monologue_prefixes.iter().any(|&p| trimmed.starts_with(p));

    if starts_with_monologue {
        if let Some(tool_start) = trimmed.find("```json").or_else(|| trimmed.find('{')) {
            let monologue = trimmed[..tool_start].trim();
            if !monologue.is_empty() {
                thinking_parts.push(monologue.to_string());
                cleaned = trimmed[tool_start..].to_string();
            }
        } else if let Some(sep) = trimmed.find("\n\n") {
            let monologue = trimmed[..sep].trim();
            let remainder = trimmed[sep + 2..].trim();
            if !monologue.is_empty() && !remainder.is_empty() {
                thinking_parts.push(monologue.to_string());
                cleaned = remainder.to_string();
            }
        } else {
            // Entire output was internal monologue with no subsequent response
            thinking_parts.push(trimmed.to_string());
            cleaned = "I am ready to help. What would you like to work on?".to_string();
        }
    }

    let final_thinking = if thinking_parts.is_empty() {
        None
    } else {
        Some(thinking_parts.join("\n\n"))
    };

    (cleaned, final_thinking)
}

/// Validates and extracts GBNF tool calls from raw model output, supporting code blocks, XML tags, and raw JSON objects.
pub fn validate_gbnf_tool_calls(output: &str) -> Result<Vec<ToolCall>, String> {
    let (clean_output, _) = isolate_thinking_scratchpad(output);
    let trimmed = clean_output.trim();

    // 1. Direct parse of whole trimmed text (if it's already a valid JSON object or array)
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed) {
        if let Some(arr) = val.as_array() {
            let calls: Vec<ToolCall> = arr.iter().filter_map(parse_single_tool_call).collect();
            if !calls.is_empty() {
                return Ok(calls);
            }
        } else if let Some(call) = parse_single_tool_call(&val) {
            return Ok(vec![call]);
        }
    }

    // 2. Scan for Markdown ```json ... ``` or ``` ... ``` code fences
    let mut search_pos = 0;
    while let Some(start_fence) = output[search_pos..].find("```") {
        let actual_start = search_pos + start_fence;
        let content_start = if output[actual_start..].starts_with("```json") {
            actual_start + 7
        } else {
            actual_start + 3
        };

        if let Some(end_fence) = output[content_start..].find("```") {
            let chunk = output[content_start..content_start + end_fence].trim();
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(chunk) {
                if let Some(arr) = val.as_array() {
                    let calls: Vec<ToolCall> = arr.iter().filter_map(parse_single_tool_call).collect();
                    if !calls.is_empty() {
                        return Ok(calls);
                    }
                } else if let Some(call) = parse_single_tool_call(&val) {
                    return Ok(vec![call]);
                }
            }
            search_pos = content_start + end_fence + 3;
        } else {
            break;
        }
    }

    // 3. Scan for <tool_call>...</tool_call> tags
    if let Some(start_tag) = output.find("<tool_call>") {
        let content_start = start_tag + 11;
        if let Some(end_tag) = output[content_start..].find("</tool_call>") {
            let chunk = output[content_start..content_start + end_tag].trim();
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(chunk) {
                if let Some(arr) = val.as_array() {
                    let calls: Vec<ToolCall> = arr.iter().filter_map(parse_single_tool_call).collect();
                    if !calls.is_empty() {
                        return Ok(calls);
                    }
                } else if let Some(call) = parse_single_tool_call(&val) {
                    return Ok(vec![call]);
                }
            }
        }
    }

    // 4. Scan for standalone raw JSON objects containing "name" or "tool"
    let mut search_idx = 0;
    while let Some(open_brace) = output[search_idx..].find('{') {
        let abs_start = search_idx + open_brace;
        if let Some(close_brace) = output[abs_start..].rfind('}') {
            let candidate = &output[abs_start..=abs_start + close_brace];
            if candidate.contains("\"name\"") || candidate.contains("\"tool\"") || candidate.contains("\"function\"") {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(candidate) {
                    if let Some(arr) = val.as_array() {
                        let calls: Vec<ToolCall> = arr.iter().filter_map(parse_single_tool_call).collect();
                        if !calls.is_empty() {
                            return Ok(calls);
                        }
                    } else if let Some(call) = parse_single_tool_call(&val) {
                        return Ok(vec![call]);
                    }
                }
            }
        }
        search_idx = abs_start + 1;
        if search_idx >= output.len() {
            break;
        }
    }

    Err("No valid GBNF tool call detected".to_string())
}

/// Validates a raw model output string against GBNF tool call grammar and returns the first tool call.
pub fn validate_gbnf_tool_call(output: &str) -> Result<ToolCall, String> {
    let calls = validate_gbnf_tool_calls(output)?;
    calls.into_iter().next().ok_or_else(|| "No tool calls found".to_string())
}

/// Formats a context prompt with a stable invariant prefix to maximize KV-cache hits in offline engines.
pub fn format_stable_kv_prompt(system_prompt: &str, tools_schema: &str, conversation_history: &str) -> String {
    format!(
        "=== CYNAPSE SYSTEM PRESET ===\n\
        {}\n\n\
        === AVAILABLE TOOL SCHEMAS ===\n\
        {}\n\n\
        === CONVERSATION LOG ===\n\
        {}",
        system_prompt.trim(),
        tools_schema.trim(),
        conversation_history.trim()
    )
}

/// Builds a clean conversational system prompt without tool schemas or JSON invocation grammar
/// for casual greetings and general dialogue.
pub fn compile_conversational_prefix(_system_persona: &str) -> String {
    "You are Cynapse, an intelligent, concise, and friendly local companion. Respond naturally and directly to the user.".to_string()
}

/// Dynamically prunes the available tools down to the tools relevant to the active user prompt.
/// If the query is conversational or a greeting, returns an empty list to prevent
/// small models from hallucinating or executing unwanted shell commands.
pub fn prune_tools_for_query(user_query: &str) -> Vec<crate::ToolDefinition> {
    let lower = user_query.to_lowercase();
    let words: Vec<&str> = lower.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-').filter(|w| !w.is_empty()).collect();

    // Pure greetings, conversational questions, and updates inquiries: ZERO tools exposed
    let is_greeting = ["hi", "hey", "hello", "yo", "sup", "howdy", "hiya", "good morning", "good afternoon", "good evening", "greetings"]
        .iter().any(|&g| lower.starts_with(g));
    let has_explicit_cmd = words.iter().any(|&w| w == "run" || w == "exec" || w == "execute" || w == "grep" || w == "patch" || w == "write")
        || lower.contains("git log") || lower.contains("git diff") || lower.contains("git status")
        || lower.contains("cargo test") || lower.contains("cargo check") || lower.contains("cargo build")
        || lower.contains("read file") || lower.contains("view file") || lower.contains("open file");

    if is_greeting && !has_explicit_cmd {
        return Vec::new();
    }

    let all_tools = crate::list_tools();
    let mut selected = Vec::new();

    let asks_file_read = (words.iter().any(|&w| w == "read" || w == "view" || w == "inspect" || w == "cat" || w == "open"))
        && (lower.contains("file") || lower.contains(".rs") || lower.contains(".toml") || lower.contains(".md") || lower.contains(".json") || lower.contains("/"));
    let asks_file_edit = (words.iter().any(|&w| w == "write" || w == "patch" || w == "modify" || w == "edit" || w == "replace" || w == "append"))
        && (lower.contains("file") || lower.contains(".rs") || lower.contains(".toml") || lower.contains(".md") || lower.contains("code"));
    let asks_search = words.iter().any(|&w| w == "grep" || w == "search" || w == "locate") || lower.contains("search for");
    let asks_exec = has_explicit_cmd;
    let asks_memory = words.iter().any(|&w| w == "memory" || w == "remember" || w == "recall" || w == "timeline")
        || lower.contains("in memory") || lower.contains("search memory") || lower.contains("memory search");
    let asks_security = words.iter().any(|&w| w == "security" || w == "forensic" || w == "threat" || w == "spyware" || w == "malware" || w == "breach" || w == "candiru" || w == "adfection")
        || lower.contains("security scan") || lower.contains("forensic scan") || lower.contains("check threat");

    for t in all_tools {
        match t.name {
            "read_file" if asks_file_read || asks_file_edit || asks_search => selected.push(t),
            "patch_file" | "write_file" if asks_file_edit => selected.push(t),
            "grep" if asks_search => selected.push(t),
            "execute_command" if asks_exec => selected.push(t),
            "memory_search" | "memory_timeline" | "memory_inspect" if asks_memory => selected.push(t),
            "security_forensic_scan" if asks_security => selected.push(t),
            "run_gates" if lower.contains("gate") || lower.contains("check gates") || lower.contains("run gates") => selected.push(t),
            _ => {}
        }
    }

    if selected.is_empty() {
        Vec::new()
    } else {
        selected.truncate(4);
        selected
    }
}

/// Returns a clean 1-shot exemplar for the given tool to enforce strict syntax on weak models.
pub fn get_tool_exemplar(tool_name: &str) -> &'static str {
    match tool_name {
        "read_file" => "```json\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"src/main.rs\"}}\n```",
        "patch_file" => "```json\n{\"name\": \"patch_file\", \"arguments\": {\"path\": \"src/main.rs\", \"target_content\": \"old_line()\", \"replacement_content\": \"new_line()\", \"allow_multiple\": false}}\n```",
        "write_file" => "```json\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"notes.txt\", \"content\": \"hello world\"}}\n```",
        "execute_command" => "```json\n{\"name\": \"execute_command\", \"arguments\": {\"command\": \"cargo check\"}}\n```",
        "grep" => "```json\n{\"name\": \"grep\", \"arguments\": {\"pattern\": \"fn main\"}}\n```",
        "memory_search" => "```json\n{\"name\": \"memory_search\", \"arguments\": {\"query\": \"auth tokens\"}}\n```",
        "memory_timeline" => "```json\n{\"name\": \"memory_timeline\", \"arguments\": {\"node_id\": \"auth_rule\"}}\n```",
        "memory_inspect" => "```json\n{\"name\": \"memory_inspect\", \"arguments\": {\"node_id\": \"auth_rule\"}}\n```",
        "security_forensic_scan" => "```json\n{\"name\": \"security_forensic_scan\", \"arguments\": {\"scan_type\": \"quick\"}}\n```",
        _ => "```json\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"Cargo.toml\"}}\n```",
    }
}

/// System 2 Schema Validator Critic: validates tool arguments before execution.
pub fn validate_tool_call_schema(call: &ToolCall, workspace_dir: &std::path::Path) -> Result<(), String> {
    match call.name.as_str() {
        "read_file" => {
            let path_str = call.arguments.get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "Missing required parameter 'path' for read_file".to_string())?;
            let p = workspace_dir.join(path_str.trim_start_matches('/'));
            if !p.exists() {
                return Err(format!("File '{}' does not exist in workspace. Verify path before reading.", path_str));
            }
        }
        "patch_file" => {
            let path_str = call.arguments.get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "Missing required parameter 'path' for patch_file".to_string())?;
            let target = call.arguments.get("target_content")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "Missing required parameter 'target_content' for patch_file".to_string())?;
            let _replacement = call.arguments.get("replacement_content")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "Missing required parameter 'replacement_content' for patch_file".to_string())?;
            let p = workspace_dir.join(path_str.trim_start_matches('/'));
            if !p.exists() {
                return Err(format!("Target file '{}' does not exist in workspace.", path_str));
            }
            if let Ok(content) = std::fs::read_to_string(&p) {
                if !content.contains(target) {
                    return Err(format!("target_content not found in '{}'. Ensure target matches exact lines in the file.", path_str));
                }
            }
        }
        "write_file" => {
            let _path = call.arguments.get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "Missing required parameter 'path' for write_file".to_string())?;
            let _content = call.arguments.get("content")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "Missing required parameter 'content' for write_file".to_string())?;
        }
        "execute_command" => {
            let cmd = call.arguments.get("command")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "Missing required parameter 'command' for execute_command".to_string())?;
            if cmd.trim().is_empty() {
                return Err("Empty command passed to execute_command".to_string());
            }
        }
        "memory_search" => {
            let q = call.arguments.get("query")
                .or_else(|| call.arguments.get("q"))
                .and_then(|v| v.as_str())
                .ok_or_else(|| "Missing required parameter 'query' for memory_search".to_string())?;
            if q.trim().is_empty() {
                return Err("Empty query passed to memory_search".to_string());
            }
        }
        "memory_timeline" | "memory_inspect" => {
            let id = call.arguments.get("node_id")
                .or_else(|| call.arguments.get("id"))
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("Missing required parameter 'node_id' for {}", call.name))?;
            if id.trim().is_empty() {
                return Err(format!("Empty node_id passed to {}", call.name));
            }
        }
        _ => {}
    }
    Ok(())
}

/// Builds dynamic Zone A prompt tailored to the subset of active tools.
pub fn compile_zone_a_prefix_with_tools(system_persona: &str, tools: &[crate::ToolDefinition]) -> String {
    if tools.is_empty() {
        return format!(
            "You are Cynapse, an autonomous local intelligent assistant.\n\
            {}\n\n\
            Operating Directives:\n\
            1. Lead with the direct answer on line 1.\n\
            2. Number multi-step tasks clearly. Cap lists at maximum 5 items.\n\
            3. End with exactly one concrete next action doable immediately.\n\
            4. State cause and fix directly for errors. Be concise and eliminate conversational filler.\n\
            5. Output ONLY your direct answer to the user. Never output internal monologue or planning prose ('The user wants...', 'Let me check...').\n\
            6. For this query, no tools are required. Answer directly in clean GitHub markdown.",
            system_persona.trim()
        );
    }

    let mut tools_desc = String::new();
    for t in tools {
        tools_desc.push_str(&format!("- `{}`: {}\n", t.name, t.description));
    }

    let exemplar = if let Some(first) = tools.first() {
        format!("\nExemplar format:\n{}\n", get_tool_exemplar(first.name))
    } else {
        String::new()
    };

    format!(
        "You are Cynapse, an autonomous local intelligent assistant.\n\
        {}\n\n\
        Operating Directives:\n\
        1. Lead with the direct answer, command, or code snippet on line 1.\n\
        2. Number multi-step tasks clearly. Cap lists at maximum 5 items.\n\
        3. End with exactly one concrete next action doable immediately.\n\
        4. State cause and fix directly for errors. Be concise and eliminate conversational filler.\n\
        5. Output ONLY your direct answer or tool calls. Never output internal monologue or planning prose ('The user is...', 'Let me check...').\n\
        6. When asked to write, create, edit, or read a file, search code, or run a shell command:\n\
           Immediately emit the tool call in a ```json code block. Do not just talk about doing it — execute it:\n\
           ```json\n\
           {{\"name\": \"<tool_name>\", \"arguments\": {{\"path\": \"...\", \"content\": \"...\"}}}}\n\
           ```\n\
           {}\
        Available Tools:\n\
        {}",
        system_persona.trim(),
        exemplar,
        tools_desc.trim()
    )
}

/// Zone A (100% byte-stable prefix): Invariant persona, behavior directives, and tool definitions.
/// Pinned in KV cache to enable 0 ms prefill latency across multi-turn sessions.
pub fn compile_zone_a_prefix(system_persona: &str) -> String {
    compile_zone_a_prefix_with_tools(system_persona, &crate::list_tools())
}

/// Zone B (Variable tail): Recalled Dendrite knowledge nodes (deterministically sorted),
/// sliding conversation history (with compressed tool results), optional operator/loop notice, and active user instruction.
pub fn compile_zone_b_tail(
    recalled_knowledge: &str,
    conversation_history: &str,
    user_query: &str,
    notice: Option<&str>,
) -> String {
    let mut tail = String::new();

    let dynamic_directive = crate::skills::format_dynamic_directive(user_query);
    tail.push_str(&dynamic_directive);

    if !recalled_knowledge.trim().is_empty() {
        if recalled_knowledge.trim().starts_with("===") {
            tail.push_str(recalled_knowledge.trim());
        } else {
            tail.push_str("=== RECALLED KNOWLEDGE ===\n");
            tail.push_str(recalled_knowledge.trim());
        }
        tail.push_str("\n\n");
    }

    if !conversation_history.trim().is_empty() {
        tail.push_str("=== CONVERSATION LOG ===\n");
        tail.push_str(conversation_history.trim());
        tail.push_str("\n\n");
    }

    if let Some(n) = notice {
        if !n.trim().is_empty() {
            tail.push_str("=== NOTICE ===\n");
            tail.push_str(n.trim());
            tail.push_str("\n\n");
        }
    }

    tail.push_str("=== USER INSTRUCTION ===\n");
    tail.push_str(user_query.trim());
    tail
}

/// Two-phase tool loop guard wrapping ToolLoopTracker.
/// Detects repeating invocations, identical result no-progress streaks, and wandering loops.
#[derive(Debug, Clone)]
pub struct LoopGuard {
    tracker: ToolLoopTracker,
}

impl Default for LoopGuard {
    fn default() -> Self {
        Self::new(ToolLoopTrackerOptions::default())
    }
}

impl LoopGuard {
    pub fn new(opts: ToolLoopTrackerOptions) -> Self {
        Self {
            tracker: ToolLoopTracker::new(opts),
        }
    }

    /// Check if a prospective tool call triggers a loop intervention.
    pub fn check(&self, tool: &ToolCall) -> Result<(), String> {
        if self.tracker.is_breaker_tripped(&tool.name) {
            return Err(format!(
                "LOOP BREAKER TRIPPED: Tool '{}' has been repeatedly vetoed without progress. Circuit broken — please report outcome directly to the user.",
                tool.name
            ));
        }

        let res = self.tracker.check(&tool.name, &tool.arguments);
        match res.level {
            LoopVerdict::Critical => Err(format!(
                "LOOP GUARD CRITICAL (No-Progress): Tool '{}' executed {} consecutive times with identical output. Please change approach or state result.",
                tool.name, res.count
            )),
            LoopVerdict::Wandering => {
                if self.tracker.is_wandering_escalated(&tool.name, &tool.arguments) {
                    Err(format!(
                        "LOOP GUARD BREAKER (Wandering): Tool '{}' executed {} distinct exploratory steps. Please summarize findings and reply.",
                        tool.name, res.count
                    ))
                } else {
                    Ok(())
                }
            }
            LoopVerdict::Warn => {
                // Advisory notice: do not veto execution (only Critical vetoes)
                Ok(())
            }
            LoopVerdict::Ok => Ok(()),
        }
    }

    /// Generates a structured advisory notice for small models when repetition or wandering is detected,
    /// nudging the model toward alternative approaches before a hard circuit breaker trips.
    pub fn advisory_notice(&self, tool: &ToolCall) -> Option<String> {
        let res = self.tracker.check(&tool.name, &tool.arguments);
        match res.level {
            LoopVerdict::Warn => {
                let preview = serde_json::to_string(&tool.arguments).unwrap_or_default();
                let short_preview = if preview.len() > 60 {
                    format!("{}...", &preview[..60])
                } else {
                    preview
                };
                Some(format!(
                    "=== ADVISORY STEERING ===\n\
                    You called tool '{}' repeatedly with identical arguments: {}.\n\
                    Inspect previous output carefully. If it did not advance the task, switch to an alternative tool or report results directly.",
                    tool.name, short_preview
                ))
            }
            LoopVerdict::Wandering => {
                Some(format!(
                    "=== ADVISORY STEERING ===\n\
                    You executed {} distinct exploratory steps with '{}'.\n\
                    Synthesize your findings and proceed to implementation rather than continued scanning.",
                    res.count, tool.name
                ))
            }
            _ => None,
        }
    }

    /// Combined check returning Ok(Some(advisory)) or Err(veto).
    pub fn check_with_advisory(&self, tool: &ToolCall) -> Result<Option<String>, String> {
        self.check(tool)?;
        Ok(self.advisory_notice(tool))
    }

    /// Record prospective tool call invocation before execution.
    pub fn record_call(&mut self, tool: &ToolCall) {
        self.tracker.record_call(&tool.name, &tool.arguments);
    }

    /// Record tool execution outcome.
    pub fn record_outcome(&mut self, tool: &ToolCall, output: &str) {
        let res_hash = hash_tool_result(&tool.name, &tool.arguments, output);
        self.tracker.record_outcome(&tool.name, &tool.arguments, Some(res_hash));
    }

    /// Record veto when loop guard trips.
    pub fn record_veto(&mut self, tool: &ToolCall) {
        self.tracker.record_veto(&tool.name, &tool.arguments);
    }

    /// Convenience single-step record and check for compatibility.
    pub fn record_and_check(&mut self, tool: &ToolCall) -> Result<(), String> {
        self.check(tool)?;
        self.record_call(tool);
        Ok(())
    }

    pub fn clear(&mut self) {
        self.tracker.clear();
    }
}

/// Step-Back Prompting Directive: prepends a structural constraint reflection
/// instruction to force small edge models to examine prerequisites before emitting tool actions.
pub fn format_step_back_prompt(query: &str) -> String {
    format!(
        "System 2 Reflection: Before selecting any tools, review: \
        (1) Is this an investigatory question or modifying action? \
        (2) Do required paths/dependencies exist? \
        (3) What is the minimal sufficient tool action?\n\
        User Query: {}",
        query.trim()
    )
}

/// Hallucination Guardrail: Validates that a requested file path exists before execution.
pub fn verify_path_exists(path_str: &str) -> Result<std::path::PathBuf, String> {
    let p = std::path::PathBuf::from(path_str);
    if p.exists() {
        Ok(p)
    } else {
        Err(format!("Hallucination Guard: Path does not exist on filesystem: '{}'", path_str))
    }
}

/// Hallucination Guardrail: Validates that the primary binary in a command line exists.
pub fn verify_command_binary_exists(cmd_str: &str) -> Result<String, String> {
    let binary = cmd_str.split_whitespace().next().unwrap_or("").trim();
    if binary.is_empty() {
        return Err("Empty command".to_string());
    }
    // Whitelisted common shell builtins
    if ["cd", "echo", "pwd", "true", "false", "test", "export"].contains(&binary) {
        return Ok(binary.to_string());
    }
    if std::path::Path::new(binary).is_file() {
        return Ok(binary.to_string());
    }
    if let Some(paths) = std::env::var_os("PATH") {
        for p in std::env::split_paths(&paths) {
            let candidate = p.join(binary);
            if candidate.is_file() {
                return Ok(binary.to_string());
            }
        }
    }
    Err(format!("Hallucination Guard: Executable binary '{}' not found in PATH", binary))
}

/// Self-Consistency Majority Voting: Takes multiple candidate tool calls sampled at T > 0,
/// grouping by tool name and canonical path/command, and returns the highest consensus call.
pub fn self_consistency_majority_vote(candidates: &[ToolCall]) -> Option<ToolCall> {
    if candidates.is_empty() {
        return None;
    }
    if candidates.len() == 1 {
        return Some(candidates[0].clone());
    }

    let mut votes: std::collections::HashMap<(String, String), (usize, ToolCall)> = std::collections::HashMap::new();
    for call in candidates {
        let key = (
            call.name.clone(),
            call.arguments.get("path")
                .or_else(|| call.arguments.get("cmd"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
        );
        let entry = votes.entry(key).or_insert((0, call.clone()));
        entry.0 += 1;
    }

    votes.into_values()
        .max_by_key(|(count, _)| *count)
        .map(|(_, call)| call)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gbnf_tool_validation() {
        let valid_json = r#"{"name": "read_file", "arguments": {"path": "src/main.rs"}}"#;
        let parsed = validate_gbnf_tool_call(valid_json).unwrap();
        assert_eq!(parsed.name, "read_file");
        assert_eq!(parsed.arguments["path"], "src/main.rs");

        // Codeblock embedded within surrounding conversational text
        let codeblock_json = "I'll read that file for you:\n```json\n{\"tool\": \"grep_search\", \"args\": {\"query\": \"fn main\"}}\n```\nLet me know if you need more.";
        let parsed_cb = validate_gbnf_tool_call(codeblock_json).unwrap();
        assert_eq!(parsed_cb.name, "grep_search");
        assert_eq!(parsed_cb.arguments["query"], "fn main");

        // XML-style tool_call tag with flattened arguments
        let tag_json = "Sure, creating the file now:\n<tool_call>\n{\"name\": \"write_file\", \"path\": \"about.md\", \"content\": \"# About Me\"}\n</tool_call>";
        let parsed_tag = validate_gbnf_tool_call(tag_json).unwrap();
        assert_eq!(parsed_tag.name, "write_file");
        assert_eq!(parsed_tag.arguments["path"], "about.md");
        assert_eq!(parsed_tag.arguments["content"], "# About Me");

        let batch_json = r#"[{"tool": "read_file", "args": {"path": "a.txt"}}, {"tool": "grep_search", "args": {"query": "foo"}}]"#;
        let batch_parsed = validate_gbnf_tool_calls(batch_json).unwrap();
        assert_eq!(batch_parsed.len(), 2);
        assert_eq!(batch_parsed[0].name, "read_file");
        assert_eq!(batch_parsed[1].name, "grep_search");
    }

    #[test]
    fn test_batch_tool_calls_in_codeblocks() {
        let text = "First, let's read the file:\n```json\n[{\"name\": \"read_file\", \"path\": \"Cargo.toml\"}, {\"name\": \"list_dir\", \"path\": \".\"}]\n```\nDone.";
        let calls = validate_gbnf_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments["path"], "Cargo.toml");
        assert_eq!(calls[1].name, "list_dir");
        assert_eq!(calls[1].arguments["path"], ".");
    }

    #[test]
    fn test_loop_guard() {
        let mut guard = LoopGuard::new(ToolLoopTrackerOptions {
            warning_threshold: 3,
            critical_threshold: 3,
            ..Default::default()
        });
        let call = ToolCall {
            name: "read_file".into(),
            arguments: serde_json::json!({"path": "foo.rs"}),
        };

        for _ in 0..2 {
            assert!(guard.check(&call).is_ok());
            guard.record_call(&call);
            guard.record_outcome(&call, "content");
        }
        let res = guard.check(&call);
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("LOOP GUARD CRITICAL"));
    }

    #[test]
    fn test_tool_loop_tracker_args_only_repeat() {
        let mut tracker = ToolLoopTracker::new(ToolLoopTrackerOptions {
            warning_threshold: 3,
            ..Default::default()
        });
        let args = serde_json::json!({"path": "foo.rs"});

        // First two calls are ok
        let v = tracker.check("read_file", &args);
        assert_eq!(v.level, LoopVerdict::Ok);
        tracker.record_call("read_file", &args);
        tracker.record_outcome("read_file", &args, Some(100));

        let v = tracker.check("read_file", &args);
        assert_eq!(v.level, LoopVerdict::Ok);
        tracker.record_call("read_file", &args);
        tracker.record_outcome("read_file", &args, Some(200));

        // Third call triggers warn
        let v = tracker.check("read_file", &args);
        assert_eq!(v.level, LoopVerdict::Warn);
        assert_eq!(v.count, 2);
    }

    #[test]
    fn test_tool_loop_tracker_no_progress_streak() {
        let mut tracker = ToolLoopTracker::new(ToolLoopTrackerOptions {
            warning_threshold: 10,
            critical_threshold: 3,
            ..Default::default()
        });
        let args = serde_json::json!({"cmd": "cargo test"});

        // Same args + same result = no progress
        for _ in 0..2 {
            let v = tracker.check("execute_command", &args);
            assert_eq!(v.level, LoopVerdict::Ok);
            tracker.record_call("execute_command", &args);
            tracker.record_outcome("execute_command", &args, Some(42));
        }

        // Third triggers critical
        let v = tracker.check("execute_command", &args);
        assert_eq!(v.level, LoopVerdict::Critical);
        assert_eq!(v.detector, LoopDetector::NoProgress);
        assert_eq!(v.count, 2);
    }

    #[test]
    fn test_tool_loop_tracker_breaker() {
        let mut tracker = ToolLoopTracker::new(ToolLoopTrackerOptions {
            breaker_veto_streak: 2,
            critical_threshold: 2,
            ..Default::default()
        });
        let args = serde_json::json!({"cmd": "cargo build"});

        // First veto
        tracker.record_call("execute_command", &args);
        tracker.record_veto("execute_command", &args);
        assert!(!tracker.is_breaker_tripped("execute_command"));

        // Second veto trips breaker
        tracker.record_call("execute_command", &args);
        tracker.record_veto("execute_command", &args);
        assert!(tracker.is_breaker_tripped("execute_command"));
    }

    #[test]
    fn test_tool_loop_tracker_wandering() {
        let mut tracker = ToolLoopTracker::new(ToolLoopTrackerOptions {
            wandering_threshold: 3,
            ..Default::default()
        });

        // 3 distinct args on same wandering-prone tool
        for i in 0..3 {
            let args = serde_json::json!({"cmd": format!("cargo run {}", i)});
            let v = tracker.check("execute_command", &args);
            if i < 2 {
                assert_eq!(v.level, LoopVerdict::Ok);
            } else {
                assert_eq!(v.level, LoopVerdict::Wandering);
                assert_eq!(v.count, 3);
            }
            tracker.record_call("execute_command", &args);
            tracker.record_outcome("execute_command", &args, Some(i));
        }
    }

    #[test]
    fn test_tool_loop_tracker_different_args_different_result_breaks_streak() {
        let mut tracker = ToolLoopTracker::new(ToolLoopTrackerOptions {
            warning_threshold: 10,
            critical_threshold: 3,
            ..Default::default()
        });

        let args1 = serde_json::json!({"cmd": "cargo test"});
        let args2 = serde_json::json!({"cmd": "cargo build"});

        // Same args, same result x2
        tracker.record_call("execute_command", &args1);
        tracker.record_outcome("execute_command", &args1, Some(1));
        tracker.record_call("execute_command", &args1);
        tracker.record_outcome("execute_command", &args1, Some(1));

        // Different args (different result) — streak resets
        tracker.record_call("execute_command", &args2);
        tracker.record_outcome("execute_command", &args2, Some(99));

        // No-progress streak should be 0 (broken by different args/result)
        let v = tracker.check("execute_command", &args1);
        assert_eq!(v.level, LoopVerdict::Ok);
    }

    #[test]
    fn test_two_zone_prompt_compilation() {
        let zone_a = compile_zone_a_prefix("Name: Cynapse\nRole: Companion");
        assert!(zone_a.contains("You are Cynapse"));
        assert!(zone_a.contains("Operating Directives:"));
        assert!(zone_a.contains("Available Tools:"));
        assert!(zone_a.contains("read_file"));

        let zone_b = compile_zone_b_tail("## fact 1: Rust is fast", "USER: hello\nASSISTANT: hi", "Can you build it?", None);
        assert!(zone_b.contains("=== OPERATING DIRECTIVE ==="));
        assert!(zone_b.contains("=== RECALLED KNOWLEDGE ==="));
        assert!(zone_b.contains("## fact 1: Rust is fast"));
        assert!(zone_b.contains("=== CONVERSATION LOG ==="));
        assert!(zone_b.contains("=== USER INSTRUCTION ==="));
        assert!(zone_b.contains("Can you build it?"));

        let zone_b_with_notice = compile_zone_b_tail(
            "## fact 1: Rust is fast",
            "USER: hello\nASSISTANT: hi",
            "Can you build it?",
            Some("Loop detector: tool repeated. Change strategy."),
        );
        assert!(zone_b_with_notice.contains("=== NOTICE ==="));
        assert!(zone_b_with_notice.contains("Loop detector: tool repeated. Change strategy."));
        // Notice must appear before USER INSTRUCTION
        let notice_idx = zone_b_with_notice.find("=== NOTICE ===").unwrap();
        let instruction_idx = zone_b_with_notice.find("=== USER INSTRUCTION ===").unwrap();
        assert!(notice_idx < instruction_idx);

        let conv_prefix = compile_conversational_prefix("Name: Cynapse\nRole: Companion");
        assert!(conv_prefix.contains("You are Cynapse"));
        assert!(!conv_prefix.contains("Available Tools:"));
        assert!(!conv_prefix.contains("read_file"));
    }

    #[test]
    fn test_loop_guard_warn_is_advisory() {
        let mut guard = LoopGuard::default();
        let tool = ToolCall {
            name: "read_file".to_string(),
            arguments: serde_json::json!({"path": "main.rs"}),
        };

        // Repeated invocations trigger Warn verdict, which is advisory (Ok(()))
        for _ in 0..4 {
            assert!(guard.check(&tool).is_ok());
            guard.record_call(&tool);
            guard.record_outcome(&tool, "fn main() {}");
        }
    }

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(levenshtein_distance("path", "path"), 0);
        assert_eq!(levenshtein_distance("pash", "path"), 1);
        assert_eq!(levenshtein_distance("filenam", "filename"), 1);
        assert_eq!(levenshtein_distance("", "test"), 4);
    }

    #[test]
    fn test_coerce_tool_arguments_aliases_and_types() {
        let args = serde_json::json!({
            "filepath": "src/main.rs",
            "allow_multiple": "true"
        });
        let coerced = coerce_tool_arguments("patch_file", &args);
        assert_eq!(coerced.get("path").and_then(|v| v.as_str()), Some("src/main.rs"));
        assert_eq!(coerced.get("allow_multiple").and_then(|v| v.as_bool()), Some(true));

        let cmd_args = serde_json::json!({
            "cmd": "cargo check"
        });
        let coerced_cmd = coerce_tool_arguments("execute_command", &cmd_args);
        assert_eq!(coerced_cmd.get("command").and_then(|v| v.as_str()), Some("cargo check"));

        // Single string auto-coercion
        let str_arg = serde_json::Value::String("README.md".to_string());
        let coerced_read = coerce_tool_arguments("read_file", &str_arg);
        assert_eq!(coerced_read.get("path").and_then(|v| v.as_str()), Some("README.md"));
    }

    #[test]
    fn test_describe_argument_error_hints() {
        let hint = describe_argument_error("read_file", "pash");
        assert_eq!(hint, Some("Unknown argument 'pash' for tool 'read_file'. Did you mean 'path'?".to_string()));

        let no_hint = describe_argument_error("read_file", "completely_unrelated_nonsense");
        assert_eq!(no_hint, None);
    }

    #[test]
    fn test_unpack_canonical_and_arg_pair() {
        let call = ToolCall {
            name: "patch_file".to_string(),
            arguments: serde_json::json!({
                "path": "src/lib.rs",
                "target_content": "old_foo()",
                "replacement_content": "new_foo()",
                "allow_multiple": false
            }),
        };
        match call.unpack_canonical() {
            CanonicalArgs::PatchFile { path, target_content, replacement_content, allow_multiple } => {
                assert_eq!(path, "src/lib.rs");
                assert_eq!(target_content, "old_foo()");
                assert_eq!(replacement_content, "new_foo()");
                assert!(!allow_multiple);
            }
            _ => panic!("Expected PatchFile"),
        }
    }

    #[test]
    fn test_isolate_thinking_scratchpad_and_tool_call() {
        let raw = "<thinking>\nI should inspect Cargo.toml first to see dependencies.\n</thinking>\n```json\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"Cargo.toml\"}}\n```";
        let (clean, thought) = isolate_thinking_scratchpad(raw);
        assert_eq!(thought, Some("I should inspect Cargo.toml first to see dependencies.".to_string()));
        assert!(!clean.contains("<thinking>"));

        let tools = validate_gbnf_tool_calls(raw).expect("Should parse tool call despite thinking tokens");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "read_file");
        assert_eq!(tools[0].arguments.get("path").and_then(|v| v.as_str()), Some("Cargo.toml"));
    }

    #[test]
    fn test_system2_guardrails_and_voting() {
        // 1. Step-back prompt
        let sb = format_step_back_prompt("delete my database");
        assert!(sb.contains("System 2 Reflection:"));
        assert!(sb.contains("delete my database"));

        // 2. Path verification
        assert!(verify_path_exists("Cargo.toml").is_ok());
        assert!(verify_path_exists("non_existent_file_xyz_123.rs").is_err());

        // 3. Command binary verification
        assert!(verify_command_binary_exists("cargo test").is_ok());
        assert!(verify_command_binary_exists("echo hello").is_ok());
        assert!(verify_command_binary_exists("fake_non_existent_binary_987").is_err());

        // 4. Majority voting
        let candidates = vec![
            ToolCall { name: "read_file".into(), arguments: serde_json::json!({"path": "src/lib.rs"}) },
            ToolCall { name: "patch_file".into(), arguments: serde_json::json!({"path": "src/lib.rs"}) },
            ToolCall { name: "read_file".into(), arguments: serde_json::json!({"path": "src/lib.rs"}) },
        ];
        let voted = self_consistency_majority_vote(&candidates).unwrap();
        assert_eq!(voted.name, "read_file");
    }

    #[test]
    fn test_advisory_steerer_nudges_before_hard_veto() {
        let mut guard = LoopGuard::new(ToolLoopTrackerOptions {
            warning_threshold: 3,
            critical_threshold: 5,
            breaker_veto_streak: 3,
            history_size: 20,
            wandering_threshold: 6,
            wandering_escalation: 12,
        });

        let call = ToolCall {
            name: "read_file".into(),
            arguments: serde_json::json!({ "path": "src/lib.rs" }),
        };

        // Call 1
        assert!(guard.check(&call).is_ok());
        assert_eq!(guard.advisory_notice(&call), None);
        guard.record_call(&call);
        guard.record_outcome(&call, "content v1");

        // Call 2
        assert!(guard.check(&call).is_ok());
        assert_eq!(guard.advisory_notice(&call), None);
        guard.record_call(&call);
        guard.record_outcome(&call, "content v2");

        // Call 3: hits warning_threshold (3). Check is still Ok (NOT vetoed), but advisory_notice returns steering guidance!
        assert!(guard.check(&call).is_ok());
        let advisory = guard.advisory_notice(&call);
        assert!(advisory.is_some(), "Advisory notice must trigger on repeat count 3");
        let adv_str = advisory.unwrap();
        assert!(adv_str.contains("=== ADVISORY STEERING ==="));
        assert!(adv_str.contains("You called tool 'read_file' repeatedly with identical arguments"));

        // check_with_advisory returns Ok(Some(advisory))
        let res = guard.check_with_advisory(&call).unwrap();
        assert!(res.is_some());
    }
}

