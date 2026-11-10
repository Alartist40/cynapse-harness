use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;
use anyhow::{bail, Context, Result};
use futures_util::StreamExt;
use regex::Regex;
pub mod backup;
pub mod bots;
pub mod compressor;
pub mod doctor;
pub mod downloader;
pub mod forensics;
pub mod gates;
pub mod offline_agent;
pub mod persona;
pub mod power;
pub mod receipts;
pub mod sbc_cluster;
pub mod security;
pub mod session;
pub mod skills;
pub mod subagent;
pub mod synapse;
pub mod undo;
pub mod updater;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolDefinition {
    pub name: &'static str,
    pub description: &'static str,
    pub concurrency_safe: bool,
}

pub fn list_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "read_file",
            description: "Read text contents of a file within permitted workspace",
            concurrency_safe: true,
        },
        ToolDefinition {
            name: "write_file",
            description: "Write content to a file within permitted workspace",
            concurrency_safe: false,
        },
        ToolDefinition {
            name: "patch_file",
            description: "Surgically patch a specific section of a file using exact target matching",
            concurrency_safe: false,
        },
        ToolDefinition {
            name: "run_gates",
            description: "Deterministically execute acceptance gates defined in GATES.md and record verifiable machine evidence",
            concurrency_safe: false,
        },
        ToolDefinition {
            name: "grep",
            description: "Search regex pattern recursively across workspace files",
            concurrency_safe: true,
        },
        ToolDefinition {
            name: "execute_command",
            description: "Execute a safe bash shell command with timeout",
            concurrency_safe: false,
        },
        ToolDefinition {
            name: "spawn_subagent",
            description: "Spawn a delegated autonomous subagent task with optional bot profile and wait mode",
            concurrency_safe: false,
        },
        ToolDefinition {
            name: "memory_search",
            description: "Search Dendrite memory graph for compact pointers (IDs, titles, tags) without context blowup",
            concurrency_safe: true,
        },
        ToolDefinition {
            name: "memory_timeline",
            description: "View chronological sequence of memory nodes around a given node ID to trace decisions",
            concurrency_safe: true,
        },
        ToolDefinition {
            name: "memory_inspect",
            description: "Inspect the full content and metadata of a specific memory node by ID",
            concurrency_safe: true,
        },
        ToolDefinition {
            name: "security_forensic_scan",
            description: "Execute read-only digital forensics scan across local browser histories, caches, cookies, and DNS artifacts for threat indicators (Candiru/AdFection) with zero-network isolation",
            concurrency_safe: true,
        },
    ]
}

/// Returns true if a tool is safe for concurrent read/inspection execution.
pub fn is_tool_concurrency_safe(name: &str) -> bool {
    list_tools()
        .into_iter()
        .find(|t| t.name == name)
        .map(|t| t.concurrency_safe)
        .unwrap_or(false)
}

/// Partitions a slice of tool calls into execution batches:
/// contiguous runs of concurrency-safe tools are grouped together,
/// while exclusive (state-mutating) tools are placed into individual singleton batches.
pub fn partition_tool_calls(calls: &[offline_agent::ToolCall]) -> Vec<Vec<offline_agent::ToolCall>> {
    let mut partitions = Vec::new();
    let mut current_safe_batch = Vec::new();

    for call in calls {
        if is_tool_concurrency_safe(&call.name) {
            current_safe_batch.push(call.clone());
        } else {
            if !current_safe_batch.is_empty() {
                partitions.push(std::mem::take(&mut current_safe_batch));
            }
            partitions.push(vec![call.clone()]);
        }
    }

    if !current_safe_batch.is_empty() {
        partitions.push(current_safe_batch);
    }

    partitions
}

/// Validate path safety: prevents path traversal and symlink bypass to sensitive system files.
fn validate_safe_path(p_str: &str, for_write: bool) -> Result<PathBuf> {
    let raw_path = if p_str.starts_with('~') {
        if let Some(home) = dirs::home_dir() {
            home.join(p_str.trim_start_matches('~').trim_start_matches('/'))
        } else {
            PathBuf::from(p_str)
        }
    } else {
        PathBuf::from(p_str)
    };
    let normalized = if raw_path.is_absolute() {
        raw_path
    } else {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")).join(raw_path)
    };

    let resolved = if normalized.exists() {
        normalized.canonicalize().unwrap_or_else(|_| normalized.clone())
    } else if let Some(parent) = normalized.parent() {
        if parent.exists() {
            parent.canonicalize().map(|p| p.join(normalized.file_name().unwrap_or_default())).unwrap_or_else(|_| normalized.clone())
        } else {
            normalized.clone()
        }
    } else {
        normalized.clone()
    };

    let p_str_clean = normalized.to_string_lossy();
    let resolved_str = resolved.to_string_lossy();

    let sensitive_patterns = [
        "/etc/shadow", "/etc/passwd", "/etc/sudoers", "/etc/master.passwd",
        "/.ssh", "/root", "/proc/kcore", "/dev/mem", "/dev/kmem",
    ];

    for sensitive in &sensitive_patterns {
        if p_str_clean.contains(sensitive) || resolved_str.contains(sensitive) {
            bail!("Access denied: path '{}' accesses sensitive system resources.", p_str);
        }
    }

    if for_write && (resolved_str == "/" || resolved_str == "/etc" || resolved_str == "/usr" || resolved_str == "/bin" || resolved_str == "/sbin") {
        bail!("Access denied: cannot write to system directory '{}'.", p_str);
    }

    Ok(resolved)
}

/// Shared safe file reading helper returning the canonical path and full content.
pub(crate) fn read_file_at(path: &str) -> Result<(PathBuf, String)> {
    let safe_path = validate_safe_path(path, false)?;
    if !safe_path.exists() {
        bail!("File not found: {}", path);
    }
    if safe_path.is_dir() {
        bail!("Path is a directory, not a file: {}", path);
    }
    let metadata = fs::metadata(&safe_path)?;
    if metadata.len() > 10 * 1024 * 1024 {
        bail!("File too large to read into context ({} bytes, limit is 10 MB)", metadata.len());
    }
    let content = fs::read_to_string(&safe_path)
        .with_context(|| format!("Failed to read file {}", safe_path.display()))?;
    Ok((safe_path, content))
}

/// Surgically patches a file by replacing a unique target slice with replacement content.
pub fn patch_file_at(
    path_str: &str,
    target_content: &str,
    replacement_content: &str,
    allow_multiple: bool,
) -> Result<String> {
    if target_content.is_empty() {
        bail!("Cannot patch file: target_content is empty");
    }
    let safe_path = validate_safe_path(path_str, true)?;
    if !safe_path.exists() {
        bail!("File not found for patching: {}", safe_path.display());
    }
    if safe_path.is_dir() {
        bail!("Target path is a directory, not a file: {}", safe_path.display());
    }
    let original = fs::read_to_string(&safe_path)
        .with_context(|| format!("Failed to read {}", safe_path.display()))?;

    let matches = original.matches(target_content).count();
    if matches == 0 {
        bail!(
            "Target content not found in {}. Ensure exact indentation and content match.",
            safe_path.display()
        );
    }
    if matches > 1 && !allow_multiple {
        bail!(
            "Target content found {} times in {}. Set allow_multiple: true or provide more surrounding context as a unique anchor.",
            matches,
            safe_path.display()
        );
    }

    let _ = undo::UndoJournal::record_patch(&safe_path, &original);

    let new_content = if allow_multiple {
        original.replace(target_content, replacement_content)
    } else {
        original.replacen(target_content, replacement_content, 1)
    };

    fs::write(&safe_path, new_content)
        .with_context(|| format!("Failed to write patched file {}", safe_path.display()))?;

    Ok(format!(
        "Successfully patched {} (replaced {} occurrence(s))",
        safe_path.display(),
        matches
    ))
}

/// Resolves the active Dendrite SQLite database path.
pub fn resolve_dendrite_db_path() -> PathBuf {
    if let Ok(custom) = std::env::var("CYNAPSE_DENDRITE_DB") {
        return PathBuf::from(custom);
    }
    if PathBuf::from("data/dendrite.db").exists() {
        return PathBuf::from("data/dendrite.db");
    }
    if let Some(home) = dirs::home_dir() {
        let p = home.join(".cynapse").join("dendrite.db");
        if p.exists() {
            return p;
        }
    }
    PathBuf::from("data/dendrite.db")
}

/// Native implementation of atomic-agent tools with sandboxing and execution timeout.
pub fn execute_tool(name: &str, arg1: &str, arg2: Option<&str>) -> Result<String> {
    match name {
        "read_file" => {
            let (_path, content) = read_file_at(arg1)?;
            Ok(content)
        }
        "write_file" => {
            let mut safe_path = validate_safe_path(arg1, true)?;
            if safe_path.is_dir() {
                safe_path = safe_path.join("about.md");
            }
            let _ = undo::UndoJournal::record_write(&safe_path);
            let content = arg2.unwrap_or_default();
            if let Some(parent) = safe_path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&safe_path, content)
                .with_context(|| format!("Failed to write file {}", safe_path.display()))?;
            Ok(format!("Successfully wrote {} bytes to {}", content.len(), safe_path.display()))
        }
        "patch_file" => {
            if let Some(json_str) = arg2 {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
                    let target = val.get("target_content")
                        .or_else(|| val.get("target"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let replacement = val.get("replacement_content")
                        .or_else(|| val.get("replacement"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let allow_multiple = val.get("allow_multiple")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    return patch_file_at(arg1, target, replacement, allow_multiple);
                }
            }
            bail!("Invalid arguments for patch_file: expected path as arg1 and JSON with target_content and replacement_content as arg2");
        }
        "run_gates" => {
            let p = if arg1.trim().is_empty() { None } else { Some(arg1) };
            gates::run_gates(p)
        }
        "grep" => {
            let pattern = arg1;
            let dir_path = arg2.unwrap_or(".");
            let safe_dir = validate_safe_path(dir_path, false)?;
            let re = Regex::new(pattern).with_context(|| format!("Invalid regex pattern: {}", pattern))?;
            let mut matches = Vec::new();

            fn walk_and_grep(dir: &Path, re: &Regex, matches: &mut Vec<String>, depth: usize) {
                if depth > 8 || matches.len() >= 50 {
                    return;
                }
                if let Ok(entries) = fs::read_dir(dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        let fname = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
                        if fname.starts_with('.') || fname == "target" || fname == "node_modules" {
                            continue;
                        }
                        if path.is_file() {
                            if let Ok(content) = fs::read_to_string(&path) {
                                for (line_no, line) in content.lines().enumerate() {
                                    if re.is_match(line) {
                                        matches.push(format!("{}:{}: {}", path.display(), line_no + 1, line.trim()));
                                        if matches.len() >= 50 {
                                            return;
                                        }
                                    }
                                }
                            }
                        } else if path.is_dir() {
                            walk_and_grep(&path, re, matches, depth + 1);
                        }
                    }
                }
            }

            if safe_dir.is_file() {
                if let Ok(content) = fs::read_to_string(&safe_dir) {
                    for (line_no, line) in content.lines().enumerate() {
                        if re.is_match(line) {
                            matches.push(format!("{}:{}: {}", safe_dir.display(), line_no + 1, line.trim()));
                        }
                    }
                }
            } else {
                walk_and_grep(&safe_dir, &re, &mut matches, 0);
            }

            if matches.is_empty() {
                Ok(format!("No matches found for pattern '{}' in {}", pattern, dir_path))
            } else {
                Ok(matches.join("\n"))
            }
        }
        "execute_command" => {
            let trimmed = arg1.trim();
            // Block dangerous commands
            let dangerous = ["rm -rf /", "rm -rf /*", "mkfs", ":(){ :|:& };:", "> /dev/sda"];
            for d in &dangerous {
                if trimmed.contains(d) {
                    bail!("Security violation: destructive command blocked: '{}'", trimmed);
                }
            }

            let mut child = Command::new("bash")
                .arg("-c")
                .arg(trimmed)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .with_context(|| format!("Failed to spawn command: {}", trimmed))?;

            // Timeout enforcement: 10 seconds
            let start = std::time::Instant::now();
            let timeout = Duration::from_secs(10);

            while start.elapsed() < timeout {
                if let Ok(Some(status)) = child.try_wait() {
                    let output = child.wait_with_output()?;
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Ok(format!("Exit Status: {}\n{}{}", status, stdout, stderr));
                }
                std::thread::sleep(Duration::from_millis(50));
            }

            let _ = child.kill();
            bail!("Command timed out after 10 seconds: {}", trimmed);
        }
        "memory_search" => {
            let query = arg1.trim();
            let limit = arg2.and_then(|s| s.parse::<usize>().ok()).unwrap_or(5).clamp(1, 10);
            let db_path = resolve_dendrite_db_path();
            if !db_path.exists() {
                return Ok(format!("No Dendrite database found at {}", db_path.display()));
            }
            let store = cynapse_memory::store::DendriteStore::open(&db_path)?;
            let results = store.fts_search_calibrated(query, limit, None)?;
            if results.is_empty() {
                return Ok(format!("No memory records found for query '{}'", query));
            }
            let graph = cynapse_memory::graph::Dendrite::new();
            store.load_all(&graph)?;
            let mut lines = Vec::new();
            for (id, rank) in results {
                if let Some(node) = graph.get(&id) {
                    let tags_str = if node.tags.is_empty() {
                        String::new()
                    } else {
                        format!(" [{}]", node.tags.join(", "))
                    };
                    let topic = node.topic_key.map(|k| format!(" key={}", k)).unwrap_or_default();
                    lines.push(format!("- #{} ({}) {}{}{} (bm25: {:.2})", node.id, node.node_type.label(), node.title, tags_str, topic, rank));
                } else {
                    lines.push(format!("- #{} (bm25: {:.2})", id, rank));
                }
            }
            Ok(lines.join("\n"))
        }
        "memory_timeline" => {
            let target_id = arg1.trim();
            let limit = arg2.and_then(|s| s.parse::<usize>().ok()).unwrap_or(5).clamp(1, 15);
            let db_path = resolve_dendrite_db_path();
            if !db_path.exists() {
                return Ok(format!("No Dendrite database found at {}", db_path.display()));
            }
            let store = cynapse_memory::store::DendriteStore::open(&db_path)?;
            let graph = cynapse_memory::graph::Dendrite::new();
            store.load_all(&graph)?;
            let target = graph.get(target_id).ok_or_else(|| anyhow::anyhow!("Node '{}' not found in memory", target_id))?;
            let mut all_nodes = graph.all();
            all_nodes.sort_by_key(|n| n.updated_at);
            let pos = all_nodes.iter().position(|n| n.id == target.id).unwrap_or(0);
            let start = pos.saturating_sub(limit / 2);
            let end = (start + limit).min(all_nodes.len());
            let mut lines = vec![format!("Timeline window around #{}:", target_id)];
            for n in &all_nodes[start..end] {
                let marker = if n.id == target.id { ">>>" } else { "   " };
                lines.push(format!("{} #{} [{}] {} (rev: {})", marker, n.id, n.node_type.label(), n.title, n.revision_count));
            }
            Ok(lines.join("\n"))
        }
        "memory_inspect" => {
            let target_id = arg1.trim();
            let db_path = resolve_dendrite_db_path();
            if !db_path.exists() {
                return Ok(format!("No Dendrite database found at {}", db_path.display()));
            }
            let store = cynapse_memory::store::DendriteStore::open(&db_path)?;
            let graph = cynapse_memory::graph::Dendrite::new();
            store.load_all(&graph)?;
            let node = graph.get(target_id).ok_or_else(|| anyhow::anyhow!("Node '{}' not found in memory", target_id))?;
            let mut out = Vec::new();
            out.push(format!("ID: #{}", node.id));
            out.push(format!("Title: {}", node.title));
            out.push(format!("Type: {}", node.node_type.label()));
            if let Some(ref tk) = node.topic_key {
                out.push(format!("Topic Key: {}", tk));
            }
            out.push(format!("Revision: {}", node.revision_count));
            if !node.tags.is_empty() {
                out.push(format!("Tags: {}", node.tags.join(", ")));
            }
            if !node.supersedes.is_empty() {
                out.push(format!("Supersedes: {}", node.supersedes.join(", ")));
            }
            if !node.conflicts_with.is_empty() {
                out.push(format!("Conflicts With: {}", node.conflicts_with.join(", ")));
            }
            out.push(format!("\nContent:\n{}", node.content));
            Ok(out.join("\n"))
        }
        "security_forensic_scan" => {
            let deep = arg1.to_lowercase().contains("deep") || arg1.to_lowercase().contains("full") || arg1 == "true";
            let report = forensics::run_forensic_scan(deep);
            if arg1.contains("json") || arg2.map(|a| a.contains("json")).unwrap_or(false) {
                Ok(report.to_json())
            } else {
                Ok(report.to_summary())
            }
        }
        _ => {
            // Check dynamic Synapse extensions
            let reg = synapse::SynapseRegistry::global();
            let json_arg = if let Some(a2) = arg2 {
                serde_json::json!({ "arg1": arg1, "arg2": a2 })
            } else {
                serde_json::from_str(arg1).unwrap_or_else(|_| serde_json::json!({ "arg": arg1 }))
            };
            reg.execute_tool(name, &json_arg)
                .map_err(|e| anyhow::anyhow!("Unknown tool '{}' or Synapse execution error: {}", name, e))
        }
    }
}

/// Executes a tool enforcing a BotProfile's allow/deny rules, workspace confinement, and security shield.
pub fn execute_tool_with_profile(
    name: &str,
    arg1: &str,
    arg2: Option<&str>,
    profile: &bots::BotProfile,
) -> Result<String> {
    if !profile.is_tool_allowed(name) {
        bail!("Permission denied: tool '{}' is not permitted by bot profile '@{}'", name, profile.slug);
    }

    if profile.requires_approval(name) {
        bail!("Approval required: tool '{}' requires interactive confirmation (in tools_ask for '@{}')", name, profile.slug);
    }

    // Security shield pre-flight audit
    let tool_call = offline_agent::ToolCall {
        name: name.to_string(),
        arguments: if name == "write_file" || name == "patch_file" {
            serde_json::json!({ "path": arg1, "content": arg2 })
        } else if name == "execute_command" {
            serde_json::json!({ "command": arg1 })
        } else {
            serde_json::json!({ "path": arg1 })
        },
    };

    match security::audit_tool_call(&tool_call) {
        security::SecurityVerdict::Blocked { reason } => {
            bail!("Security shield violation: {}", reason);
        }
        security::SecurityVerdict::RequiresApproval { reason } => {
            bail!("Approval required by security shield: {}", reason);
        }
        security::SecurityVerdict::Safe => {}
    }

    // Path confinement check for file tools
    match name {
        "read_file" | "write_file" | "patch_file" | "run_gates" => {
            if !profile.is_path_in_workspace(arg1) {
                bail!(
                    "Workspace violation: path '{}' is outside the restricted workspace '{:?}' of bot profile '@{}'",
                    arg1,
                    profile.workspace_restrict,
                    profile.slug
                );
            }
        }
        "grep" => {
            let dir = arg2.unwrap_or(".");
            if !profile.is_path_in_workspace(dir) {
                bail!(
                    "Workspace violation: directory '{}' is outside the restricted workspace '{:?}' of bot profile '@{}'",
                    dir,
                    profile.workspace_restrict,
                    profile.slug
                );
            }
        }
        _ => {}
    }

    execute_tool(name, arg1, arg2)
}

/// Executes a tool enforcing a BotProfile's allow/deny rules and workspace confinement,
/// bypassing the `requires_approval` check because explicit user approval was already granted.
pub fn execute_tool_with_profile_approved(
    name: &str,
    arg1: &str,
    arg2: Option<&str>,
    profile: &bots::BotProfile,
) -> Result<String> {
    if !profile.is_tool_allowed(name) {
        bail!("Permission denied: tool '{}' is not permitted by bot profile '@{}'", name, profile.slug);
    }

    let tool_call = offline_agent::ToolCall {
        name: name.to_string(),
        arguments: if name == "write_file" || name == "patch_file" {
            serde_json::json!({ "path": arg1, "content": arg2 })
        } else if name == "execute_command" {
            serde_json::json!({ "command": arg1 })
        } else {
            serde_json::json!({ "path": arg1 })
        },
    };

    if let security::SecurityVerdict::Blocked { reason } = security::audit_tool_call(&tool_call) {
        bail!("Security shield violation: {}", reason);
    }

    // Path confinement check for file tools
    match name {
        "read_file" | "write_file" | "patch_file" | "run_gates" => {
            if !profile.is_path_in_workspace(arg1) {
                bail!(
                    "Workspace violation: path '{}' is outside the restricted workspace '{:?}' of bot profile '@{}'",
                    arg1,
                    profile.workspace_restrict,
                    profile.slug
                );
            }
        }
        "grep" => {
            let dir = arg2.unwrap_or(".");
            if !profile.is_path_in_workspace(dir) {
                bail!(
                    "Workspace violation: directory '{}' is outside the restricted workspace '{:?}' of bot profile '@{}'",
                    dir,
                    profile.workspace_restrict,
                    profile.slug
                );
            }
        }
        _ => {}
    }

    execute_tool(name, arg1, arg2)
}

/// Executes a tool call directly using its canonical unpacked arguments.
pub fn execute_tool_call(call: &offline_agent::ToolCall) -> Result<String> {
    match call.unpack_canonical() {
        offline_agent::CanonicalArgs::ReadFile { path } => {
            let (_path, content) = read_file_at(&path)?;
            Ok(content)
        }
        offline_agent::CanonicalArgs::WriteFile { path, content } => {
            execute_tool("write_file", &path, Some(&content))
        }
        offline_agent::CanonicalArgs::PatchFile { path, target_content, replacement_content, allow_multiple } => {
            patch_file_at(&path, &target_content, &replacement_content, allow_multiple)
        }
        offline_agent::CanonicalArgs::RunGates { path } => {
            gates::run_gates(path.as_deref())
        }
        offline_agent::CanonicalArgs::Grep { pattern, path } => {
            execute_tool("grep", &pattern, path.as_deref())
        }
        offline_agent::CanonicalArgs::ExecuteCommand { command } => {
            execute_tool("execute_command", &command, None)
        }
        offline_agent::CanonicalArgs::SpawnSubagent { instruction, role } => {
            execute_tool("spawn_subagent", &instruction, role.as_deref())
        }
        offline_agent::CanonicalArgs::MemorySearch { query, limit } => {
            execute_tool("memory_search", &query, limit.map(|l| l.to_string()).as_deref())
        }
        offline_agent::CanonicalArgs::MemoryTimeline { node_id, limit } => {
            execute_tool("memory_timeline", &node_id, limit.map(|l| l.to_string()).as_deref())
        }
        offline_agent::CanonicalArgs::MemoryInspect { node_id } => {
            execute_tool("memory_inspect", &node_id, None)
        }
        offline_agent::CanonicalArgs::Generic { arg1, arg2 } => {
            execute_tool(&call.name, &arg1, arg2.as_deref())
        }
    }
}

/// Executes a tool call with profile validation using canonical unpacked arguments.
pub fn execute_tool_call_with_profile(
    call: &offline_agent::ToolCall,
    profile: &bots::BotProfile,
) -> Result<String> {
    let (arg1, arg2) = call.to_arg_pair();
    execute_tool_with_profile(&call.name, &arg1, arg2.as_deref(), profile)
}

/// Executes an approved tool call with profile validation using canonical unpacked arguments.
pub fn execute_tool_call_with_profile_approved(
    call: &offline_agent::ToolCall,
    profile: &bots::BotProfile,
) -> Result<String> {
    let (arg1, arg2) = call.to_arg_pair();
    execute_tool_with_profile_approved(&call.name, &arg1, arg2.as_deref(), profile)
}

/// Real HuggingFace model downloader streaming target .gguf or .safetensors files into models_dir.
pub async fn pull_huggingface_model(url_or_repo: &str, models_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(models_dir)?;
    let (download_url, target_filename) = downloader::resolve_hf_download_url_async(url_or_repo, "Q4_K_M").await;
    let target_path = models_dir.join(&target_filename);

    println!("📥 Downloading HuggingFace Model...");
    println!("   URL: {}", download_url);
    println!("   Target: {}", target_path.display());

    let client = reqwest::Client::new();
    let res = client
        .get(&download_url)
        .header("User-Agent", "Cynapse-Agent-Downloader/1.0")
        .send()
        .await
        .with_context(|| format!("Failed to connect to HuggingFace URL: {}", download_url))?;

    if !res.status().is_success() {
        bail!("HuggingFace server returned HTTP status: {}", res.status());
    }

    let mut file = File::create(&target_path)
        .with_context(|| format!("Failed to create destination file: {}", target_path.display()))?;

    let mut stream = res.bytes_stream();
    let mut downloaded_bytes = 0u64;

    while let Some(chunk) = stream.next().await {
        let bytes = chunk.context("Error downloading chunk from stream")?;
        file.write_all(&bytes)?;
        downloaded_bytes += bytes.len() as u64;
    }

    println!("✓ Download complete: {:.2} MB saved to {}", downloaded_bytes as f64 / 1_048_576.0, target_path.display());
    Ok(target_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_sanitization() {
        assert!(validate_safe_path("/etc/shadow", false).is_err());
        assert!(validate_safe_path("/root/.ssh/id_rsa", false).is_err());
        assert!(validate_safe_path("src/lib.rs", false).is_ok());
    }

    #[test]
    fn test_symlink_safety() {
        let temp_dir = std::env::temp_dir();
        let symlink_path = temp_dir.join("cynapse_test_symlink_passwd");
        let _ = fs::remove_file(&symlink_path);
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            if symlink("/etc/passwd", &symlink_path).is_ok() {
                let res = validate_safe_path(&symlink_path.to_string_lossy(), false);
                let _ = fs::remove_file(&symlink_path);
                assert!(res.is_err());
            }
        }
    }

    #[test]
    fn test_tool_concurrency_classification() {
        assert!(is_tool_concurrency_safe("read_file"));
        assert!(is_tool_concurrency_safe("grep"));
        assert!(!is_tool_concurrency_safe("write_file"));
        assert!(!is_tool_concurrency_safe("execute_command"));
        assert!(!is_tool_concurrency_safe("spawn_subagent"));
        assert!(!is_tool_concurrency_safe("unknown_tool"));
    }

    #[test]
    fn test_partition_tool_calls_mixed() {
        use serde_json::json;
        let calls = vec![
            offline_agent::ToolCall {
                name: "read_file".into(),
                arguments: json!({"path": "a.txt"}),
            },
            offline_agent::ToolCall {
                name: "grep".into(),
                arguments: json!({"pattern": "foo"}),
            },
            offline_agent::ToolCall {
                name: "write_file".into(),
                arguments: json!({"path": "b.txt", "content": "hi"}),
            },
            offline_agent::ToolCall {
                name: "execute_command".into(),
                arguments: json!({"command": "ls"}),
            },
            offline_agent::ToolCall {
                name: "read_file".into(),
                arguments: json!({"path": "c.txt"}),
            },
        ];

        let partitions = partition_tool_calls(&calls);
        assert_eq!(partitions.len(), 4);
        // Batch 1: contiguous safe [read_file, grep]
        assert_eq!(partitions[0].len(), 2);
        assert_eq!(partitions[0][0].name, "read_file");
        assert_eq!(partitions[0][1].name, "grep");
        // Batch 2: singleton exclusive [write_file]
        assert_eq!(partitions[1].len(), 1);
        assert_eq!(partitions[1][0].name, "write_file");
        // Batch 3: singleton exclusive [execute_command]
        assert_eq!(partitions[2].len(), 1);
        assert_eq!(partitions[2][0].name, "execute_command");
        // Batch 4: singleton safe [read_file]
        assert_eq!(partitions[3].len(), 1);
        assert_eq!(partitions[3][0].name, "read_file");
    }

    #[test]
    fn test_execute_command_timeout_and_block() {
        assert!(execute_tool("execute_command", "rm -rf /", None).is_err());
        let res = execute_tool("execute_command", "echo 'hello cynapse'", None);
        assert!(res.is_ok());
        assert!(res.unwrap().contains("hello cynapse"));
    }

    #[test]
    fn test_execute_tool_with_profile_permissions_and_confinement() {
        let temp = tempfile::tempdir().unwrap();
        let ws = temp.path().join("sub_ws");
        fs::create_dir_all(&ws).unwrap();
        let safe_file = ws.join("test.txt");
        fs::write(&safe_file, "hello from safe ws").unwrap();

        let restricted_profile = bots::BotProfile {
            slug: "isolated".into(),
            display_name: "Isolated Bot".into(),
            description: "".into(),
            persona_file: None,
            model_preset: None,
            tools_allow: vec!["read_file".into()],
            tools_deny: vec!["write_file".into()],
            tools_ask: vec![],
            workspace_restrict: Some(ws.to_string_lossy().to_string()),
            max_concurrent: None,
        };

        // Allowed inside workspace
        let res = execute_tool_with_profile("read_file", &safe_file.to_string_lossy(), None, &restricted_profile);
        assert!(res.is_ok());
        assert_eq!(res.unwrap(), "hello from safe ws");

        // Denied tool (write_file)
        let res_write = execute_tool_with_profile("write_file", &safe_file.to_string_lossy(), Some("new"), &restricted_profile);
        assert!(res_write.is_err());
        assert!(res_write.unwrap_err().to_string().contains("Permission denied"));

        // Tool outside workspace (read outside)
        let outside_file = temp.path().join("outside.txt");
        fs::write(&outside_file, "secret").unwrap();
        let res_outside = execute_tool_with_profile("read_file", &outside_file.to_string_lossy(), None, &restricted_profile);
        assert!(res_outside.is_err());
        assert!(res_outside.unwrap_err().to_string().contains("Workspace violation"));

        // Tool requiring interactive approval (tools_ask)
        let ask_profile = bots::BotProfile {
            slug: "cautious".into(),
            display_name: "Cautious Bot".into(),
            description: "".into(),
            persona_file: None,
            model_preset: None,
            tools_allow: vec!["*".into()],
            tools_deny: vec![],
            tools_ask: vec!["execute_command".into()],
            workspace_restrict: None,
            max_concurrent: None,
        };
        let res_ask = execute_tool_with_profile("execute_command", "ls", None, &ask_profile);
        assert!(res_ask.is_err());
        assert!(res_ask.unwrap_err().to_string().contains("Approval required"));
    }

    #[test]
    fn test_patch_file_surgical_precision() {
        let temp = tempfile::tempdir().unwrap();
        let target_file = temp.path().join("source.rs");
        let initial_code = "fn alpha() -> i32 { 1 }\nfn beta() -> i32 { 2 }\nfn gamma() -> i32 { 3 }\n";
        fs::write(&target_file, initial_code).unwrap();

        // 1. Surgical patch of middle function
        let patch_res = patch_file_at(
            &target_file.to_string_lossy(),
            "fn beta() -> i32 { 2 }",
            "fn beta() -> i32 { 42 }",
            false,
        );
        assert!(patch_res.is_ok());

        let updated = fs::read_to_string(&target_file).unwrap();
        assert_eq!(
            updated,
            "fn alpha() -> i32 { 1 }\nfn beta() -> i32 { 42 }\nfn gamma() -> i32 { 3 }\n"
        );

        // 2. Fails safely when target is ambiguous
        fs::write(&target_file, "foo bar foo").unwrap();
        let ambig = patch_file_at(&target_file.to_string_lossy(), "foo", "baz", false);
        assert!(ambig.is_err());
        assert!(ambig.unwrap_err().to_string().contains("found 2 times"));

        // 3. allow_multiple replaces all
        let multi = patch_file_at(&target_file.to_string_lossy(), "foo", "baz", true);
        assert!(multi.is_ok());
        let updated_multi = fs::read_to_string(&target_file).unwrap();
        assert_eq!(updated_multi, "baz bar baz");
    }

    #[test]
    fn test_execute_tool_call_routing() {
        let temp = tempfile::tempdir().unwrap();
        let file_path = temp.path().join("canonical.txt");
        fs::write(&file_path, "hello world").unwrap();

        let call = offline_agent::ToolCall {
            name: "read_file".to_string(),
            arguments: serde_json::json!({ "path": file_path.to_string_lossy() }),
        };
        let out = execute_tool_call(&call).unwrap();
        assert_eq!(out, "hello world");
    }

    #[test]
    fn test_security_shield_blocking_in_profile_execution() {
        let open_profile = bots::BotProfile::default();
        let blocked_res = execute_tool_with_profile("execute_command", "rm -rf /", None, &open_profile);
        assert!(blocked_res.is_err());
        assert!(blocked_res.unwrap_err().to_string().contains("Security shield violation"));
    }

    #[test]
    fn test_progressive_memory_disclosure_tools() {
        let temp = tempfile::tempdir().unwrap();
        let db_path = temp.path().join("test_dendrite.db");
        std::env::set_var("CYNAPSE_DENDRITE_DB", db_path.to_str().unwrap());

        let store = cynapse_memory::store::DendriteStore::open(&db_path).unwrap();
        let mut n1 = cynapse_memory::graph::Node::placeholder("node_auth".to_string(), 100);
        n1.title = "Authentication Token Architecture".to_string();
        n1.content = "Detailed secret signing and rotation procedure for JWTs.".to_string();
        n1.node_type = cynapse_memory::graph::NodeType::Procedure;
        n1.topic_key = Some("architecture/auth-tokens".to_string());
        n1.tags = vec!["auth".to_string(), "security".to_string()];
        store.save(&n1).unwrap();

        let mut n2 = cynapse_memory::graph::Node::placeholder("node_db".to_string(), 200);
        n2.title = "Database Engine Selection".to_string();
        n2.content = "SQLite with WAL mode and FTS5 ranking.".to_string();
        n2.node_type = cynapse_memory::graph::NodeType::Concept;
        n2.topic_key = Some("architecture/database".to_string());
        store.save(&n2).unwrap();

        // 1. Layer 1: memory_search (compact list, no raw content dump)
        let call_search = offline_agent::ToolCall {
            name: "memory_search".to_string(),
            arguments: serde_json::json!({ "query": "Authentication" }),
        };
        let search_res = execute_tool_call(&call_search).unwrap();
        assert!(search_res.contains("#node_auth"));
        assert!(search_res.contains("Authentication Token Architecture"));
        assert!(!search_res.contains("Detailed secret signing"), "Raw content must not leak into search results");

        // 2. Layer 2: memory_timeline (chronological sequence around node)
        let call_timeline = offline_agent::ToolCall {
            name: "memory_timeline".to_string(),
            arguments: serde_json::json!({ "node_id": "node_auth" }),
        };
        let timeline_res = execute_tool_call(&call_timeline).unwrap();
        assert!(timeline_res.contains("Timeline window around #node_auth:"));
        assert!(timeline_res.contains("#node_auth"));

        // 3. Layer 3: memory_inspect (full content on demand)
        let call_inspect = offline_agent::ToolCall {
            name: "memory_inspect".to_string(),
            arguments: serde_json::json!({ "node_id": "node_auth" }),
        };
        let inspect_res = execute_tool_call(&call_inspect).unwrap();
        assert!(inspect_res.contains("ID: #node_auth"));
        assert!(inspect_res.contains("Topic Key: architecture/auth-tokens"));
        assert!(inspect_res.contains("Detailed secret signing and rotation procedure for JWTs."));

        std::env::remove_var("CYNAPSE_DENDRITE_DB");
    }

    #[test]
    fn test_security_forensic_scan_tool_routing() {
        let call = offline_agent::ToolCall {
            name: "security_forensic_scan".to_string(),
            arguments: serde_json::json!({ "scan_type": "quick" }),
        };
        let out = execute_tool_call(&call).unwrap();
        assert!(out.contains("ZERO-BREACH DIGITAL FORENSICS REPORT"));
        assert!(out.contains("Doctrine D2 Zero-Network: Guaranteed"));
        assert!(out.contains("Loaded IOCs:"));
    }
}
