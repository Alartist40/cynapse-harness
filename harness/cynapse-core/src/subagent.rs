//! Subagent Spawn Runtime & Supervisor for Cynapse Agency Capabilities
//!
//! Reimplemented in pure Rust from nanobot's `agent/subagent.py` & `agent/tools/spawn.py`.
//! Provides task lifecycle management, concurrency capping via `tokio::sync::Semaphore`,
//! cancellation tokens, status reporting, and background announce-back messaging.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};

/// Lifecycle status of an autonomous delegated subagent task.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SubagentStatus {
    Pending,
    Running,
    Done,
    Failed(String),
    Cancelled,
}

/// Metadata and outcome record of a subagent task.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SubagentTask {
    pub id: usize,
    pub task: String,
    pub bot_slug: String,
    pub wait: bool,
    pub status: SubagentStatus,
    pub result: Option<String>,
    pub started_at_epoch_ms: u64,
    pub finished_at_epoch_ms: Option<u64>,
}

/// Announcement message emitted when a background subagent finishes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SubagentAnnouncement {
    pub task_id: usize,
    pub bot_slug: String,
    pub status: SubagentStatus,
    pub result: String,
}

/// Supervisor and lifecycle manager for concurrent delegated subagents.
#[derive(Debug, Clone)]
pub struct SubagentManager {
    tasks: Arc<RwLock<HashMap<usize, SubagentTask>>>,
    semaphore: Arc<tokio::sync::Semaphore>,
    cancellation_flags: Arc<Mutex<HashMap<usize, Arc<AtomicBool>>>>,
    next_id: Arc<AtomicUsize>,
    max_concurrent: usize,
}

impl Default for SubagentManager {
    fn default() -> Self {
        Self::new(2)
    }
}

impl SubagentManager {
    /// Creates a new SubagentManager with a global concurrency cap (default: 2).
    pub fn new(max_concurrent: usize) -> Self {
        let cap = if max_concurrent == 0 { 2 } else { max_concurrent };
        Self {
            tasks: Arc::new(RwLock::new(HashMap::new())),
            semaphore: Arc::new(tokio::sync::Semaphore::new(cap)),
            cancellation_flags: Arc::new(Mutex::new(HashMap::new())),
            next_id: Arc::new(AtomicUsize::new(1)),
            max_concurrent: cap,
        }
    }

    /// Returns the global concurrency cap.
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }

    /// Access the shared semaphore for background worker scheduling.
    pub fn semaphore(&self) -> Arc<tokio::sync::Semaphore> {
        Arc::clone(&self.semaphore)
    }

    /// Current timestamp in milliseconds since UNIX epoch.
    pub fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }

    /// Helper for testing stall detection by overriding a task's start timestamp.
    pub fn set_task_started_at_for_test(&self, id: usize, timestamp_ms: u64) {
        if let Ok(mut tasks) = self.tasks.write() {
            if let Some(t) = tasks.get_mut(&id) {
                t.started_at_epoch_ms = timestamp_ms;
            }
        }
    }

    /// Registers a new subagent task and returns its ID and a thread-safe cancellation flag.
    pub fn create_task(&self, task_instruction: &str, bot_slug: &str, wait: bool) -> (usize, Arc<AtomicBool>) {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let cancel_flag = Arc::new(AtomicBool::new(false));

        let task = SubagentTask {
            id,
            task: task_instruction.to_string(),
            bot_slug: bot_slug.to_string(),
            wait,
            status: SubagentStatus::Pending,
            result: None,
            started_at_epoch_ms: Self::now_ms(),
            finished_at_epoch_ms: None,
        };

        if let Ok(mut tasks) = self.tasks.write() {
            tasks.insert(id, task);
        }
        if let Ok(mut flags) = self.cancellation_flags.lock() {
            flags.insert(id, Arc::clone(&cancel_flag));
        }

        Self::append_task_log(id, &format!("SPAWN: bot=@{} wait={} task=\"{}\"", bot_slug, wait, task_instruction));

        (id, cancel_flag)
    }

    /// Updates the execution status and optional result payload of a subagent task.
    pub fn update_status(&self, id: usize, status: SubagentStatus, result: Option<String>) {
        let is_terminal = matches!(status, SubagentStatus::Done | SubagentStatus::Failed(_) | SubagentStatus::Cancelled);
        Self::append_task_log(id, &format!("STATUS: {:?} (result_len={})", status, result.as_ref().map(|r| r.len()).unwrap_or(0)));
        if let Ok(mut tasks) = self.tasks.write() {
            if let Some(task) = tasks.get_mut(&id) {
                task.status = status;
                if result.is_some() {
                    task.result = result;
                }
                if is_terminal {
                    task.finished_at_epoch_ms = Some(Self::now_ms());
                }
            }
        }
    }

    /// Signals cancellation to a running subagent task by ID.
    pub fn cancel_task(&self, id: usize) -> bool {
        let flag_set = if let Ok(flags) = self.cancellation_flags.lock() {
            if let Some(flag) = flags.get(&id) {
                flag.store(true, Ordering::SeqCst);
                true
            } else {
                false
            }
        } else {
            false
        };

        if flag_set {
            self.update_status(id, SubagentStatus::Cancelled, Some("Subagent execution cancelled by user request.".into()));
        }
        flag_set
    }

    /// Checks whether a task has received a cancellation signal.
    pub fn is_cancelled(&self, id: usize) -> bool {
        if let Ok(flags) = self.cancellation_flags.lock() {
            flags.get(&id).map(|f| f.load(Ordering::SeqCst)).unwrap_or(false)
        } else {
            false
        }
    }

    /// Retrieves the status of a specific task.
    pub fn get_status(&self, id: usize) -> Option<SubagentStatus> {
        self.tasks.read().ok()?.get(&id).map(|t| t.status.clone())
    }

    /// Retrieves a snapshot of a specific task record.
    pub fn get_task(&self, id: usize) -> Option<SubagentTask> {
        self.tasks.read().ok()?.get(&id).cloned()
    }

    /// Returns a list of all tracked subagent tasks ordered by ID.
    pub fn list_tasks(&self) -> Vec<SubagentTask> {
        if let Ok(tasks) = self.tasks.read() {
            let mut list: Vec<SubagentTask> = tasks.values().cloned().collect();
            list.sort_by_key(|t| t.id);
            list
        } else {
            Vec::new()
        }
    }

    /// Returns the count of currently running subagent tasks.
    pub fn running_count(&self) -> usize {
        if let Ok(tasks) = self.tasks.read() {
            tasks.values().filter(|t| matches!(t.status, SubagentStatus::Running | SubagentStatus::Pending)).count()
        } else {
            0
        }
    }

    /// Checks for tasks that have been in Running state longer than `timeout_ms`,
    /// automatically cancels them, and returns their task IDs.
    pub fn check_and_cancel_stalled_tasks(&self, timeout_ms: u64) -> Vec<usize> {
        let now = Self::now_ms();
        let mut stalled = Vec::new();
        if let Ok(tasks) = self.tasks.read() {
            for task in tasks.values() {
                if matches!(task.status, SubagentStatus::Running) && now.saturating_sub(task.started_at_epoch_ms) > timeout_ms {
                    stalled.push(task.id);
                }
            }
        }
        for &id in &stalled {
            self.cancel_task(id);
            self.update_status(id, SubagentStatus::Cancelled, Some(format!("Auto-cancelled: task stalled for > {}s without completing.", timeout_ms / 1000)));
        }
        stalled
    }

    /// Path to task log file in ~/.cynapse/tasks/<id>.log
    pub fn task_log_path(task_id: usize) -> PathBuf {
        let dir = dirs::home_dir()
            .map(|h| h.join(".cynapse").join("tasks"))
            .unwrap_or_else(|| PathBuf::from(".cynapse/tasks"));
        let _ = std::fs::create_dir_all(&dir);
        dir.join(format!("{}.log", task_id))
    }

    /// Appends a timestamped line to ~/.cynapse/tasks/<id>.log
    pub fn append_task_log(task_id: usize, event: &str) {
        let path = Self::task_log_path(task_id);
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&path) {
            let ts = Self::now_ms();
            let _ = writeln!(file, "[{}] {}", ts, event);
        }
    }
}

/// Runs pre-flight verification on modified code files.
/// If `file_path` ends with `.rs` and Cargo.toml is in parent hierarchy, runs `cargo check --message-format=short`.
pub fn run_preflight_verification(file_path: &str) -> Option<String> {
    let path = Path::new(file_path);
    if path.extension().and_then(|s| s.to_str()) == Some("rs") {
        // Find ancestor with Cargo.toml
        let mut curr = path.parent();
        while let Some(dir) = curr {
            if dir.join("Cargo.toml").exists() {
                let out = Command::new("cargo")
                    .arg("check")
                    .arg("--message-format=short")
                    .current_dir(dir)
                    .output();
                if let Ok(output) = out {
                    if !output.status.success() {
                        let err_text = String::from_utf8_lossy(&output.stderr);
                        let short_err = err_text
                            .lines()
                            .take(15)
                            .collect::<Vec<_>>()
                            .join("\n");
                        return Some(format!("\n[Auto-Verification Error - cargo check failed]:\n{}", short_err));
                    }
                }
                break;
            }
            curr = dir.parent();
        }
    }
    None
}

/// Result returned from executing a subagent offline loop.
#[derive(Debug, Clone)]
pub struct SubagentLoopResult {
    pub final_text: String,
    pub step_count: usize,
    pub tools_executed: usize,
    pub completed: bool,
    pub cancelled: bool,
}

/// Core autonomous subagent loop.
///
/// Runs up to `max_steps` iterations:
/// 1. Queries the model using `query_fn`.
/// 2. Checks cancellation flag before & after generation and tool execution.
/// 3. Parses tool calls with `validate_gbnf_tool_calls`.
/// 4. Executes allowed tools under `BotProfile` via `execute_tool_with_profile`.
/// 5. Validates with `LoopGuard` to prevent repetitive loops.
/// 6. Re-prompts until the subagent completes or exhausts `max_steps`.
pub async fn run_subagent_loop<F, Fut, A, AFut>(
    task: &str,
    profile: &crate::bots::BotProfile,
    system_persona: &str,
    max_steps: usize,
    cancel_flag: Option<Arc<AtomicBool>>,
    mut query_fn: F,
    mut approval_fn: Option<A>,
) -> SubagentLoopResult
where
    F: FnMut(String, String) -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<String>>,
    A: FnMut(String, String, Option<String>) -> AFut,
    AFut: std::future::Future<Output = bool>,
{
    let mut history_messages: Vec<(String, String)> = Vec::new(); // (role, content)
    let mut loop_guard = crate::offline_agent::LoopGuard::default();
    let mut step_count = 0;
    let mut tools_executed = 0;
    let mut final_response = String::new();

    let system_prompt = crate::offline_agent::compile_zone_a_prefix(system_persona);

    loop {
        if let Some(ref cf) = cancel_flag {
            if cf.load(Ordering::SeqCst) {
                return SubagentLoopResult {
                    final_text: "Task cancelled before completion.".to_string(),
                    step_count,
                    tools_executed,
                    completed: false,
                    cancelled: true,
                };
            }
        }

        if step_count >= max_steps {
            // Finalization notice when steps exhausted
            let notice = format!("TOOL BUDGET EXHAUSTED ({} steps). Provide summary.", max_steps);
            let history_context: String = history_messages
                .iter()
                .rev()
                .take(6)
                .cloned()
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .map(|(role, content)| format!("{}: {}", role.to_uppercase(), content))
                .collect::<Vec<_>>()
                .join("\n\n");

            let prompt = crate::offline_agent::compile_zone_b_tail("", &history_context, task, Some(&notice));
            if let Ok(res) = query_fn(prompt, system_prompt.clone()).await {
                final_response = res;
            }
            break;
        }

        let history_context: String = history_messages
            .iter()
            .rev()
            .take(6)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .map(|(role, content)| format!("{}: {}", role.to_uppercase(), content))
            .collect::<Vec<_>>()
            .join("\n\n");

        let prompt = crate::offline_agent::compile_zone_b_tail("", &history_context, task, None);
        let model_out = match query_fn(prompt, system_prompt.clone()).await {
            Ok(out) => out,
            Err(e) => {
                final_response = format!("Model query error: {}", e);
                break;
            }
        };

        if let Some(ref cf) = cancel_flag {
            if cf.load(Ordering::SeqCst) {
                return SubagentLoopResult {
                    final_text: "Task cancelled during execution.".to_string(),
                    step_count,
                    tools_executed,
                    completed: false,
                    cancelled: true,
                };
            }
        }

        // Check for GBNF tool calls
        let detected = crate::offline_agent::validate_gbnf_tool_calls(&model_out);
        match detected {
            Ok(tool_calls) if !tool_calls.is_empty() => {
                step_count += 1;
                history_messages.push(("assistant".into(), model_out.clone()));

                let mut batch_results: Vec<String> = Vec::new();
                let mut all_vetoed = true;

                for call in &tool_calls {
                    match loop_guard.check(call) {
                        Ok(()) => {
                            all_vetoed = false;
                            loop_guard.record_call(call);

                            let (arg1, arg2) = extract_tool_args(call);

                            let mut output_str = if !profile.is_tool_allowed(&call.name) {
                                format!("Tool error: Permission denied: tool '{}' is not permitted by bot profile '@{}'", call.name, profile.slug)
                            } else if profile.requires_approval(&call.name) {
                                if let Some(ref mut ask) = approval_fn {
                                    let approved = ask(call.name.clone(), arg1.clone(), arg2.clone()).await;
                                    if approved {
                                        match crate::execute_tool_with_profile_approved(&call.name, &arg1, arg2.as_deref(), profile) {
                                            Ok(out) => {
                                                tools_executed += 1;
                                                out
                                            }
                                            Err(err) => format!("Tool error: {}", err),
                                        }
                                    } else {
                                        "Tool error: [tool denied by user]".to_string()
                                    }
                                } else {
                                    format!("Tool error: Approval required: tool '{}' requires interactive confirmation, but interactive approval is unavailable inline", call.name)
                                }
                            } else {
                                match crate::execute_tool_with_profile(&call.name, &arg1, arg2.as_deref(), profile) {
                                    Ok(out) => {
                                        tools_executed += 1;
                                        out
                                    }
                                    Err(err) => format!("Tool error: {}", err),
                                }
                            };

                            // Auto-Verification for self-correcting code generation
                            if call.name == "write_file" && !output_str.starts_with("Tool error:") {
                                if let Some(verification_err) = run_preflight_verification(&arg1) {
                                    output_str.push_str(&verification_err);
                                }
                            }

                            loop_guard.record_outcome(call, &output_str);
                            let mut res_str = format!("[{}]\n{}", call.name, output_str);
                            if let Some(advisory) = loop_guard.advisory_notice(call) {
                                res_str.push_str(&format!("\n\n{}", advisory));
                            }
                            batch_results.push(res_str);
                        }
                        Err(veto) => {
                            loop_guard.record_veto(call);
                            batch_results.push(format!("[{}] Vetoed by LoopGuard: {}", call.name, veto));
                        }
                    }
                }

                let combined = batch_results.join("\n\n");
                history_messages.push(("system".into(), format!("Tool Results (Step {}/{}):\n{}", step_count, max_steps, combined)));

                if all_vetoed {
                    final_response = format!("Subagent stopped due to loop veto: {}", combined);
                    break;
                }
            }
            _ => {
                // No tools detected -> model delivered plain text final response
                final_response = model_out;
                break;
            }
        }
    }

    SubagentLoopResult {
        final_text: final_response,
        step_count,
        tools_executed,
        completed: true,
        cancelled: false,
    }
}

/// Helper to extract arg1 and arg2 from a ToolCall arguments json.
fn extract_tool_args(call: &crate::offline_agent::ToolCall) -> (String, Option<String>) {
    let args = &call.arguments;
    match call.name.as_str() {
        "read_file" => {
            let p = args.get("path")
                .or_else(|| args.get("file"))
                .or_else(|| args.get("filename"))
                .or_else(|| args.get("arg1"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            (p.to_string(), None)
        }
        "write_file" => {
            let p = args.get("path")
                .or_else(|| args.get("file"))
                .or_else(|| args.get("filename"))
                .or_else(|| args.get("arg1"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let c = args.get("content")
                .or_else(|| args.get("text"))
                .or_else(|| args.get("body"))
                .or_else(|| args.get("arg2"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            (p.to_string(), c)
        }
        "grep" => {
            let pat = args.get("pattern")
                .or_else(|| args.get("query"))
                .or_else(|| args.get("regex"))
                .or_else(|| args.get("arg1"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let d = args.get("dir")
                .or_else(|| args.get("path"))
                .or_else(|| args.get("arg2"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            (pat.to_string(), d)
        }
        "execute_command" => {
            let cmd = args.get("command")
                .or_else(|| args.get("cmd"))
                .or_else(|| args.get("script"))
                .or_else(|| args.get("arg1"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            (cmd.to_string(), None)
        }
        _ => {
            let a1 = args.get("arg1").and_then(|v| v.as_str()).unwrap_or("");
            let a2 = args.get("arg2").and_then(|v| v.as_str()).map(|s| s.to_string());
            (a1.to_string(), a2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subagent_task_lifecycle_and_status() {
        let manager = SubagentManager::new(2);
        let (id, _cancel) = manager.create_task("Inspect src/lib.rs", "coder", false);
        assert_eq!(id, 1);
        assert_eq!(manager.get_status(id), Some(SubagentStatus::Pending));

        manager.update_status(id, SubagentStatus::Running, None);
        assert_eq!(manager.get_status(id), Some(SubagentStatus::Running));
        assert_eq!(manager.running_count(), 1);

        manager.update_status(id, SubagentStatus::Done, Some("Found 4 functions.".into()));
        assert_eq!(manager.get_status(id), Some(SubagentStatus::Done));
        assert_eq!(manager.running_count(), 0);

        let task = manager.get_task(id).unwrap();
        assert_eq!(task.result.as_deref(), Some("Found 4 functions."));
        assert!(task.finished_at_epoch_ms.is_some());
    }

    #[test]
    fn test_subagent_cancellation() {
        let manager = SubagentManager::new(2);
        let (id, cancel_flag) = manager.create_task("Long search", "researcher", true);
        assert!(!cancel_flag.load(Ordering::SeqCst));
        assert!(!manager.is_cancelled(id));

        assert!(manager.cancel_task(id));
        assert!(cancel_flag.load(Ordering::SeqCst));
        assert!(manager.is_cancelled(id));
        assert_eq!(manager.get_status(id), Some(SubagentStatus::Cancelled));
    }

    #[tokio::test]
    async fn test_subagent_semaphore_concurrency_cap() {
        let manager = SubagentManager::new(2);
        let sem = manager.semaphore();

        let p1 = sem.try_acquire();
        assert!(p1.is_ok());
        let p2 = sem.try_acquire();
        assert!(p2.is_ok());

        // Capacity is 2 -> third acquisition fails immediately
        let p3 = sem.try_acquire();
        assert!(p3.is_err());

        drop(p1);
        let p3_retry = sem.try_acquire();
        assert!(p3_retry.is_ok());
    }

    #[tokio::test]
    async fn test_subagent_loop_executes_tools_under_profile() {
        let temp = tempfile::tempdir().unwrap();
        let target_file = temp.path().join("sub_test.txt");
        std::fs::write(&target_file, "hello from subagent file").unwrap();

        let profile = crate::bots::BotProfile {
            slug: "tester".into(),
            display_name: "Tester".into(),
            description: "".into(),
            persona_file: None,
            model_preset: None,
            tools_allow: vec!["read_file".into()],
            tools_deny: vec![],
            tools_ask: vec![],
            workspace_restrict: None,
            max_concurrent: None,
        };

        let file_path_str = target_file.to_string_lossy().to_string();
        let mut turn = 0;
        let query_mock = move |_prompt: String, _sys: String| {
            turn += 1;
            let path_clone = file_path_str.clone();
            async move {
                if turn == 1 {
                    // Turn 1: Emit GBNF tool call to read_file
                    Ok(format!(
                        "Let me read the file.\n```json\n{{\"name\": \"read_file\", \"arguments\": {{\"path\": \"{}\"}}}}\n```",
                        path_clone
                    ))
                } else {
                    // Turn 2: Emit plain text answer
                    Ok("The file content is 'hello from subagent file'.".to_string())
                }
            }
        };

        let res = run_subagent_loop(
            "Read sub_test.txt and report",
            &profile,
            "You are a test bot",
            5,
            None,
            query_mock,
            None::<fn(String, String, Option<String>) -> std::future::Ready<bool>>,
        )
        .await;

        assert_eq!(res.step_count, 1);
        assert_eq!(res.tools_executed, 1);
        assert!(res.completed);
        assert!(!res.cancelled);
        assert!(res.final_text.contains("hello from subagent file"));
    }

    #[tokio::test]
    async fn test_subagent_loop_cancellation_during_run() {
        let profile = crate::bots::BotProfile::default();
        let cancel_flag = Arc::new(AtomicBool::new(false));
        let flag_clone = Arc::clone(&cancel_flag);

        let query_mock = move |_prompt: String, _sys: String| {
            flag_clone.store(true, Ordering::SeqCst);
            async move {
                Ok("```json\n{\"name\": \"execute_command\", \"arguments\": {\"command\": \"ls\"}}\n```".into())
            }
        };

        let res = run_subagent_loop(
            "Long running task",
            &profile,
            "Persona",
            5,
            Some(cancel_flag),
            query_mock,
            None::<fn(String, String, Option<String>) -> std::future::Ready<bool>>,
        )
        .await;

        assert!(res.cancelled);
        assert!(!res.completed);
    }

    #[tokio::test]
    async fn test_subagent_loop_approval_granted() {
        let temp = tempfile::tempdir().unwrap();
        let target_file = temp.path().join("sub_write.txt");

        let profile = crate::bots::BotProfile {
            slug: "coder".into(),
            display_name: "Coder".into(),
            description: "".into(),
            persona_file: None,
            model_preset: None,
            tools_allow: vec!["write_file".into()],
            tools_deny: vec![],
            tools_ask: vec!["write_file".into()],
            workspace_restrict: None,
            max_concurrent: None,
        };

        let file_path_str = target_file.to_string_lossy().to_string();
        let mut turn = 0;
        let query_mock = move |_prompt: String, _sys: String| {
            turn += 1;
            let path_clone = file_path_str.clone();
            async move {
                if turn == 1 {
                    Ok(format!(
                        "Writing file.\n```json\n{{\"name\": \"write_file\", \"arguments\": {{\"path\": \"{}\", \"content\": \"approved content\"}}}}\n```",
                        path_clone
                    ))
                } else {
                    Ok("File written successfully.".to_string())
                }
            }
        };

        let approval_mock = |_tool: String, _arg1: String, _arg2: Option<String>| async {
            true // User approves
        };

        let res = run_subagent_loop(
            "Write file",
            &profile,
            "Persona",
            5,
            None,
            query_mock,
            Some(approval_mock),
        )
        .await;

        assert_eq!(res.step_count, 1);
        assert_eq!(res.tools_executed, 1);
        assert!(res.completed);
        assert_eq!(std::fs::read_to_string(&target_file).unwrap(), "approved content");
    }

    #[tokio::test]
    async fn test_subagent_loop_approval_denied() {
        let temp = tempfile::tempdir().unwrap();
        let target_file = temp.path().join("sub_deny.txt");

        let profile = crate::bots::BotProfile {
            slug: "coder".into(),
            display_name: "Coder".into(),
            description: "".into(),
            persona_file: None,
            model_preset: None,
            tools_allow: vec!["write_file".into()],
            tools_deny: vec![],
            tools_ask: vec!["write_file".into()],
            workspace_restrict: None,
            max_concurrent: None,
        };

        let file_path_str = target_file.to_string_lossy().to_string();
        let mut turn = 0;
        let query_mock = move |prompt: String, _sys: String| {
            turn += 1;
            let path_clone = file_path_str.clone();
            async move {
                if turn == 1 {
                    Ok(format!(
                        "Writing file.\n```json\n{{\"name\": \"write_file\", \"arguments\": {{\"path\": \"{}\", \"content\": \"forbidden content\"}}}}\n```",
                        path_clone
                    ))
                } else {
                    assert!(prompt.contains("[tool denied by user]"));
                    Ok("Aborting because user denied tool.".to_string())
                }
            }
        };

        let approval_mock = |_tool: String, _arg1: String, _arg2: Option<String>| async {
            false // User denies
        };

        let res = run_subagent_loop(
            "Write file",
            &profile,
            "Persona",
            5,
            None,
            query_mock,
            Some(approval_mock),
        )
        .await;

        assert_eq!(res.step_count, 1);
        assert_eq!(res.tools_executed, 0);
        assert!(res.completed);
        assert!(!target_file.exists());
        assert!(res.final_text.contains("Aborting because user denied tool"));
    }

    #[tokio::test]
    async fn test_subagent_loop_approval_inline_fails_closed() {
        let temp = tempfile::tempdir().unwrap();
        let target_file = temp.path().join("sub_inline.txt");

        let profile = crate::bots::BotProfile {
            slug: "coder".into(),
            display_name: "Coder".into(),
            description: "".into(),
            persona_file: None,
            model_preset: None,
            tools_allow: vec!["write_file".into()],
            tools_deny: vec![],
            tools_ask: vec!["write_file".into()],
            workspace_restrict: None,
            max_concurrent: None,
        };

        let file_path_str = target_file.to_string_lossy().to_string();
        let mut turn = 0;
        let query_mock = move |prompt: String, _sys: String| {
            turn += 1;
            let path_clone = file_path_str.clone();
            async move {
                if turn == 1 {
                    Ok(format!(
                        "Writing file.\n```json\n{{\"name\": \"write_file\", \"arguments\": {{\"path\": \"{}\", \"content\": \"data\"}}}}\n```",
                        path_clone
                    ))
                } else {
                    assert!(prompt.contains("interactive approval is unavailable inline"));
                    Ok("Inline execution cannot prompt user.".to_string())
                }
            }
        };

        let res = run_subagent_loop(
            "Write file inline",
            &profile,
            "Persona",
            5,
            None,
            query_mock,
            None::<fn(String, String, Option<String>) -> std::future::Ready<bool>>,
        )
        .await;

        assert_eq!(res.step_count, 1);
        assert_eq!(res.tools_executed, 0);
        assert!(res.completed);
        assert!(!target_file.exists());
    }

    #[test]
    fn test_subagent_stalled_task_auto_cancellation() {
        let manager = SubagentManager::new(2);
        let (id1, _cancel1) = manager.create_task("Fast task", "coder", false);
        let (id2, cancel2) = manager.create_task("Stalled task", "coder", false);

        manager.update_status(id1, SubagentStatus::Done, Some("completed".into()));
        manager.update_status(id2, SubagentStatus::Running, None);

        // Manually adjust started_at_epoch_ms of task 2 to simulate stall
        if let Ok(mut tasks) = manager.tasks.write() {
            if let Some(t) = tasks.get_mut(&id2) {
                t.started_at_epoch_ms = SubagentManager::now_ms().saturating_sub(150_000); // 150s ago
            }
        }

        let cancelled = manager.check_and_cancel_stalled_tasks(120_000); // 120s threshold
        assert_eq!(cancelled, vec![id2]);
        assert!(cancel2.load(Ordering::SeqCst));
        assert_eq!(manager.get_status(id2), Some(SubagentStatus::Cancelled));
    }

    #[test]
    fn test_subagent_task_logger() {
        let manager = SubagentManager::new(2);
        let (id, _cancel) = manager.create_task("Test log task", "coder", false);
        manager.update_status(id, SubagentStatus::Running, None);
        manager.update_status(id, SubagentStatus::Done, Some("Finished successfully".into()));

        let log_path = SubagentManager::task_log_path(id);
        assert!(log_path.exists());
        let content = std::fs::read_to_string(&log_path).expect("read log");
        assert!(content.contains("SPAWN: bot=@coder"));
        assert!(content.contains("STATUS: Running"));
        assert!(content.contains("STATUS: Done"));
    }
}
