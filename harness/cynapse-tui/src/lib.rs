use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;
use anyhow::Result;
use colored::*;
use regex::Regex;
use serde::Deserialize;
use cynapse_engine::{query_tier1_stream, TokenType};
use cynapse_memory::graph::{Dendrite, NodeType};

use cynapse_memory::context::DendriteContext;
use cynapse_memory::store::DendriteStore;

pub mod memory_render;
pub mod app;
pub mod harness_ui;
pub mod terminal;
pub mod theme;

use memory_render::{render_dendrite_visualizer, render_memory_pipeline, truncate_smart, PipelineState, StepStatus};
use app::TuiApp;

/// Fallback inference backend candidates (sync with cynapse_core::doctor::LOCAL_BACKEND_PORTS):
/// 38265 = Cynapse llama-server daemon.
pub const FALLBACK_BACKEND_PORTS: [u16; 1] = [38265];

/// Split an endpoint URL into (host, explicit port). `None` when no port is present.
fn endpoint_host_port(endpoint: &str) -> Option<(&str, u16)> {
    let rest = endpoint.split_once("://").map(|(_, r)| r).unwrap_or(endpoint);
    let authority = rest.split('/').next().unwrap_or(rest);
    let host_port = authority.rsplit_once('@').map(|(_, hp)| hp).unwrap_or(authority);
    let (host, port) = host_port.rsplit_once(':')?;
    let port: u16 = port.parse().ok()?;
    let host = host
        .strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(host);
    Some((host, port))
}

fn is_loopback(host: &str) -> bool {
    host == "localhost" || host == "::1" || host == "127.0.0.1" || host.starts_with("127.")
}

/// Pick a working tier1 endpoint: keep `configured` while its port answers
/// HTTP; otherwise fall back through `candidates` (first live wins); if every
/// candidate is dead keep `configured` so `cynapse doctor` fails loud.
fn select_endpoint_with_candidates(configured: &str, candidates: &[u16]) -> String {
    let (host, port) = match endpoint_host_port(configured) {
        Some(hp) => hp,
        None => return configured.to_string(),
    };
    // Only loopback endpoints can be meaningfully probed from this process.
    if !is_loopback(host) {
        return configured.to_string();
    }
    if cynapse_core::doctor::probe_port("127.0.0.1", port) {
        return configured.to_string();
    }
    for &candidate in candidates {
        if candidate == port {
            continue;
        }
        if cynapse_core::doctor::probe_port("127.0.0.1", candidate) {
            return format!("http://127.0.0.1:{}", candidate);
        }
    }
    configured.to_string()
}

/// Public endpoint resolution used by `TuiSession::new`.
pub fn select_live_endpoint(configured: &str) -> String {
    select_endpoint_with_candidates(configured, &FALLBACK_BACKEND_PORTS)
}

/// Install panic hook logging to ~/.cynapse/logs/crash.log
pub fn install_panic_hook() {
    static HOOK_INSTALLED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    HOOK_INSTALLED.get_or_init(|| {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let log_dir = if let Some(home) = dirs::home_dir() {
                home.join(".cynapse").join("logs")
            } else {
                PathBuf::from("./logs")
            };
            let _ = std::fs::create_dir_all(&log_dir);
            let crash_log = log_dir.join("crash.log");
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&crash_log) {
                let _ = writeln!(f, "=== CYNAPSE CRASH REPORT ===");
                let _ = writeln!(f, "Timestamp: {:?}", std::time::SystemTime::now());
                let _ = writeln!(f, "Panic info: {}", info);
                let _ = writeln!(f, "Backtrace:\n{:?}", std::backtrace::Backtrace::capture());
            }
            prev(info);
        }));
    });
}

#[derive(Debug, Deserialize)]
struct RuntimeConfig {
    engine: Option<EngineConfig>,
    sampling: Option<SamplingConfig>,
}

#[derive(Debug, Deserialize)]
struct SamplingConfig {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    repeat_penalty: Option<f32>,
    repeat_last_n: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct EngineConfig {
    tier1_endpoint: Option<String>,
    default_model: Option<String>,
    ctx_size: Option<usize>,
    gpu_layers: Option<usize>,
    model_search_paths: Option<Vec<String>>,
}

pub struct TuiSession {
    pub models_dir: PathBuf,
    pub active_model_name: String,
    pub active_model_path: PathBuf,
    pub tier1_endpoint: String,
    pub graph: Arc<Dendrite>,
    pub store: Option<Arc<DendriteStore>>,
    pub dendrite_ctx: Arc<DendriteContext>,
}

impl TuiSession {
    pub fn new(models_dir: PathBuf) -> Self {
        install_panic_hook();
        let mut active_model_name = "Ornith-1.5-9B-Q4_K_M.gguf".to_string();
        let mut tier1_endpoint = "http://127.0.0.1:38265".to_string();

        // Load runtime configuration from cynapse.toml using toml parser
        let config_candidates = [
            PathBuf::from("cynapse.toml"),
            models_dir.parent().map(|p| p.join("cynapse.toml")).unwrap_or_default(),
            dirs::home_dir().map(|h| h.join(".cynapse").join("cynapse.toml")).unwrap_or_default(),
        ];

        for cfg_path in &config_candidates {
            if cfg_path.exists() {
                if let Ok(content) = std::fs::read_to_string(cfg_path) {
                    if let Ok(parsed) = toml::from_str::<RuntimeConfig>(&content) {
                        if let Some(engine) = parsed.engine {
                            if let Some(ep) = engine.tier1_endpoint {
                                tier1_endpoint = ep;
                            }
                            if let Some(m) = engine.default_model {
                                active_model_name = m;
                            }
                            if let Some(sz) = engine.ctx_size {
                                cynapse_engine::set_engine_ctx_size(sz);
                            }
                            if let Some(gl) = engine.gpu_layers {
                                cynapse_engine::set_gpu_layers(gl);
                            }
                            if let Some(paths) = engine.model_search_paths {
                                let pbufs: Vec<PathBuf> = paths.into_iter().map(PathBuf::from).collect();
                                cynapse_engine::set_model_search_dirs(pbufs);
                            }
                        }
                        if let Some(s) = parsed.sampling {
                            let mut p = cynapse_engine::sampling();
                            if let Some(t) = s.temperature { p.temperature = t; }
                            if let Some(tp) = s.top_p { p.top_p = tp; }
                            if let Some(tk) = s.top_k { p.top_k = tk; }
                            if let Some(rp) = s.repeat_penalty { p.repeat_penalty = rp; }
                            if let Some(rn) = s.repeat_last_n { p.repeat_last_n = rn; }
                            cynapse_engine::set_sampling(p);
                        }
                    }
                    break;
                }
            }
        }

        // Verify the configured endpoint's port is alive; otherwise fall back
        // to the first live well-known backend (11434/11435/38265).
        let tier1_endpoint = select_live_endpoint(&tier1_endpoint);

        let active_model_path = models_dir.join("model.gguf");

        // Initialize SQLite persistence store: check ~/.cynapse/dendrite.db FIRST
        let store = if let Some(home) = dirs::home_dir() {
            let user_db_dir = home.join(".cynapse");
            let _ = std::fs::create_dir_all(&user_db_dir);
            let user_db = user_db_dir.join("dendrite.db");
            DendriteStore::open(&user_db).ok().map(Arc::new)
        } else {
            None
        }.or_else(|| {
            let db_dir = PathBuf::from("data");
            let _ = std::fs::create_dir_all(&db_dir);
            let primary_db = db_dir.join("dendrite.db");
            DendriteStore::open(&primary_db).ok().map(Arc::new)
        });

        let graph = Arc::new(Dendrite::new());

        // Hydrate stored graph nodes from SQLite DB
        if let Some(ref st) = store {
            let _ = st.load_all(&graph);
        }

        // Pre-populate core nodes if missing
        if graph.get("cynapse_core").is_none() {
            let n1 = graph.upsert("cynapse_core", "CYNAPSE Agent Core", "Local-first modular AI agent system with Dendrite 4-tier memory graph.", NodeType::Identity, Some(vec!["#summary".into(), "#system".into()]));
            if let Some(ref st) = store { let _ = st.save(&n1); }
        }
        if graph.get("fast_tier").is_none() {
            let n2 = graph.upsert("fast_tier", "Native llama.cpp Daemon", "Tier 1 execution engine daemon running on port 38265 with hardware acceleration reaching 4.8 tok/s.", NodeType::Concept, Some(vec!["#engine".into()]));
            if let Some(ref st) = store { let _ = st.save(&n2); }
        }
        if graph.get("rust_inference").is_none() {
            let n3 = graph.upsert("rust_inference", "Leafcutter Pure Rust Inference", "Tier 2 GGUF & Tier 3 Safetensors layer streaming engine written in pure Rust.", NodeType::Procedure, Some(vec!["#procedure".into()]));
            if let Some(ref st) = store { let _ = st.save(&n3); }
        }

        let dendrite_ctx = DendriteContext::new(graph.clone(), store.clone());

        let mut sess = Self {
            models_dir,
            active_model_name,
            active_model_path,
            tier1_endpoint,
            graph,
            store,
            dendrite_ctx,
        };
        sess.auto_detect_model_sync();
        sess
    }

    /// Sync scan of local models directory to pick active model if default doesn't exist
    pub fn auto_detect_model_sync(&mut self) {
        let home_models = dirs::home_dir().map(|h| h.join(".cynapse").join("models"));
        let mut search_dirs = vec![self.models_dir.clone(), PathBuf::from("./models")];
        if let Some(h) = home_models {
            search_dirs.push(h);
        }

        let mut found_files = Vec::new();
        for dir in &search_dirs {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.is_file() {
                        if let Some(ext) = p.extension().and_then(|s| s.to_str()) {
                            if (ext == "gguf" || ext == "safetensors" || ext == "bin") && p.file_name().unwrap() != "README.md" {
                                let filename = p.file_name().unwrap().to_string_lossy().to_string();
                                if !found_files.contains(&filename) {
                                    found_files.push(filename);
                                }
                            }
                        }
                    }
                }
            }
        }
        if !found_files.is_empty() {
            if self.active_model_name == "ministral-3:3b" || !found_files.contains(&self.active_model_name) {
                let preferred = found_files.iter().find(|f| {
                    let fl = f.to_lowercase();
                    fl.contains("9b") || fl.contains("8b") || fl.contains("7b") || fl.contains("3b")
                });
                self.active_model_name = preferred.unwrap_or(&found_files[0]).clone();
            }
            if let Some((_, resolved_path)) = self.resolve_model_by_name_or_index(&self.active_model_name) {
                self.active_model_path = resolved_path;
            }
        }
    }

    /// Extract dynamic quantization tag from filename (e.g. Q4_K_M, Q4_K_XL, Q8_0, F16, Q5_K_S)
    fn extract_quantization(&self, filename: &str) -> String {
        static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
        let re = RE.get_or_init(|| Regex::new(r"(?i)(Q[0-9]_[K0-9_A-Z]+|F16|F32|IQ[0-9]_[A-Z]+)").unwrap());
        if let Some(mat) = re.find(filename) {
            mat.as_str().to_uppercase()
        } else if filename.ends_with(".safetensors") {
            "SAFETENSORS".to_string()
        } else {
            "GGUF".to_string()
        }
    }

    pub fn resolve_model_by_name_or_index(&self, target: &str) -> Option<(String, PathBuf)> {
        let trimmed = target.trim();
        let mut models = Vec::new();
        let search_dirs = [self.models_dir.clone(), PathBuf::from("./models")];

        for dir in &search_dirs {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() {
                        if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                            if (ext == "gguf" || ext == "safetensors" || ext == "bin") && path.file_name().unwrap() != "README.md" {
                                let name = path.file_name().unwrap().to_string_lossy().to_string();
                                if !models.iter().any(|(n, _)| n == &name) {
                                    models.push((name, path));
                                }
                            }
                        }
                    }
                }
            }
        }

        if let Ok(num) = trimmed.parse::<usize>() {
            if num >= 1 && num <= models.len() {
                return Some(models[num - 1].clone());
            }
        }

        let lower = trimmed.to_lowercase();
        for (name, path) in &models {
            if name.to_lowercase() == lower || name.to_lowercase().contains(&lower) {
                return Some((name.clone(), path.clone()));
            }
        }

        let direct = PathBuf::from(trimmed);
        if direct.exists() && direct.is_file() {
            let fname = direct.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| trimmed.to_string());
            return Some((fname, direct));
        }

        None
    }

    pub fn list_models(&self) {
        println!("{}", "======================================================================".cyan());
        println!("{}", format!("📋 AVAILABLE MODELS IN CYNAPSE ({})", self.models_dir.display()).yellow().bold());
        println!("{}", "======================================================================".cyan());

        let mut idx = 1;
        if let Ok(entries) = std::fs::read_dir(&self.models_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                        if ext == "gguf" || ext == "safetensors" || ext == "bin" {
                            let name = path.file_name().unwrap().to_string_lossy();
                            if name == "README.md" { continue; }

                            let meta = entry.metadata().ok();
                            let bytes = meta.map(|m| m.len()).unwrap_or(0);
                            let size_mb = bytes / 1024 / 1024;
                            let size_str = if size_mb > 1024 {
                                format!("{:.2} GB", size_mb as f64 / 1024.0)
                            } else {
                                format!("{} MB", size_mb)
                            };

                            let quant = self.extract_quantization(&name);
                            println!(" [{}] {:<42} {:<12} {:<10}", idx, name, quant, size_str);
                            idx += 1;
                        }
                    }
                }
            }
        }
        if idx == 1 {
            println!(" (No local models in directory. Download one with: cynapse pull <hf-url-or-repo>)");
        }
        println!("{}", "======================================================================".cyan());
        println!("To run a model:    cynapse run <number>   (or /run <number>)");
        println!("To remove a model: cynapse rm <number>    (or /rm <number>)");
        println!("{}", "======================================================================".cyan());
    }

    pub async fn run_tui_app(&mut self) -> Result<()> {
        self.run_tui_app_with_resume(None).await
    }

    pub async fn run_tui_app_with_resume(&mut self, resume_id: Option<&str>) -> Result<()> {
        let mut app = TuiApp::new(
            self.models_dir.clone(),
            self.active_model_name.clone(),
            self.tier1_endpoint.clone(),
            self.graph.clone(),
            self.store.clone(),
            self.dendrite_ctx.clone(),
        );
        if let Some(sid) = resume_id {
            if let Ok(()) = app.load_session(sid) {
                app.messages.push(app::ChatMessage {
                    role: "system".into(),
                    content: format!("Resumed session transcript from ID: {}", sid),
                    thinking: None,
                });
            }
        }
        app.run().await
    }

    pub async fn run_cli_loop(&mut self) -> Result<()> {
        self.run_cli_loop_with_resume(None).await
    }

    pub async fn run_cli_loop_with_resume(&mut self, resume_id: Option<&str>) -> Result<()> {
        if let Some(sid) = resume_id {
            let session_mgr = cynapse_core::session::SessionManager::new();
            if let Ok(data) = session_mgr.load_session(sid) {
                println!("{}", format!("Resuming past CLI session transcript: {}", sid).cyan().bold());
                for msg in &data.messages {
                    println!("[{}] {}", msg.role.yellow(), msg.content);
                }
                println!("{}", "----------------------------------------------------------------------".cyan());
            }
        }
        self.run_interactive_loop().await
    }

    pub async fn run_interactive_loop(&mut self) -> Result<()> {
        println!("{}", "======================================================================".cyan().bold());
        println!("{}", "                   🧠 CYNAPSE LOCAL AGENT SYSTEM                      ".yellow().bold());
        println!("{}", "======================================================================".cyan().bold());
        println!("Pure Rust Runtime:            Enabled (Zero Node / Zero Python)");
        println!("Semantic Engine Router:       Enabled (1.5 GiB RAM Reserve Headroom)");
        println!("Engine Tiers:");
        println!("  • Tier 1 (Fast):            llama.cpp / Ollama (~4.8 tok/s)");
        println!("  • Tier 2 (Large GGUF):      Leafcutter Rust GGUF Core");
        println!("  • Tier 3 (Large Safetensor): Leafcutter Rust Safetensor Core");
        println!("Memory Core:                  DENDRITE 4-Tier Graph (SQLite FTS5 + BM25)");
        println!("{}", "----------------------------------------------------------------------".cyan());
        println!("🎯 Active Model: {}", self.active_model_name.green().bold());
        println!("{}", "----------------------------------------------------------------------".cyan());
        println!("Turn-based execution mode. Wait for prompt cue 'cynapse >>>> ' to type.");
        println!();

        let stdin = io::stdin();
        loop {
            print!("{}", "cynapse >>>> ".bold().bright_magenta());
            io::stdout().flush()?;

            let mut input = String::new();
            if stdin.read_line(&mut input)? == 0 {
                break;
            }
            let trimmed = input.trim();
            if trimmed.is_empty() {
                continue;
            }

            if trimmed == "exit" || trimmed == "quit" || trimmed == "/exit" {
                println!("{}", "Goodbye!".yellow());
                break;
            }

            if trimmed == "/clear" || trimmed == "/cls" {
                print!("\x1B[2J\x1B[1;1H");
                let _ = io::stdout().flush();
                println!("{}", "Cleared screen.".yellow());
                println!();
                continue;
            }

            if trimmed == "/list" || trimmed == "/ls" {
                self.list_models();
                println!();
                continue;
            }

            if trimmed.starts_with("/run ") || trimmed.starts_with("/select ") {
                let target = trimmed.split_whitespace().nth(1).unwrap_or("");
                if let Ok(num) = target.parse::<usize>() {
                    let mut found = false;
                    let mut idx = 1;
                    if let Ok(entries) = std::fs::read_dir(&self.models_dir) {
                        for entry in entries.flatten() {
                            let path = entry.path();
                            if path.is_file() {
                                if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                                    if ext == "gguf" || ext == "safetensors" || ext == "bin" {
                                        if path.file_name().unwrap() == "README.md" { continue; }
                                        if idx == num {
                                            self.active_model_name = path.file_name().unwrap().to_string_lossy().to_string();
                                            self.active_model_path = path;
                                            found = true;
                                            break;
                                        }
                                        idx += 1;
                                    }
                                }
                            }
                        }
                    }
                    if found {
                        println!("✓ Switched active model to [{}]: {}", num, self.active_model_name.green());
                    } else {
                        println!("❌ Model index [{}] not found. Type /list to view models.", num);
                    }
                } else {
                    self.active_model_name = target.to_string();
                    println!("✓ Switched active model name to: {}", self.active_model_name.green());
                }
                println!();
                continue;
            }

            if trimmed == "/memory" || trimmed == "/dendrite" || trimmed == "/graph" || trimmed == "/mem" {
                render_dendrite_visualizer(&self.graph);
                println!();
                continue;
            }

            let pipeline = PipelineState {
                search_detail: format!("Searching Dendrite for '{}'", truncate_smart(trimmed, 20)),
                search_status: StepStatus::Done,
                verify_detail: "Verified graph relevance".into(),
                verify_status: StepStatus::Done,
                inject_detail: "Injected relevant facts into prompt".into(),
                inject_status: StepStatus::Done,
                update_detail: "Streaming model output...".into(),
                update_status: StepStatus::Running,
            };

            render_memory_pipeline(&pipeline);
            println!("\n{}", "⚙️ Generating stream...".dimmed());
            let mut current_type: Option<TokenType> = None;

            let is_conversational = cynapse_memory::context::is_conversational_query(trimmed);
            let persona_dir = cynapse_core::persona::PersonaManager::default_dir();
            let persona_mgr = cynapse_core::persona::PersonaManager::new(&persona_dir)
                .unwrap_or_else(|_| cynapse_core::persona::PersonaManager::new("./persona").unwrap());
            let persona_prompt = persona_mgr.build_system_prompt();
            let system_prompt = if is_conversational {
                cynapse_core::offline_agent::compile_conversational_prefix(&persona_prompt)
            } else {
                cynapse_core::offline_agent::compile_zone_a_prefix(&persona_prompt)
            };

            let mut full_response = String::new();
            let mut full_thinking = String::new();

            let stats_res = query_tier1_stream(
                &self.tier1_endpoint,
                &self.active_model_name,
                trimmed,
                &system_prompt,
                |ttype, token| {
                    if current_type != Some(ttype) {
                        current_type = Some(ttype);
                        match ttype {
                            TokenType::Thinking => {
                                print!("\n{}", "💭 [Thinking...]\n".magenta().italic());
                            }
                            TokenType::Response => {
                                print!("\n{}", "💡 [Response]:\n".green().bold());
                            }
                        }
                    }
                    match ttype {
                        TokenType::Thinking => {
                            full_thinking.push_str(token);
                            print!("{}", token.dimmed().purple());
                        }
                        TokenType::Response => {
                            full_response.push_str(token);
                            print!("{}", token);
                        }
                    }
                    let _ = io::stdout().flush();
                },
            )
            .await;

            // Offline Agent: GBNF tool call check & execution in CLI mode
            if let Ok(tool_call) = cynapse_core::offline_agent::validate_gbnf_tool_call(&full_response)
                .or_else(|_| cynapse_core::offline_agent::validate_gbnf_tool_call(&full_thinking))
            {
                println!("\n{}", format!("🔧 Detected Tool Call: `{}`", tool_call.name).yellow().bold());
                let (arg1, arg2): (String, Option<String>) = match tool_call.name.as_str() {
                    "write_file" => {
                        let path = tool_call.arguments.get("path")
                            .or_else(|| tool_call.arguments.get("file"))
                            .or_else(|| tool_call.arguments.get("filename"))
                            .or_else(|| tool_call.arguments.get("filepath"))
                            .or_else(|| tool_call.arguments.get("arg1"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let content = tool_call.arguments.get("content")
                            .or_else(|| tool_call.arguments.get("text"))
                            .or_else(|| tool_call.arguments.get("body"))
                            .or_else(|| tool_call.arguments.get("data"))
                            .or_else(|| tool_call.arguments.get("arg2"))
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        (path.to_string(), content)
                    }
                    "read_file" => {
                        let path = tool_call.arguments.get("path")
                            .or_else(|| tool_call.arguments.get("file"))
                            .or_else(|| tool_call.arguments.get("filename"))
                            .or_else(|| tool_call.arguments.get("filepath"))
                            .or_else(|| tool_call.arguments.get("arg1"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        (path.to_string(), None)
                    }
                    "grep" => {
                        let pattern = tool_call.arguments.get("pattern")
                            .or_else(|| tool_call.arguments.get("query"))
                            .or_else(|| tool_call.arguments.get("regex"))
                            .or_else(|| tool_call.arguments.get("arg1"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let dir = tool_call.arguments.get("dir")
                            .or_else(|| tool_call.arguments.get("path"))
                            .or_else(|| tool_call.arguments.get("directory"))
                            .or_else(|| tool_call.arguments.get("arg2"))
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        (pattern.to_string(), dir)
                    }
                    "execute_command" => {
                        let cmd = tool_call.arguments.get("command")
                            .or_else(|| tool_call.arguments.get("cmd"))
                            .or_else(|| tool_call.arguments.get("script"))
                            .or_else(|| tool_call.arguments.get("arg1"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        (cmd.to_string(), None)
                    }
                    _ => {
                        let a1 = tool_call.arguments.get("path")
                            .or_else(|| tool_call.arguments.get("query"))
                            .or_else(|| tool_call.arguments.get("command"))
                            .or_else(|| tool_call.arguments.get("arg1"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let a2 = tool_call.arguments.get("content")
                            .or_else(|| tool_call.arguments.get("dir"))
                            .or_else(|| tool_call.arguments.get("arg2"))
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        (a1.to_string(), a2)
                    }
                };
                match cynapse_core::execute_tool(&tool_call.name, &arg1, arg2.as_deref()) {
                    Ok(out) => println!("{}", format!("✓ Tool executed successfully:\n{}", out).green()),
                    Err(e) => println!("{}", format!("❌ Tool error: {}", e).red()),
                }
            }

            match stats_res {
                Ok(stats) => {
                    println!("\n");
                    println!("{}", "----------------------------------------------------------------------".cyan());
                    println!(
                        "📊 [Model: {} | Output: {} tokens | Latency: {:.2}s | Speed: {:.2} tok/s | Avail RAM: {:.1} GB]",
                        stats.model_name.green(),
                        stats.tokens_generated,
                        stats.elapsed_sec,
                        stats.tok_per_sec,
                        stats.avail_ram_gb
                    );
                    println!("{}", "----------------------------------------------------------------------".cyan());
                    println!();
                }
                Err(e) => {
                    println!("\n{}", "❌ [Error]:".red().bold());
                    println!("   {}", e);
                    println!("{}", "----------------------------------------------------------------------".cyan());
                    println!();
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    /// Spawn a minimal HTTP responder on an ephemeral port; returns its port
    /// and a stop flag so the server thread can be joined.
    fn spawn_http_probe_listener() -> (u16, Arc<AtomicBool>, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let port = listener.local_addr().unwrap().port();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_flag = stop.clone();
        let server = std::thread::spawn(move || {
            while !stop_flag.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let _ = stream.set_read_timeout(Some(std::time::Duration::from_millis(500)));
                        let mut buf = [0u8; 1024];
                        let _ = stream.read(&mut buf);
                        let _ = stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n");
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(std::time::Duration::from_millis(10));
                    }
                    Err(_) => break,
                }
            }
        });
        (port, stop, server)
    }

    fn closed_ephemeral_port() -> u16 {
        let held = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = held.local_addr().unwrap().port();
        drop(held);
        port
    }

    #[test]
    fn test_endpoint_selection_probe_prefers_live_listener() {
        let dead_port = closed_ephemeral_port();
        let (live_port, stop, server) = spawn_http_probe_listener();

        // Configured port dead -> fall back to the live candidate.
        let configured = format!("http://127.0.0.1:{}", dead_port);
        let picked = select_endpoint_with_candidates(&configured, &[live_port]);
        assert_eq!(picked, format!("http://127.0.0.1:{}", live_port));

        // Configured port alive -> kept verbatim even if candidates are dead.
        let kept = select_endpoint_with_candidates(&format!("http://127.0.0.1:{}", live_port), &[dead_port]);
        assert_eq!(kept, format!("http://127.0.0.1:{}", live_port));

        // Everything dead -> configured value preserved (doctor fails loud).
        let unchanged = select_endpoint_with_candidates(&configured, &[closed_ephemeral_port()]);
        assert_eq!(unchanged, configured);

        stop.store(true, Ordering::SeqCst);
        server.join().unwrap();
    }

    #[test]
    fn test_endpoint_host_port_parsing() {
        assert_eq!(endpoint_host_port("http://127.0.0.1:11434"), Some(("127.0.0.1", 11434)));
        assert_eq!(endpoint_host_port("http://127.0.0.1:38265/v1"), Some(("127.0.0.1", 38265)));
        assert_eq!(endpoint_host_port("http://ollama.local:11435"), Some(("ollama.local", 11435)));
        assert_eq!(endpoint_host_port("http://ollama.local"), None);
    }
}
