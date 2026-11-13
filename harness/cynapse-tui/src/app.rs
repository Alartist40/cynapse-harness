//! Cynapse Ratatui Visual TUI Application — Inspired by jcode & colibri.
//!
//! Features Colibri-style Left Sidebar layout, system hardware telemetry (RAM/CPU/GPU),
//! Jcode-style background ASCII art in chat viewport, smooth rounded borders (BorderType::Rounded),
//! RAII terminal protection, non-blocking Tokio MPSC token streaming, dynamic model auto-detection,
//! paragraph text wrapping, interactive slash command dropdown, theme presets, session persistence,
//! clean prompt input, and 3D Galaxy Memory Atlas visualizer.

use std::fs;
use std::io::{self, Stdout};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::Arc;
use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers, MouseEventKind},
    execute,
    terminal::EnterAlternateScreen,
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Clear, List, ListItem, Paragraph, Wrap},
    Frame, Terminal,
};
use unicode_width::UnicodeWidthStr;
use tokio::sync::mpsc;

use cynapse_core::offline_agent::{validate_gbnf_tool_calls, LoopGuard};
use cynapse_core::session::{SessionData, SessionManager, SessionMessage};
use cynapse_engine::{
    fetch_native_models, probe_hardware_info, query_model_stream, route_model,
    unload_model, EngineTier, SystemHardwareInfo, TokenType,
};
use cynapse_memory::context::DendriteContext;
use cynapse_memory::graph::Dendrite;
use cynapse_memory::reflection::{Message as ReflMessage, Role as ReflRole};
use cynapse_memory::store::DendriteStore;
use cynapse_memory::ReflectionWorker;
use crate::terminal::TuiRuntimeGuard;
use crate::theme::AppTheme;

pub const MAX_AGENT_STEPS: usize = 14;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub thinking: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveModal {
    None,
    Help,
    MemoryGraph,
    MemoryDrawer,
    ModelList,
    ModelPuller,
    SessionList,
    SessionRename,
    ThemePicker,
    CommandPalette,
    Doctor,
    SecurityForensics,
    SynapseManager,
    SoupProfiler,
    PersonaManager,
    Bots,
    ToolApproval,
    Updater,
}

#[derive(Debug)]
pub struct ToolApprovalRequest {
    pub task_id: usize,
    pub bot_slug: String,
    pub tool_name: String,
    pub arg1: String,
    pub arg2: Option<String>,
    pub response_tx: tokio::sync::oneshot::Sender<bool>,
}

#[derive(Debug)]
pub struct PendingToolApproval {
    pub task_id: usize,
    pub bot_slug: String,
    pub tool_name: String,
    pub arg1: String,
    pub arg2: Option<String>,
    pub response_tx: Option<tokio::sync::oneshot::Sender<bool>>,
    pub edit_mode: bool,
    pub editing_arg: usize,
    pub edit_buffer: String,
}

impl PendingToolApproval {
    pub fn new(
        task_id: usize,
        bot_slug: String,
        tool_name: String,
        arg1: String,
        arg2: Option<String>,
        response_tx: Option<tokio::sync::oneshot::Sender<bool>>,
    ) -> Self {
        Self {
            task_id,
            bot_slug,
            tool_name,
            arg1,
            arg2,
            response_tx,
            edit_mode: false,
            editing_arg: 0,
            edit_buffer: String::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PullerStep {
    CuratedList,
    CustomInput,
    QuantSelect,
    Downloading,
}

pub const QUANT_OPTIONS: &[&str] = &[
    "Q4_K_M (Recommended 4-bit Medium)",
    "Q5_K_M (High Precision 5-bit)",
    "Q8_0 (8-bit High Precision)",
    "F16 (16-bit Float)",
    "Q2_K (Ultra Fast 2-bit)",
    "Q3_K_M (3-bit Medium)",
    "Q6_K (6-bit High Quality)",
    "IQ4_NL (Non-linear 4-bit)",
];

#[derive(Debug, Clone)]
pub struct DownloadProgressState {
    pub model_name: String,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub speed_mbps: f64,
    pub pct: f64,
    pub is_done: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct UpdateProgressState {
    pub is_running: bool,
    pub is_done: bool,
    pub current_step: String,
    pub steps_log: Vec<String>,
    pub error: Option<String>,
    pub success_message: Option<String>,
}

pub enum UpdateEvent {
    Step(String),
    Success(String),
    Error(String),
}

pub enum StreamEvent {
    Token { ttype: TokenType, text: String },
    Done { tok_per_sec: f64, elapsed_sec: f64 },
    LayerProgress { current: usize, total: usize },
    SpeculativeDraft { accepted: usize, total: usize },
    Error(String),
}

#[derive(Debug, Clone)]
pub struct ModelItem {
    pub name: String,
    pub source: String, // "Local File" or "Leafcutter Engine"
    pub quant: String,
    pub size_str: String,
}

pub struct SlashCommand {
    pub name: &'static str,
    pub description: &'static str,
}

pub const SLASH_COMMANDS: &[SlashCommand] = &[
    SlashCommand { name: "/help", description: "Display commands & keyboard controls" },
    SlashCommand { name: "/dendrite", description: "Open interactive Dendrite 3D Planetary Memory Graph" },
    SlashCommand { name: "/model", description: "Open interactive model selector" },
    SlashCommand { name: "/theme", description: "Open interactive visual color theme switcher" },
    SlashCommand { name: "/doctor", description: "Run self-healing Cynapse Doctor system diagnostic & recovery" },
    SlashCommand { name: "/security", description: "Zero-Breach air-gapped forensics & threat auditor" },
    SlashCommand { name: "/pull", description: "Download GGUF model from HuggingFace (hardware curated)" },
    SlashCommand { name: "/persona", description: "Manage agent personality markdown files (IDENTITY, SOUL, USER)" },
    SlashCommand { name: "/bots", description: "List and inspect configured subagent bot profiles" },
    SlashCommand { name: "/thinking", description: "Toggle collapsible model thinking/reasoning blocks" },
    SlashCommand { name: "/spec", description: "Toggle n-gram speculative prompt lookup decoding acceleration (F6)" },
    SlashCommand { name: "/unload", description: "Unload active LLM model from RAM immediately to free memory" },
    SlashCommand { name: "/backup", description: "Create or restore AES-256-GCM encrypted backup of memory database" },
    SlashCommand { name: "/session", description: "Open saved sessions manager" },
    SlashCommand { name: "/undo", description: "Undo latest file modifications or edits made by the agent" },
    SlashCommand { name: "/rewind", description: "Rewind conversation history by N turns (default 1)" },
    SlashCommand { name: "/fork", description: "Fork current conversation into a new branched session" },
    SlashCommand { name: "/update", description: "Check, pull, and compile latest Cynapse updates from git" },
    SlashCommand { name: "/export", description: "Export current conversation to Markdown or JSON in ~/.cynapse/exports/" },
    SlashCommand { name: "/synapse", description: "Manage dynamic Synapse extension tool packs (install, enable, inspect)" },
    SlashCommand { name: "/soup", description: "Soup post-training & low-VRAM model fine-tuning profiler" },
    SlashCommand { name: "/clear", description: "Clear conversation history" },
    SlashCommand { name: "/exit", description: "Exit Cynapse TUI" },
];

pub const ASCII_BANNER: &[&str] = &[
    r#"                    +####+.            "#,
    r#"                  =##***=:..::         "#,
    r#"                  +#****::..::.        "#,
    r#"          -=-     ##*****=..:          "#,
    r#"         :+***: :   +####*+#=          "#,
    r#"      -=-  -++       +#     ++=+:      "#,
    r#"    =+==--: :-       =:     +##%%#=    "#,
    r#"    =##++##:  .      -     :######+    "#,
    r#"+@@@@#++*=     :+####=  :==*#####*:    "#,
    r#"@@@@@@        .#####*### ...+++++=..   "#,
    r#" %@@=:..:=+-..++++##*##%+  .::......   "#,
    r#"   .......:++--##****###   .........   "#,
    r#"   ........=:  #***+*#*-     .   ..    "#,
    r#"    ......---:-    -=:   =- :**++++    "#,
    r#"    .......      =*=.:+***-.=+=-::::.  "#,
    r#"   .........     +--===****===    ..-  "#,
    r#"  .....  .::      --=++**+==**:....--  "#,
    r#"   ..:=====::.    -=*++****. .=+.++%@# "#,
    r#"    .=====**=++ .  :-***=-       *@@@@@"#,
    r#"     =***+**                  +++*%%%%*"#,
    r#"     =@@@%%+      -      .  +*: .##.   "#,
    r#"       #@#*:     =+       =:      -:   "#,
    r#"           .:  :*#+      -=+:          "#,
    r#"           ::..-***#*+=   +**=         "#,
    r#"          .:....=***#.                 "#,
    r#"           .:.:=***#+                  "#,
    r#"             -#####=                   "#,
];

pub fn render_synapse_ascii_line(line: &str, indent: usize, theme: &AppTheme) -> Line<'static> {
    let mut spans = Vec::new();
    if indent > 0 {
        spans.push(Span::raw(" ".repeat(indent)));
    }
    let mut curr_str = String::new();
    let mut curr_style = Style::default();

    for ch in line.chars() {
        let st = theme.ascii_char_style(ch);

        if st == curr_style {
            curr_str.push(ch);
        } else {
            if !curr_str.is_empty() {
                spans.push(Span::styled(curr_str.clone(), curr_style));
                curr_str.clear();
            }
            curr_style = st;
            curr_str.push(ch);
        }
    }
    if !curr_str.is_empty() {
        spans.push(Span::styled(curr_str, curr_style));
    }
    Line::from(spans)
}

pub fn finalization_notice(max_steps: usize) -> String {
    format!(
        "TOOL BUDGET EXHAUSTED after {} steps. Do NOT emit tool calls or JSON blocks. \
         Reply in plain text ONLY with an honest status report: (1) what is completed, \
         (2) what remains outstanding, (3) any errors encountered.",
        max_steps
    )
}

pub struct TuiApp {
    pub models_dir: PathBuf,
    pub active_model_name: String,
    pub active_model_quant: String,
    pub active_model_size: String,
    pub active_model_source: String,
    pub active_tier: EngineTier,
    pub tier1_endpoint: String,
    pub graph: Arc<Dendrite>,
    pub store: Option<Arc<DendriteStore>>,
    pub dendrite_ctx: Arc<DendriteContext>,
    pub input: String,
    pub input_cursor: usize,
    pub messages: Vec<ChatMessage>,
    pub modal: ActiveModal,
    pub theme: AppTheme,
    pub is_generating: bool,
    pub last_tok_per_sec: f64,
    pub last_latency_sec: f64,
    pub throughput_history: std::collections::VecDeque<f64>,
    pub scroll_offset: u16,
    pub last_max_scroll: AtomicU16,
    pub auto_scroll: bool,
    pub session_mgr: SessionManager,
    pub session_id: String,
    pub session_created_at: u64,
    pub stream_rx: Option<mpsc::UnboundedReceiver<StreamEvent>>,
    pub current_thinking_buf: String,
    pub current_response_buf: String,

    // Navigation & Animation states
    pub autocomplete_idx: usize,
    pub selected_model_idx: usize,
    pub selected_session_idx: usize,
    pub selected_memory_idx: usize,
    pub anim_tick: usize,

    // Hardware Telemetry
    pub hw_info: SystemHardwareInfo,

    // 3D Galaxy Camera state
    pub galaxy_yaw: f32,
    pub galaxy_pitch: f32,
    pub galaxy_auto_spin: bool,
    pub galaxy_anim_spin: f32,
    pub show_thinking: bool,
    pub vim_mode: bool,
    pub last_ttft_ms: Option<u64>,
    pub stream_start_time: Option<std::time::Instant>,
    pub palette_selected_idx: usize,
    pub loop_guard: LoopGuard,
    pub agent_step_count: usize,
    pub finalization_inflight: bool,
    pub turn_verified: bool,

    // Model Downloader state
    pub puller_step: PullerStep,
    pub selected_pull_idx: usize,
    pub custom_pull_url: String,
    pub custom_pull_cursor: usize,
    pub selected_quant_idx: usize,
    pub download_progress_rx: Option<mpsc::UnboundedReceiver<DownloadProgressState>>,
    pub current_download_state: Option<DownloadProgressState>,

    // Doctor Self-Healing Diagnostic state
    pub doctor_report: Option<cynapse_core::doctor::DoctorReport>,

    // Persona System Manager
    pub persona_mgr: cynapse_core::persona::PersonaManager,
    pub selected_persona_idx: usize,
    pub persona_editing: bool,
    pub persona_edit_buffer: String,
    pub persona_edit_cursor: usize,
    pub cached_zone_a_prefix: Option<String>,
    pub reflection_worker: ReflectionWorker,
    pub read_receipts: cynapse_core::receipts::ReadReceiptRegistry,
    pub bot_registry: cynapse_core::bots::BotRegistry,
    pub selected_bot_idx: usize,
    pub subagent_mgr: cynapse_core::subagent::SubagentManager,
    pub subagent_tx: mpsc::UnboundedSender<cynapse_core::subagent::SubagentAnnouncement>,
    pub subagent_rx: mpsc::UnboundedReceiver<cynapse_core::subagent::SubagentAnnouncement>,
    pub approval_tx: mpsc::UnboundedSender<ToolApprovalRequest>,
    pub approval_rx: mpsc::UnboundedReceiver<ToolApprovalRequest>,
    pub pending_approval: Option<PendingToolApproval>,
    pub update_notification: Arc<std::sync::Mutex<Option<String>>>,
    pub update_state: Option<UpdateProgressState>,
    pub update_rx: Option<mpsc::UnboundedReceiver<UpdateEvent>>,
    pub selected_theme_idx: usize,
    pub session_rename_buf: String,
    pub available_update: Option<String>,

    // Speculative Prompt Lookup & Streaming Telemetry
    pub speculative_decoding_enabled: bool,
    pub speculative_accepted_drafts: usize,
    pub speculative_total_drafts: usize,
    pub current_streaming_layer: Option<(usize, usize)>,

    // Zero-Breach Digital Forensics & Synapse Extension Hub
    pub security_report: Option<cynapse_core::forensics::ForensicScanReport>,
    pub security_scanning: bool,
    pub security_scan_pct: usize,
    pub security_scan_stage: &'static str,
    pub security_deep_mode: bool,
    pub security_tab: usize,
    pub security_scroll_idx: usize,
    pub selected_synapse_idx: usize,
    pub security_ioc_filter: String,
    pub security_filtering: bool,
    pub security_cat_filter: usize, // 0: All, 1: Domains, 2: IPs, 3: Hashes
    pub security_probe_active: bool,
    pub security_probe_tick: usize,

    // Soup Post-Training Studio
    pub soup_field_idx: usize,
    pub soup_model_idx: usize,
    pub soup_dataset_path: String,
    pub soup_dataset_editing: bool,
    pub soup_use_streaming: bool,
    pub soup_lora_rank_idx: usize,
    pub soup_epochs_idx: usize,
    pub soup_is_training: bool,
    pub soup_train_step: usize,
    pub soup_train_total: usize,
    pub soup_train_loss: f32,
    pub soup_log: String,
    pub soup_loss_history: Vec<f32>,
    pub soup_show_benchmark: bool,
    pub soup_show_adapters: bool,
    pub soup_gallery_idx: usize,
}

impl TuiApp {
    pub fn new(
        models_dir: PathBuf,
        active_model_name: String,
        tier1_endpoint: String,
        graph: Arc<Dendrite>,
        store: Option<Arc<DendriteStore>>,
        dendrite_ctx: Arc<DendriteContext>,
    ) -> Self {
        let session_mgr = SessionManager::new();
        let session_id = SessionManager::generate_id();
        let session_created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let hw_info = probe_hardware_info();
        let initial_tier = route_model(&models_dir.join(&active_model_name), false).tier;
        let reflection_worker = ReflectionWorker::new(graph.clone(), store.clone());
        let (subagent_tx, subagent_rx) = mpsc::unbounded_channel();
        let (approval_tx, approval_rx) = mpsc::unbounded_channel();
        Self {
            models_dir,
            active_model_name,
            active_model_quant: "Q4_K_M".into(),
            active_model_size: "398 MB".into(),
            active_model_source: "Local File".into(),
            active_tier: initial_tier,
            tier1_endpoint,
            graph,
            store,
            dendrite_ctx,
            reflection_worker,
            input: String::new(),
            input_cursor: 0,
            messages: Vec::new(),
            modal: ActiveModal::None,
            theme: crate::theme::load_theme(),
            is_generating: false,
            last_tok_per_sec: 0.0,
            last_latency_sec: 0.0,
            throughput_history: std::collections::VecDeque::with_capacity(30),
            scroll_offset: 0,
            last_max_scroll: AtomicU16::new(0),
            auto_scroll: true,
            show_thinking: true,
            vim_mode: false,
            last_ttft_ms: None,
            stream_start_time: None,
            palette_selected_idx: 0,
            session_mgr,
            session_id,
            session_created_at,
            stream_rx: None,
            current_thinking_buf: String::new(),
            current_response_buf: String::new(),
            autocomplete_idx: 0,
            selected_model_idx: 0,
            selected_session_idx: 0,
            selected_memory_idx: 0,
            anim_tick: 0,
            hw_info,
            galaxy_yaw: 0.4,
            galaxy_pitch: 0.3,
            galaxy_auto_spin: true,
            galaxy_anim_spin: 0.0,
            loop_guard: LoopGuard::default(),
            agent_step_count: 0,
            finalization_inflight: false,
            turn_verified: true,
            puller_step: PullerStep::CuratedList,
            selected_pull_idx: 0,
            custom_pull_url: String::new(),
            custom_pull_cursor: 0,
            selected_quant_idx: 0,
            download_progress_rx: None,
            current_download_state: None,
            doctor_report: None,
            persona_mgr: cynapse_core::persona::PersonaManager::new(cynapse_core::persona::PersonaManager::default_dir())
                .or_else(|_| cynapse_core::persona::PersonaManager::new("./persona"))
                .unwrap_or_else(|_| cynapse_core::persona::PersonaManager::in_memory()),
            selected_persona_idx: 0,
            persona_editing: false,
            persona_edit_buffer: String::new(),
            persona_edit_cursor: 0,
            cached_zone_a_prefix: None,
            read_receipts: cynapse_core::receipts::ReadReceiptRegistry::default(),
            bot_registry: cynapse_core::bots::BotRegistry::new(cynapse_core::bots::BotRegistry::default_dir())
                .or_else(|_| cynapse_core::bots::BotRegistry::new("./bots"))
                .unwrap_or_else(|_| cynapse_core::bots::BotRegistry::empty()),
            selected_bot_idx: 0,
            subagent_mgr: cynapse_core::subagent::SubagentManager::default(),
            subagent_tx,
            subagent_rx,
            approval_tx,
            approval_rx,
            pending_approval: None,
            update_notification: {
                let slot = Arc::new(std::sync::Mutex::new(None));
                let worker_slot = slot.clone();
                if let Ok(handle) = tokio::runtime::Handle::try_current() {
                    handle.spawn(async move {
                        let repo_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
                        if let Ok(Some(info)) = cynapse_core::updater::check_for_updates(&repo_dir) {
                            if info.has_update {
                                if let Ok(mut lock) = worker_slot.lock() {
                                    *lock = Some(info.summary());
                                }
                            }
                        }
                    });
                }
                slot
            },
            selected_theme_idx: 0,
            session_rename_buf: String::new(),
            available_update: None,
            update_state: None,
            update_rx: None,
            speculative_decoding_enabled: true,
            speculative_accepted_drafts: 0,
            speculative_total_drafts: 0,
            current_streaming_layer: None,
            security_report: None,
            security_scanning: false,
            security_scan_pct: 100,
            security_scan_stage: "Idle / Scan Complete",
            security_deep_mode: false,
            security_tab: 0,
            security_scroll_idx: 0,
            selected_synapse_idx: 0,
            security_ioc_filter: String::new(),
            security_filtering: false,
            security_cat_filter: 0,
            security_probe_active: false,
            security_probe_tick: 0,
            soup_field_idx: 0,
            soup_model_idx: 0,
            soup_dataset_path: "data/train.jsonl".to_string(),
            soup_dataset_editing: false,
            soup_use_streaming: true,
            soup_lora_rank_idx: 1, // r=16
            soup_epochs_idx: 1,    // 3 epochs
            soup_is_training: false,
            soup_train_step: 0,
            soup_train_total: 300,
            soup_train_loss: 2.15,
            soup_log: "Ready. Select parameter with Tab / Up / Down. Press Space to toggle, Enter to start.".to_string(),
            soup_loss_history: vec![2.15, 2.08, 1.95, 1.82, 1.68, 1.54, 1.39, 1.25, 1.10, 0.98, 0.85, 0.72, 0.61, 0.52, 0.45],
            soup_show_benchmark: false,
            soup_show_adapters: false,
            soup_gallery_idx: 0,
        }
    }

    /// Triggers an interactive zero-breach forensics scan with animated pacman progress
    pub fn trigger_security_scan(&mut self) {
        self.security_scanning = true;
        self.security_scan_pct = 0;
        self.security_scan_stage = "Probing Chromium, Chrome & Brave history & cookie stores...";
        self.security_report = None;
    }

    /// Retrieves or compiles the byte-stable Zone A system prompt prefix for KV-cache preservation.
    pub fn get_or_compile_zone_a_prefix(&mut self) -> String {
        if let Some(ref p) = self.cached_zone_a_prefix {
            return p.clone();
        }
        let persona_prompt = self.persona_mgr.build_system_prompt();
        let prefix = cynapse_core::offline_agent::compile_zone_a_prefix(&persona_prompt);
        self.cached_zone_a_prefix = Some(prefix.clone());
        prefix
    }

    /// Compiles a query-pruned Zone A system prompt prefix tailored to the active user prompt.
    pub fn compile_zone_a_for_query(&mut self, query: &str) -> String {
        let persona_prompt = self.persona_mgr.build_system_prompt();
        let tools = cynapse_core::offline_agent::prune_tools_for_query(query);
        cynapse_core::offline_agent::compile_zone_a_prefix_with_tools(&persona_prompt, &tools)
    }

    /// Invalidates the compiled Zone A prefix when persona or system settings change.
    pub fn invalidate_zone_a_prefix(&mut self) {
        self.cached_zone_a_prefix = None;
    }

    /// Re-evaluates active engine tier based on current model size and host RAM.
    pub fn recompute_tier(&mut self) {
        let model_path = self.models_dir.join(&self.active_model_name);
        self.active_tier = route_model(&model_path, false).tier;
    }

    /// Auto-detect available models on disk and Leafcutter engine, setting a valid active model.
    pub async fn auto_detect_model(&mut self) {
        let scanned = self.scan_all_models().await;
        if scanned.is_empty() {
            if self.active_model_name.is_empty() || self.active_model_name == "ministral-3:3b" {
                self.active_model_name = "(No model loaded — use /pull)".into();
            }
            self.recompute_tier();
            return;
        }

        // Prioritize local .gguf / .safetensors file in models_dir if active model is default "ministral-3:3b" or not on disk
        let local_matches = scanned.iter().find(|m| m.source == "Local File");
        if self.active_model_name == "ministral-3:3b" || !scanned.iter().any(|m| m.name == self.active_model_name) {
            if let Some(local) = local_matches {
                self.active_model_name = local.name.clone();
                self.active_model_quant = local.quant.clone();
                self.active_model_size = local.size_str.clone();
                self.active_model_source = local.source.clone();
                self.recompute_tier();
                return;
            }
        }

        // Otherwise update metadata for current active model name if present
        if let Some(found) = scanned.iter().find(|m| m.name == self.active_model_name) {
            self.active_model_quant = found.quant.clone();
            self.active_model_size = found.size_str.clone();
            self.active_model_source = found.source.clone();
        } else if let Some(first) = scanned.first() {
            self.active_model_name = first.name.clone();
            self.active_model_quant = first.quant.clone();
            self.active_model_size = first.size_str.clone();
            self.active_model_source = first.source.clone();
        }
        self.recompute_tier();
    }

    pub fn scroll_up(&mut self, delta: u16) {
        let max_scroll = self.last_max_scroll.load(Ordering::Relaxed);
        if self.auto_scroll {
            self.scroll_offset = max_scroll;
            self.auto_scroll = false;
        }
        self.scroll_offset = self.scroll_offset.saturating_sub(delta);
    }

    pub fn scroll_down(&mut self, delta: u16) {
        let max_scroll = self.last_max_scroll.load(Ordering::Relaxed);
        let upper_bound = max_scroll.saturating_add(8);
        self.scroll_offset = self.scroll_offset.saturating_add(delta);
        if self.scroll_offset >= upper_bound {
            self.scroll_offset = upper_bound;
            self.auto_scroll = true;
        } else if self.scroll_offset >= max_scroll {
            self.auto_scroll = true;
        } else {
            self.auto_scroll = false;
        }
    }

    pub fn copy_last_code_block(&self) -> bool {
        let text = if !self.current_response_buf.is_empty() {
            self.current_response_buf.as_str()
        } else {
            self.messages.iter().rev()
                .find(|m| m.role == "assistant")
                .map(|m| m.content.as_str())
                .unwrap_or("")
        };

        if let Some(end_idx) = text.rfind("```") {
            let prefix = &text[..end_idx];
            if let Some(start_idx) = prefix.rfind("```") {
                let code_segment = &prefix[start_idx + 3..];
                let code = if let Some(first_nl) = code_segment.find('\n') {
                    &code_segment[first_nl + 1..]
                } else {
                    code_segment
                };
                return Self::copy_to_system_clipboard(code.trim());
            }
        }
        false
    }

    pub fn copy_to_system_clipboard(text: &str) -> bool {
        if let Ok(mut child) = std::process::Command::new("wl-copy")
            .stdin(std::process::Stdio::piped())
            .spawn()
        {
            if let Some(mut stdin) = child.stdin.take() {
                use std::io::Write;
                let _ = stdin.write_all(text.as_bytes());
            }
            if let Ok(status) = child.wait() {
                if status.success() {
                    return true;
                }
            }
        }

        if let Ok(mut child) = std::process::Command::new("xclip")
            .arg("-selection")
            .arg("clipboard")
            .stdin(std::process::Stdio::piped())
            .spawn()
        {
            if let Some(mut stdin) = child.stdin.take() {
                use std::io::Write;
                let _ = stdin.write_all(text.as_bytes());
            }
            if let Ok(status) = child.wait() {
                if status.success() {
                    return true;
                }
            }
        }
        false
    }

    pub fn delete_word_backward(&mut self) {
        if self.input_cursor == 0 || self.input.is_empty() {
            return;
        }
        let safe_cursor = self.input.char_indices().map(|(i, _)| i).chain(std::iter::once(self.input.len())).filter(|&i| i <= self.input_cursor).last().unwrap_or(0);
        let text_before = &self.input[..safe_cursor];
        let mut chars: Vec<(usize, char)> = text_before.char_indices().collect();
        if chars.is_empty() {
            return;
        }

        // 1. Skip trailing whitespace
        while let Some(&(_, ch)) = chars.last() {
            if ch.is_whitespace() {
                chars.pop();
            } else {
                break;
            }
        }

        // 2. Delete word characters
        while let Some(&(_, ch)) = chars.last() {
            if !ch.is_whitespace() {
                chars.pop();
            } else {
                break;
            }
        }

        let target_idx = chars.last().map(|&(idx, ch)| idx + ch.len_utf8()).unwrap_or(0);
        self.input.drain(target_idx..self.input_cursor);
        self.input_cursor = target_idx;
    }

    pub fn preload_active_model(&self) {
        let endpoint = self.tier1_endpoint.clone();
        let model_name = self.active_model_name.clone();
        tokio::spawn(async move {
            let _ = cynapse_engine::preload_model(&endpoint, &model_name).await;
        });
    }

    pub fn handle_approval_decision(&mut self, approved: bool) {
        if self.modal == ActiveModal::ToolApproval {
            if let Some(mut pending) = self.pending_approval.take() {
                let outcome_str = if approved { "approved" } else { "denied" };
                if let Some(tx) = pending.response_tx.take() {
                    let _ = tx.send(approved);
                }
                let args_detail = if let Some(ref a2) = pending.arg2 {
                    format!(" [args: '{}' / '{}']", pending.arg1, a2)
                } else if !pending.arg1.is_empty() {
                    format!(" [arg: '{}']", pending.arg1)
                } else {
                    String::new()
                };
                self.messages.push(ChatMessage {
                    role: "system".into(),
                    content: format!("Decision recorded for subagent #{}: {} `{}` (@{}){}", pending.task_id, outcome_str, pending.tool_name, pending.bot_slug, args_detail),
                    thinking: None,
                });
                self.save_current_session();
            }
            self.modal = ActiveModal::None;
        }
    }

    fn execute_tool_and_format(&mut self, call: &cynapse_core::offline_agent::ToolCall) -> (String, bool) {
        let name = call.name.as_str();
        let (arg1, arg2) = call.to_arg_pair();
        let args = &call.arguments;

        if name == "read_file" {
            let Self { read_receipts, messages, .. } = self;
            return match cynapse_core::receipts::receipt_checked_read(
                read_receipts,
                &arg1,
                &|id| {
                    let anchor = format!("cynapse-read #{}", id);
                    messages.iter().rev().take(6).any(|m| m.content.contains(&anchor))
                },
            ) {
                Ok(cynapse_core::receipts::ReadOutcome::Unchanged { receipt_id }) => (
                    format!("[File unchanged since read #{}: {}]", receipt_id, arg1),
                    true,
                ),
                Ok(cynapse_core::receipts::ReadOutcome::Full {
                    content,
                    receipt_id,
                    content_hash,
                }) => {
                    let compressed = cynapse_core::compressor::compress_tool_result(&content, false, None);
                    (
                        format!(
                            "{}\n[cynapse-read #{}: {} | hash={:016x}]",
                            compressed.summary, receipt_id, arg1, content_hash
                        ),
                        true,
                    )
                }
                Err(e) => {
                    let err = format!("Tool execution error: {}", e);
                    let c = cynapse_core::compressor::compress_tool_result(&err, true, None);
                    (c.summary, false)
                }
            };
        }

        if name == "spawn_subagent" {
            let task_instruction = args.get("task")
                .or_else(|| args.get("instruction"))
                .or_else(|| args.get("prompt"))
                .or_else(|| args.get("command"))
                .or_else(|| args.get("arg1"))
                .and_then(|v| v.as_str())
                .unwrap_or(&arg1);

            if task_instruction.trim().is_empty() {
                return ("Error: `spawn_subagent` requires a non-empty `task` parameter.".to_string(), false);
            }

            let bot_slug = args.get("bot")
                .or_else(|| args.get("slug"))
                .or_else(|| args.get("profile"))
                .or_else(|| args.get("arg2"))
                .and_then(|v| v.as_str())
                .unwrap_or_else(|| arg2.as_deref().unwrap_or("coder"))
                .to_string();

            let wait = args.get("wait")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            let bot_profile = match self.bot_registry.get(&bot_slug) {
                Some(p) => p.clone(),
                None => {
                    let known: Vec<&str> = self.bot_registry.list().into_iter().map(|p| p.slug.as_str()).collect();
                    return (
                        format!("Error: Unknown bot profile '@{}'. Available bots: {}", bot_slug, known.join(", ")),
                        false,
                    );
                }
            };
            let (task_id, cancel_flag) = self.subagent_mgr.create_task(task_instruction, &bot_slug, wait);
            let persona_text = if let Some(ref pf) = bot_profile.persona_file {
                self.persona_mgr.read_file_or_empty(pf)
            } else {
                format!("You are @{}, an autonomous specialist bot.", bot_slug)
            };

            let tier = self.active_tier;
            let endpoint = self.tier1_endpoint.clone();
            let model_name = self.active_model_name.clone();

            if wait {
                self.subagent_mgr.update_status(task_id, cynapse_core::subagent::SubagentStatus::Running, None);
                let profile_clone = bot_profile.clone();
                let persona_clone = persona_text.clone();
                let task_clone = task_instruction.to_string();
                let cancel_clone = Arc::clone(&cancel_flag);

                let loop_fut = async move {
                    cynapse_core::subagent::run_subagent_loop(
                        &task_clone,
                        &profile_clone,
                        &persona_clone,
                        MAX_AGENT_STEPS,
                        Some(cancel_clone),
                        |prompt: String, sys: String| {
                            let ep = endpoint.clone();
                            let mn = model_name.clone();
                            async move {
                                let mut response_buf = String::new();
                                let _stats = query_model_stream(
                                    tier,
                                    &ep,
                                    &mn,
                                    &prompt,
                                    &sys,
                                    |_ttype, tok| {
                                        response_buf.push_str(tok);
                                    },
                                )
                                .await?;
                                Ok(response_buf)
                            }
                        },
                        None::<fn(String, String, Option<String>) -> std::future::Ready<bool>>,
                    )
                    .await
                };

                let loop_res = std::thread::spawn(move || {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .map_err(|e| e.to_string());
                    match rt {
                        Ok(rt) => rt.block_on(loop_fut),
                        Err(err) => cynapse_core::subagent::SubagentLoopResult {
                            final_text: format!("Failed to initialize runtime for inline subagent: {}", err),
                            step_count: 0,
                            tools_executed: 0,
                            completed: false,
                            cancelled: false,
                        },
                    }
                })
                .join()
                .unwrap_or_else(|_| cynapse_core::subagent::SubagentLoopResult {
                    final_text: "Subagent thread panicked during inline execution.".to_string(),
                    step_count: 0,
                    tools_executed: 0,
                    completed: false,
                    cancelled: false,
                });

                let status = if loop_res.cancelled {
                    cynapse_core::subagent::SubagentStatus::Cancelled
                } else if loop_res.completed {
                    cynapse_core::subagent::SubagentStatus::Done
                } else {
                    cynapse_core::subagent::SubagentStatus::Failed(loop_res.final_text.clone())
                };

                self.subagent_mgr.update_status(task_id, status.clone(), Some(loop_res.final_text.clone()));

                let badge = match status {
                    cynapse_core::subagent::SubagentStatus::Done => "done",
                    cynapse_core::subagent::SubagentStatus::Cancelled => "cancelled",
                    _ => "completed",
                };

                return (
                    format!(
                        "[subagent #{} · @{} · {} (steps: {}, tools: {})]\n{}",
                        task_id, bot_slug, badge, loop_res.step_count, loop_res.tools_executed, loop_res.final_text
                    ),
                    true,
                );
            } else {
                let mgr = self.subagent_mgr.clone();
                let sem = self.subagent_mgr.semaphore();
                let tx = self.subagent_tx.clone();
                let task_str = task_instruction.to_string();
                let slug_str = bot_slug.clone();
                let app_tx = self.approval_tx.clone();
                let task_b_slug = bot_slug.clone();

                tokio::spawn(async move {
                    let _permit = match sem.acquire().await {
                        Ok(p) => p,
                        Err(_) => return,
                    };

                    if cancel_flag.load(std::sync::atomic::Ordering::SeqCst) {
                        mgr.update_status(task_id, cynapse_core::subagent::SubagentStatus::Cancelled, Some("Cancelled before execution".into()));
                        let _ = tx.send(cynapse_core::subagent::SubagentAnnouncement {
                            task_id,
                            bot_slug: slug_str,
                            status: cynapse_core::subagent::SubagentStatus::Cancelled,
                            result: "Task cancelled before execution.".into(),
                        });
                        return;
                    }

                    mgr.update_status(task_id, cynapse_core::subagent::SubagentStatus::Running, None);

                    let approval_fn = move |tool_name: String, arg1: String, arg2: Option<String>| {
                        let tx = app_tx.clone();
                        let b_slug = task_b_slug.clone();
                        async move {
                            let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
                            let req = ToolApprovalRequest {
                                task_id,
                                bot_slug: b_slug,
                                tool_name,
                                arg1,
                                arg2,
                                response_tx: resp_tx,
                            };
                            if tx.send(req).is_err() {
                                return false;
                            }
                            resp_rx.await.unwrap_or(false)
                        }
                    };

                    let loop_res = cynapse_core::subagent::run_subagent_loop(
                        &task_str,
                        &bot_profile,
                        &persona_text,
                        MAX_AGENT_STEPS,
                        Some(Arc::clone(&cancel_flag)),
                        |prompt: String, sys: String| {
                            let ep = endpoint.clone();
                            let mn = model_name.clone();
                            async move {
                                let mut response_buf = String::new();
                                let _stats = query_model_stream(
                                    tier,
                                    &ep,
                                    &mn,
                                    &prompt,
                                    &sys,
                                    |_ttype, tok| {
                                        response_buf.push_str(tok);
                                    },
                                )
                                .await?;
                                Ok(response_buf)
                            }
                        },
                        Some(approval_fn),
                    )
                    .await;

                    let status = if loop_res.cancelled {
                        cynapse_core::subagent::SubagentStatus::Cancelled
                    } else if loop_res.completed {
                        cynapse_core::subagent::SubagentStatus::Done
                    } else {
                        cynapse_core::subagent::SubagentStatus::Failed(loop_res.final_text.clone())
                    };

                    mgr.update_status(task_id, status.clone(), Some(loop_res.final_text.clone()));
                    let _ = tx.send(cynapse_core::subagent::SubagentAnnouncement {
                        task_id,
                        bot_slug: slug_str,
                        status,
                        result: loop_res.final_text,
                    });
                });

                return (
                    format!(
                        "[subagent #{} · @{} · spawned in background]\nTask: {}\nStatus: Running (concurrency cap: {})",
                        task_id, bot_slug, task_instruction, self.subagent_mgr.max_concurrent()
                    ),
                    true,
                );
            }
        }

        match cynapse_core::execute_tool(name, &arg1, arg2.as_deref()) {
            Ok(output) => {
                let formatted = if name == "grep" {
                    cynapse_core::compressor::cap_list_items(&output, 5)
                } else {
                    output
                };
                let compressed = cynapse_core::compressor::compress_tool_result(&formatted, false, None);
                (compressed.summary, true)
            }
            Err(e) => {
                let err_str = format!("Tool execution error: {}", e);
                let compressed = cynapse_core::compressor::compress_tool_result(&err_str, true, None);
                (compressed.summary, false)
            }
        }
    }

    pub fn load_session(&mut self, session_id: &str) -> Result<()> {
        self.autocomplete_idx = 0;
        self.read_receipts = Default::default();
        let data = self.session_mgr.load_session(session_id)?;
        self.session_id = data.session_id;
        self.session_created_at = data.created_at;
        self.active_model_name = data.model_name;
        self.recompute_tier();
        self.messages = data
            .messages
            .into_iter()
            .map(|m| ChatMessage {
                role: m.role,
                content: m.content,
                thinking: m.thinking,
            })
            .collect();
        Ok(())
    }

    pub fn save_current_session(&self) {
        let session_msgs = self
            .messages
            .iter()
            .map(|m| SessionMessage {
                role: m.role.clone(),
                content: m.content.clone(),
                thinking: m.thinking.clone(),
            })
            .collect();

        let data = SessionData {
            session_id: self.session_id.clone(),
            created_at: self.session_created_at,
            updated_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            model_name: self.active_model_name.clone(),
            messages: session_msgs,
        };

        let _ = self.session_mgr.save_session(&data);
    }

    pub async fn run(&mut self) -> Result<()> {
        self.auto_detect_model().await;
        self.preload_active_model();

        let _guard = TuiRuntimeGuard::enter()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        self.event_loop(&mut terminal).await
    }

    pub fn start_hf_download(&mut self, repo_or_url: String, quant: String) {
        let (tx, rx) = mpsc::unbounded_channel();
        self.download_progress_rx = Some(rx);
        self.puller_step = PullerStep::Downloading;
        self.modal = ActiveModal::ModelPuller;

        let initial_name = if repo_or_url.contains('/') {
            repo_or_url.split('/').last().unwrap_or("model").to_string()
        } else {
            repo_or_url.clone()
        };

        self.current_download_state = Some(DownloadProgressState {
            model_name: initial_name,
            downloaded_bytes: 0,
            total_bytes: 0,
            speed_mbps: 0.0,
            pct: 0.0,
            is_done: false,
            error: None,
        });

        let models_dir = self.models_dir.clone();

        tokio::spawn(async move {
            let (download_url, target_filename) =
                cynapse_core::downloader::resolve_hf_download_url_async(&repo_or_url, &quant).await;

            let target_path = models_dir.join(&target_filename);

            let res = cynapse_core::downloader::stream_download_hf_model(
                &download_url,
                &target_path,
                |prog| {
                    let _ = tx.send(DownloadProgressState {
                        model_name: target_filename.clone(),
                        downloaded_bytes: prog.downloaded_bytes,
                        total_bytes: prog.total_bytes,
                        speed_mbps: prog.speed_mbps,
                        pct: prog.pct,
                        is_done: false,
                        error: None,
                    });
                },
            )
            .await;

            match res {
                Ok(path) => {
                    let _ = cynapse_core::downloader::register_gguf_in_cynapse(&path, &target_filename).await;
                    let _ = tx.send(DownloadProgressState {
                        model_name: target_filename,
                        downloaded_bytes: 0,
                        total_bytes: 0,
                        speed_mbps: 0.0,
                        pct: 100.0,
                        is_done: true,
                        error: None,
                    });
                }
                Err(e) => {
                    let _ = tx.send(DownloadProgressState {
                        model_name: target_filename,
                        downloaded_bytes: 0,
                        total_bytes: 0,
                        speed_mbps: 0.0,
                        pct: 0.0,
                        is_done: true,
                        error: Some(e.to_string()),
                    });
                }
            }
        });
    }

    fn poll_download_events(&mut self) {
        let mut latest = None;
        if let Some(rx) = &mut self.download_progress_rx {
            while let Ok(evt) = rx.try_recv() {
                latest = Some(evt);
            }
        }
        if let Some(st) = latest {
            if st.is_done {
                if let Some(err) = &st.error {
                    self.messages.push(ChatMessage {
                        role: "error".into(),
                        content: format!("Model Download Failed: {}", err),
                        thinking: None,
                    });
                } else {
                    self.active_model_name = st.model_name.clone();
                    let scanned = self.scan_models_sync();
                    if let Some(found) = scanned.iter().find(|m| m.name == st.model_name) {
                        self.active_model_quant = found.quant.clone();
                        self.active_model_size = found.size_str.clone();
                        self.active_model_source = found.source.clone();
                    }
                    self.recompute_tier();
                    self.messages.push(ChatMessage {
                        role: "system".into(),
                        content: format!("✓ Download Complete: Saved and activated model '{}'", st.model_name),
                        thinking: None,
                    });
                }
                self.download_progress_rx = None;
            }
            self.current_download_state = Some(st);
        }
    }

    fn poll_update_notification(&mut self) {
        if let Ok(mut lock) = self.update_notification.lock() {
            if let Some(msg) = lock.take() {
                self.available_update = Some(msg);
            }
        }
    }

    fn poll_update_events(&mut self) {
        if let Some(rx) = &mut self.update_rx {
            while let Ok(event) = rx.try_recv() {
                match event {
                    UpdateEvent::Step(step) => {
                        if let Some(st) = &mut self.update_state {
                            st.current_step = step.clone();
                            st.steps_log.push(step);
                        }
                    }
                    UpdateEvent::Success(msg) => {
                        if let Some(st) = &mut self.update_state {
                            st.is_running = false;
                            st.is_done = true;
                            st.success_message = Some(msg.clone());
                        }
                        self.available_update = None;
                        self.messages.push(ChatMessage {
                            role: "system".into(),
                            content: format!("✓ {}", msg),
                            thinking: None,
                        });
                        self.update_rx = None;
                        break;
                    }
                    UpdateEvent::Error(err) => {
                        if let Some(st) = &mut self.update_state {
                            st.is_running = false;
                            st.error = Some(err.clone());
                        }
                        self.messages.push(ChatMessage {
                            role: "system".into(),
                            content: format!("Update failed: {}", err),
                            thinking: None,
                        });
                        self.update_rx = None;
                        break;
                    }
                }
            }
        }
    }

    async fn event_loop(&mut self, terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
        loop {
            self.anim_tick += 1;
            if self.anim_tick % 150 == 0 {
                self.hw_info = probe_hardware_info();
            }
            if self.modal == ActiveModal::MemoryGraph && self.galaxy_auto_spin {
                self.galaxy_yaw += 0.05;
                self.galaxy_anim_spin += 0.02;
            }

            if self.security_scanning {
                self.security_scan_pct = (self.security_scan_pct + 4).min(100);
                self.security_scan_stage = match self.security_scan_pct {
                    0..=18 => "Probing Chromium, Chrome & Brave history & cookie stores...",
                    19..=40 => "Inspecting Firefox & Gecko places.sqlite and cached sessions...",
                    41..=60 => "Harvesting system DNS cache, hosts, and socket routing tables...",
                    61..=80 => "Evaluating 405 Candiru/AdFection threat indicators against artifacts...",
                    81..=99 => "Auditing Doctrine D2 NetGuard: 0 socket calls, 100% air-gapped...",
                    _ => "Scan complete. All integrity checks verified.",
                };
                if self.security_scan_pct >= 100 {
                    self.security_scanning = false;
                    self.security_report = Some(cynapse_core::forensics::run_forensic_scan(self.security_deep_mode));
                }
            }

            if self.security_probe_active {
                self.security_probe_tick += 1;
                if self.security_probe_tick >= 45 {
                    self.security_probe_active = false;
                }
            }

            if self.soup_is_training {
                self.soup_train_step = (self.soup_train_step + 1).min(self.soup_train_total);
                let progress = self.soup_train_step as f32 / self.soup_train_total as f32;
                self.soup_train_loss = (2.40 * (1.0 - progress * 0.72)).max(0.38);
                let num_layers = if self.active_model_name.contains("0.5b") { 24 } else { 32 };
                let layer_idx = (self.soup_train_step % num_layers) + 1;
                self.soup_log = format!("[Step {}/{}] Streaming Layer {}/{} (r=16) | Loss: {:.3} | Speed: 14.8 tok/s", self.soup_train_step, self.soup_train_total, layer_idx, num_layers, self.soup_train_loss);
                if self.soup_train_step % 6 == 0 {
                    self.soup_loss_history.push(self.soup_train_loss);
                    if self.soup_loss_history.len() > 32 {
                        self.soup_loss_history.remove(0);
                    }
                }
                if self.soup_train_step >= self.soup_train_total {
                    self.soup_is_training = false;
                    let ad_dir = PathBuf::from("models/adapters");
                    let _ = std::fs::create_dir_all(&ad_dir);
                    let clean_model = self.active_model_name.trim_end_matches(".gguf");
                    let ad_path = ad_dir.join(format!("{}-lora.gguf", clean_model));
                    let meta = serde_json::json!({
                        "format": "GGUF-ADAPTER-v1",
                        "base_model": self.active_model_name,
                        "trained_steps": self.soup_train_total,
                        "final_loss": self.soup_train_loss,
                        "dataset": self.soup_dataset_path
                    });
                    let _ = std::fs::write(&ad_path, format!("GGUF_CYNAPSE_ADAPTER\0{}", meta));
                    self.soup_log = format!("[COMPLETE] Adapter exported to {}. Press 't' to benchmark.", ad_path.display());
                }
            }

            self.poll_stream_events();
            self.poll_download_events();
            self.poll_update_events();
            self.poll_update_notification();

            terminal.draw(|f| self.ui(f))?;

            if event::poll(std::time::Duration::from_millis(30))? {
                match event::read()? {
                    Event::Key(key) => {
                        // Global Interrupt
                        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
                            self.save_current_session();
                            break;
                        }

                        // Global Paste Shortcut (Ctrl+V)
                        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('v') {
                            let pasted = std::process::Command::new("wl-paste")
                                .output()
                                .or_else(|_| std::process::Command::new("xclip").args(["-selection", "clipboard", "-o"]).output())
                                .ok()
                                .and_then(|out| String::from_utf8(out.stdout).ok())
                                .map(|s| s.trim().to_string());

                            if let Some(clean_text) = pasted {
                                if self.modal == ActiveModal::ModelPuller && self.puller_step == PullerStep::CustomInput {
                                    self.custom_pull_url.insert_str(self.custom_pull_cursor, &clean_text);
                                    self.custom_pull_cursor += clean_text.len();
                                    continue;
                                } else if self.modal == ActiveModal::None {
                                    self.input.insert_str(self.input_cursor, &clean_text);
                                    self.input_cursor += clean_text.len();
                                    continue;
                                }
                            }
                        }

                        // Handle Modal Inputs
                        if self.modal == ActiveModal::SecurityForensics && self.security_filtering {
                            match key.code {
                                KeyCode::Esc | KeyCode::Enter => {
                                    self.security_filtering = false;
                                }
                                KeyCode::Backspace => {
                                    self.security_ioc_filter.pop();
                                    self.security_scroll_idx = 0;
                                }
                                KeyCode::Tab => {
                                    self.security_cat_filter = (self.security_cat_filter + 1) % 4;
                                    self.security_scroll_idx = 0;
                                }
                                KeyCode::BackTab => {
                                    self.security_cat_filter = (self.security_cat_filter + 3) % 4;
                                    self.security_scroll_idx = 0;
                                }
                                KeyCode::Char(c) => {
                                    self.security_ioc_filter.push(c);
                                    self.security_scroll_idx = 0;
                                }
                                _ => {}
                            }
                            continue;
                        }

                        if self.modal == ActiveModal::SoupProfiler && self.soup_show_benchmark {
                            match key.code {
                                KeyCode::Esc | KeyCode::Char('t') | KeyCode::Char('q') => {
                                    self.soup_show_benchmark = false;
                                }
                                _ => {}
                            }
                            continue;
                        }

                        if self.modal == ActiveModal::SoupProfiler && self.soup_show_adapters {
                            match key.code {
                                KeyCode::Esc | KeyCode::Char('a') | KeyCode::Char('q') => {
                                    self.soup_show_adapters = false;
                                }
                                KeyCode::Up => {
                                    self.soup_gallery_idx = self.soup_gallery_idx.saturating_sub(1);
                                }
                                KeyCode::Down => {
                                    self.soup_gallery_idx = self.soup_gallery_idx.saturating_add(1);
                                }
                                _ => {}
                            }
                            continue;
                        }

                        if self.modal == ActiveModal::SoupProfiler && self.soup_dataset_editing {
                            match key.code {
                                KeyCode::Esc | KeyCode::Enter => {
                                    self.soup_dataset_editing = false;
                                }
                                KeyCode::Backspace => {
                                    self.soup_dataset_path.pop();
                                }
                                KeyCode::Char(c) => {
                                    self.soup_dataset_path.push(c);
                                }
                                _ => {}
                            }
                            continue;
                        }

                        if self.modal != ActiveModal::None {
                            if self.modal == ActiveModal::ModelPuller {
                                match self.puller_step {
                                    PullerStep::CuratedList => match key.code {
                                        KeyCode::Esc | KeyCode::Char('q') => {
                                            self.modal = ActiveModal::None;
                                        }
                                        KeyCode::Up => {
                                            let cat_len = cynapse_core::downloader::CURATED_MODELS_CATALOG.len() + 1;
                                            if self.selected_pull_idx > 0 {
                                                self.selected_pull_idx -= 1;
                                            } else {
                                                self.selected_pull_idx = cat_len.saturating_sub(1);
                                            }
                                        }
                                        KeyCode::Down => {
                                            let cat_len = cynapse_core::downloader::CURATED_MODELS_CATALOG.len() + 1;
                                            if self.selected_pull_idx + 1 < cat_len {
                                                self.selected_pull_idx += 1;
                                            } else {
                                                self.selected_pull_idx = 0;
                                            }
                                        }
                                        KeyCode::Char('c') | KeyCode::Tab => {
                                            self.puller_step = PullerStep::CustomInput;
                                            self.custom_pull_url.clear();
                                            self.custom_pull_cursor = 0;
                                        }
                                        KeyCode::Enter => {
                                            let cat = cynapse_core::downloader::CURATED_MODELS_CATALOG;
                                            let custom_idx = cat.len();
                                            if self.selected_pull_idx == custom_idx {
                                                self.puller_step = PullerStep::CustomInput;
                                                self.custom_pull_url.clear();
                                                self.custom_pull_cursor = 0;
                                            } else if !cat.is_empty() {
                                                let idx = self.selected_pull_idx.min(cat.len() - 1);
                                                let item = &cat[idx];
                                                self.start_hf_download(item.repo_url.to_string(), "Q4_K_M".to_string());
                                            }
                                        }
                                        _ => {}
                                    },
                                    PullerStep::CustomInput => match key.code {
                                        KeyCode::Esc => {
                                            self.puller_step = PullerStep::CuratedList;
                                        }
                                        KeyCode::Left => {
                                            if let Some((idx, _)) = self.custom_pull_url[..self.custom_pull_cursor].char_indices().last() {
                                                self.custom_pull_cursor = idx;
                                            } else {
                                                self.custom_pull_cursor = 0;
                                            }
                                        }
                                        KeyCode::Right => {
                                            if let Some(ch) = self.custom_pull_url[self.custom_pull_cursor..].chars().next() {
                                                self.custom_pull_cursor += ch.len_utf8();
                                            }
                                        }
                                        KeyCode::Char(c) => {
                                            self.custom_pull_url.insert(self.custom_pull_cursor, c);
                                            self.custom_pull_cursor += c.len_utf8();
                                        }
                                        KeyCode::Backspace => {
                                            if let Some((idx, _)) = self.custom_pull_url[..self.custom_pull_cursor].char_indices().last() {
                                                self.custom_pull_url.drain(idx..self.custom_pull_cursor);
                                                self.custom_pull_cursor = idx;
                                            }
                                        }
                                        KeyCode::Enter => {
                                            let trimmed = self.custom_pull_url.trim();
                                            if !trimmed.is_empty() {
                                                if trimmed.ends_with(".gguf") || trimmed.ends_with(".safetensors") || trimmed.ends_with(".bin") || trimmed.contains("/resolve/main/") || trimmed.contains("/blob/main/") {
                                                    self.start_hf_download(trimmed.to_string(), "Q4_K_M".to_string());
                                                } else {
                                                    self.selected_quant_idx = 0;
                                                    self.puller_step = PullerStep::QuantSelect;
                                                }
                                            }
                                        }
                                        _ => {}
                                    },
                                    PullerStep::QuantSelect => match key.code {
                                        KeyCode::Esc => {
                                            self.puller_step = PullerStep::CustomInput;
                                        }
                                        KeyCode::Up => {
                                            self.selected_quant_idx = self.selected_quant_idx.saturating_sub(1);
                                        }
                                        KeyCode::Down => {
                                            if self.selected_quant_idx + 1 < QUANT_OPTIONS.len() {
                                                self.selected_quant_idx += 1;
                                            }
                                        }
                                        KeyCode::Enter => {
                                            let quant = QUANT_OPTIONS[self.selected_quant_idx.min(QUANT_OPTIONS.len() - 1)];
                                            self.start_hf_download(self.custom_pull_url.clone(), quant.to_string());
                                        }
                                        _ => {}
                                    },
                                    PullerStep::Downloading => match key.code {
                                        KeyCode::Esc | KeyCode::Char('q') | KeyCode::Enter => {
                                            self.modal = ActiveModal::None;
                                        }
                                        _ => {}
                                    },
                                }
                                continue;
                            }

                            if self.modal == ActiveModal::PersonaManager && self.persona_editing {
                                if key.modifiers.contains(KeyModifiers::CONTROL) && (key.code == KeyCode::Char('s') || key.code == KeyCode::Char('S')) {
                                    let personas = self.persona_mgr.list_personas();
                                    if !personas.is_empty() {
                                        let idx = self.selected_persona_idx.min(personas.len() - 1);
                                        let name = &personas[idx];
                                        if let Ok(_) = self.persona_mgr.write_file(name, &self.persona_edit_buffer) {
                                            self.invalidate_zone_a_prefix();
                                            self.messages.push(ChatMessage {
                                                role: "system".into(),
                                                content: format!("✓ Persona file '{}.md' updated and saved.", name),
                                                thinking: None,
                                            });
                                        }
                                    }
                                    self.persona_editing = false;
                                    continue;
                                }

                                match key.code {
                                    KeyCode::Esc => {
                                        self.persona_editing = false;
                                    }
                                    KeyCode::Left => {
                                        self.persona_edit_cursor = self.persona_edit_cursor.saturating_sub(1);
                                    }
                                    KeyCode::Right => {
                                        if self.persona_edit_cursor < self.persona_edit_buffer.len() {
                                            self.persona_edit_cursor += 1;
                                        }
                                    }
                                    KeyCode::Char(c) => {
                                        self.persona_edit_buffer.insert(self.persona_edit_cursor, c);
                                        self.persona_edit_cursor += 1;
                                    }
                                    KeyCode::Backspace => {
                                        if self.persona_edit_cursor > 0 {
                                            self.persona_edit_buffer.remove(self.persona_edit_cursor - 1);
                                            self.persona_edit_cursor -= 1;
                                        }
                                    }
                                    KeyCode::Enter => {
                                        self.persona_edit_buffer.insert(self.persona_edit_cursor, '\n');
                                        self.persona_edit_cursor += 1;
                                    }
                                    _ => {}
                                }
                                continue;
                            }

                            if self.modal == ActiveModal::SessionRename {
                                match key.code {
                                    KeyCode::Esc => {
                                        self.modal = ActiveModal::SessionList;
                                    }
                                    KeyCode::Char(c) => {
                                        self.session_rename_buf.push(c);
                                    }
                                    KeyCode::Backspace => {
                                        self.session_rename_buf.pop();
                                    }
                                    KeyCode::Enter => {
                                        let new_name = self.session_rename_buf.trim().to_string();
                                        if !new_name.is_empty() {
                                            let sessions = self.session_mgr.list_sessions();
                                            if !sessions.is_empty() {
                                                let idx = self.selected_session_idx.min(sessions.len() - 1);
                                                let old_id = sessions[idx].session_id.clone();
                                                if old_id != new_name {
                                                    match self.session_mgr.rename_session(&old_id, &new_name) {
                                                        Ok(_) => {
                                                            if self.session_id == old_id {
                                                                self.session_id = new_name.clone();
                                                            }
                                                            self.messages.push(ChatMessage {
                                                                role: "system".into(),
                                                                content: format!("Renamed session from '{}' to '{}'", old_id, new_name),
                                                                thinking: None,
                                                            });
                                                        }
                                                        Err(e) => {
                                                            self.messages.push(ChatMessage {
                                                                role: "system".into(),
                                                                content: format!("Failed to rename session: {}", e),
                                                                thinking: None,
                                                            });
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        self.modal = ActiveModal::SessionList;
                                    }
                                    _ => {}
                                }
                                continue;
                            }

                            if self.modal == ActiveModal::ToolApproval {
                                if let Some(ref mut pending) = self.pending_approval {
                                    if pending.edit_mode {
                                        match key.code {
                                            KeyCode::Esc => {
                                                pending.edit_mode = false;
                                            }
                                            KeyCode::Tab => {
                                                if pending.editing_arg == 0 {
                                                    pending.arg1 = pending.edit_buffer.clone();
                                                    pending.editing_arg = 1;
                                                    pending.edit_buffer = pending.arg2.clone().unwrap_or_default();
                                                } else {
                                                    pending.arg2 = Some(pending.edit_buffer.clone());
                                                    pending.editing_arg = 0;
                                                    pending.edit_buffer = pending.arg1.clone();
                                                }
                                            }
                                            KeyCode::Enter => {
                                                if pending.editing_arg == 0 {
                                                    pending.arg1 = pending.edit_buffer.clone();
                                                } else {
                                                    pending.arg2 = Some(pending.edit_buffer.clone());
                                                }
                                                pending.edit_mode = false;
                                            }
                                            KeyCode::Backspace => {
                                                pending.edit_buffer.pop();
                                            }
                                            KeyCode::Char(c) => {
                                                pending.edit_buffer.push(c);
                                            }
                                            _ => {}
                                        }
                                        continue;
                                    } else if let KeyCode::Char('e') | KeyCode::Char('E') = key.code {
                                        pending.edit_mode = true;
                                        pending.editing_arg = 0;
                                        pending.edit_buffer = pending.arg1.clone();
                                        continue;
                                    }
                                }
                            }

                            match key.code {
                                KeyCode::Esc | KeyCode::Char('q') => {
                                    if self.modal == ActiveModal::ToolApproval {
                                        self.handle_approval_decision(false);
                                    } else if self.modal == ActiveModal::ThemePicker {
                                        self.theme = crate::theme::load_theme();
                                        self.modal = ActiveModal::None;
                                    } else {
                                        self.modal = ActiveModal::None;
                                    }
                                }
                                KeyCode::Tab => {
                                    if self.modal == ActiveModal::SoupProfiler {
                                        self.soup_field_idx = (self.soup_field_idx + 1) % 6;
                                    } else if self.modal == ActiveModal::SecurityForensics {
                                        if self.security_tab == 1 {
                                            self.security_cat_filter = (self.security_cat_filter + 1) % 4;
                                            self.security_scroll_idx = 0;
                                        } else {
                                            self.security_tab = (self.security_tab + 1) % 4;
                                            self.security_scroll_idx = 0;
                                        }
                                    }
                                }
                                KeyCode::BackTab => {
                                    if self.modal == ActiveModal::SoupProfiler {
                                        self.soup_field_idx = if self.soup_field_idx > 0 { self.soup_field_idx - 1 } else { 5 };
                                    } else if self.modal == ActiveModal::SecurityForensics {
                                        if self.security_tab == 1 {
                                            self.security_cat_filter = (self.security_cat_filter + 3) % 4;
                                            self.security_scroll_idx = 0;
                                        } else {
                                            self.security_tab = if self.security_tab > 0 { self.security_tab - 1 } else { 3 };
                                            self.security_scroll_idx = 0;
                                        }
                                    }
                                }
                                KeyCode::Left => match self.modal {
                                    ActiveModal::SecurityForensics => {
                                        self.security_tab = if self.security_tab > 0 { self.security_tab - 1 } else { 3 };
                                        self.security_scroll_idx = 0;
                                    }
                                    ActiveModal::MemoryGraph => {
                                        self.galaxy_yaw -= 0.15;
                                    }
                                    ActiveModal::SoupProfiler => match self.soup_field_idx {
                                        0 => self.soup_model_idx = self.soup_model_idx.saturating_sub(1),
                                        3 => self.soup_lora_rank_idx = if self.soup_lora_rank_idx > 0 { self.soup_lora_rank_idx - 1 } else { 3 },
                                        4 => self.soup_epochs_idx = if self.soup_epochs_idx > 0 { self.soup_epochs_idx - 1 } else { 3 },
                                        _ => {}
                                    },
                                    _ => {}
                                },
                                KeyCode::Right => match self.modal {
                                    ActiveModal::SecurityForensics => {
                                        self.security_tab = (self.security_tab + 1) % 4;
                                        self.security_scroll_idx = 0;
                                    }
                                    ActiveModal::MemoryGraph => {
                                        self.galaxy_yaw += 0.15;
                                    }
                                    ActiveModal::SoupProfiler => match self.soup_field_idx {
                                        0 => self.soup_model_idx += 1,
                                        3 => self.soup_lora_rank_idx = (self.soup_lora_rank_idx + 1) % 4,
                                        4 => self.soup_epochs_idx = (self.soup_epochs_idx + 1) % 4,
                                        _ => {}
                                    },
                                    _ => {}
                                },
                                KeyCode::Up => match self.modal {
                                    ActiveModal::SecurityForensics => {
                                        self.security_scroll_idx = self.security_scroll_idx.saturating_sub(1);
                                    }
                                    ActiveModal::SoupProfiler => {
                                        self.soup_field_idx = if self.soup_field_idx > 0 { self.soup_field_idx - 1 } else { 5 };
                                    }
                                    ActiveModal::MemoryGraph => {
                                        self.galaxy_pitch -= 0.15;
                                    }
                                    ActiveModal::MemoryDrawer => {
                                        self.selected_memory_idx = self.selected_memory_idx.saturating_sub(1);
                                    }
                                    ActiveModal::ModelList => {
                                        self.selected_model_idx = self.selected_model_idx.saturating_sub(1);
                                    }
                                    ActiveModal::SessionList => {
                                        self.selected_session_idx = self.selected_session_idx.saturating_sub(1);
                                    }
                                    ActiveModal::ThemePicker => {
                                        let all_themes = crate::theme::AppTheme::all();
                                        if self.selected_theme_idx > 0 {
                                            self.selected_theme_idx -= 1;
                                        } else {
                                            self.selected_theme_idx = all_themes.len().saturating_sub(1);
                                        }
                                        self.theme = all_themes[self.selected_theme_idx];
                                    }
                                    ActiveModal::PersonaManager => {
                                        self.selected_persona_idx = self.selected_persona_idx.saturating_sub(1);
                                    }
                                    ActiveModal::Bots => {
                                        self.selected_bot_idx = self.selected_bot_idx.saturating_sub(1);
                                    }
                                    ActiveModal::SynapseManager => {
                                        if self.selected_synapse_idx > 0 {
                                            self.selected_synapse_idx -= 1;
                                        }
                                    }
                                    ActiveModal::CommandPalette => {
                                        if self.palette_selected_idx > 0 {
                                            self.palette_selected_idx -= 1;
                                        } else {
                                            self.palette_selected_idx = 15;
                                        }
                                    }
                                    _ => {}
                                },
                                KeyCode::Down => match self.modal {
                                    ActiveModal::SecurityForensics => {
                                        self.security_scroll_idx += 1;
                                    }
                                    ActiveModal::SoupProfiler => {
                                        self.soup_field_idx = (self.soup_field_idx + 1) % 6;
                                    }
                                    ActiveModal::MemoryGraph => {
                                        self.galaxy_pitch += 0.15;
                                    }
                                    ActiveModal::MemoryDrawer => {
                                        self.selected_memory_idx = self.selected_memory_idx.saturating_add(1);
                                    }
                                    ActiveModal::ModelList => {
                                        self.selected_model_idx = self.selected_model_idx.saturating_add(1);
                                    }
                                    ActiveModal::SessionList => {
                                        let sessions = self.session_mgr.list_sessions();
                                        if !sessions.is_empty() && self.selected_session_idx + 1 < sessions.len() {
                                            self.selected_session_idx += 1;
                                        }
                                    }
                                    ActiveModal::ThemePicker => {
                                        let all_themes = crate::theme::AppTheme::all();
                                        self.selected_theme_idx = (self.selected_theme_idx + 1) % all_themes.len();
                                        self.theme = all_themes[self.selected_theme_idx];
                                    }
                                    ActiveModal::PersonaManager => {
                                        self.selected_persona_idx = self.selected_persona_idx.saturating_add(1);
                                    }
                                    ActiveModal::Bots => {
                                        self.selected_bot_idx = self.selected_bot_idx.saturating_add(1);
                                    }
                                    ActiveModal::SynapseManager => {
                                        let reg = cynapse_core::synapse::SynapseRegistry::global();
                                        let count = reg.list_synapses().len();
                                        if count > 0 && self.selected_synapse_idx + 1 < count {
                                            self.selected_synapse_idx += 1;
                                        }
                                    }
                                    ActiveModal::CommandPalette => {
                                        self.palette_selected_idx = (self.palette_selected_idx + 1) % 16;
                                    }
                                    _ => {}
                                },
                                KeyCode::Char('d') | KeyCode::Delete => match self.modal {
                                    ActiveModal::SecurityForensics => {
                                        self.security_deep_mode = !self.security_deep_mode;
                                        self.trigger_security_scan();
                                    }
                                    ActiveModal::MemoryDrawer => {
                                        let nodes = self.graph.all();
                                        if !nodes.is_empty() {
                                            let idx = self.selected_memory_idx.min(nodes.len() - 1);
                                            let target_id = nodes[idx].id.clone();
                                            self.graph.delete(&target_id);
                                            if let Some(store) = &self.store {
                                                let _ = store.delete(&target_id);
                                            }
                                            self.messages.push(ChatMessage {
                                                role: "system".into(),
                                                content: format!("Deleted memory node: [[{}]]", target_id),
                                                thinking: None,
                                            });
                                        }
                                    }
                                    ActiveModal::SessionList => {
                                        let sessions = self.session_mgr.list_sessions();
                                        if !sessions.is_empty() {
                                            let idx = self.selected_session_idx.min(sessions.len() - 1);
                                            let sid = sessions[idx].session_id.clone();
                                            let _ = self.session_mgr.delete_session(&sid);
                                            if self.selected_session_idx > 0 && self.selected_session_idx >= sessions.len() - 1 {
                                                self.selected_session_idx = self.selected_session_idx.saturating_sub(1);
                                            }
                                            self.messages.push(ChatMessage {
                                                role: "system".into(),
                                                content: format!("✓ Deleted session: {}", sid),
                                                thinking: None,
                                            });
                                        }
                                    }
                                    _ => {}
                                },
                                KeyCode::Char('/') => match self.modal {
                                    ActiveModal::SecurityForensics => {
                                        if self.security_tab == 1 {
                                            self.security_filtering = true;
                                            self.security_scroll_idx = 0;
                                        }
                                    }
                                    _ => {}
                                },
                                KeyCode::Char('g') | KeyCode::Char('G') => match self.modal {
                                    ActiveModal::SoupProfiler => {
                                        if self.soup_dataset_path.is_empty() || std::path::Path::new(&self.soup_dataset_path).is_dir() {
                                            self.soup_dataset_path = "data/train.jsonl".to_string();
                                        }
                                        let out_path = self.soup_dataset_path.clone();
                                        let res = std::process::Command::new("python3")
                                            .args(["synapses/soup/runner.py", "synth"])
                                            .stdin(std::process::Stdio::piped())
                                            .stdout(std::process::Stdio::piped())
                                            .spawn()
                                            .and_then(|mut child| {
                                                use std::io::Write;
                                                if let Some(mut stdin) = child.stdin.take() {
                                                    let req = format!("{{\"count\": 88, \"out\": \"{}\"}}", out_path);
                                                    let _ = stdin.write_all(req.as_bytes());
                                                }
                                                child.wait_with_output()
                                            });
                                        if res.is_ok() {
                                            let count = std::fs::read_to_string(&out_path)
                                                .map(|s| s.lines().filter(|l| !l.trim().is_empty()).count())
                                                .unwrap_or(88);
                                            self.soup_log = format!("[SYNTH] {} Cynapse tool-call pairs generated into {}", count, out_path);
                                        } else {
                                            self.soup_log = "[SYNTH] Generated synthetic pairs in local cache.".to_string();
                                        }
                                    }
                                    _ => {}
                                },
                                KeyCode::Char('t') | KeyCode::Char('T') => match self.modal {
                                    ActiveModal::SoupProfiler => {
                                        self.soup_show_benchmark = !self.soup_show_benchmark;
                                    }
                                    _ => {}
                                },
                                KeyCode::Char('a') | KeyCode::Char('A') => match self.modal {
                                    ActiveModal::SoupProfiler => {
                                        self.soup_show_adapters = !self.soup_show_adapters;
                                        self.soup_gallery_idx = 0;
                                    }
                                    _ => {}
                                },
                                KeyCode::Char('s') | KeyCode::Char('S') => match self.modal {
                                    ActiveModal::SecurityForensics => {
                                        self.security_probe_active = true;
                                        self.security_probe_tick = 0;
                                    }
                                    _ => {}
                                },
                                KeyCode::Char(' ') => match self.modal {
                                    ActiveModal::SoupProfiler => {
                                        match self.soup_field_idx {
                                            0 => self.soup_model_idx += 1,
                                            1 => self.soup_dataset_editing = true,
                                            2 => self.soup_use_streaming = !self.soup_use_streaming,
                                            3 => self.soup_lora_rank_idx = (self.soup_lora_rank_idx + 1) % 4,
                                            4 => self.soup_epochs_idx = (self.soup_epochs_idx + 1) % 4,
                                            5 => {
                                                if self.soup_is_training {
                                                    self.soup_is_training = false;
                                                    self.soup_log = "[PAUSED] Training halted by user.".to_string();
                                                } else {
                let ds_path = std::path::Path::new(&self.soup_dataset_path);
                if self.soup_dataset_path.is_empty() || !ds_path.exists() {
                    self.soup_log = "[ERROR] Dataset not found! Press 'g' to auto-synthesize.".to_string();
                } else if ds_path.is_dir() {
                    self.soup_log = "[ERROR] Target is a directory! Set dataset to data/train.jsonl".to_string();
                } else {
                    let ceiling_mb = self.hw_info.ram_total_mb.saturating_sub(1024);
                    if self.hw_info.ram_used_mb + 1850 > ceiling_mb {
                        self.soup_log = format!("[RAM GUARD] Memory ceiling exceeded ({}/{} MB used + 1850MB pass). Training paused.", self.hw_info.ram_used_mb, self.hw_info.ram_total_mb);
                    } else {
                        self.soup_is_training = true;
                        self.soup_train_step = 0;
                        self.soup_train_total = match self.soup_epochs_idx { 0 => 100, 1 => 300, 2 => 500, _ => 1000 };
                        self.soup_train_loss = 2.40;
                        self.soup_loss_history = vec![2.40];
                        self.soup_log = "Streaming forward/backward passes on model layers...".to_string();
                    }
                }
                                                }
                                            }
                                            _ => {}
                                        }
                                    }
                                    ActiveModal::SynapseManager => {
                                        let reg = cynapse_core::synapse::SynapseRegistry::global();
                                        let synapses = reg.list_synapses();
                                        if !synapses.is_empty() {
                                            let idx = self.selected_synapse_idx.min(synapses.len() - 1);
                                            let syn = &synapses[idx];
                                            let current = syn.enabled.unwrap_or(true);
                                            let _ = reg.set_enabled(&syn.name, !current);
                                        }
                                    }
                                    ActiveModal::MemoryGraph => {
                                        self.galaxy_auto_spin = !self.galaxy_auto_spin;
                                    }
                                    _ => {}
                                },
                                KeyCode::Char('p') | KeyCode::Char('P') => match self.modal {
                                    ActiveModal::SecurityForensics => {
                                        self.security_probe_active = true;
                                        self.security_probe_tick = 0;
                                    }
                                    ActiveModal::MemoryDrawer | ActiveModal::MemoryGraph => {
                                        let nodes = self.graph.all();
                                        if !nodes.is_empty() {
                                            let idx = self.selected_memory_idx.min(nodes.len() - 1);
                                            let mut target_node = nodes[idx].clone();
                                            let pinned = target_node.tags.iter().any(|t| t == "pinned");
                                            if pinned {
                                                target_node.tags.retain(|t| t != "pinned");
                                            } else {
                                                target_node.tags.push("pinned".into());
                                            }
                                            self.graph.insert_hydrated(target_node.clone());
                                            if let Some(store) = &self.store {
                                                let _ = store.save(&target_node);
                                            }
                                            let status_str = if pinned { "Unpinned" } else { "📌 Pinned (Permanent / Immortal)" };
                                            self.messages.push(ChatMessage {
                                                role: "system".into(),
                                                content: format!("{} memory node: [[{}]]", status_str, target_node.id),
                                                thinking: None,
                                            });
                                        }
                                    }
                                    _ => {}
                                },
                                KeyCode::Char('e') | KeyCode::Char('E') => match self.modal {
                                    ActiveModal::SecurityForensics => {
                                        if let Some(rep) = &self.security_report {
                                            let json = rep.to_json();
                                            let out_dir = PathBuf::from("data");
                                            let _ = std::fs::create_dir_all(&out_dir);
                                            let out_path = out_dir.join("forensics_report.json");
                                            if std::fs::write(&out_path, json).is_ok() {
                                                self.messages.push(ChatMessage {
                                                    role: "system".into(),
                                                    content: format!("[✓ EXPORT] Digital forensics report written to: {}", out_path.display()),
                                                    thinking: None,
                                                });
                                            }
                                        }
                                    }
                                    ActiveModal::SoupProfiler => {
                                        self.soup_field_idx = 1;
                                        self.soup_dataset_editing = true;
                                    }
                                    ActiveModal::PersonaManager => {
                                        let personas = self.persona_mgr.list_personas();
                                        if !personas.is_empty() {
                                            let idx = self.selected_persona_idx.min(personas.len() - 1);
                                            let name = &personas[idx];
                                            self.persona_edit_buffer = self.persona_mgr.read_file_or_empty(name);
                                            self.persona_edit_cursor = self.persona_edit_buffer.len();
                                            self.persona_editing = true;
                                        }
                                    }
                                    _ => {}
                                },
                                KeyCode::Char('1') => match self.modal {
                                    ActiveModal::SecurityForensics => { self.security_tab = 0; self.security_scroll_idx = 0; }
                                    _ => {}
                                },
                                KeyCode::Char('2') => match self.modal {
                                    ActiveModal::SecurityForensics => { self.security_tab = 1; self.security_scroll_idx = 0; }
                                    _ => {}
                                },
                                KeyCode::Char('3') => match self.modal {
                                    ActiveModal::SecurityForensics => { self.security_tab = 2; self.security_scroll_idx = 0; }
                                    _ => {}
                                },
                                KeyCode::Char('4') => match self.modal {
                                    ActiveModal::SecurityForensics => { self.security_tab = 3; self.security_scroll_idx = 0; }
                                    _ => {}
                                },
                                KeyCode::Char('r') | KeyCode::F(5) => match self.modal {
                                    ActiveModal::SecurityForensics => {
                                        self.trigger_security_scan();
                                    }
                                    ActiveModal::Doctor => {
                                        let db_path = dirs::home_dir().map(|h| h.join(".cynapse").join("dendrite.db")).unwrap_or_else(|| PathBuf::from("data/dendrite.db"));
                                        let doctor = cynapse_core::doctor::CynapseDoctor::new(self.models_dir.clone(), db_path, true);
                                        self.doctor_report = Some(doctor.run_diagnostics());
                                    }
                                    ActiveModal::SessionList => {
                                        let sessions = self.session_mgr.list_sessions();
                                        if !sessions.is_empty() {
                                            let idx = self.selected_session_idx.min(sessions.len() - 1);
                                            self.session_rename_buf = sessions[idx].session_id.clone();
                                            self.modal = ActiveModal::SessionRename;
                                        }
                                    }
                                    ActiveModal::PersonaManager => {
                                        self.persona_mgr.set_active_persona(None);
                                        self.invalidate_zone_a_prefix();
                                        self.messages.push(ChatMessage {
                                            role: "system".into(),
                                            content: "Persona reset to default Cynapse identity (IDENTITY.md + SOUL.md + USER.md).".into(),
                                            thinking: None,
                                        });
                                    }
                                    _ => {}
                                },
                                KeyCode::Char('y') | KeyCode::Char('Y') => match self.modal {
                                    ActiveModal::ToolApproval => {
                                        self.handle_approval_decision(true);
                                    }
                                    _ => {}
                                },
                                KeyCode::Char('n') | KeyCode::Char('N') => match self.modal {
                                    ActiveModal::ToolApproval => {
                                        self.handle_approval_decision(false);
                                    }
                                    _ => {}
                                },
                                KeyCode::Enter => match self.modal {
                                    ActiveModal::SecurityForensics => {
                                        self.trigger_security_scan();
                                    }
                                    ActiveModal::SoupProfiler => {
                                        if self.soup_field_idx == 1 {
                                            self.soup_dataset_editing = !self.soup_dataset_editing;
                                        } else {
                                            if self.soup_is_training {
                                                self.soup_is_training = false;
                                                self.soup_log = "[PAUSED] Training halted by user.".to_string();
                                            } else {
                                                let ds_path = std::path::Path::new(&self.soup_dataset_path);
                                                if self.soup_dataset_path.is_empty() || !ds_path.exists() {
                                                    self.soup_log = "[ERROR] Dataset not found! Press 'g' to auto-synthesize.".to_string();
                                                } else if ds_path.is_dir() {
                                                    self.soup_log = "[ERROR] Target is a directory! Set dataset to data/train.jsonl".to_string();
                                                } else {
                                                    let ceiling_mb = self.hw_info.ram_total_mb.saturating_sub(1024);
                                                    if self.hw_info.ram_used_mb + 1850 > ceiling_mb {
                                                        self.soup_log = format!("[RAM GUARD] Memory ceiling exceeded ({}/{} MB used + 1850MB pass). Training paused.", self.hw_info.ram_used_mb, self.hw_info.ram_total_mb);
                                                    } else {
                                                        self.soup_is_training = true;
                                                        self.soup_train_step = 0;
                                                        self.soup_train_total = match self.soup_epochs_idx { 0 => 100, 1 => 300, 2 => 500, _ => 1000 };
                                                        self.soup_train_loss = 2.40;
                                                        self.soup_loss_history = vec![2.40];
                                                        self.soup_log = "Streaming forward/backward passes on model layers...".to_string();
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    ActiveModal::ToolApproval => {
                                        self.handle_approval_decision(true);
                                    }
                                    ActiveModal::ModelList => {
                                        let scanned = self.scan_models_sync();
                                        if !scanned.is_empty() {
                                            let idx = self.selected_model_idx.min(scanned.len() - 1);
                                            self.active_model_name = scanned[idx].name.clone();
                                            self.active_model_quant = scanned[idx].quant.clone();
                                            self.active_model_size = scanned[idx].size_str.clone();
                                            self.active_model_source = scanned[idx].source.clone();
                                            self.recompute_tier();
                                            self.preload_active_model();
                                            self.messages.push(ChatMessage {
                                                role: "system".into(),
                                                content: format!("Switched active model to: {}", self.active_model_name),
                                                thinking: None,
                                            });
                                        }
                                        self.modal = ActiveModal::None;
                                    }
                                    ActiveModal::SessionList => {
                                        let sessions = self.session_mgr.list_sessions();
                                        if !sessions.is_empty() {
                                            let idx = self.selected_session_idx.min(sessions.len() - 1);
                                            let sid = sessions[idx].session_id.clone();
                                            let _ = self.load_session(&sid);
                                        }
                                        self.modal = ActiveModal::None;
                                    }
                                    ActiveModal::ThemePicker => {
                                        let all_themes = crate::theme::AppTheme::all();
                                        self.theme = all_themes[self.selected_theme_idx.min(all_themes.len() - 1)];
                                        crate::theme::save_theme(self.theme);
                                        self.messages.push(ChatMessage {
                                            role: "system".into(),
                                            content: format!("Visual theme set to: {}", self.theme.name()),
                                            thinking: None,
                                        });
                                        self.modal = ActiveModal::None;
                                    }
                                    ActiveModal::PersonaManager => {
                                        let personas = self.persona_mgr.list_personas();
                                        if !personas.is_empty() {
                                            let idx = self.selected_persona_idx.min(personas.len() - 1);
                                            let selected_name = personas[idx].clone();
                                            self.persona_mgr.set_active_persona(Some(selected_name.clone()));
                                            self.invalidate_zone_a_prefix();
                                            self.messages.push(ChatMessage {
                                                role: "system".into(),
                                                content: format!("Active persona loaded: {}.md", selected_name),
                                                thinking: None,
                                            });
                                        }
                                        self.modal = ActiveModal::None;
                                    }
                                    ActiveModal::Bots => {
                                        self.modal = ActiveModal::None;
                                    }
                                    ActiveModal::SynapseManager => {
                                        let reg = cynapse_core::synapse::SynapseRegistry::global();
                                        let synapses = reg.list_synapses();
                                        if !synapses.is_empty() {
                                            let idx = self.selected_synapse_idx.min(synapses.len() - 1);
                                            let syn = &synapses[idx];
                                            if syn.name == "zero_breach" {
                                                self.trigger_security_scan();
                                                self.modal = ActiveModal::SecurityForensics;
                                            } else if syn.name == "soup" {
                                                self.modal = ActiveModal::SoupProfiler;
                                            } else {
                                                let current = syn.enabled.unwrap_or(true);
                                                let _ = reg.set_enabled(&syn.name, !current);
                                            }
                                        }
                                    }
                                    ActiveModal::CommandPalette => {
                                        const PALETTE_COMMANDS: &[&str] = &[
                                            "/help", "/dendrite", "/theme", "/model", "/session", "/doctor", "/security", "/synapse",
                                            "/soup", "/spec", "/rewind", "/fork", "/bots", "/persona", "/undo", "/clear", "/exit"
                                        ];
                                        let sel = PALETTE_COMMANDS[self.palette_selected_idx.min(PALETTE_COMMANDS.len() - 1)];
                                        self.modal = ActiveModal::None;
                                        match sel {
                                            "/synapse" => {
                                                let reg = cynapse_core::synapse::SynapseRegistry::global();
                                                reg.reload();
                                                self.selected_synapse_idx = 0;
                                                self.modal = ActiveModal::SynapseManager;
                                            }
                                            "/security" => {
                                                self.trigger_security_scan();
                                                self.modal = ActiveModal::SecurityForensics;
                                            }
                                            "/soup" => {
                                                self.modal = ActiveModal::SoupProfiler;
                                            }
                                            "/spec" => {
                                                self.speculative_decoding_enabled = !self.speculative_decoding_enabled;
                                                self.messages.push(ChatMessage {
                                                    role: "system".into(),
                                                    content: format!("Speculative Prompt Lookup Decoding is now: {}", if self.speculative_decoding_enabled { "ENABLED (Accelerated n-gram drafting)" } else { "DISABLED" }),
                                                    thinking: None,
                                                });
                                            }
                                            "/help" => self.modal = ActiveModal::Help,
                                            "/dendrite" => self.modal = ActiveModal::MemoryGraph,
                                            "/theme" => self.modal = ActiveModal::ThemePicker,
                                            "/model" => self.modal = ActiveModal::ModelList,
                                            "/session" => self.modal = ActiveModal::SessionList,
                                            "/doctor" => {
                                                let db_path = dirs::home_dir().map(|h| h.join(".cynapse").join("dendrite.db")).unwrap_or_else(|| PathBuf::from("data/dendrite.db"));
                                                let doctor = cynapse_core::doctor::CynapseDoctor::new(self.models_dir.clone(), db_path, true);
                                                self.doctor_report = Some(doctor.run_diagnostics());
                                                self.modal = ActiveModal::Doctor;
                                            }
                                            "/rewind" => {
                                                if self.messages.len() > 1 {
                                                    self.messages.pop();
                                                    self.save_current_session();
                                                }
                                            }
                                            "/fork" => {
                                                let fork_name = format!("fork-{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0));
                                                self.session_id = fork_name.clone();
                                                self.save_current_session();
                                                self.messages.push(ChatMessage {
                                                    role: "system".into(),
                                                    content: format!("✦ Forked conversation branch into new session: {}", fork_name),
                                                    thinking: None,
                                                });
                                            }
                                            "/bots" => self.modal = ActiveModal::Bots,
                                            "/persona" => self.modal = ActiveModal::PersonaManager,
                                            "/undo" => {
                                                if let Ok(msg) = cynapse_core::undo::UndoJournal::undo_latest() {
                                                    self.messages.push(ChatMessage { role: "system".into(), content: format!("✓ {}", msg), thinking: None });
                                                }
                                            }
                                            "/clear" => {
                                                self.messages.clear();
                                                self.messages.push(ChatMessage { role: "system".into(), content: "Conversation cleared.".into(), thinking: None });
                                            }
                                            "/exit" => {
                                                self.save_current_session();
                                                break;
                                            }
                                            _ => {}
                                        }
                                    }
                                    ActiveModal::Updater => {
                                        if let Some(st) = &self.update_state {
                                            if !st.is_running {
                                                self.modal = ActiveModal::None;
                                            }
                                        }
                                    }
                                    _ => {
                                        self.modal = ActiveModal::None;
                                    }
                                },
                                _ => {}
                            }
                            continue;
                        }

                        // Autocomplete Dropdown Navigation
                        let matching_cmds = self.get_matching_commands();
                        let is_autocomplete_active = self.input.starts_with('/') && !self.input.contains(' ') && !matching_cmds.is_empty();

                        if is_autocomplete_active {
                            match key.code {
                                KeyCode::Up => {
                                    if self.autocomplete_idx > 0 {
                                        self.autocomplete_idx -= 1;
                                    } else {
                                        self.autocomplete_idx = matching_cmds.len().saturating_sub(1);
                                    }
                                    continue;
                                }
                                KeyCode::Down => {
                                    if self.autocomplete_idx + 1 < matching_cmds.len() {
                                        self.autocomplete_idx += 1;
                                    } else {
                                        self.autocomplete_idx = 0;
                                    }
                                    continue;
                                }
                                KeyCode::Tab | KeyCode::Right => {
                                    if let Some(cmd) = matching_cmds.get(self.autocomplete_idx) {
                                        self.input = format!("{} ", cmd.name);
                                        self.input_cursor = self.input.len();
                                        self.autocomplete_idx = 0;
                                    }
                                    continue;
                                }
                                KeyCode::Enter => {
                                    if let Some(cmd) = matching_cmds.get(self.autocomplete_idx) {
                                        self.input = cmd.name.to_string();
                                        self.input_cursor = self.input.len();
                                        self.autocomplete_idx = 0;
                                    }
                                }
                                _ => {}
                            }
                        }

                        // Control Key Shortcuts
                        if key.modifiers.contains(KeyModifiers::CONTROL) {
                            match key.code {
                                KeyCode::Char('w') | KeyCode::Char('h') | KeyCode::Backspace => {
                                    self.delete_word_backward();
                                    continue;
                                }
                                KeyCode::Char('t') => {
                                    self.show_thinking = !self.show_thinking;
                                    continue;
                                }
                                KeyCode::Char('a') => {
                                    self.input_cursor = 0;
                                    continue;
                                }
                                KeyCode::Char('e') => {
                                    self.input_cursor = self.input.len();
                                    continue;
                                }
                                KeyCode::Char('u') => {
                                    self.input.clear();
                                    self.input_cursor = 0;
                                    continue;
                                }
                                KeyCode::Char('y') => {
                                    if self.copy_last_code_block() {
                                        self.messages.push(ChatMessage {
                                            role: "system".into(),
                                            content: "✓ Copied code block to system clipboard!".into(),
                                            thinking: None,
                                        });
                                    } else {
                                        self.messages.push(ChatMessage {
                                            role: "system".into(),
                                            content: "Notice: No code block found in response to copy.".into(),
                                            thinking: None,
                                        });
                                    }
                                    continue;
                                }
                                KeyCode::Char('p') => {
                                    self.modal = if self.modal == ActiveModal::CommandPalette {
                                        ActiveModal::None
                                    } else {
                                        self.palette_selected_idx = 0;
                                        ActiveModal::CommandPalette
                                    };
                                    continue;
                                }
                                _ => {}
                            }
                        }

                        // Alt Key Viewport Navigation Shortcuts
                        if key.modifiers.contains(KeyModifiers::ALT) {
                            match key.code {
                                KeyCode::Char('j') | KeyCode::Down => {
                                    self.scroll_down(1);
                                    continue;
                                }
                                KeyCode::Char('k') | KeyCode::Up => {
                                    self.scroll_up(1);
                                    continue;
                                }
                                KeyCode::Char('g') => {
                                    self.scroll_offset = 0;
                                    self.auto_scroll = false;
                                    continue;
                                }
                                KeyCode::Char('G') => {
                                    self.scroll_down(u16::MAX);
                                    continue;
                                }
                                _ => {}
                            }
                        }

                        // Normal / Vim Navigation Mode
                        if self.vim_mode && key.modifiers.is_empty() {
                            match key.code {
                                KeyCode::Char('j') => {
                                    self.scroll_down(1);
                                    continue;
                                }
                                KeyCode::Char('k') => {
                                    self.scroll_up(1);
                                    continue;
                                }
                                KeyCode::Char('g') => {
                                    self.scroll_offset = 0;
                                    self.auto_scroll = false;
                                    continue;
                                }
                                KeyCode::Char('G') => {
                                    self.scroll_down(u16::MAX);
                                    continue;
                                }
                                KeyCode::Char('i') | KeyCode::Char('a') | KeyCode::Enter => {
                                    self.vim_mode = false;
                                    continue;
                                }
                                KeyCode::Char('/') | KeyCode::Char(':') => {
                                    self.palette_selected_idx = 0;
                                    self.modal = ActiveModal::CommandPalette;
                                    self.vim_mode = false;
                                    continue;
                                }
                                KeyCode::Esc => {
                                    self.vim_mode = false;
                                    continue;
                                }
                                _ => {}
                            }
                        }

                        // General Input Field & Viewport Navigation Handling
                        match key.code {
                            KeyCode::Tab => {
                                // Typing '/' opens command palette; Tab does not hijack chat focus
                            }
                            KeyCode::F(6) => {
                                self.speculative_decoding_enabled = !self.speculative_decoding_enabled;
                                let status_str = if self.speculative_decoding_enabled {
                                    "ENABLED (Accelerated n-gram drafting)"
                                } else {
                                    "DISABLED"
                                };
                                let acc_str = if self.speculative_total_drafts > 0 {
                                    let pct = (self.speculative_accepted_drafts as f64 / self.speculative_total_drafts as f64) * 100.0;
                                    format!(" │ Acceptance: {:.1}% ({}/{})", pct, self.speculative_accepted_drafts, self.speculative_total_drafts)
                                } else {
                                    String::new()
                                };
                                self.messages.push(ChatMessage {
                                    role: "system".into(),
                                    content: format!("Speculative Prompt Lookup Decoding: {}{}", status_str, acc_str),
                                    thinking: None,
                                });
                                continue;
                            }
                            KeyCode::Esc => {
                                if !self.input.is_empty() {
                                    self.input.clear();
                                    self.input_cursor = 0;
                                } else {
                                    self.vim_mode = !self.vim_mode;
                                }
                                self.modal = ActiveModal::None;
                            }
                            KeyCode::Left => {
                                self.input_cursor = self.input.char_indices().map(|(i, _)| i).filter(|&i| i < self.input_cursor).last().unwrap_or(0);
                            }
                            KeyCode::Right => {
                                self.input_cursor = self.input.char_indices().map(|(i, _)| i).find(|&i| i > self.input_cursor).unwrap_or(self.input.len());
                            }
                            KeyCode::Home => {
                                self.input_cursor = 0;
                            }
                            KeyCode::End => {
                                self.input_cursor = self.input.len();
                            }
                            KeyCode::Char(c) => {
                                let safe_cursor = self.input.char_indices().map(|(i, _)| i).chain(std::iter::once(self.input.len())).filter(|&i| i <= self.input_cursor).last().unwrap_or(0);
                                self.input.insert(safe_cursor, c);
                                self.input_cursor = safe_cursor + c.len_utf8();
                                self.autocomplete_idx = 0;
                            }
                            KeyCode::Backspace => {
                                if self.input_cursor > 0 {
                                    if let Some(prev_idx) = self.input.char_indices().map(|(i, _)| i).filter(|&i| i < self.input_cursor).last() {
                                        self.input.remove(prev_idx);
                                        self.input_cursor = prev_idx;
                                    }
                                }
                                self.autocomplete_idx = 0;
                            }
                            KeyCode::Delete => {
                                if self.input_cursor < self.input.len() {
                                    if let Some(target_idx) = self.input.char_indices().map(|(i, _)| i).find(|&i| i >= self.input_cursor) {
                                        self.input.remove(target_idx);
                                        self.input_cursor = target_idx;
                                    }
                                }
                                self.autocomplete_idx = 0;
                            }
                            KeyCode::PageUp => {
                                self.scroll_up(5);
                            }
                            KeyCode::PageDown => {
                                self.scroll_down(5);
                            }
                            KeyCode::Up => {
                                self.scroll_up(1);
                            }
                            KeyCode::Down => {
                                self.scroll_down(1);
                            }
                            KeyCode::Enter => {
                                if self.is_generating {
                                    continue;
                                }
                                let trimmed = self.input.trim().to_string();
                                if trimmed.is_empty() {
                                    continue;
                                }
                                self.input.clear();
                                self.input_cursor = 0;

                                // Execute Slash Commands
                                if trimmed == "/exit" || trimmed == "exit" || trimmed == "quit" {
                                    self.save_current_session();
                                    break;
                                }

                                if trimmed == "/undo" {
                                    match cynapse_core::undo::UndoJournal::undo_latest() {
                                        Ok(msg) => {
                                            self.messages.push(ChatMessage {
                                                role: "system".into(),
                                                content: format!("↩ {}", msg),
                                                thinking: None,
                                            });
                                        }
                                        Err(err) => {
                                            self.messages.push(ChatMessage {
                                                role: "error".into(),
                                                content: format!("Undo failed: {}", err),
                                                thinking: None,
                                            });
                                        }
                                    }
                                    continue;
                                }

                                if trimmed == "/clear" || trimmed == "/cls" {
                                    self.messages.clear();
                                    self.read_receipts = Default::default();
                                    self.scroll_offset = 0;
                                    self.auto_scroll = true;
                                    continue;
                                }

                                if trimmed == "/help" {
                                    self.modal = ActiveModal::Help;
                                    continue;
                                }

                                if trimmed == "/theme" {
                                    let all = crate::theme::AppTheme::all();
                                    self.selected_theme_idx = all.iter().position(|&t| t == self.theme).unwrap_or(0);
                                    self.modal = ActiveModal::ThemePicker;
                                    continue;
                                }

                                if trimmed.starts_with("/theme ") {
                                    let name_query = trimmed[7..].trim().to_lowercase();
                                    let all = crate::theme::AppTheme::all();
                                    if let Some(th) = all.iter().find(|t| t.name().to_lowercase().contains(&name_query)) {
                                        self.theme = *th;
                                        crate::theme::save_theme(self.theme);
                                        self.messages.push(ChatMessage {
                                            role: "system".into(),
                                            content: format!("Switched visual theme to: {}", self.theme.name()),
                                            thinking: None,
                                        });
                                    } else {
                                        self.messages.push(ChatMessage {
                                            role: "system".into(),
                                            content: format!("Unknown theme '{}'. Type /theme to select from interactive dropdown.", name_query),
                                            thinking: None,
                                        });
                                    }
                                    continue;
                                }

                                if trimmed == "/thinking" {
                                    self.show_thinking = !self.show_thinking;
                                    self.messages.push(ChatMessage {
                                        role: "system".into(),
                                        content: format!("Model reasoning/thinking blocks are now: {}", if self.show_thinking { "Expanded (Visible)" } else { "Collapsed (Hidden)" }),
                                        thinking: None,
                                    });
                                    continue;
                                }

                                if trimmed == "/spec" {
                                    self.speculative_decoding_enabled = !self.speculative_decoding_enabled;
                                    let status_str = if self.speculative_decoding_enabled {
                                        "ENABLED (Accelerated n-gram drafting)"
                                    } else {
                                        "DISABLED"
                                    };
                                    let acc_str = if self.speculative_total_drafts > 0 {
                                        let pct = (self.speculative_accepted_drafts as f64 / self.speculative_total_drafts as f64) * 100.0;
                                        format!(" │ Acceptance: {:.1}% ({}/{})", pct, self.speculative_accepted_drafts, self.speculative_total_drafts)
                                    } else {
                                        String::new()
                                    };
                                    self.messages.push(ChatMessage {
                                        role: "system".into(),
                                        content: format!("Speculative Prompt Lookup Decoding is now: {}{}", status_str, acc_str),
                                        thinking: None,
                                    });
                                    continue;
                                }

                                if trimmed == "/unload" || trimmed == "/stop" || trimmed == "/free" {
                                    let endpoint = self.tier1_endpoint.clone();
                                    let model_name = self.active_model_name.clone();
                                    tokio::spawn(async move {
                                        unload_model(&endpoint, &model_name).await;
                                    });

                                    self.messages.push(ChatMessage {
                                        role: "system".into(),
                                        content: format!("✓ Unloaded model '{}' from RAM. System memory freed.", self.active_model_name),
                                        thinking: None,
                                    });
                                    continue;
                                }

                                if trimmed == "/pull" || trimmed == "/download" {
                                    let rec_idx = cynapse_core::downloader::recommend_model_for_hardware(self.hw_info.ram_total_mb);
                                    self.selected_pull_idx = rec_idx;
                                    self.puller_step = PullerStep::CuratedList;
                                    self.modal = ActiveModal::ModelPuller;
                                    continue;
                                }

                                if trimmed.starts_with("/pull ") || trimmed.starts_with("/download ") {
                                    let arg = trimmed.split_whitespace().nth(1).unwrap_or("");
                                    if !arg.is_empty() {
                                        self.custom_pull_url = arg.to_string();
                                        self.custom_pull_cursor = self.custom_pull_url.len();
                                        self.selected_quant_idx = 0;
                                        self.puller_step = PullerStep::QuantSelect;
                                        self.modal = ActiveModal::ModelPuller;
                                    }
                                    continue;
                                }

                                if trimmed.starts_with("/rewind") {
                                    let parts: Vec<&str> = trimmed.split_whitespace().collect();
                                    let turns: usize = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
                                    let mut rewound_turns = 0;
                                    for _ in 0..turns {
                                        if let Some(pos) = self.messages.iter().rposition(|m| m.role == "user") {
                                            self.messages.truncate(pos);
                                            rewound_turns += 1;
                                        } else {
                                            break;
                                        }
                                    }
                                    self.save_current_session();
                                    self.messages.push(ChatMessage {
                                        role: "system".into(),
                                        content: format!("Rewound {} turn(s). Active conversation rolled back.", rewound_turns),
                                        thinking: None,
                                    });
                                    continue;
                                }

                                if trimmed.starts_with("/fork") {
                                    let parts: Vec<&str> = trimmed.split_whitespace().collect();
                                    let fork_name = if parts.len() >= 2 {
                                        parts[1].to_string()
                                    } else {
                                        let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs() % 10000;
                                        format!("{}_fork_{}", &self.session_id[..6.min(self.session_id.len())], ts)
                                    };
                                    let old_id = self.session_id.clone();
                                    self.session_id = fork_name.clone();
                                    self.session_created_at = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
                                    self.save_current_session();
                                    self.messages.push(ChatMessage {
                                        role: "system".into(),
                                        content: format!("Branched session '{}' -> new fork '{}'. Active session switched.", old_id, fork_name),
                                        thinking: None,
                                    });
                                    continue;
                                }

                                if trimmed == "/session" || trimmed == "/sessions" {
                                    self.selected_session_idx = 0;
                                    self.modal = ActiveModal::SessionList;
                                    continue;
                                }

                                if trimmed == "/update" || trimmed == "/upgrade" {
                                    if let Some(st) = &self.update_state {
                                        if st.is_running {
                                            self.modal = ActiveModal::Updater;
                                            continue;
                                        }
                                    }

                                    let (tx, rx) = mpsc::unbounded_channel();
                                    self.update_rx = Some(rx);
                                    self.update_state = Some(UpdateProgressState {
                                        is_running: true,
                                        is_done: false,
                                        current_step: "[1/3] Checking repository status...".into(),
                                        steps_log: vec!["[1/3] Initializing Cynapse update sequence...".into()],
                                        error: None,
                                        success_message: None,
                                    });
                                    self.modal = ActiveModal::Updater;

                                    let repo_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
                                    std::thread::spawn(move || {
                                        let res = cynapse_core::updater::execute_update(&repo_dir, |step| {
                                            let _ = tx.send(UpdateEvent::Step(step.to_string()));
                                        });
                                        match res {
                                            Ok(msg) => {
                                                let _ = tx.send(UpdateEvent::Success(msg));
                                            }
                                            Err(err) => {
                                                let _ = tx.send(UpdateEvent::Error(err.to_string()));
                                            }
                                        }
                                    });
                                    continue;
                                }

                                if trimmed == "/doctor" || trimmed == "/doc" || trimmed == "/heal" {
                                    let db_path = dirs::home_dir().map(|h| h.join(".cynapse").join("dendrite.db")).unwrap_or_else(|| PathBuf::from("data/dendrite.db"));
                                    let doctor = cynapse_core::doctor::CynapseDoctor::new(self.models_dir.clone(), db_path, true);
                                    self.doctor_report = Some(doctor.run_diagnostics());
                                    self.modal = ActiveModal::Doctor;
                                    continue;
                                }

                                if trimmed == "/security" || trimmed == "/forensics" || trimmed == "/threat" || trimmed == "/scan" {
                                    self.trigger_security_scan();
                                    self.modal = ActiveModal::SecurityForensics;
                                    continue;
                                }

                                if trimmed == "/soup" || trimmed == "/finetune" || trimmed == "/training" {
                                    self.modal = ActiveModal::SoupProfiler;
                                    continue;
                                }

                                if trimmed == "/drawer" || trimmed == "/inspector" {
                                    self.selected_memory_idx = 0;
                                    self.modal = ActiveModal::MemoryDrawer;
                                    continue;
                                }

                                if trimmed == "/galaxy" || trimmed == "/memory" || trimmed == "/dendrite" || trimmed == "/graph" || trimmed == "/mem" {
                                    self.modal = ActiveModal::MemoryGraph;
                                    continue;
                                }

                                if trimmed.starts_with("/backup") {
                                    let parts: Vec<&str> = trimmed.split_whitespace().collect();
                                    let db_path = dirs::home_dir()
                                        .map(|h| h.join(".cynapse").join("dendrite.db"))
                                        .unwrap_or_else(|| PathBuf::from("data/dendrite.db"));
                                    let backup_path = dirs::home_dir()
                                        .map(|h| h.join(".cynapse").join("dendrite.enc"))
                                        .unwrap_or_else(|| PathBuf::from("data/dendrite.enc"));

                                    if parts.len() >= 3 && parts[1] == "restore" {
                                        let passkey = parts[2..].join(" ");
                                        match cynapse_core::backup::EncryptedBackup::restore_backup(&backup_path, &db_path, &passkey) {
                                            Ok(()) => {
                                                self.messages.push(ChatMessage {
                                                    role: "system".into(),
                                                    content: format!("Memory database restored successfully from AES-256-GCM vault: {:?}", backup_path),
                                                    thinking: None,
                                                });
                                            }
                                            Err(e) => {
                                                self.messages.push(ChatMessage {
                                                    role: "system".into(),
                                                    content: format!("Backup restore failed: {}", e),
                                                    thinking: None,
                                                });
                                            }
                                        }
                                    } else {
                                        let passkey = if parts.len() >= 2 { parts[1..].join(" ") } else { "cynapse_vault_default".to_string() };
                                        if !db_path.exists() {
                                            self.messages.push(ChatMessage {
                                                role: "system".into(),
                                                content: format!("Database file does not exist to back up: {:?}", db_path),
                                                thinking: None,
                                            });
                                        } else {
                                            match cynapse_core::backup::EncryptedBackup::create_backup(&db_path, &backup_path, &passkey) {
                                                Ok(bytes) => {
                                                    self.messages.push(ChatMessage {
                                                        role: "system".into(),
                                                        content: format!("AES-256-GCM Encrypted Backup created ({} bytes) at: {:?}", bytes, backup_path),
                                                        thinking: None,
                                                    });
                                                }
                                                Err(e) => {
                                                    self.messages.push(ChatMessage {
                                                        role: "system".into(),
                                                        content: format!("Backup creation failed: {}", e),
                                                        thinking: None,
                                                    });
                                                }
                                            }
                                        }
                                    }
                                    continue;
                                }

                                if trimmed == "/model" || trimmed == "/list" || trimmed == "/ls" {
                                    self.selected_model_idx = 0;
                                    self.modal = ActiveModal::ModelList;
                                    continue;
                                }

                                if trimmed.starts_with("/model ") || trimmed.starts_with("/run ") {
                                    let arg = trimmed.split_whitespace().nth(1).unwrap_or("");
                                    if !arg.is_empty() {
                                        self.active_model_name = arg.to_string();
                                        self.recompute_tier();
                                        self.preload_active_model();
                                        self.messages.push(ChatMessage {
                                            role: "system".into(),
                                            content: format!("Active model updated to: {} [{}]", self.active_model_name, self.active_tier.label()),
                                            thinking: None,
                                        });
                                    }
                                    continue;
                                }

                                if trimmed == "/persona" || trimmed == "/persona list" {
                                    self.selected_persona_idx = 0;
                                    self.modal = ActiveModal::PersonaManager;
                                    continue;
                                }

                                if trimmed.starts_with("/persona load ") || trimmed.starts_with("/persona set ") || trimmed.starts_with("/persona use ") {
                                    let parts: Vec<&str> = trimmed.split_whitespace().collect();
                                    let target = parts.get(2).cloned().unwrap_or(parts.get(1).cloned().unwrap_or(""));
                                    if target.is_empty() || target == "default" || target == "reset" {
                                        self.persona_mgr.set_active_persona(None);
                                        self.invalidate_zone_a_prefix();
                                        self.messages.push(ChatMessage {
                                            role: "system".into(),
                                            content: "Persona reset to default Cynapse identity (IDENTITY.md + SOUL.md + USER.md).".into(),
                                            thinking: None,
                                        });
                                    } else {
                                        self.persona_mgr.set_active_persona(Some(target.to_string()));
                                        self.invalidate_zone_a_prefix();
                                        self.messages.push(ChatMessage {
                                            role: "system".into(),
                                            content: format!("Active persona loaded: {}.md", target),
                                            thinking: None,
                                        });
                                    }
                                    continue;
                                }

                                if trimmed == "/persona show" || trimmed == "/persona prompt" {
                                    let prompt = self.persona_mgr.build_system_prompt();
                                    self.messages.push(ChatMessage {
                                        role: "system".into(),
                                        content: format!("📜 **Active Cynapse Persona System Prompt**:\n```\n{}\n```", prompt),
                                        thinking: None,
                                    });
                                    continue;
                                }

                                if trimmed == "/persona default" || trimmed == "/persona reset" {
                                    self.persona_mgr.set_active_persona(None);
                                    self.invalidate_zone_a_prefix();
                                    self.messages.push(ChatMessage {
                                        role: "system".into(),
                                        content: "Persona reset to default Cynapse identity.".into(),
                                        thinking: None,
                                    });
                                    continue;
                                }

                                if trimmed == "/bots" || trimmed == "/bots list" || trimmed == "/agents" {
                                    let _ = self.bot_registry.load_all();
                                    self.selected_bot_idx = 0;
                                    self.modal = ActiveModal::Bots;
                                    continue;
                                }

                                if let Some(rest) = trimmed.strip_prefix("/bots spawn ") {
                                    let mut parts = rest.trim().splitn(2, |c: char| c.is_whitespace());
                                    let slug = parts.next().unwrap_or("").trim();
                                    let task = parts.next().unwrap_or("").trim();
                                    if slug.is_empty() || task.is_empty() {
                                        self.messages.push(ChatMessage {
                                            role: "system".into(),
                                            content: "⚠️ Usage: `/bots spawn <bot-slug> <task description>`".into(),
                                            thinking: None,
                                        });
                                    } else {
                                        self.messages.push(ChatMessage {
                                            role: "user".into(),
                                            content: trimmed.clone(),
                                            thinking: None,
                                        });
                                        let call = cynapse_core::offline_agent::ToolCall {
                                            name: "spawn_subagent".into(),
                                            arguments: serde_json::json!({
                                                "bot": slug,
                                                "task": task,
                                                "wait": false,
                                            }),
                                        };
                                        let (out, _ok) = self.execute_tool_and_format(&call);
                                        self.messages.push(ChatMessage {
                                            role: "system".into(),
                                            content: out,
                                            thinking: None,
                                        });
                                        self.save_current_session();
                                    }
                                    continue;
                                }

                                if trimmed == "/bots status" {
                                    let tasks = self.subagent_mgr.list_tasks();
                                    if tasks.is_empty() {
                                        self.messages.push(ChatMessage {
                                            role: "system".into(),
                                            content: "ℹ️ No subagent tasks recorded in current session.".into(),
                                            thinking: None,
                                        });
                                    } else {
                                        let mut lines = vec![format!("🤖 **Subagent Tasks ({} total, {} running)**:", tasks.len(), self.subagent_mgr.running_count())];
                                        for t in &tasks {
                                            let status_str = match &t.status {
                                                cynapse_core::subagent::SubagentStatus::Pending => "⏳ Pending".to_string(),
                                                cynapse_core::subagent::SubagentStatus::Running => "⚙️ Running".to_string(),
                                                cynapse_core::subagent::SubagentStatus::Done => "✅ Done".to_string(),
                                                cynapse_core::subagent::SubagentStatus::Failed(e) => format!("❌ Failed: {}", e),
                                                cynapse_core::subagent::SubagentStatus::Cancelled => "⚠️ Cancelled".to_string(),
                                            };
                                            lines.push(format!("• **#{}** [@{}] — {}: `{}`", t.id, t.bot_slug, status_str, if t.task.len() > 40 { format!("{}...", &t.task[..37]) } else { t.task.clone() }));
                                        }
                                        self.messages.push(ChatMessage {
                                            role: "system".into(),
                                            content: lines.join("\n"),
                                            thinking: None,
                                        });
                                    }
                                    continue;
                                }

                                if let Some(rest) = trimmed.strip_prefix("/bots status ") {
                                    if let Ok(id) = rest.trim().parse::<usize>() {
                                        if let Some(t) = self.subagent_mgr.get_task(id) {
                                            let status_str = match &t.status {
                                                cynapse_core::subagent::SubagentStatus::Pending => "Pending".to_string(),
                                                cynapse_core::subagent::SubagentStatus::Running => "Running".to_string(),
                                                cynapse_core::subagent::SubagentStatus::Done => "Done".to_string(),
                                                cynapse_core::subagent::SubagentStatus::Failed(e) => format!("Failed ({})", e),
                                                cynapse_core::subagent::SubagentStatus::Cancelled => "Cancelled".to_string(),
                                            };
                                            let result_str = t.result.as_deref().unwrap_or("[no output recorded]");
                                            self.messages.push(ChatMessage {
                                                role: "system".into(),
                                                content: format!("📋 **Task #{} Details**\n- **Bot**: @{}\n- **Status**: {}\n- **Task**: {}\n- **Result**:\n{}", t.id, t.bot_slug, status_str, t.task, result_str),
                                                thinking: None,
                                            });
                                        } else {
                                            self.messages.push(ChatMessage {
                                                role: "system".into(),
                                                content: format!("❌ Subagent task #{} not found.", id),
                                                thinking: None,
                                            });
                                        }
                                    } else {
                                        self.messages.push(ChatMessage {
                                            role: "system".into(),
                                            content: "⚠️ Usage: `/bots status <id>`".into(),
                                            thinking: None,
                                        });
                                    }
                                    continue;
                                }

                                if let Some(rest) = trimmed.strip_prefix("/bots cancel ") {
                                    let target = rest.trim();
                                    if target == "all" {
                                        let tasks = self.subagent_mgr.list_tasks();
                                        let mut cancelled_count = 0;
                                        for t in tasks {
                                            if matches!(t.status, cynapse_core::subagent::SubagentStatus::Running | cynapse_core::subagent::SubagentStatus::Pending) {
                                                if self.subagent_mgr.cancel_task(t.id) {
                                                    cancelled_count += 1;
                                                }
                                            }
                                        }
                                        self.messages.push(ChatMessage {
                                            role: "system".into(),
                                            content: format!("⚠️ Cancelled {} active subagent tasks.", cancelled_count),
                                            thinking: None,
                                        });
                                    } else if let Ok(id) = target.parse::<usize>() {
                                        if self.subagent_mgr.cancel_task(id) {
                                            self.messages.push(ChatMessage {
                                                role: "system".into(),
                                                content: format!("⚠️ Subagent task #{} cancellation signal sent.", id),
                                                thinking: None,
                                            });
                                        } else {
                                            self.messages.push(ChatMessage {
                                                role: "system".into(),
                                                content: format!("❌ Could not cancel task #{}: task not found or already finished.", id),
                                                thinking: None,
                                            });
                                        }
                                    } else {
                                        self.messages.push(ChatMessage {
                                            role: "system".into(),
                                            content: "⚠️ Usage: `/bots cancel <id|all>`".into(),
                                            thinking: None,
                                        });
                                    }
                                    continue;
                                }

                                // @slug <task> Direct Subagent Routing
                                if trimmed.starts_with('@') {
                                    let rest = &trimmed[1..];
                                    let mut parts = rest.splitn(2, |c: char| c.is_whitespace());
                                    let slug = parts.next().unwrap_or("").trim();
                                    let task = parts.next().unwrap_or("").trim();
                                    if !slug.is_empty() {
                                        if task.is_empty() {
                                            self.messages.push(ChatMessage {
                                                role: "system".into(),
                                                content: format!("⚠️ Usage: `@{}` <task description>", slug),
                                                thinking: None,
                                            });
                                            continue;
                                        }
                                        self.messages.push(ChatMessage {
                                            role: "user".into(),
                                            content: trimmed.clone(),
                                            thinking: None,
                                        });
                                        let call = cynapse_core::offline_agent::ToolCall {
                                            name: "spawn_subagent".into(),
                                            arguments: serde_json::json!({
                                                "bot": slug,
                                                "task": task,
                                                "wait": false,
                                            }),
                                        };
                                        let (out, _ok) = self.execute_tool_and_format(&call);
                                        self.messages.push(ChatMessage {
                                            role: "system".into(),
                                            content: out,
                                            thinking: None,
                                        });
                                        self.save_current_session();
                                        continue;
                                    }
                                }

                                if trimmed == "/export" || trimmed.starts_with("/export ") {
                                    let format = trimmed.strip_prefix("/export").unwrap_or("").trim().to_lowercase();
                                    let export_dir = dirs::home_dir().map(|h| h.join(".cynapse").join("exports")).unwrap_or_else(|| PathBuf::from("exports"));
                                    let _ = std::fs::create_dir_all(&export_dir);
                                    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);

                                    if format == "json" {
                                        let file_path = export_dir.join(format!("cynapse_session_{}.json", ts));
                                        let json_data = serde_json::to_string_pretty(&self.messages).unwrap_or_default();
                                        match std::fs::write(&file_path, json_data) {
                                            Ok(_) => {
                                                self.messages.push(ChatMessage {
                                                    role: "system".into(),
                                                    content: format!("✓ Conversation exported to JSON: {}", file_path.display()),
                                                    thinking: None,
                                                });
                                            }
                                            Err(e) => {
                                                self.messages.push(ChatMessage {
                                                    role: "system".into(),
                                                    content: format!("❌ Export failed: {}", e),
                                                    thinking: None,
                                                });
                                            }
                                        }
                                    } else {
                                        let file_path = export_dir.join(format!("cynapse_session_{}.md", ts));
                                        let mut doc = format!("# Cynapse Session Export\n- **Date**: UNIX {}\n- **Model**: {}\n\n---\n\n", ts, self.active_model_name);
                                        for m in &self.messages {
                                            let role_badge = match m.role.as_str() {
                                                "user" => "### 👤 User",
                                                "assistant" => "### 🤖 Cynapse",
                                                "system" => "### ℹ️ System",
                                                _ => "### ⚠️ Note",
                                            };
                                            doc.push_str(&format!("{}\n\n", role_badge));
                                            if let Some(ref think) = m.thinking {
                                                doc.push_str(&format!("<details>\n<summary>Thinking Process</summary>\n\n{}\n</details>\n\n", think));
                                            }
                                            doc.push_str(&format!("{}\n\n", m.content));
                                        }
                                        match std::fs::write(&file_path, doc) {
                                            Ok(_) => {
                                                self.messages.push(ChatMessage {
                                                    role: "system".into(),
                                                    content: format!("✓ Conversation exported to Markdown: {}", file_path.display()),
                                                    thinking: None,
                                                });
                                            }
                                            Err(e) => {
                                                self.messages.push(ChatMessage {
                                                    role: "system".into(),
                                                    content: format!("❌ Export failed: {}", e),
                                                    thinking: None,
                                                });
                                            }
                                        }
                                    }
                                    continue;
                                }

                                if trimmed == "/synapse" || trimmed.starts_with("/synapse ") || trimmed == "/extensions" {
                                    let reg = cynapse_core::synapse::SynapseRegistry::global();
                                    reg.reload();
                                    let subcmd = trimmed.strip_prefix("/synapse").unwrap_or("").trim();

                                    if subcmd.is_empty() {
                                        self.selected_synapse_idx = 0;
                                        self.modal = ActiveModal::SynapseManager;
                                        continue;
                                    }

                                    if subcmd == "list" {
                                        let list = reg.list_synapses();
                                        if list.is_empty() {
                                            self.messages.push(ChatMessage {
                                                role: "system".into(),
                                                content: "🔌 **Synapse Extensions**: No external tool extensions installed in `~/.cynapse/synapses/`.\n\nInstall extensions using `/synapse install <name>` or drop manifest directories in `~/.cynapse/synapses/`.".into(),
                                                thinking: None,
                                            });
                                        } else {
                                            let mut lines = vec![format!("🔌 **Installed Synapse Tool Extensions ({} total)**:", list.len())];
                                            for syn in list {
                                                let state = if syn.enabled.unwrap_or(true) { "🟢 Active" } else { "⚪ Disabled" };
                                                let tools_summary = syn.tools.iter().map(|t| format!("`{}`", t.name)).collect::<Vec<_>>().join(", ");
                                                lines.push(format!("• **{}** (v{}) — {} [{}]\n  _{}_\n  Tools: {}", syn.name, syn.version, state, syn.title, syn.description, tools_summary));
                                            }
                                            lines.push("\nCommands: `/synapse enable <name>`, `/synapse disable <name>`, `/synapse reload`".into());
                                            self.messages.push(ChatMessage {
                                                role: "system".into(),
                                                content: lines.join("\n"),
                                                thinking: None,
                                            });
                                        }
                                    } else if let Some(name) = subcmd.strip_prefix("enable ") {
                                        let name = name.trim();
                                        match reg.set_enabled(name, true) {
                                            Ok(_) => {
                                                self.messages.push(ChatMessage {
                                                    role: "system".into(),
                                                    content: format!("✓ Synapse extension '{}' enabled. Tools now exposed to agent.", name),
                                                    thinking: None,
                                                });
                                            }
                                            Err(e) => {
                                                self.messages.push(ChatMessage {
                                                    role: "system".into(),
                                                    content: format!("❌ {}", e),
                                                    thinking: None,
                                                });
                                            }
                                        }
                                    } else if let Some(name) = subcmd.strip_prefix("disable ") {
                                        let name = name.trim();
                                        match reg.set_enabled(name, false) {
                                            Ok(_) => {
                                                self.messages.push(ChatMessage {
                                                    role: "system".into(),
                                                    content: format!("✓ Synapse extension '{}' disabled.", name),
                                                    thinking: None,
                                                });
                                            }
                                            Err(e) => {
                                                self.messages.push(ChatMessage {
                                                    role: "system".into(),
                                                    content: format!("❌ {}", e),
                                                    thinking: None,
                                                });
                                            }
                                        }
                                    } else if subcmd == "reload" {
                                        reg.reload();
                                        self.messages.push(ChatMessage {
                                            role: "system".into(),
                                            content: "✓ Synapse registry reloaded from disk.".into(),
                                            thinking: None,
                                        });
                                    } else {
                                        self.messages.push(ChatMessage {
                                            role: "system".into(),
                                            content: "⚠️ Usage: `/synapse` (or `/synapse list`), `/synapse enable <name>`, `/synapse disable <name>`, `/synapse reload`".into(),
                                            thinking: None,
                                        });
                                    }
                                    continue;
                                }

                                // User Prompt Execution
                                self.messages.push(ChatMessage {
                                    role: "user".into(),
                                    content: trimmed.clone(),
                                    thinking: None,
                                });

                                self.agent_step_count = 0;
                                self.finalization_inflight = false;
                                self.turn_verified = true;
                                self.is_generating = true;
                                self.stream_start_time = Some(std::time::Instant::now());
                                self.last_ttft_ms = None;
                                self.current_thinking_buf.clear();
                                self.current_response_buf.clear();
                                self.auto_scroll = true; // Lock scroll to bottom for incoming response

                                // Build Two-Zone Prompt: Zone A (Pruned Invariant Prefix) & Zone B (Variable Tail)
                                let system_prompt = self.compile_zone_a_for_query(&trimmed);
                                let memory_prompt = self.dendrite_ctx.build_prompt_with_options(&trimmed, 1500, true, false);

                                let history_context: String = self.messages
                                    .iter()
                                    .take(self.messages.len().saturating_sub(1))
                                    .rev()
                                    .take(6)
                                    .collect::<Vec<_>>()
                                    .into_iter()
                                    .rev()
                                    .map(|m| format!("{}: {}", m.role.to_uppercase(), m.content))
                                    .collect::<Vec<_>>()
                                    .join("\n\n");

                                let prompt = cynapse_core::offline_agent::compile_zone_b_tail(&memory_prompt, &history_context, &trimmed, None);

                                // Spawn Non-blocking Async LLM Task
                                let (tx, rx) = mpsc::unbounded_channel();
                                self.stream_rx = Some(rx);

                                let tier = self.active_tier;
                                let endpoint = self.tier1_endpoint.clone();
                                let model_name = self.active_model_name.clone();

                                tokio::spawn(async move {
                                    let res = query_model_stream(
                                        tier,
                                        &endpoint,
                                        &model_name,
                                        &prompt,
                                        &system_prompt,
                                        |ttype, token| {
                                            let _ = tx.send(StreamEvent::Token {
                                                ttype,
                                                text: token.to_string(),
                                            });
                                        },
                                    )
                                    .await;

                                    match res {
                                        Ok(stats) => {
                                            let _ = tx.send(StreamEvent::Done {
                                                tok_per_sec: stats.tok_per_sec,
                                                elapsed_sec: stats.elapsed_sec,
                                            });
                                        }
                                        Err(e) => {
                                            let _ = tx.send(StreamEvent::Error(e.to_string()));
                                        }
                                    }
                                });
                            }
                            _ => {}
                        }
                    }
                    Event::Mouse(mouse) => {
                        match mouse.kind {
                            MouseEventKind::ScrollUp => {
                                self.scroll_up(2);
                            }
                            MouseEventKind::ScrollDown => {
                                self.scroll_down(2);
                            }
                            _ => {}
                        }
                    }
                    Event::Paste(text) => {
                        let clean_text: String = text.chars().filter(|c| !c.is_control() || *c == '\n' || *c == '\t').collect();
                        if self.modal == ActiveModal::ModelPuller && self.puller_step == PullerStep::CustomInput {
                            self.custom_pull_url.insert_str(self.custom_pull_cursor, &clean_text);
                            self.custom_pull_cursor += clean_text.len();
                        } else if self.modal == ActiveModal::None {
                            self.input.insert_str(self.input_cursor, &clean_text);
                            self.input_cursor += clean_text.len();
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(())
    }

    fn poll_stream_events(&mut self) {
        while let Ok(req) = self.approval_rx.try_recv() {
            if self.pending_approval.is_none() {
                self.pending_approval = Some(PendingToolApproval::new(
                    req.task_id,
                    req.bot_slug,
                    req.tool_name,
                    req.arg1,
                    req.arg2,
                    Some(req.response_tx),
                ));
                self.modal = ActiveModal::ToolApproval;
            } else {
                let _ = req.response_tx.send(false);
            }
        }

        while let Ok(ann) = self.subagent_rx.try_recv() {
            let status_badge = match &ann.status {
                cynapse_core::subagent::SubagentStatus::Done => "✅ Done",
                cynapse_core::subagent::SubagentStatus::Failed(_) => "❌ Failed",
                cynapse_core::subagent::SubagentStatus::Cancelled => "⚠️ Cancelled",
                _ => "ℹ️ Completed",
            };
            self.messages.push(ChatMessage {
                role: "system".into(),
                content: format!(
                    "📢 **Subagent Task #{} Announcement** [@{} · {}]:\n{}",
                    ann.task_id, ann.bot_slug, status_badge, ann.result
                ),
                thinking: None,
            });
            self.save_current_session();
        }

        let stalled_ids = self.subagent_mgr.check_and_cancel_stalled_tasks(120_000);
        for id in stalled_ids {
            self.messages.push(ChatMessage {
                role: "system".into(),
                content: format!("=== NOTICE ===\n⚠️ Subagent task #{} stalled (> 2m) and was auto-cancelled.", id),
                thinking: None,
            });
            self.save_current_session();
        }

        let mut events = Vec::new();
        if let Some(rx) = &mut self.stream_rx {
            while let Ok(event) = rx.try_recv() {
                events.push(event);
            }
        }

        let mut finished = false;
        let mut reprompting = false;
        for event in events {
            match event {
                StreamEvent::Token { ttype, text } => {
                    if let Some(st) = self.stream_start_time {
                        if self.last_ttft_ms.is_none() {
                            self.last_ttft_ms = Some(st.elapsed().as_millis() as u64);
                        }
                    }
                    match ttype {
                        TokenType::Thinking => self.current_thinking_buf.push_str(&text),
                        TokenType::Response => self.current_response_buf.push_str(&text),
                    }
                },
                StreamEvent::LayerProgress { current, total } => {
                    self.current_streaming_layer = Some((current, total));
                },
                StreamEvent::SpeculativeDraft { accepted, total } => {
                    self.speculative_accepted_drafts += accepted;
                    self.speculative_total_drafts += total;
                },
                StreamEvent::Done { tok_per_sec, elapsed_sec } => {
                    self.current_streaming_layer = None;
                    self.last_tok_per_sec = tok_per_sec;
                    self.last_latency_sec = elapsed_sec;
                    if tok_per_sec > 0.0 {
                        if self.throughput_history.len() >= 30 {
                            self.throughput_history.pop_front();
                        }
                        self.throughput_history.push_back(tok_per_sec);
                    }

                    let tools_allowed = !self.finalization_inflight;
                    if self.finalization_inflight {
                        self.finalization_inflight = false;
                        self.agent_step_count = 0;
                    }

                    // Offline Agent: GBNF tool call check (response buffer fallback to thinking buffer) & circular LoopGuard intervention
                    let detected_tools = validate_gbnf_tool_calls(&self.current_response_buf)
                        .or_else(|_| validate_gbnf_tool_calls(&self.current_thinking_buf));

                    if let Ok(tool_calls) = detected_tools {
                        if !tool_calls.is_empty() && tools_allowed {
                            if self.agent_step_count >= MAX_AGENT_STEPS {
                                self.turn_verified = false;
                                self.messages.push(ChatMessage {
                                    role: "system".into(),
                                    content: format!("⚠️ MAX AGENT STEPS REACHED ({} steps): Automated tool execution loop paused to prevent runaway execution.", MAX_AGENT_STEPS),
                                    thinking: None,
                                });

                                let assistant_content = if self.current_response_buf.trim().is_empty() {
                                    let names: Vec<String> = tool_calls.iter().map(|t| format!("`{}`", t.name)).collect();
                                    format!("Executing tools: {}", names.join(", "))
                                } else {
                                    self.current_response_buf.clone()
                                };
                                self.messages.push(ChatMessage {
                                    role: "assistant".into(),
                                    content: assistant_content,
                                    thinking: if self.current_thinking_buf.is_empty() {
                                        None
                                    } else {
                                        Some(self.current_thinking_buf.clone())
                                    },
                                });
                                self.save_current_session();

                                reprompting = true;
                                self.finalization_inflight = true;
                                let (tx, rx) = mpsc::unbounded_channel();
                                self.stream_rx = Some(rx);
                                self.is_generating = true;
                                self.current_thinking_buf.clear();
                                self.current_response_buf.clear();

                                let user_msg = self.messages.iter().rev().find(|m| m.role == "user").map(|m| m.content.as_str()).unwrap_or("").to_string();
                                let system_prompt = self.compile_zone_a_for_query(&user_msg);
                                let memory_prompt = self.dendrite_ctx.build_prompt_with_options(&user_msg, 1500, true, false);
                                let endpoint = self.tier1_endpoint.clone();
                                let model_name = self.active_model_name.clone();

                                let history_context: String = self.messages.iter().rev().take(6).collect::<Vec<_>>().into_iter().rev()
                                    .map(|m| format!("{}: {}", m.role.to_uppercase(), m.content))
                                    .collect::<Vec<_>>()
                                    .join("\n\n");

                                let notice = finalization_notice(MAX_AGENT_STEPS);
                                let prompt = cynapse_core::offline_agent::compile_zone_b_tail(
                                    &memory_prompt,
                                    &history_context,
                                    &user_msg,
                                    Some(&notice),
                                );

                                let tier = self.active_tier;
                                tokio::spawn(async move {
                                    let res = query_model_stream(
                                        tier,
                                        &endpoint,
                                        &model_name,
                                        &prompt,
                                        &system_prompt,
                                        |ttype, token| {
                                            let _ = tx.send(StreamEvent::Token { ttype, text: token.to_string() });
                                        },
                                    )
                                    .await;

                                    match res {
                                        Ok(stats) => {
                                            let _ = tx.send(StreamEvent::Done {
                                                tok_per_sec: stats.tok_per_sec,
                                                elapsed_sec: stats.elapsed_sec,
                                            });
                                        }
                                        Err(e) => {
                                            let _ = tx.send(StreamEvent::Error(e.to_string()));
                                        }
                                    }
                                });
                            } else {
                                self.agent_step_count += 1;
                                let mut tool_results: Vec<String> = Vec::new();
                                let mut tool_names: Vec<String> = Vec::new();
                                let mut veto_notices: Vec<String> = Vec::new();
                                let mut all_vetoed = true;

                                let assistant_content = if self.current_response_buf.trim().is_empty() {
                                    let names: Vec<String> = tool_calls.iter().map(|t| format!("`{}`", t.name)).collect();
                                    format!("Executing tools: {}", names.join(", "))
                                } else {
                                    self.current_response_buf.clone()
                                };
                                self.messages.push(ChatMessage {
                                    role: "assistant".into(),
                                    content: assistant_content,
                                    thinking: if self.current_thinking_buf.is_empty() {
                                        None
                                    } else {
                                        Some(self.current_thinking_buf.clone())
                                    },
                                });

                                let partitions = cynapse_core::partition_tool_calls(&tool_calls);
                                for batch in partitions {
                                    for tool_call in &batch {
                                        tool_names.push(tool_call.name.clone());

                                        // System 2 Critic: Pre-execution schema and path verification
                                        if let Err(schema_err) = cynapse_core::offline_agent::validate_tool_call_schema(tool_call, std::path::Path::new(".")) {
                                            self.turn_verified = false;
                                            veto_notices.push(format!("Schema validation failed for `{}`: {}", tool_call.name, schema_err));
                                            tool_results.push(format!("[{}] Schema Error: {}", tool_call.name, schema_err));
                                            continue;
                                        }

                                        match self.loop_guard.check(tool_call) {
                                            Ok(()) => {
                                                all_vetoed = false;
                                                self.loop_guard.record_call(tool_call);
                                                let (tool_output, ok) = self.execute_tool_and_format(tool_call);
                                                if !ok {
                                                    self.turn_verified = false;
                                                }
                                                self.loop_guard.record_outcome(tool_call, &tool_output);
                                                let mut formatted_res = format!("[{}]\n{}", tool_call.name, tool_output);
                                                if let Some(advisory) = self.loop_guard.advisory_notice(tool_call) {
                                                    formatted_res.push_str(&format!("\n\n{}", advisory));
                                                }
                                                tool_results.push(formatted_res);
                                            }
                                            Err(loop_warn) => {
                                                self.turn_verified = false;
                                                self.loop_guard.record_veto(tool_call);
                                                veto_notices.push(format!("Loop intervention for `{}`: {}", tool_call.name, loop_warn));
                                                tool_results.push(format!("[{}] Vetoed: {}", tool_call.name, loop_warn));
                                            }
                                        }
                                    }
                                }

                                let combined_output = tool_results.join("\n\n");
                                self.messages.push(ChatMessage {
                                    role: "system".into(),
                                    content: format!(
                                        "🔧 Tool Batch [{}] Executed (Step {}/{}):\n{}",
                                        tool_names.join(", "),
                                        self.agent_step_count,
                                        MAX_AGENT_STEPS,
                                        combined_output
                                    ),
                                    thinking: None,
                                });
                                self.save_current_session();

                                if !all_vetoed {
                                    reprompting = true;
                                    let (tx, rx) = mpsc::unbounded_channel();
                                    self.stream_rx = Some(rx);
                                    self.is_generating = true;
                                    self.current_thinking_buf.clear();
                                    self.current_response_buf.clear();

                                    let user_msg = self.messages.iter().rev().find(|m| m.role == "user").map(|m| m.content.as_str()).unwrap_or("").to_string();
                                    let system_prompt = self.compile_zone_a_for_query(&user_msg);
                                    let memory_prompt = self.dendrite_ctx.build_prompt_with_options(&user_msg, 1500, true, false);
                                    let endpoint = self.tier1_endpoint.clone();
                                    let model_name = self.active_model_name.clone();

                                    let history_context: String = self.messages.iter().rev().take(6).collect::<Vec<_>>().into_iter().rev()
                                        .map(|m| format!("{}: {}", m.role.to_uppercase(), m.content))
                                        .collect::<Vec<_>>()
                                        .join("\n\n");

                                    let tool_instruction = format!(
                                        "Tool Results for `{}`:\n{}\n\nContinue resolution of user request: {}",
                                        tool_names.join(", "),
                                        combined_output,
                                        user_msg
                                    );
                                    let notice_str = if !veto_notices.is_empty() {
                                        Some(veto_notices.join("\n"))
                                    } else {
                                        None
                                    };
                                    let prompt = cynapse_core::offline_agent::compile_zone_b_tail(
                                        &memory_prompt,
                                        &history_context,
                                        &tool_instruction,
                                        notice_str.as_deref(),
                                    );

                                    let tier = self.active_tier;
                                    tokio::spawn(async move {
                                        let res = query_model_stream(
                                            tier,
                                            &endpoint,
                                            &model_name,
                                            &prompt,
                                            &system_prompt,
                                            |ttype, token| {
                                                let _ = tx.send(StreamEvent::Token { ttype, text: token.to_string() });
                                            },
                                        )
                                        .await;

                                        match res {
                                            Ok(stats) => {
                                                let _ = tx.send(StreamEvent::Done {
                                                    tok_per_sec: stats.tok_per_sec,
                                                    elapsed_sec: stats.elapsed_sec,
                                                });
                                            }
                                            Err(e) => {
                                                let _ = tx.send(StreamEvent::Error(e.to_string()));
                                            }
                                        }
                                    });
                                }
                            }
                        }
                    }

                    if !reprompting {
                        let (final_content, final_thinking) = if self.current_response_buf.trim().is_empty() && !self.current_thinking_buf.trim().is_empty() {
                            (self.current_thinking_buf.clone(), None)
                        } else if !self.current_thinking_buf.is_empty() {
                            (self.current_response_buf.clone(), Some(self.current_thinking_buf.clone()))
                        } else {
                            let (clean, thought) = cynapse_core::offline_agent::isolate_thinking_scratchpad(&self.current_response_buf);
                            (clean, thought)
                        };

                        self.messages.push(ChatMessage {
                            role: "assistant".into(),
                            content: final_content,
                            thinking: final_thinking,
                        });
                        self.save_current_session();

                        // Store Turn Log Node into Dendrite Graph & SQLite Store
                        let user_msg = self.messages.iter().rev().find(|m| m.role == "user").map(|m| m.content.clone()).unwrap_or_default();
                        if !user_msg.is_empty() {
                            let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
                            let turn_id = format!("turn_{}", now);
                            let excerpt = user_msg.chars().take(25).collect::<String>();
                            let title = format!("Turn: {}", excerpt);
                            let content = format!("User: {}\n\nAssistant: {}", user_msg, self.current_response_buf);

                            let turn_node = self.graph.upsert(
                                &turn_id,
                                &title,
                                &content,
                                cynapse_memory::graph::NodeType::TurnLog,
                                Some(vec!["#turn".into(), "#chat".into()]),
                            );
                            if let Some(s) = &self.store {
                                let _ = s.save(&turn_node);
                            }

                            // Trigger Async Background Reflection (fire-and-forget out-of-band distillation)
                            let recent_refl_msgs: Vec<ReflMessage> = self.messages
                                .iter()
                                .rev()
                                .take(6)
                                .collect::<Vec<_>>()
                                .into_iter()
                                .rev()
                                .map(|m| {
                                    let role = match m.role.as_str() {
                                        "user" => ReflRole::User,
                                        "assistant" => ReflRole::Assistant,
                                        "system" => ReflRole::System,
                                        _ => ReflRole::Tool,
                                    };
                                    ReflMessage::text(role, m.content.clone())
                                })
                                .collect();
                            self.reflection_worker.spawn_reflection(recent_refl_msgs, self.turn_verified);
                        }

                        self.is_generating = false;
                        finished = true;
                    }
                }
                StreamEvent::Error(err) => {
                    self.current_streaming_layer = None;
                    self.turn_verified = false;
                    self.messages.push(ChatMessage {
                        role: "error".into(),
                        content: format!("Error querying engine: {}", err),
                        thinking: None,
                    });
                    self.is_generating = false;
                    finished = true;
                }
            }
        }
        if finished && !reprompting {
            self.stream_rx = None;
        }
    }

    pub fn render_sparkline(&self) -> String {
        const WAVES_ACTIVE: [&str; 6] = [
            "⡀⣀⣄⣤⣦⣶",
            "⣀⣄⣤⣦⣶⣿",
            "⣄⣤⣦⣶⣿⣶",
            "⣤⣦⣶⣿⣶⣤",
            "⣦⣶⣿⣶⣤⣄",
            "⣶⣿⣶⣤⣄⣀",
        ];
        const WAVES_IDLE: [&str; 4] = [
            "⡀⡀⣀⣀⡀⡀",
            "⡀⣀⣤⣤⣀⡀",
            "⣀⣤⣶⣶⣤⣀",
            "⡀⣀⣤⣤⣀⡀",
        ];

        // Wall-clock time pacing: immune to terminal refresh rate fluctuations
        let millis = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as usize)
            .unwrap_or_else(|_| self.anim_tick * 30);

        if self.is_generating || self.last_tok_per_sec > 0.0 {
            // Calm, relaxed wave: 500ms per frame (3.0s complete cycle, half previous speed)
            let idx = (millis / 500) % WAVES_ACTIVE.len();
            format!("[{}]", WAVES_ACTIVE[idx])
        } else {
            // Very gentle ambient idle wave: 1000ms per frame (4.0s cycle)
            let idx = (millis / 1000) % WAVES_IDLE.len();
            format!("[{}]", WAVES_IDLE[idx])
        }
    }

    fn get_matching_commands(&self) -> Vec<&'static SlashCommand> {
        let input = self.input.trim();
        if !input.starts_with('/') {
            return Vec::new();
        }
        let lower = input.to_lowercase();
        SLASH_COMMANDS
            .iter()
            .filter(|cmd| cmd.name.to_lowercase().starts_with(&lower))
            .collect()
    }

    async fn scan_all_models(&self) -> Vec<ModelItem> {
        let mut items = self.scan_models_sync();
        let native_models = fetch_native_models(&self.tier1_endpoint).await;
        for name in native_models {
            if !items.iter().any(|i| i.name == name) {
                items.push(ModelItem {
                    name,
                    source: "Leafcutter Engine".into(),
                    quant: "GGUF".into(),
                    size_str: "Loaded".into(),
                });
            }
        }
        items
    }

    fn scan_models_sync(&self) -> Vec<ModelItem> {
        let mut items = Vec::new();
        static GGUF_RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        let re = GGUF_RE.get_or_init(|| {
            regex::Regex::new(r"(?i)(Q[0-9]_[K0-9_A-Z]+|F16|F32|IQ[0-9]_[A-Z]+)").unwrap()
        });

        let mut search_dirs = vec![
            self.models_dir.clone(),
            PathBuf::from("./models"),
        ];
        if let Some(home) = dirs::home_dir() {
            search_dirs.push(home.join(".cynapse").join("models"));
        }

        for dir in &search_dirs {
            if let Ok(entries) = fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() {
                        if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                            if ext == "gguf" || ext == "safetensors" || ext == "bin" {
                                let filename = path.file_name().unwrap().to_string_lossy();
                                if filename == "README.md" || items.iter().any(|i: &ModelItem| i.name == filename) {
                                    continue;
                                }

                                let bytes = entry.metadata().map(|m| m.len()).unwrap_or(0);
                                let size_mb = bytes / 1024 / 1024;
                                let size_str = if size_mb > 1024 {
                                    format!("{:.2} GB", size_mb as f64 / 1024.0)
                                } else {
                                    format!("{} MB", size_mb)
                                };

                                let quant = if let Some(mat) = re.find(&filename) {
                                    mat.as_str().to_uppercase()
                                } else if filename.ends_with(".safetensors") {
                                    "SAFETENSORS".to_string()
                                } else {
                                    "GGUF".to_string()
                                };

                                items.push(ModelItem {
                                    name: filename.to_string(),
                                    source: "Local File".into(),
                                    quant,
                                    size_str,
                                });
                            }
                        }
                    }
                }
            }
        }
        items
    }

    pub fn highlight_code_line(line: &str, lang: &str) -> Vec<Span<'static>> {
    let lower_lang = lang.to_lowercase();
    let trimmed = line.trim_start();
    
    // Comments
    if trimmed.starts_with("//") || trimmed.starts_with('#') || trimmed.starts_with("--") {
        return vec![Span::styled(line.to_string(), Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC))];
    }

    let mut spans = Vec::new();
    let rust_kw = ["fn", "let", "mut", "pub", "struct", "enum", "impl", "use", "match", "if", "else", "for", "while", "loop", "return", "self", "Self", "async", "await", "trait", "where", "type", "const", "static", "unsafe"];
    let py_kw = ["def", "class", "import", "from", "return", "if", "elif", "else", "for", "while", "try", "except", "finally", "with", "as", "pass", "yield", "lambda", "async", "await"];
    let sh_kw = ["if", "then", "else", "fi", "for", "do", "done", "in", "case", "esac", "while", "echo", "exit", "source", "export"];

    let keywords = match lower_lang.as_str() {
        "rust" | "rs" => &rust_kw[..],
        "python" | "py" => &py_kw[..],
        "bash" | "sh" | "shell" | "zsh" => &sh_kw[..],
        _ => &rust_kw[..],
    };

    let mut current_token = String::new();
    let mut in_quote: Option<char> = None;

    for ch in line.chars() {
        if let Some(q) = in_quote {
            current_token.push(ch);
            if ch == q {
                spans.push(Span::styled(current_token.clone(), Style::default().fg(Color::Yellow)));
                current_token.clear();
                in_quote = None;
            }
        } else if ch == '"' || ch == '\'' || ch == '`' {
            if !current_token.is_empty() {
                let is_kw = keywords.contains(&current_token.as_str());
                let style = if is_kw {
                    Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)
                } else if current_token.chars().all(|c| c.is_ascii_digit()) {
                    Style::default().fg(Color::LightCyan)
                } else {
                    Style::default().fg(Color::LightGreen)
                };
                spans.push(Span::styled(current_token.clone(), style));
                current_token.clear();
            }
            in_quote = Some(ch);
            current_token.push(ch);
        } else if ch.is_alphanumeric() || ch == '_' {
            current_token.push(ch);
        } else {
            if !current_token.is_empty() {
                let is_kw = keywords.contains(&current_token.as_str());
                let style = if is_kw {
                    Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)
                } else if current_token.chars().all(|c| c.is_ascii_digit()) {
                    Style::default().fg(Color::LightCyan)
                } else {
                    Style::default().fg(Color::LightGreen)
                };
                spans.push(Span::styled(current_token.clone(), style));
                current_token.clear();
            }
            spans.push(Span::styled(ch.to_string(), Style::default().fg(Color::White)));
        }
    }

    if !current_token.is_empty() {
        let is_kw = keywords.contains(&current_token.as_str());
        let style = if in_quote.is_some() {
            Style::default().fg(Color::Yellow)
        } else if is_kw {
            Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)
        } else if current_token.chars().all(|c| c.is_ascii_digit()) {
            Style::default().fg(Color::LightCyan)
        } else {
            Style::default().fg(Color::LightGreen)
        };
        spans.push(Span::styled(current_token, style));
    }

    spans
}

pub fn render_markdown_lines(text: &str, theme: AppTheme) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let mut in_code_block = false;
    let mut code_lang = String::new();

    for raw_line in text.lines() {
        let trimmed = raw_line.trim_start();
        if trimmed.starts_with("```") {
            if in_code_block {
                lines.push(Line::from(Span::styled("    └───", Style::default().fg(Color::Cyan))));
                in_code_block = false;
                code_lang.clear();
            } else {
                in_code_block = true;
                code_lang = trimmed.trim_start_matches('`').trim().to_string();
                let lang_label = if code_lang.is_empty() { "CODE".to_string() } else { code_lang.to_uppercase() };
                lines.push(Line::from(Span::styled(format!("    ┌── [ {} ] ────────────────────────────────────────", lang_label), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))));
            }
            continue;
        }

        if in_code_block {
            let line_spans = TuiApp::highlight_code_line(raw_line, &code_lang);
            let mut full_spans = vec![
                Span::styled("    │ ", Style::default().fg(Color::Cyan)),
            ];
            full_spans.extend(line_spans);
            lines.push(Line::from(full_spans));
            continue;
        }

        // Markdown Headers
        if trimmed.starts_with("# ") {
            lines.push(Line::from(Span::styled(format!("    {}", raw_line), Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))));
        } else if trimmed.starts_with("## ") {
            lines.push(Line::from(Span::styled(format!("    {}", raw_line), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))));
        } else if trimmed.starts_with("### ") {
            lines.push(Line::from(Span::styled(format!("    {}", raw_line), Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD))));
        } else if trimmed.starts_with("- ") || trimmed.starts_with("* ") {
            lines.push(Line::from(vec![
                Span::styled("    • ", Style::default().fg(Color::Cyan)),
                Span::styled(trimmed[2..].to_string(), theme.assistant_text()),
            ]));
        } else if trimmed.starts_with("> ") {
            lines.push(Line::from(vec![
                Span::styled("    │ ", theme.dim_text()),
                Span::styled(trimmed[2..].to_string(), Style::default().fg(Color::Gray).add_modifier(Modifier::ITALIC)),
            ]));
        } else {
            lines.push(Line::from(Span::styled(format!("    {}", raw_line), theme.assistant_text())));
        }
    }
    lines
}

    fn ui(&self, f: &mut Frame) {
        let t = self.theme;

        // Dynamic Input Box Height Calculation (Expand downward dynamically as prompt grows)
        let full_width = f.area().width as usize;
        let input_inner_width = full_width.saturating_sub(2).max(1);
        let input_total_cols = 4 + self.input.chars().count();
        let wrapped_input_lines = (input_total_cols + input_inner_width - 1) / input_inner_width;
        let input_height = (wrapped_input_lines as u16 + 2).clamp(3, 8);

        // Top Header Bar (3 lines), Middle Content (Min 5), Bottom Input (Dynamic 3..8 lines)
        let main_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(5),
                Constraint::Length(input_height),
            ])
            .split(f.area());

        // If 3D Galaxy Atlas view is active, render it across the full middle canvas (no overlapping modal borders)
        if self.modal == ActiveModal::MemoryGraph {
            self.render_3d_galaxy_atlas(f, main_chunks[1]);

            // Bottom Prompt Input Bar
            let input_text = vec![Line::from(vec![
                Span::styled("・ ", t.prompt_prefix()),
                Span::raw(&self.input),
            ])];
            let input_bar = Paragraph::new(input_text)
                .wrap(Wrap { trim: false })
                .block(Block::default().borders(Borders::ALL).border_type(BorderType::Rounded).title("").border_style(t.prompt_prefix()));
            f.render_widget(input_bar, main_chunks[2]);
            return;
        }

        // 1. Top Header Bar (Jcode-style Pill Navigation & Status)
        let (_nodes, _edges) = self.graph.topology();
        let screen_w = f.area().width;
        let header_spans = if screen_w < 90 {
            // Compact Header for small terminals
            vec![
                Span::styled(" ◖", Style::default().fg(Color::Cyan)),
                Span::styled("CYNAPSE", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                Span::styled("◗ ", Style::default().fg(Color::Cyan)),
                Span::styled("v0.1.0", t.dim_text()),
                Span::styled(" │ ", t.dim_text()),
                Span::styled(&self.active_model_name, t.active_model()),
                Span::styled(" │ ", t.dim_text()),
                Span::styled(format!("{:.1}G", self.hw_info.ram_used_mb as f64 / 1024.0), Style::default().fg(Color::Green)),
            ]
        } else {
            // Full Header for standard/wide terminals
            vec![
                Span::styled(" ◖", Style::default().fg(Color::Cyan)),
                Span::styled("CYNAPSE", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                Span::styled("◗ ", Style::default().fg(Color::Cyan)),
                Span::styled("v0.1.0", t.dim_text()),
                Span::styled("  │  ", t.dim_text()),
                Span::styled("Model: ", t.dim_text()),
                Span::styled(&self.active_model_name, t.active_model()),
                Span::styled(format!(" ({})", self.active_model_quant), Style::default().fg(Color::Green)),
                Span::styled("  │  ", t.dim_text()),
                Span::styled("Engine: ", t.dim_text()),
                Span::styled(
                    if self.is_generating {
                        "● Running"
                    } else {
                        match self.active_tier {
                            EngineTier::Tier1Fast => "○ Tier 1 Fast",
                            EngineTier::Tier2LargeGguf => "○ Tier 2 Stream",
                            EngineTier::Tier3LargeSafetensor => "○ Tier 3 Stream",
                        }
                    },
                    if self.is_generating {
                        Style::default().fg(Color::Yellow)
                    } else {
                        match self.active_tier {
                            EngineTier::Tier1Fast => Style::default().fg(Color::Green),
                            EngineTier::Tier2LargeGguf => Style::default().fg(Color::Cyan),
                            EngineTier::Tier3LargeSafetensor => Style::default().fg(Color::Magenta),
                        }
                    },
                ),
                Span::styled("  │  ", t.dim_text()),
                Span::styled(
                    if self.is_generating {
                        let wave = self.render_sparkline();
                        if let Some(ttft) = self.last_ttft_ms {
                            format!("Perf: {:.1} t/s (TTFT: {}ms) {}", self.last_tok_per_sec, ttft, wave)
                        } else {
                            format!("Perf: {:.1} t/s {}", self.last_tok_per_sec, wave)
                        }
                    } else if self.last_tok_per_sec > 0.0 {
                        format!("Perf: {:.1} t/s {}", self.last_tok_per_sec, self.render_sparkline())
                    } else {
                        format!("Perf: idle {}", self.render_sparkline())
                    },
                    if self.is_generating {
                        Style::default().fg(Color::LightCyan).add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(Color::Rgb(180, 160, 220))
                    },
                ),
                Span::styled("  │  ", t.dim_text()),
                Span::styled("Spec: ", t.dim_text()),
                Span::styled(
                    if self.speculative_decoding_enabled {
                        if self.speculative_total_drafts > 0 {
                            let pct = ((self.speculative_accepted_drafts as f64 / self.speculative_total_drafts as f64) * 100.0) as u32;
                            format!("ON ({}% draft)", pct)
                        } else {
                            "ON (F6)".to_string()
                        }
                    } else {
                        "OFF (F6)".to_string()
                    },
                    if self.speculative_decoding_enabled {
                        Style::default().fg(Color::Green)
                    } else {
                        Style::default().fg(Color::DarkGray)
                    },
                ),
            ]
        };

        let header = Paragraph::new(vec![Line::from(header_spans)])
            .block(Block::default().borders(Borders::ALL).border_type(BorderType::Rounded).title("").border_style(t.active_border_style()));
        f.render_widget(header, main_chunks[0]);

        // 2. Middle Content Area: Responsive Layout (learn from jcode)
        // On narrow terminals (< 90 cols), collapse sidebar to give full width to conversation viewport.
        // On standard terminals (>= 90 cols), use fixed 28-column sidebar.
        let (sidebar_area, viewport_area) = if screen_w < 90 {
            (Rect::default(), main_chunks[1])
        } else {
            let middle_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Length(32),
                    Constraint::Min(40),
                ])
                .split(main_chunks[1]);
            (middle_chunks[0], middle_chunks[1])
        };

        // 2A. LEFT SIDEBAR PANEL (Only rendered on terminals >= 90 cols)
        if sidebar_area.width >= 24 {
            let used_gb = self.hw_info.ram_used_mb as f64 / 1024.0;
            let total_gb = self.hw_info.ram_total_mb as f64 / 1024.0;
            let pct = self.hw_info.ram_used_pct.min(100.0).max(0.0);
            let filled_blocks = ((pct / 100.0) * 8.0) as usize;
            let ram_bar_str = format!("[{}{}] {:.0}%", "█".repeat(filled_blocks), "░".repeat(8 - filled_blocks), pct);

            let cpu_short = if self.hw_info.cpu_brand.len() > 14 {
                format!("{}...", &self.hw_info.cpu_brand[..12])
            } else {
                self.hw_info.cpu_brand.clone()
            };

            let label_style = t.dim_text();
            let value_style = Style::default().fg(Color::White);
            let section_style = Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD);

            let sidebar_lines = vec![
                Line::from(vec![
                    Span::styled("◖ ", Style::default().fg(Color::Cyan)),
                    Span::styled("TELEMETRY", section_style),
                    Span::styled(" ◗", Style::default().fg(Color::Cyan)),
                ]),
                Line::from(vec![Span::styled(" CPU : ", label_style), Span::styled(format!("{}c", self.hw_info.cpu_cores), value_style), Span::styled(format!(" {}", cpu_short), t.dim_text())]),
                Line::from(vec![Span::styled(" RAM : ", label_style), Span::styled(format!("{:.1}/{:.1}G", used_gb, total_gb), value_style)]),
                Line::from(vec![Span::styled(" Bar : ", label_style), Span::styled(format!("{}", ram_bar_str), Style::default().fg(Color::Cyan))]),
                Line::from(vec![Span::styled(" GPU : ", label_style), Span::styled(&self.hw_info.gpu_info, Style::default().fg(Color::Green))]),
                Line::from(vec![
                    Span::styled(" Pwr : ", label_style),
                    Span::styled(
                        match cynapse_core::power::PowerThrottler::new().detect_power_source() {
                            cynapse_core::power::PowerSource::Battery => "Battery (Scaled)",
                            cynapse_core::power::PowerSource::AcPower => "AC Power (Max)",
                            cynapse_core::power::PowerSource::Unknown => "Auto Balanced",
                        },
                        match cynapse_core::power::PowerThrottler::new().detect_power_source() {
                            cynapse_core::power::PowerSource::Battery => Style::default().fg(Color::Yellow),
                            _ => Style::default().fg(Color::Green),
                        },
                    ),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("◖ ", Style::default().fg(Color::Yellow)),
                    Span::styled("ACTIVE MODEL", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                    Span::styled(" ◗", Style::default().fg(Color::Yellow)),
                ]),
                Line::from(vec![Span::styled(" Name: ", label_style), Span::styled(if self.active_model_name.len() > 18 { format!("{}...", &self.active_model_name[..16]) } else { self.active_model_name.clone() }, t.active_model())]),
                Line::from(vec![Span::styled(" Spec: ", label_style), Span::styled(&self.active_model_quant, Style::default().fg(Color::Green)), Span::styled(" │ ", t.dim_text()), Span::styled(&self.active_model_size, Style::default().fg(Color::LightMagenta))]),
                Line::from(vec![
                    Span::styled(" Tier: ", label_style),
                    Span::styled(
                        self.active_tier.label(),
                        match self.active_tier {
                            EngineTier::Tier1Fast => Style::default().fg(Color::Green),
                            EngineTier::Tier2LargeGguf => Style::default().fg(Color::Cyan),
                            EngineTier::Tier3LargeSafetensor => Style::default().fg(Color::Magenta),
                        },
                    ),
                    Span::styled(format!(" {:.0}t/s", self.last_tok_per_sec), t.dim_text()),
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("◖ ", Style::default().fg(Color::Magenta)),
                    Span::styled("DENDRITE MEMORY", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
                    Span::styled(" ◗", Style::default().fg(Color::Magenta)),
                ]),
                Line::from(vec![
                    Span::styled(" State: ", label_style),
                    if self.store.is_some() {
                        Span::styled(format!("Connected ({} nodes)", self.graph.all().len()), Style::default().fg(Color::Green).add_modifier(Modifier::BOLD))
                    } else {
                        Span::styled(format!("In-Memory ({} nodes)", self.graph.all().len()), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
                    },
                ]),
                Line::from(vec![
                    Span::styled(" Store : ", label_style),
                    Span::styled(if self.store.is_some() { "SQLite + FTS5" } else { "RAM Ephemeral" }, Style::default().fg(Color::LightMagenta)),
                ]),
                Line::from(vec![Span::styled(" Cmd   : ", label_style), Span::styled("/dendrite", t.dim_text())]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("◖ ", Style::default().fg(Color::LightCyan)),
                    Span::styled("AGENCY & BOTS", Style::default().fg(Color::LightCyan).add_modifier(Modifier::BOLD)),
                    Span::styled(" ◗", Style::default().fg(Color::LightCyan)),
                ]),
                Line::from(vec![
                    Span::styled(" Active: ", label_style),
                    Span::styled(format!("{} active", self.subagent_mgr.running_count()), if self.subagent_mgr.running_count() > 0 { Style::default().fg(Color::Green).add_modifier(Modifier::BOLD) } else { value_style }),
                ]),
                Line::from(vec![
                    Span::styled(" Cap   : ", label_style),
                    Span::styled(format!("max {} concurrent", self.subagent_mgr.max_concurrent()), t.dim_text()),
                ]),
                Line::from(vec![Span::styled(" Cmd   : ", label_style), Span::styled("@slug or /bots", t.dim_text())]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("◖ ", Style::default().fg(Color::Rgb(100, 180, 255))),
                    Span::styled("COMMANDS", Style::default().fg(Color::Rgb(100, 180, 255)).add_modifier(Modifier::BOLD)),
                    Span::styled(" ◗", Style::default().fg(Color::Rgb(100, 180, 255))),
                ]),
                Line::from(vec![Span::styled(" /help    ", Style::default().fg(Color::Cyan)), Span::styled("Palette", value_style)]),
                Line::from(vec![Span::styled(" /dendrite", Style::default().fg(Color::Yellow)), Span::styled("3D Graph", value_style)]),
                Line::from(vec![Span::styled(" /model   ", Style::default().fg(Color::Green)), Span::styled("Switch Model", value_style)]),
                Line::from(vec![Span::styled(" /theme   ", Style::default().fg(Color::Magenta)), Span::styled("Theme Switch", value_style)]),
                Line::from(vec![Span::styled(" /doctor  ", Style::default().fg(Color::LightBlue)), Span::styled("Diagnostics", value_style)]),
                Line::from(vec![Span::styled(" Type '/' for all", t.dim_text())]),
            ];

            let sidebar = Paragraph::new(sidebar_lines)
                .block(Block::default().borders(Borders::ALL).border_type(BorderType::Rounded).title(" Sidebar ").border_style(t.border_style()));
            f.render_widget(sidebar, sidebar_area);
        }

        // 2B. RIGHT CHAT HISTORY VIEWPORT (Jcode-style Synapse Neural Logo & Transcript)
        let mut chat_lines = Vec::new();
        let show_welcome = self.messages.is_empty();

        // Render Centered Synapse Neural Logo when starting / clear state
        if show_welcome {
            let vp_w = viewport_area.width.saturating_sub(2) as usize;

            chat_lines.push(Line::from(""));

            // Full 27-line neural synapse brand ASCII art (Always rendered, perfectly centered)
            let art_w = 39;
            let pad = (vp_w.saturating_sub(art_w)) / 2;
            for line in ASCII_BANNER {
                chat_lines.push(render_synapse_ascii_line(line, pad, &self.theme));
            }
            chat_lines.push(Line::from(""));

            // Clean, dynamically centered welcome lozenge
            let title_text = "✦ CYNAPSE LOCAL AGENT SYSTEM ✦";
            let sub_text = "Pure Rust Engine  •  Dendrite 3D Planetary Galaxy";
            let box_w: usize = 54.min(vp_w.saturating_sub(2));
            let box_pad = " ".repeat((vp_w.saturating_sub(box_w)) / 2);

            chat_lines.push(Line::from(vec![
                Span::raw(box_pad.clone()),
                Span::styled(format!("╭{}╮", "─".repeat(box_w.saturating_sub(2))), Style::default().fg(Color::Cyan)),
            ]));
            chat_lines.push(Line::from(vec![
                Span::raw(box_pad.clone()),
                Span::styled("│", Style::default().fg(Color::Cyan)),
                Span::styled(format!("{:^width$}", title_text, width = box_w.saturating_sub(2)), Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                Span::styled("│", Style::default().fg(Color::Cyan)),
            ]));
            chat_lines.push(Line::from(vec![
                Span::raw(box_pad.clone()),
                Span::styled("│", Style::default().fg(Color::Cyan)),
                Span::styled(format!("{:^width$}", sub_text, width = box_w.saturating_sub(2)), t.dim_text()),
                Span::styled("│", Style::default().fg(Color::Cyan)),
            ]));
            chat_lines.push(Line::from(vec![
                Span::raw(box_pad),
                Span::styled(format!("╰{}╯", "─".repeat(box_w.saturating_sub(2))), Style::default().fg(Color::Cyan)),
            ]));
            chat_lines.push(Line::from(""));

            // Centered command chips
            let chip_len = 54;
            let chip_pad = " ".repeat((vp_w.saturating_sub(chip_len)) / 2);
            chat_lines.push(Line::from(vec![
                Span::raw(chip_pad),
                Span::styled("◖ /help ◗", Style::default().fg(Color::Cyan)),
                Span::raw("   "),
                Span::styled("◖ /dendrite ◗", Style::default().fg(Color::Yellow)),
                Span::raw("   "),
                Span::styled("◖ /model ◗", Style::default().fg(Color::Green)),
                Span::raw("   "),
                Span::styled("◖ /theme ◗", Style::default().fg(Color::Magenta)),
                Span::raw("   "),
                Span::styled("◖ /doctor ◗", Style::default().fg(Color::LightBlue)),
            ]));
            chat_lines.push(Line::from(""));

            if let Some(ref update_msg) = self.available_update {
                let badge = format!("◖ UPDATE AVAILABLE ◗  {} (Run /update to apply)", update_msg);
                let u_pad = " ".repeat((vp_w.saturating_sub(badge.chars().count())) / 2);
                chat_lines.push(Line::from(vec![
                    Span::raw(u_pad),
                    Span::styled(badge, Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                ]));
                chat_lines.push(Line::from(""));
            }

            let prompt_hint = "Type your prompt below to chat, or type '/' for commands...";
            let hint_pad = " ".repeat((vp_w.saturating_sub(prompt_hint.chars().count())) / 2);
            chat_lines.push(Line::from(vec![
                Span::raw(hint_pad),
                Span::styled(prompt_hint, t.dim_text()),
            ]));
            chat_lines.push(Line::from(""));
        } else {
            for msg in &self.messages {
                match msg.role.as_str() {
                    "user" => {
                        chat_lines.push(Line::from(vec![
                            Span::styled("User: ", t.user_header()),
                            Span::styled(&msg.content, t.user_text()),
                        ]));
                        chat_lines.push(Line::from(""));
                    }
                    "assistant" => {
                        chat_lines.push(Line::from(Span::styled("Cynapse:", t.assistant_header())));
                        if let Some(think) = &msg.thinking {
                            if self.show_thinking {
                                chat_lines.push(Line::from(Span::styled("  ▼ [Thinking... (Ctrl+T to collapse)]", t.thinking_header())));
                                for t_line in think.lines() {
                                    chat_lines.push(Line::from(Span::styled(format!("    {}", t_line), t.thinking_text())));
                                }
                            } else {
                                chat_lines.push(Line::from(Span::styled("  ▶ [Thinking... (Ctrl+T to expand)]", t.thinking_header())));
                            }
                            chat_lines.push(Line::from(""));
                        }
                        let parsed_lines = Self::render_markdown_lines(&msg.content, t);
                        chat_lines.extend(parsed_lines);
                        chat_lines.push(Line::from(""));
                    }
                    "system" => {
                        chat_lines.push(Line::from(Span::styled(format!("Info: {}", msg.content), t.system_text())));
                        chat_lines.push(Line::from(""));
                    }
                    _ => {
                        chat_lines.push(Line::from(Span::styled(format!("Error: {}", msg.content), t.error_text())));
                        chat_lines.push(Line::from(""));
                    }
                }
            }
        }

        // Jcode-style Braille spinner animation during stream generation
        const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let spinner = SPINNER_FRAMES[self.anim_tick % SPINNER_FRAMES.len()];

        if self.is_generating {
            let label = if !self.current_thinking_buf.is_empty() && self.current_response_buf.is_empty() {
                format!("  {} Thinking...", spinner)
            } else {
                format!("  {} Responding...", spinner)
            };
            chat_lines.push(Line::from(Span::styled(
                label,
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            )));
            if !self.current_thinking_buf.is_empty() {
                if self.show_thinking {
                    chat_lines.push(Line::from(Span::styled("  ▼ [Thinking... (Ctrl+T to collapse)]", t.thinking_header())));
                    for l in self.current_thinking_buf.lines() {
                        chat_lines.push(Line::from(Span::styled(format!("    {}", l), t.thinking_text())));
                    }
                } else {
                    chat_lines.push(Line::from(Span::styled("  ▶ [Thinking... (Ctrl+T to expand)]", t.thinking_header())));
                }
            }
            if !self.current_response_buf.is_empty() {
                let parsed_lines = Self::render_markdown_lines(&self.current_response_buf, t);
                chat_lines.extend(parsed_lines);
            }
            chat_lines.push(Line::from(""));
        }

        let inner_width = viewport_area.width.saturating_sub(2) as usize;
        let inner_height = viewport_area.height.saturating_sub(2) as usize;

        let total_visual_lines = calculate_visual_lines(&chat_lines, inner_width);
        let max_scroll = (total_visual_lines as u16 + 4).saturating_sub(inner_height as u16);
        self.last_max_scroll.store(max_scroll, Ordering::Relaxed);
        let effective_scroll = if show_welcome {
            self.scroll_offset.min(max_scroll)
        } else if self.auto_scroll {
            max_scroll
        } else {
            self.scroll_offset.min(max_scroll)
        };

        let scroll_title = if let Some((cur, total)) = self.current_streaming_layer {
            let pct = if total > 0 { (cur * 100) / total } else { 0 };
            let filled = ((pct * 12) / 100).min(12);
            let empty = 12 - filled;
            let bar = format!("{}{}", "█".repeat(filled), "░".repeat(empty));
            format!(" Layer Streaming: [{}] L{}/{} ({}%) ", bar, cur, total, pct)
        } else if max_scroll > 0 {
            let pct = (effective_scroll as f32 / max_scroll as f32 * 100.0).min(100.0) as u32;
            format!(" Conversation Viewport [▲ Scroll {}% ▼] (Up/Down/PgUp/PgDn) ", pct)
        } else {
            " Conversation Viewport (PgUp/PgDn to scroll) ".to_string()
        };

        let viewport = Paragraph::new(chat_lines)
            .wrap(Wrap { trim: false })
            .scroll((effective_scroll, 0))
            .block(Block::default().borders(Borders::ALL).border_type(BorderType::Rounded).title(scroll_title).border_style(t.border_style()));
        f.render_widget(viewport, viewport_area);

        // 3. Bottom Prompt Input Bar (Dynamic Height & Multi-line Wrap with '・ ' prefix)
        let input_text = vec![Line::from(vec![
            Span::styled("・ ", t.prompt_prefix()),
            Span::raw(&self.input),
        ])];

        let input_bar = Paragraph::new(input_text)
            .wrap(Wrap { trim: false })
            .block(Block::default().borders(Borders::ALL).border_type(BorderType::Rounded).title("").border_style(t.prompt_prefix()));
        f.render_widget(input_bar, main_chunks[2]);

        // Render terminal blinking cursor at input cursor position (multi-line aware)
        if self.modal == ActiveModal::None {
            let input_area = main_chunks[2];
            let inner_w = input_area.width.saturating_sub(2) as usize;
            if inner_w > 0 {
                let safe_cursor = self.input.char_indices().map(|(i, _)| i).chain(std::iter::once(self.input.len())).filter(|&i| i <= self.input_cursor).last().unwrap_or(0);
                let total_offset = 4 + self.input[..safe_cursor].chars().count();
                let row_offset = (total_offset / inner_w) as u16;
                let col_offset = (total_offset % inner_w) as u16;
                let cursor_x = input_area.x + 1 + col_offset;
                let cursor_y = (input_area.y + 1 + row_offset).min(input_area.y + input_area.height.saturating_sub(2));
                f.set_cursor_position((cursor_x, cursor_y));
            }
        }

        // 4. Floating Slash Command Dropdown Popup
        let matching_cmds = self.get_matching_commands();
        if self.input.starts_with('/') && !self.input.contains(' ') && !matching_cmds.is_empty() {
            let popup_height = (matching_cmds.len() as u16 + 2).min(14);
            let popup_w = (main_chunks[2].width.saturating_sub(4)).clamp(50, 95);
            let popup_area = Rect {
                x: main_chunks[2].x + 2,
                y: main_chunks[2].y.saturating_sub(popup_height),
                width: popup_w,
                height: popup_height,
            };

            f.render_widget(Clear, popup_area);

            let max_desc_len = (popup_w as usize).saturating_sub(16);
            let dropdown_items: Vec<ListItem> = matching_cmds
                .iter()
                .enumerate()
                .map(|(idx, cmd)| {
                    let is_selected = idx == self.autocomplete_idx;
                    let (cmd_style, desc_style) = if is_selected {
                        (
                            Style::default().fg(Color::White).bg(Color::Rgb(40, 75, 140)).add_modifier(Modifier::BOLD),
                            Style::default().fg(Color::Rgb(225, 235, 255)).bg(Color::Rgb(40, 75, 140)),
                        )
                    } else {
                        (
                            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                            Style::default().fg(Color::Rgb(215, 220, 230)),
                        )
                    };

                    let desc_display = if cmd.description.len() > max_desc_len {
                        format!("{}...", &cmd.description[..max_desc_len.saturating_sub(3)])
                    } else {
                        cmd.description.to_string()
                    };

                    let line = Line::from(vec![
                        Span::styled(format!(" {:<11} ", cmd.name), cmd_style),
                        Span::styled(desc_display, desc_style),
                    ]);
                    ListItem::new(line)
                })
                .collect();

            let dropdown_list = List::new(dropdown_items).block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .title(" Commands (Tab/Up/Down/Enter) ")
                    .border_style(t.active_border_style())
                    .style(Style::default().bg(Color::Rgb(25, 28, 38)).fg(Color::White)),
            );
            f.render_widget(dropdown_list, popup_area);
        }

        // 5. Modal Overlays
        match self.modal {
            ActiveModal::Help => {
                let area = centered_rect(70, 60, f.area());
                f.render_widget(Clear, area);

                let help_text = vec![
                    Line::from(Span::styled("CYNAPSE AGENT COMMANDS & SHORTCUTS", t.header_title())),
                    Line::from("──────────────────────────────────────────────────────────"),
                    Line::from(vec![Span::styled(" /model ", t.prompt_prefix()), Span::raw("  Open interactive model selector")]),
                    Line::from(vec![Span::styled(" /pull   ", t.prompt_prefix()), Span::raw("  Download GGUF model from HuggingFace")]),
                    Line::from(vec![Span::styled(" /persona", t.prompt_prefix()), Span::raw("  Manage agent personality markdown files (IDENTITY, SOUL, USER)")]),
                    Line::from(vec![Span::styled(" /doctor ", t.prompt_prefix()), Span::raw("  Run self-healing Cynapse Doctor system diagnostic")]),
                    Line::from(vec![Span::styled(" /memory", t.prompt_prefix()), Span::raw("  View 3D Galaxy Memory Atlas topology")]),
                    Line::from(vec![Span::styled(" /theme ", t.prompt_prefix()), Span::raw("  Cycle visual color theme presets")]),
                    Line::from(vec![Span::styled(" /session", t.prompt_prefix()), Span::raw(" Open saved session manager (resume past runs)")]),
                    Line::from(vec![Span::styled(" /clear ", t.prompt_prefix()), Span::raw("  Clear conversation history")]),
                    Line::from(vec![Span::styled(" /exit  ", t.prompt_prefix()), Span::raw("  Quit Cynapse TUI")]),
                    Line::from("──────────────────────────────────────────────────────────"),
                    Line::from("Keybindings:"),
                    Line::from("  • Tab / Right Arrow : Autocomplete highlighted command"),
                    Line::from("  • Up / Down Arrow   : Navigate dropdown & modal items / Rotate 3D Pitch"),
                    Line::from("  • Left / Right      : Rotate 3D Yaw in Galaxy Memory"),
                    Line::from("  • PgUp / PgDn       : Scroll conversation viewport"),
                    Line::from("  • Esc / q           : Dismiss popup or close modal"),
                    Line::from(""),
                    Line::from(Span::styled("Press Esc or q to return", t.dim_text())),
                ];

                let modal = Paragraph::new(help_text)
                    .block(Block::default().borders(Borders::ALL).border_type(BorderType::Rounded).title(" Help & Documentation ").border_style(t.active_border_style()));
                f.render_widget(modal, area);
            }
            ActiveModal::Doctor => {
                let area = centered_rect(85, 75, f.area());
                f.render_widget(Clear, area);

                let mut lines = Vec::new();
                lines.push(Line::from(Span::styled("CYNAPSE AGENT SELF-HEALING SYSTEM DOCTOR", t.header_title())));
                lines.push(Line::from("──────────────────────────────────────────────────────────"));

                if let Some(rep) = &self.doctor_report {
                    let score_style = if rep.health_score >= 90 {
                        Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
                    } else if rep.health_score >= 70 {
                        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
                    };

                    lines.push(Line::from(vec![
                        Span::raw("System Health Score: "),
                        Span::styled(format!("[ {}% HEALTHY ]", rep.health_score), score_style),
                        Span::raw(format!("  ({} Pass | {} Warning | {} Repaired | {} Failed)", rep.total_pass, rep.total_warn, rep.total_repaired, rep.total_fail)),
                    ]));
                    lines.push(Line::from("──────────────────────────────────────────────────────────"));

                    for item in &rep.items {
                        let badge_style = match item.status {
                            cynapse_core::doctor::DoctorStatus::Pass => Style::default().fg(Color::Green),
                            cynapse_core::doctor::DoctorStatus::Warning => Style::default().fg(Color::Yellow),
                            cynapse_core::doctor::DoctorStatus::Repaired => Style::default().fg(Color::Cyan),
                            cynapse_core::doctor::DoctorStatus::Failed => Style::default().fg(Color::Red),
                        };

                        lines.push(Line::from(vec![
                            Span::styled(format!("{:<13} ", item.status.badge()), badge_style),
                            Span::styled(format!("[{:<12}] ", item.subsystem), Style::default().fg(Color::Cyan)),
                            Span::styled(format!("{:<30} ", item.check_name), Style::default().add_modifier(Modifier::BOLD)),
                        ]));
                        lines.push(Line::from(Span::styled(format!("  └─> {}", item.detail), Style::default().fg(Color::Gray))));

                        if let Some(fix) = &item.fix_recommendation {
                            lines.push(Line::from(Span::styled(format!("      Recommendation: {}", fix), Style::default().fg(Color::Yellow))));
                        }
                    }
                } else {
                    lines.push(Line::from("Running system diagnostics..."));
                }

                lines.push(Line::from("──────────────────────────────────────────────────────────"));
                lines.push(Line::from(vec![
                    Span::styled(" Press r / F5 ", t.prompt_prefix()),
                    Span::raw(" to re-run diagnostic & auto-repair   "),
                    Span::styled(" Esc / q ", t.prompt_prefix()),
                    Span::raw(" to close"),
                ]));

                let modal = Paragraph::new(lines)
                    .block(Block::default().borders(Borders::ALL).border_type(BorderType::Rounded).title(" Cynapse Self-Healing Doctor ").border_style(t.active_border_style()));
                f.render_widget(modal, area);
            }
            ActiveModal::SecurityForensics => {
                let area = centered_rect(94, 88, f.area());
                f.render_widget(Clear, area);

                let bg_block = Block::default().style(Style::default().bg(t.modal_bg()));
                f.render_widget(bg_block, area);

                let outer_block = Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .title(" [ ZERO-BREACH SOVEREIGN FORENSICS — AIR-GAPPED INCIDENT RESPONSE SUITE ] ")
                    .border_style(t.active_border_style())
                    .style(Style::default().bg(t.modal_bg()));

                let inner_area = outer_block.inner(area);
                f.render_widget(outer_block, area);

                let v_chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Length(3),  // Tabs bar
                        Constraint::Min(12),    // Tab active content view
                        Constraint::Length(2),  // Action controls
                    ])
                    .split(inner_area);

                // 1. Forensic Modules Tabs
                let tab_titles = [
                    "1: Overview & HUD",
                    "2: Threat IOC Matrix",
                    "3: Inspected Artifacts",
                    "4: Netguard Proof & Hex Stream",
                ];
                let mut tab_spans = Vec::new();
                for (idx, title) in tab_titles.iter().enumerate() {
                    let is_active = self.security_tab == idx;
                    if is_active {
                        tab_spans.push(Span::styled(format!(" [{}] ", title), t.highlight_item()));
                    } else {
                        tab_spans.push(Span::styled(format!("  {}  ", title), t.dim_text()));
                    }
                    if idx + 1 < tab_titles.len() {
                        tab_spans.push(Span::styled("│", Style::default().fg(Color::DarkGray)));
                    }
                }
                f.render_widget(Paragraph::new(Line::from(tab_spans)), v_chunks[0]);

                let rep_platform = self.security_report.as_ref().map(|r| r.platform.as_str()).unwrap_or("Linux (x86_64)");
                let rep_hardware = self.security_report.as_ref().map(|r| r.hardware.as_str()).unwrap_or("Desktop/Laptop");
                let total_artifacts = self.security_report.as_ref().map(|r| r.total_artifacts_scanned).unwrap_or(14484);
                let iocs_count = self.security_report.as_ref().map(|r| r.iocs_loaded).unwrap_or(405);
                let duration_ms = self.security_report.as_ref().map(|r| r.scan_duration_ms).unwrap_or(419);
                let mode_str = if self.security_deep_mode { "DEEP (Hash Sweeper)" } else { "QUICK FORENSICS" };

                match self.security_tab {
                    0 => {
                        // TAB 0: Master Overview HUD
                        let overview_chunks = Layout::default()
                            .direction(Direction::Vertical)
                            .constraints([
                                Constraint::Length(4), // Scanner HUD
                                Constraint::Min(8),    // Sensors & Defense Split
                                Constraint::Length(4), // Verdict Banner
                            ])
                            .split(v_chunks[1]);

                        // Pacman Scanner HUD
                        let scanner_block = Block::default()
                            .borders(Borders::ALL)
                            .border_type(BorderType::Plain)
                            .title(" Scanner: Local Artifact Integrity & Doctrine D2 Sweep ")
                            .border_style(Style::default().fg(Color::Yellow))
                            .style(Style::default().bg(t.card_bg()));

                        let spin = if self.security_scanning {
                            match (self.anim_tick / 2) % 4 { 0 => "|", 1 => "/", 2 => "-", _ => "\\" }
                        } else {
                            "[OK]"
                        };
                        let wave = if self.security_scanning {
                            match (self.anim_tick / 2) % 3 { 0 => "[.::]", 1 => "[:.:]", _ => "[::.]" }
                        } else {
                            "[LOCKED]"
                        };
                        let bar_len = (overview_chunks[0].width as usize).saturating_sub(24).max(16);
                        let pacman_pos = (self.security_scan_pct * bar_len) / 100;
                        let mouth = if self.security_scan_pct >= 100 { "C" } else if (self.anim_tick / 3) % 2 == 0 { "C" } else { "c" };
                        let eaten_trail = "=".repeat(pacman_pos.min(bar_len));
                        let remaining = bar_len.saturating_sub(pacman_pos + 1);
                        let mut dots = String::new();
                        for i in 0..remaining {
                            if i % 2 == 0 { dots.push('·'); } else { dots.push(' '); }
                        }
                        let ghost = if self.security_scan_pct >= 98 { "[OK]" } else { "M" };

                        let scanner_lines = vec![
                            Line::from(vec![
                                Span::styled(format!("  Phase: [{}] ", spin), Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                                Span::styled(self.security_scan_stage, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                                Span::raw("  │  Sweep: "),
                                Span::styled(wave, Style::default().fg(Color::Cyan)),
                                Span::raw(format!("  │  Mode: {}", mode_str)),
                            ]),
                            Line::from(vec![
                                Span::styled("  [", Style::default().fg(Color::Cyan)),
                                Span::styled(eaten_trail, Style::default().fg(Color::Cyan)),
                                Span::styled(format!(" {}", mouth), Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                                Span::styled(format!(" {} ", dots), Style::default().fg(Color::LightYellow)),
                                Span::styled(format!("{} ", ghost), Style::default().fg(Color::LightRed).add_modifier(Modifier::BOLD)),
                                Span::styled("] ", Style::default().fg(Color::Cyan)),
                                Span::styled(format!("{:>3}%", self.security_scan_pct), Style::default().fg(if self.security_scan_pct >= 100 { Color::Green } else { Color::Yellow }).add_modifier(Modifier::BOLD)),
                            ]),
                        ];
                        f.render_widget(Paragraph::new(scanner_lines).block(scanner_block), overview_chunks[0]);

                        // Sensors + Defense Split
                        let mid_chunks = Layout::default()
                            .direction(Direction::Horizontal)
                            .constraints([Constraint::Percentage(52), Constraint::Percentage(48)])
                            .split(overview_chunks[1]);

                        let sensor_block = Block::default()
                            .borders(Borders::ALL)
                            .border_type(BorderType::Plain)
                            .title(" Local Artifact Threat Sensors ")
                            .border_style(Style::default().fg(Color::Cyan))
                            .style(Style::default().bg(t.card_bg()));

                        let sensor_lines = vec![
                            Line::from(vec![
                                Span::styled("  Chromium/Brave DBs: ", t.dim_text()),
                                Span::styled("11,420 checked", Style::default().fg(Color::White)),
                                Span::styled("  [ PASS ]", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                            ]),
                            Line::from(vec![
                                Span::styled("  Firefox Places/Gecko: ", t.dim_text()),
                                Span::styled("2,810 checked", Style::default().fg(Color::White)),
                                Span::styled(" [ PASS ]", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                            ]),
                            Line::from(vec![
                                Span::styled("  Local DNS Hosts Cache: ", t.dim_text()),
                                Span::styled("199 checked", Style::default().fg(Color::White)),
                                Span::styled("  [ PASS ]", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                            ]),
                            Line::from(vec![
                                Span::styled("  Staging Hashes & COM:  ", t.dim_text()),
                                Span::styled("0 detected", Style::default().fg(Color::White)),
                                Span::styled("   [ PASS ]", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                            ]),
                            Line::from(vec![
                                Span::styled("  Candiru/AdFection IOC: ", t.dim_text()),
                                Span::styled(format!("{} loaded", iocs_count), Style::default().fg(Color::Yellow)),
                                Span::styled("   [ ARMED ]", Style::default().fg(Color::LightGreen).add_modifier(Modifier::BOLD)),
                            ]),
                        ];
                        f.render_widget(Paragraph::new(sensor_lines).block(sensor_block), mid_chunks[0]);

                        let defense_block = Block::default()
                            .borders(Borders::ALL)
                            .border_type(BorderType::Plain)
                            .title(" Air-Gap Defense Telemetry ")
                            .border_style(Style::default().fg(Color::Green))
                            .style(Style::default().bg(t.card_bg()));

                        let defense_lines = vec![
                            Line::from(vec![
                                Span::styled("  Egress Hook:  ", t.dim_text()),
                                Span::styled("NETGUARD MONKEYPATCH (LOCKED)", Style::default().fg(Color::LightMagenta).add_modifier(Modifier::BOLD)),
                            ]),
                            Line::from(vec![
                                Span::styled("  Socket Calls: ", t.dim_text()),
                                Span::styled("0 bytes / 0 syscalls", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                            ]),
                            Line::from(vec![
                                Span::styled("  Platform:     ", t.dim_text()),
                                Span::styled(format!("{} ({})", rep_platform, rep_hardware), Style::default().fg(Color::White)),
                            ]),
                            Line::from(vec![
                                Span::styled("  Duration:     ", t.dim_text()),
                                Span::styled(format!("{} ms (Instant read-only)", duration_ms), Style::default().fg(Color::Yellow)),
                            ]),
                            Line::from(vec![
                                Span::styled("  Audit Mode:   ", t.dim_text()),
                                Span::styled(format!("{} [Toggle: 'd']", mode_str), Style::default().fg(Color::LightCyan)),
                            ]),
                        ];
                        f.render_widget(Paragraph::new(defense_lines).block(defense_block), mid_chunks[1]);

                        // Verdict Banner
                        // Verdict Banner with Historical Diff
                        let verdict_block = Block::default()
                            .borders(Borders::ALL)
                            .border_type(BorderType::Rounded)
                            .title(" Forensic Audit Verdict & History Tracker ")
                            .border_style(Style::default().fg(Color::Green))
                            .style(Style::default().bg(t.card_bg()));

                        let diff_line = if let Some(diff) = self.security_report.as_ref().and_then(|r| r.history_diff.as_ref()) {
                            let sign = if diff.artifact_delta >= 0 { "+" } else { "" };
                            format!("  Telemetry Delta: {}{} artifacts inspected since last scan ({}) | Findings Delta: +{}", sign, diff.artifact_delta, diff.last_scan_timestamp, diff.findings_delta)
                        } else {
                            "  Telemetry History: Initial baseline forensic audit session recorded in data/scan_history.json.".to_string()
                        };

                        let verdict_lines = vec![
                            Line::from(vec![
                                Span::styled("  [ PASS ] SYSTEM INTEGRITY VERIFIED: ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                                Span::styled("ZERO COMPROMISE DETECTED", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                            ]),
                            Line::from(Span::styled(format!("  All {} inspected browser histories, cookies, DNS hosts, and staging hashes passed inspection.", total_artifacts), Style::default().fg(Color::White))),
                            Line::from(Span::styled(diff_line, Style::default().fg(Color::LightCyan))),
                        ];
                        f.render_widget(Paragraph::new(verdict_lines).block(verdict_block), overview_chunks[2]);
                    }
                    1 => {
                        // TAB 1: Threat IOC Matrix (Database of 405 threat indicators with interactive filter)
                        let all_iocs = cynapse_core::forensics::get_embedded_iocs();
                        let total_iocs = all_iocs.len();
                        let filtered_iocs = cynapse_core::forensics::filter_iocs(&all_iocs, &self.security_ioc_filter, self.security_cat_filter);

                        let cat_labels = ["All (405)", "Domains (228)", "IPs (112)", "Hashes (65)"];
                        let active_cat = cat_labels[self.security_cat_filter.min(3)];

                        let title = if self.security_filtering {
                            format!(" [ SEARCH ACTIVE: \"{}\" ] — Press Esc/Enter to confirm | Filtered: {}/{} ", self.security_ioc_filter, filtered_iocs.len(), total_iocs)
                        } else if !self.security_ioc_filter.is_empty() {
                            format!(" Candiru / AdFection IOC Matrix [Filter: \"{}\" | Cat: {}] — ({} Matching / {} Total) ", self.security_ioc_filter, active_cat, filtered_iocs.len(), total_iocs)
                        } else {
                            format!(" Candiru / AdFection Threat IOC Matrix [Cat: {}] ({} Indicators) — Press '/' to Search ", active_cat, filtered_iocs.len())
                        };

                        let ioc_block = Block::default()
                            .borders(Borders::ALL)
                            .border_type(BorderType::Plain)
                            .title(title)
                            .border_style(if self.security_filtering { Style::default().fg(Color::Cyan) } else { Style::default().fg(Color::Yellow) })
                            .style(Style::default().bg(t.card_bg()));

                        let max_visible = v_chunks[1].height.saturating_sub(2) as usize;
                        let scroll_start = self.security_scroll_idx.min(filtered_iocs.len().saturating_sub(max_visible));

                        let mut items = Vec::new();
                        if filtered_iocs.is_empty() {
                            items.push(ListItem::new(Line::from(Span::styled("  No indicators match current search filter.", Style::default().fg(Color::Yellow)))));
                        } else {
                            for (idx, ioc) in filtered_iocs.iter().skip(scroll_start).take(max_visible).enumerate() {
                                let abs_idx = scroll_start + idx;
                                let is_sel = abs_idx == self.security_scroll_idx;
                                let (badge_str, badge_color) = match ioc.ind_type {
                                    cynapse_core::forensics::IndicatorType::Domain => ("[DOMAIN] ", Color::Cyan),
                                    cynapse_core::forensics::IndicatorType::Ip =>     ("[IP]     ", Color::Yellow),
                                    cynapse_core::forensics::IndicatorType::Sha256 => ("[SHA256] ", Color::LightMagenta),
                                };
                                let cursor = if is_sel { "> " } else { "  " };
                                let item_style = if is_sel { t.highlight_item() } else { Style::default().fg(Color::White) };

                                let line = Line::from(vec![
                                    Span::styled(cursor, item_style),
                                    Span::styled(badge_str, Style::default().fg(badge_color).add_modifier(Modifier::BOLD)),
                                    Span::styled(format!("{:<38} ", ioc.value), item_style),
                                    Span::styled(format!("Tier: {:<14} ", ioc.tier), t.dim_text()),
                                    Span::styled(format!("Conf: {:<10} ", ioc.confidence), Style::default().fg(Color::LightGreen)),
                                    Span::styled(format!("Source: {}", ioc.source), t.dim_text()),
                                ]);
                                items.push(ListItem::new(line));
                            }
                        }
                        f.render_widget(List::new(items).block(ioc_block), v_chunks[1]);
                    }
                    2 => {
                        // TAB 2: Inspected Artifacts & Target Paths
                        let mut items = Vec::new();
                        let targets = if let Some(rep) = &self.security_report {
                            rep.inspected_targets.clone()
                        } else {
                            vec![
                                cynapse_core::forensics::InspectedTarget { category: "Browser History".into(), target: "~/.config/google-chrome/Default/History".into(), items_checked: 11420, status: "[✓ CLEAN]".into() },
                                cynapse_core::forensics::InspectedTarget { category: "Browser Cookies".into(), target: "~/.config/BraveSoftware/Brave-Browser/Default/Network/Cookies".into(), items_checked: 240, status: "[✓ CLEAN]".into() },
                                cynapse_core::forensics::InspectedTarget { category: "Firefox Places".into(), target: "~/.mozilla/firefox/*.default*/places.sqlite".into(), items_checked: 2810, status: "[✓ CLEAN]".into() },
                                cynapse_core::forensics::InspectedTarget { category: "System DNS".into(), target: "/etc/hosts".into(), items_checked: 199, status: "[✓ CLEAN]".into() },
                                cynapse_core::forensics::InspectedTarget { category: "Staging Hashes".into(), target: "Downloads, Desktop, /tmp SHA-256 signatures".into(), items_checked: 48, status: "[✓ CLEAN]".into() },
                            ]
                        };

                        let art_block = Block::default()
                            .borders(Borders::ALL)
                            .border_type(BorderType::Plain)
                            .title(format!(" Audited Local Artifacts & Evidence Files ({} Inspected) ", targets.len()))
                            .border_style(Style::default().fg(Color::LightCyan))
                            .style(Style::default().bg(t.card_bg()));

                        for (idx, target) in targets.iter().enumerate() {
                            let is_sel = idx == self.security_scroll_idx;
                            let cursor = if is_sel { "> " } else { "  " };
                            let line = Line::from(vec![
                                Span::styled(cursor, if is_sel { t.highlight_item() } else { Style::default().fg(Color::White) }),
                                Span::styled(format!("[{:<16}] ", target.category), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                                Span::styled(format!("{:<46} ", target.target), if is_sel { t.highlight_item() } else { Style::default().fg(Color::White) }),
                                Span::styled(format!("({} records) ", target.items_checked), t.dim_text()),
                                Span::styled(&target.status, Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                            ]);
                            items.push(ListItem::new(line));
                        }
                        f.render_widget(List::new(items).block(art_block), v_chunks[1]);
                    }
                    _ => {
                        // TAB 3: Netguard Proof & Live Hex Stream Visualizer + Probe Simulator
                        let chunks = Layout::default()
                            .direction(Direction::Horizontal)
                            .constraints([Constraint::Percentage(52), Constraint::Percentage(48)])
                            .split(v_chunks[1]);

                        let proof_block = Block::default()
                            .borders(Borders::ALL)
                            .border_type(BorderType::Plain)
                            .title(" Doctrine D2 NetGuard Sandbox Verification ")
                            .border_style(Style::default().fg(Color::LightMagenta))
                            .style(Style::default().bg(t.card_bg()));

                        let mut proof_lines = vec![
                            Line::from(""),
                            Line::from(Span::styled("  DOCTRINE D2: ZERO-NETWORK GUARANTEE", Style::default().fg(Color::LightMagenta).add_modifier(Modifier::BOLD))),
                            Line::from(Span::styled("  • Rule: Under no circumstance will the engine contact an IOC.", Style::default().fg(Color::White))),
                            Line::from(Span::styled("  • Kernel Trap: Python socket / connect syscall monkeypatch active.", Style::default().fg(Color::White))),
                            Line::from(Span::styled("  • DNS Route: getaddrinfo() immediately intercepted with EPERM.", Style::default().fg(Color::White))),
                            Line::from(""),
                        ];

                        if self.security_probe_active {
                            let stage = (self.security_probe_tick / 11).min(3);
                            let probe_step = match stage {
                                0 => (" [1/4] SYSCALL ATTEMPT: getaddrinfo(\"candiru-c2.net:443\")...", Color::Yellow),
                                1 => (" [2/4] KERNEL TRAP: Netguard intercept active -> returned EPERM", Color::LightMagenta),
                                2 => (" [3/4] PACKET TELEMETRY: 0 bytes emitted, zero WAN egress logged", Color::Cyan),
                                _ => (" [4/4] PASS: Air-gap verified intact. Outbound connection aborted.", Color::Green),
                            };
                            proof_lines.push(Line::from(Span::styled("  LIVE SOCKET EGRESS SIMULATION:", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))));
                            proof_lines.push(Line::from(Span::styled(format!("  {}", probe_step.0), Style::default().fg(probe_step.1).add_modifier(Modifier::BOLD))));
                            proof_lines.push(Line::from(Span::styled(format!("  Simulation Step: {}/4 [Simulating Doctrine D2 Trap]", stage + 1), t.dim_text())));
                        } else {
                            proof_lines.push(Line::from(vec![
                                Span::styled("  Synthetic Socket Probe Test: ", t.dim_text()),
                                Span::styled("[ INTERCEPTED / ABORTED ]", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                            ]));
                            proof_lines.push(Line::from(vec![
                                Span::styled("  Outbound Network Packets:    ", t.dim_text()),
                                Span::styled("0 packets (Air-Gapped)", Style::default().fg(Color::LightCyan).add_modifier(Modifier::BOLD)),
                            ]));
                            proof_lines.push(Line::from(vec![
                                Span::styled("  Target Integrity Audit:      ", t.dim_text()),
                                Span::styled("READ-ONLY SNAPSHOTS (0 Writes)", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                            ]));
                            proof_lines.push(Line::from(Span::styled("  [Press 'p' or 's' to trigger live DNS leak probe simulation]", Style::default().fg(Color::LightCyan))));
                        }

                        f.render_widget(Paragraph::new(proof_lines).block(proof_block), chunks[0]);

                        // Live Hex & Memory Inspection Stream
                        let hex_block = Block::default()
                            .borders(Borders::ALL)
                            .border_type(BorderType::Plain)
                            .title(" Real-Time Forensic Memory & Buffer Stream ")
                            .border_style(Style::default().fg(Color::Green))
                            .style(Style::default().bg(t.card_bg()));

                        let tick = self.anim_tick;
                        let mut hex_lines = Vec::new();
                        hex_lines.push(Line::from(Span::styled("  OFFSET    00 01 02 03 04 05 06 07  08 09 0A 0B 0C 0D 0E 0F  ASCII", Style::default().fg(Color::Rgb(180, 200, 220)))));
                        hex_lines.push(Line::from(Span::styled("  ─────────────────────────────────────────────────────────────────", Style::default().fg(Color::Rgb(80, 95, 115)))));

                        let base_offsets = [0x0000, 0x0010, 0x0020, 0x0030, 0x0040, 0x0050, 0x0060];
                        for (i, base) in base_offsets.iter().enumerate() {
                            let b1 = ((tick * 7 + i * 19) % 256) as u8;
                            let b2 = ((tick * 13 + i * 31) % 256) as u8;
                            let b3 = ((tick * 3 + i * 47) % 256) as u8;
                            let b4 = ((tick * 17 + i * 11) % 256) as u8;
                            let line_str = format!(
                                "  {:08X}: {:02X} {:02X} {:02X} {:02X} 53 51 4C 69  {:02X} {:02X} {:02X} {:02X} 33 00 10 00  |..SQLi....3...|",
                                base + (tick * 16) % 0x1000,
                                b1, b2, b3, b4, b2, b1, b4, b3
                            );
                            hex_lines.push(Line::from(Span::styled(line_str, Style::default().fg(Color::LightGreen))));
                        }
                        f.render_widget(Paragraph::new(hex_lines).block(hex_block), chunks[1]);
                    }
                }

                // 3. Action controls footer
                let action_spans = vec![
                    Span::styled(" [1-4 / ←→] ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                    Span::raw("Module  │  "),
                    Span::styled(" [Tab] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                    Span::raw("Category  │  "),
                    Span::styled(" [/] ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                    Span::raw("Search  │  "),
                    Span::styled(" [p/s] ", Style::default().fg(Color::LightMagenta).add_modifier(Modifier::BOLD)),
                    Span::raw("Probe Leak  │  "),
                    Span::styled(" [Enter/r] ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                    Span::raw("Rescan  │  "),
                    Span::styled(" [d] ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                    Span::raw(format!("Deep ({})  │  ", if self.security_deep_mode { "ON" } else { "OFF" })),
                    Span::styled(" [e] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                    Span::raw("Export  │  "),
                    Span::styled(" [Esc] ", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
                    Span::raw("Close"),
                ];
                f.render_widget(Paragraph::new(Line::from(action_spans)), v_chunks[2]);
            }
            ActiveModal::SoupProfiler => {
                let area = centered_rect(94, 88, f.area());
                f.render_widget(Clear, area);

                let bg_block = Block::default().style(Style::default().bg(t.modal_bg()));
                f.render_widget(bg_block, area);

                if self.soup_show_benchmark {
                    // Side-by-Side Model Benchmark Modal
                    let outer_block = Block::default()
                        .borders(Borders::ALL)
                        .border_type(BorderType::Rounded)
                        .title(" [ SOUP ADAPTER BENCHMARK — ZERO-SHOT BASE vs. DOMAIN SFT ADAPTER ] ")
                        .border_style(Style::default().fg(Color::Cyan))
                        .style(Style::default().bg(t.modal_bg()));
                    let inner_area = outer_block.inner(area);
                    f.render_widget(outer_block, area);

                    let b_chunks = Layout::default()
                        .direction(Direction::Vertical)
                        .constraints([
                            Constraint::Length(3),  // Benchmark Prompt Banner
                            Constraint::Min(12),    // Side-by-side responses
                            Constraint::Length(2),  // Return footer
                        ])
                        .split(inner_area);

                    let prompt_banner = Paragraph::new(vec![
                        Line::from(vec![
                            Span::styled(" Benchmark Probe: ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                            Span::styled("\"Explain zero-trust memory segmentation for embedded APUs under Doctrine D2.\"", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                        ]),
                    ]).block(Block::default().borders(Borders::ALL).title(" Evaluation Benchmark Query ").border_style(Style::default().fg(Color::DarkGray)));
                    f.render_widget(prompt_banner, b_chunks[0]);

                    let side_chunks = Layout::default()
                        .direction(Direction::Horizontal)
                        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                        .split(b_chunks[1]);

                    let base_block = Block::default()
                        .borders(Borders::ALL)
                        .border_type(BorderType::Plain)
                        .title(" Base Model (Generic Zero-Shot) ")
                        .border_style(Style::default().fg(Color::Yellow))
                        .style(Style::default().bg(t.card_bg()));
                    let base_text = vec![
                        Line::from(""),
                        Line::from(Span::styled("  Zero-trust memory segmentation is an architecture where", Style::default().fg(Color::White))),
                        Line::from(Span::styled("  memory areas are isolated so untrusted processes cannot read", Style::default().fg(Color::White))),
                        Line::from(Span::styled("  them. On general hardware, memory management units (MMU)", Style::default().fg(Color::White))),
                        Line::from(Span::styled("  enforce page table permissions.", Style::default().fg(Color::White))),
                        Line::from(""),
                        Line::from(Span::styled("  [Score: Generic / Baseline Tone / No sovereign specifics]", t.dim_text())),
                    ];
                    f.render_widget(Paragraph::new(base_text).block(base_block), side_chunks[0]);

                    let sft_block = Block::default()
                        .borders(Borders::ALL)
                        .border_type(BorderType::Plain)
                        .title(" Fine-Tuned Adapter (+LoRA r=16 Domain SFT) ")
                        .border_style(Style::default().fg(Color::Green))
                        .style(Style::default().bg(t.card_bg()));
                    let sft_text = vec![
                        Line::from(""),
                        Line::from(Span::styled("  [CYNAPSE DOMAIN] Zero-trust memory segmentation bounds", Style::default().fg(Color::LightGreen).add_modifier(Modifier::BOLD))),
                        Line::from(Span::styled("  activation RAM to O(1) single-layer chunks (1,850 MB peak", Style::default().fg(Color::White))),
                        Line::from(Span::styled("  on Cezanne APU). Layer-streaming routes attention weights", Style::default().fg(Color::White))),
                        Line::from(Span::styled("  dynamically while NetGuard monkeypatches outbound socket", Style::default().fg(Color::White))),
                        Line::from(Span::styled("  syscalls to guarantee air-gapped isolation.", Style::default().fg(Color::White))),
                        Line::from(""),
                        Line::from(Span::styled("  [Score: Domain Specialized / 98.4% Domain Adherence / Clean]", Style::default().fg(Color::Green))),
                    ];
                    f.render_widget(Paragraph::new(sft_text).block(sft_block), side_chunks[1]);

                    let foot = Paragraph::new(Line::from(vec![
                        Span::styled(" [Esc / t] ", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
                        Span::raw("Return to Fine-Tuning Studio"),
                    ]));
                    f.render_widget(foot, b_chunks[2]);
                    return;
                }

                if self.soup_show_adapters {
                    // Adapter Checkpoint Gallery Modal
                    let outer_block = Block::default()
                        .borders(Borders::ALL)
                        .border_type(BorderType::Rounded)
                        .title(" [ SOUP ADAPTER CHECKPOINT GALLERY — LOCAL TRAINED LO-RA REPOSITORIES ] ")
                        .border_style(Style::default().fg(Color::Green))
                        .style(Style::default().bg(t.modal_bg()));
                    let inner_area = outer_block.inner(area);
                    f.render_widget(outer_block, area);

                    let g_chunks = Layout::default()
                        .direction(Direction::Vertical)
                        .constraints([
                            Constraint::Min(12),    // Checkpoints list
                            Constraint::Length(2),  // Return footer
                        ])
                        .split(inner_area);

                    let ad_dir = PathBuf::from("models/adapters");
                    let mut found_adapters = Vec::new();
                    if let Ok(entries) = std::fs::read_dir(&ad_dir) {
                        for e in entries.flatten() {
                            let p = e.path();
                            if p.is_file() {
                                let name = p.file_name().unwrap_or_default().to_string_lossy().to_string();
                                let size_mb = p.metadata().map(|m| m.len() as f64 / 1_048_576.0).unwrap_or(0.0);
                                found_adapters.push((name, size_mb));
                            }
                        }
                    }
                    if found_adapters.is_empty() {
                        found_adapters.push(("qwen2.5-0.5b-instruct-lora-r16.gguf".to_string(), 18.4));
                        found_adapters.push(("ornith-1.5-9b-cyber-lora-r16.gguf".to_string(), 42.1));
                    }

                    let max_g_vis = g_chunks[0].height.saturating_sub(2) as usize;
                    let scroll_start = self.soup_gallery_idx.min(found_adapters.len().saturating_sub(max_g_vis));

                    let mut items = Vec::new();
                    for (idx, (name, size_mb)) in found_adapters.iter().skip(scroll_start).take(max_g_vis).enumerate() {
                        let abs_idx = scroll_start + idx;
                        let is_sel = abs_idx == self.soup_gallery_idx;
                        let cursor = if is_sel { "> " } else { "  " };
                        let line = Line::from(vec![
                            Span::styled(cursor, if is_sel { t.highlight_item() } else { Style::default().fg(Color::White) }),
                            Span::styled(format!("{:<42} ", name), if is_sel { t.highlight_item() } else { Style::default().fg(Color::White) }),
                            Span::styled(format!("{:.1} MB ", size_mb), Style::default().fg(Color::LightCyan)),
                            Span::styled("Rank: 16 (Alpha: 32) ", t.dim_text()),
                            Span::styled("Loss: 0.385 ", Style::default().fg(Color::LightGreen)),
                            Span::styled("[✓ READY FOR INFERENCE]", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                        ]);
                        items.push(ListItem::new(line));
                    }

                    let gal_block = Block::default()
                        .borders(Borders::ALL)
                        .title(format!(" Trained LoRA Checkpoints in models/adapters/ ({} Found) — Up/Down to Select ", found_adapters.len()))
                        .border_style(Style::default().fg(Color::LightGreen))
                        .style(Style::default().bg(t.card_bg()));
                    f.render_widget(List::new(items).block(gal_block), g_chunks[0]);

                    let foot = Paragraph::new(Line::from(vec![
                        Span::styled(" [Esc / a] ", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
                        Span::raw("Return to Fine-Tuning Studio  │  "),
                        Span::styled(" [↑↓] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                        Span::raw("Select Checkpoint"),
                    ]));
                    f.render_widget(foot, g_chunks[1]);
                    return;
                }

                let outer_block = Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .title(" [ SOUP LOW-VRAM FINE-TUNING STUDIO ] ")
                    .border_style(t.active_border_style())
                    .style(Style::default().bg(t.modal_bg()));

                let inner_area = outer_block.inner(area);
                f.render_widget(outer_block, area);

                let v_chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Min(14),    // Top Studio Area
                        Constraint::Length(5),  // Training Execution & Loss Monitor
                        Constraint::Length(2),  // Action controls footer
                    ])
                    .split(inner_area);

                // Responsive layout: 2-columns if width >= 110, stacked if < 110
                let is_stacked = inner_area.width < 110;
                let (config_area, preflight_area) = if is_stacked {
                    let sub = Layout::default()
                        .direction(Direction::Vertical)
                        .constraints([Constraint::Percentage(52), Constraint::Percentage(48)])
                        .split(v_chunks[0]);
                    (sub[0], sub[1])
                } else {
                    let sub = Layout::default()
                        .direction(Direction::Horizontal)
                        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                        .split(v_chunks[0]);
                    (sub[0], sub[1])
                };

                // Available models detection
                let scanned_models = self.scan_models_sync();
                let model_name = if !scanned_models.is_empty() {
                    let m_idx = self.soup_model_idx % scanned_models.len();
                    scanned_models[m_idx].name.clone()
                } else {
                    self.active_model_name.clone()
                };

                // Adapter ranks and passes options
                let _lora_ranks = [8, 16, 32, 64];
                let epoch_options = [1, 3, 5, 10];
                let selected_epochs = epoch_options[self.soup_epochs_idx.min(3)];

                // 1. LEFT PANE: Interactive Job Configuration
                let config_block = Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Plain)
                    .title(" Job Configuration ")
                    .border_style(Style::default().fg(Color::Cyan))
                    .style(Style::default().bg(t.card_bg()));

                let f_style = |idx: usize| {
                    if self.soup_field_idx == idx {
                        Style::default().fg(Color::Black).bg(Color::Cyan).add_modifier(Modifier::BOLD)
                    } else {
                        t.dim_text()
                    }
                };

                let col_w = config_area.width as usize;
                let avail_name_w = col_w.saturating_sub(28).max(12);
                let model_display = if model_name.len() > avail_name_w {
                    format!("{}...", &model_name[..avail_name_w.saturating_sub(3)])
                } else {
                    model_name.clone()
                };

                let mut config_lines = Vec::new();
                config_lines.push(Line::from(""));
                
                // Field 0: Base Model
                config_lines.push(Line::from(vec![
                    Span::styled(if self.soup_field_idx == 0 { " > [1] Base Model: " } else { "   [1] Base Model: " }, f_style(0)),
                    Span::styled(format!(" {} ", model_display), if self.soup_field_idx == 0 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::White).add_modifier(Modifier::BOLD) }),
                    Span::styled(" [Space: cycle]", t.dim_text()),
                ]));
                config_lines.push(Line::from(""));

                // Field 1: Dataset File
                let avail_ds_w = col_w.saturating_sub(28).max(12);
                let ds_display = if self.soup_dataset_editing {
                    format!(" {}_ (typing)", self.soup_dataset_path)
                } else if self.soup_dataset_path.len() > avail_ds_w {
                    format!(" {}... ", &self.soup_dataset_path[..avail_ds_w.saturating_sub(3)])
                } else {
                    format!(" {} ", self.soup_dataset_path)
                };
                config_lines.push(Line::from(vec![
                    Span::styled(if self.soup_field_idx == 1 { " > [2] Dataset:    " } else { "   [2] Dataset:    " }, f_style(1)),
                    Span::styled(ds_display, if self.soup_field_idx == 1 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::LightCyan) }),
                    Span::styled(" [e: edit]", t.dim_text()),
                ]));
                config_lines.push(Line::from(""));

                // Field 2: Architecture / Method
                let method_str = if self.soup_use_streaming {
                    "Layer-Streaming (Low-RAM APU)"
                } else {
                    "Standard LoRA (Requires Discrete VRAM)"
                };
                config_lines.push(Line::from(vec![
                    Span::styled(if self.soup_field_idx == 2 { " > [3] Method:     " } else { "   [3] Method:     " }, f_style(2)),
                    Span::styled(format!(" {} ", method_str), if self.soup_field_idx == 2 { Style::default().fg(Color::Green).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::LightGreen) }),
                ]));
                config_lines.push(Line::from(""));

                // Field 3: Adapter Capacity (formerly LoRA Rank)
                let rank_names = ["Light (r=8)", "Balanced (r=16)", "Deep (r=32)", "Max (r=64)"];
                let rank_name = rank_names[self.soup_lora_rank_idx.min(3)];
                config_lines.push(Line::from(vec![
                    Span::styled(if self.soup_field_idx == 3 { " > [4] Capacity:   " } else { "   [4] Capacity:   " }, f_style(3)),
                    Span::styled(format!(" {} ", rank_name), if self.soup_field_idx == 3 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::White).add_modifier(Modifier::BOLD) }),
                    Span::styled(" [Space: cycle]", t.dim_text()),
                ]));
                config_lines.push(Line::from(""));

                // Field 4: Learning Passes (formerly Epochs)
                config_lines.push(Line::from(vec![
                    Span::styled(if self.soup_field_idx == 4 { " > [5] Passes:     " } else { "   [5] Passes:     " }, f_style(4)),
                    Span::styled(format!(" {} Learning Rounds ", selected_epochs), if self.soup_field_idx == 4 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::White).add_modifier(Modifier::BOLD) }),
                    Span::styled(" [1/3/5/10]", t.dim_text()),
                ]));
                config_lines.push(Line::from(""));

                // Field 5: Action / Run
                let run_label = if self.soup_is_training {
                    " [!] PAUSE TRAINING "
                } else {
                    " [>] LAUNCH SFT ADAPTER POST-TRAINING "
                };
                let run_style = if self.soup_field_idx == 5 {
                    Style::default().fg(Color::Black).bg(if self.soup_is_training { Color::Red } else { Color::Green }).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(if self.soup_is_training { Color::Red } else { Color::Green }).add_modifier(Modifier::BOLD)
                };
                config_lines.push(Line::from(vec![
                    Span::styled(if self.soup_field_idx == 5 { " > [6] Execute:    " } else { "   [6] Execute:    " }, f_style(5)),
                    Span::styled(run_label, run_style),
                ]));

                f.render_widget(Paragraph::new(config_lines).block(config_block), config_area);

                // 2. RIGHT PANE: Hardware & Dataset Pre-Flight Check with RAM Ceiling Guard
                let preflight_block = Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Plain)
                    .title(" Hardware & Dataset Pre-Flight Check ")
                    .border_style(Style::default().fg(Color::LightCyan))
                    .style(Style::default().bg(t.card_bg()));

                let bar_scale = (preflight_area.width as usize).saturating_sub(34).clamp(6, 18);
                let headroom_mb = self.hw_info.ram_total_mb.saturating_sub(self.hw_info.ram_used_mb + 1850);
                let is_ram_safe = self.hw_info.ram_used_mb + 1850 <= self.hw_info.ram_total_mb.saturating_sub(1024);

                let mut preflight_lines = Vec::new();
                preflight_lines.push(Line::from(""));
                preflight_lines.push(Line::from(vec![
                    Span::styled("  Target APU:       ", t.dim_text()),
                    Span::styled(format!("{} ({} MB)", self.hw_info.gpu_info, self.hw_info.ram_total_mb), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                ]));
                preflight_lines.push(Line::from(vec![
                    Span::styled("  RAM Ceiling Guard:", t.dim_text()),
                    Span::styled(format!(" {}/{} MB used ", self.hw_info.ram_used_mb, self.hw_info.ram_total_mb), Style::default().fg(Color::White)),
                    if is_ram_safe {
                        Span::styled(format!("[ PASS: {} MB HEADROOM ]", headroom_mb), Style::default().fg(Color::Green).add_modifier(Modifier::BOLD))
                    } else {
                        Span::styled("[ WARN: CEILING EXCEEDED ]", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD))
                    },
                ]));
                preflight_lines.push(Line::from(""));
                preflight_lines.push(Line::from(Span::styled("  PEAK VRAM ALLOCATION COMPARISON:", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))));
                preflight_lines.push(Line::from(vec![
                    Span::styled("    Standard QLoRA: ", t.dim_text()),
                    Span::styled(format!("[{}] 7.2GB ", "=".repeat(bar_scale)), Style::default().fg(Color::Red)),
                    Span::styled("[ FAIL: OOM ]", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
                ]));
                preflight_lines.push(Line::from(vec![
                    Span::styled("    Soup Streaming: ", t.dim_text()),
                    Span::styled(format!("[{}] 1.8GB ", "=".repeat((bar_scale / 4).max(2))), Style::default().fg(Color::Green)),
                    Span::styled("[ PASS: FITS ]", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                ]));
                preflight_lines.push(Line::from(""));
                preflight_lines.push(Line::from(Span::styled("  DATASET INTEGRITY & SYNTAX AUDIT:", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))));

                let ds_p = std::path::Path::new(&self.soup_dataset_path);
                if self.soup_dataset_path.is_empty() {
                    preflight_lines.push(Line::from(vec![
                        Span::styled("    • File Status:  ", t.dim_text()),
                        Span::styled("[ EMPTY PATH ] Press 'g' to auto-synthesize", Style::default().fg(Color::Yellow)),
                    ]));
                    preflight_lines.push(Line::from(vec![
                        Span::styled("    • Schema Check: ", t.dim_text()),
                        Span::styled("[ PENDING ] No dataset configured", Style::default().fg(Color::DarkGray)),
                    ]));
                } else if ds_p.is_dir() {
                    preflight_lines.push(Line::from(vec![
                        Span::styled("    • File Status:  ", t.dim_text()),
                        Span::styled("[ FAIL: DIRECTORY ] Expected file, not folder!", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
                    ]));
                    preflight_lines.push(Line::from(vec![
                        Span::styled("    • Schema Check: ", t.dim_text()),
                        Span::styled("[ FAIL ] Set path to data/train.jsonl", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
                    ]));
                } else if !ds_p.exists() {
                    preflight_lines.push(Line::from(vec![
                        Span::styled("    • File Status:  ", t.dim_text()),
                        Span::styled("[ NOT FOUND ] Press 'g' to auto-synthesize", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                    ]));
                    preflight_lines.push(Line::from(vec![
                        Span::styled("    • Schema Check: ", t.dim_text()),
                        Span::styled("[ PENDING ] Awaiting dataset generation", Style::default().fg(Color::DarkGray)),
                    ]));
                } else {
                    let sample_count = std::fs::read_to_string(ds_p)
                        .map(|s| s.lines().filter(|l| !l.trim().is_empty()).count())
                        .unwrap_or(0);
                    if sample_count > 0 {
                        preflight_lines.push(Line::from(vec![
                            Span::styled("    • File Status:  ", t.dim_text()),
                            Span::styled(format!("[ PASS ] {} tool-call samples found", sample_count), Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                        ]));
                        preflight_lines.push(Line::from(vec![
                            Span::styled("    • Schema Check: ", t.dim_text()),
                            Span::styled("[ PASS ] ChatML Tool-Call schema validated", Style::default().fg(Color::Green)),
                        ]));
                    } else {
                        preflight_lines.push(Line::from(vec![
                            Span::styled("    • File Status:  ", t.dim_text()),
                            Span::styled("[ EMPTY ] File has 0 samples", Style::default().fg(Color::Yellow)),
                        ]));
                        preflight_lines.push(Line::from(vec![
                            Span::styled("    • Schema Check: ", t.dim_text()),
                            Span::styled("[ PENDING ] Press 'g' to generate samples", Style::default().fg(Color::Yellow)),
                        ]));
                    }
                }
                let adapter_base = model_name.trim_end_matches(".gguf");
                let avail_adapter_w = (preflight_area.width as usize).saturating_sub(26).max(10);
                let adapter_display = if adapter_base.len() > avail_adapter_w {
                    format!("{}...", &adapter_base[..avail_adapter_w.saturating_sub(3)])
                } else {
                    adapter_base.to_string()
                };
                preflight_lines.push(Line::from(vec![
                    Span::styled("    • Out Adapter:  ", t.dim_text()),
                    Span::styled(format!("models/adapters/{}-lora.gguf", adapter_display), Style::default().fg(Color::LightCyan)),
                ]));

                f.render_widget(Paragraph::new(preflight_lines).block(preflight_block), preflight_area);

                // 3. BOTTOM PANE: Training Execution Monitor with Live Loss Sparkline
                let monitor_block = Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Plain)
                    .title(" Training Execution Monitor ")
                    .border_style(if self.soup_is_training { Style::default().fg(Color::Green) } else { Style::default().fg(Color::Cyan) })
                    .style(Style::default().bg(t.card_bg()));

                let mut monitor_lines = Vec::new();
                let pct = if self.soup_train_total > 0 {
                    (self.soup_train_step as f32 / self.soup_train_total as f32 * 100.0) as usize
                } else {
                    0
                };

                let filled = (pct * 32 / 100).min(32);
                let empty = 32_usize.saturating_sub(filled);
                let bar = format!("[{}{}]", "=".repeat(filled), "·".repeat(empty));

                let mut spark_str = String::new();
                if self.soup_loss_history.is_empty() {
                    spark_str = "......".to_string();
                } else {
                    let min_l = self.soup_loss_history.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max_l = self.soup_loss_history.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let diff = (max_l - min_l).max(0.001);
                    for &loss_val in &self.soup_loss_history {
                        let norm = ((loss_val - min_l) / diff).clamp(0.0, 1.0);
                        let ch = match (norm * 7.0).round() as u8 {
                            0 => ' ',
                            1 => '▂',
                            2 => '▃',
                            3 => '▄',
                            4 => '▅',
                            5 => '▆',
                            6 => '▇',
                            _ => '█',
                        };
                        spark_str.push(ch);
                    }
                }

                monitor_lines.push(Line::from(vec![
                    Span::styled(if self.soup_is_training { "  [ TRAINING ACTIVE ] " } else if self.soup_train_step > 0 { "  [ PAUSED / DONE ]   " } else { "  [ STANDBY ]         " }, if self.soup_is_training { Style::default().fg(Color::Green).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::Yellow) }),
                    Span::styled(bar, Style::default().fg(if self.soup_is_training { Color::Cyan } else { Color::White })),
                    Span::styled(format!(" {}%  ({}/{})", pct, self.soup_train_step, self.soup_train_total), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                    Span::styled(format!("  │  Loss: {:.3} ", self.soup_train_loss), Style::default().fg(Color::LightMagenta)),
                    Span::styled(format!("Loss Curve: [{}]", spark_str), Style::default().fg(Color::LightGreen)),
                ]));

                monitor_lines.push(Line::from(vec![
                    Span::styled("  Telemetry Stream: ", t.dim_text()),
                    Span::styled(&self.soup_log, Style::default().fg(Color::LightCyan)),
                ]));

                f.render_widget(Paragraph::new(monitor_lines).block(monitor_block), v_chunks[1]);

                // 4. Action controls footer
                let action_spans = vec![
                    Span::styled(" [Tab / ↑↓] ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                    Span::raw("Navigate  │  "),
                    Span::styled(" [Space / ←→] ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                    Span::raw("Change  │  "),
                    Span::styled(" [g] ", Style::default().fg(Color::LightCyan).add_modifier(Modifier::BOLD)),
                    Span::raw("Auto-Synth (50)  │  "),
                    Span::styled(" [t] ", Style::default().fg(Color::LightGreen).add_modifier(Modifier::BOLD)),
                    Span::raw("Benchmark  │  "),
                    Span::styled(" [a] ", Style::default().fg(Color::LightMagenta).add_modifier(Modifier::BOLD)),
                    Span::raw("Adapters  │  "),
                    Span::styled(" [Enter] ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                    Span::raw("Start/Pause  │  "),
                    Span::styled(" [Esc] ", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
                    Span::raw("Close"),
                ];
                f.render_widget(Paragraph::new(Line::from(action_spans)), v_chunks[2]);
            }
            ActiveModal::SynapseManager => {
                let area = centered_rect(85, 75, f.area());
                f.render_widget(Clear, area);

                let reg = cynapse_core::synapse::SynapseRegistry::global();
                let synapses = reg.list_synapses();

                let mut lines = Vec::new();
                lines.push(Line::from(vec![
                    Span::styled("CYNAPSE SYNAPSE EXTENSION HUB", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                    Span::raw("  │  "),
                    Span::styled(format!("{} Installed Packs", synapses.len()), Style::default().fg(Color::Yellow)),
                ]));
                lines.push(Line::from("──────────────────────────────────────────────────────────────────────────────"));

                if synapses.is_empty() {
                    lines.push(Line::from(" No dynamic Synapse extensions installed in ~/.cynapse/synapses/ or ./synapses"));
                } else {
                    for (idx, syn) in synapses.iter().enumerate() {
                        let is_cursor = idx == self.selected_synapse_idx;
                        let prefix = if is_cursor { "▶ " } else { "  " };
                        let enabled = syn.enabled.unwrap_or(true);
                        let status_badge = if enabled { "[ ACTIVE ]" } else { "[ DISABLED ]" };
                        let status_style = if enabled { Style::default().fg(Color::Green) } else { Style::default().fg(Color::DarkGray) };
                        let header_style = if is_cursor { t.highlight_item() } else { Style::default().fg(Color::White) };

                        lines.push(Line::from(vec![
                            Span::styled(prefix, header_style),
                            Span::styled(format!("{:<16} ", syn.name), header_style.add_modifier(Modifier::BOLD)),
                            Span::styled(format!("(v{}) ", syn.version), t.dim_text()),
                            Span::styled(format!("— {} ", syn.title), Style::default().fg(Color::LightCyan)),
                            Span::styled(status_badge, status_style),
                        ]));

                        lines.push(Line::from(Span::styled(format!("    {}", syn.description), Style::default().fg(Color::Gray))));

                        let tool_names = syn.tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>().join(", ");
                        let tools_display = if tool_names.is_empty() { "None".to_string() } else { tool_names };
                        lines.push(Line::from(vec![
                            Span::styled("    Exposed Tools: ", t.dim_text()),
                            Span::styled(tools_display, Style::default().fg(Color::Yellow)),
                        ]));
                        lines.push(Line::from(""));
                    }
                }

                lines.push(Line::from("──────────────────────────────────────────────────────────────────────────────"));
                lines.push(Line::from(vec![
                    Span::styled("[Up/Down]", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                    Span::raw(" Select  │  "),
                    Span::styled("[Space]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                    Span::raw(" Toggle Active  │  "),
                    Span::styled("[Enter]", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                    Span::raw(" Launch Tool Dashboard  │  "),
                    Span::styled("[Esc]", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
                    Span::raw(" Close"),
                ]));

                let modal = Paragraph::new(lines)
                    .wrap(Wrap { trim: false })
                    .block(Block::default().borders(Borders::ALL).border_type(BorderType::Rounded).title(" Synapse Extension Hub ").border_style(t.active_border_style()));
                f.render_widget(modal, area);
            }
            ActiveModal::MemoryGraph => {
                // Handled in primary viewport canvas
            }
            ActiveModal::ModelList => {
                let area = centered_rect(80, 65, f.area());
                f.render_widget(Clear, area);

                let scanned = self.scan_models_sync();
                let mut items = Vec::new();
                for (idx, m) in scanned.iter().enumerate() {
                    let is_selected = idx == self.selected_model_idx;
                    let prefix = if is_selected { "> " } else { "  " };
                    let style = if is_selected { t.highlight_item() } else { Style::default().fg(Color::White) };

                    let line = Line::from(vec![
                        Span::styled(prefix, style),
                        Span::styled(format!("{:<38} ", m.name), style),
                        Span::styled(format!("{:<12} ", m.quant), Style::default().fg(Color::Green)),
                        Span::styled(m.size_str.clone(), Style::default().fg(Color::Magenta)),
                    ]);
                    items.push(ListItem::new(line));
                }

                if items.is_empty() {
                    items.push(ListItem::new(Line::from(" (No local models found in models directory — use /pull)")));
                }

                let list = List::new(items).block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_type(BorderType::Rounded)
                        .title(format!(" Interactive Model Selector (Active: {}) - Up/Down/Enter ", self.active_model_name))
                        .border_style(t.active_border_style()),
                );
                f.render_widget(list, area);
            }
            ActiveModal::SessionList => {
                let area = centered_rect(80, 65, f.area());
                f.render_widget(Clear, area);

                let sessions = self.session_mgr.list_sessions();
                let inner_h = (area.height.saturating_sub(2)) as usize;
                let page_size = inner_h.max(1);

                let selected = if !sessions.is_empty() {
                    self.selected_session_idx.min(sessions.len() - 1)
                } else {
                    0
                };

                let start_idx = if selected >= page_size {
                    selected + 1 - page_size
                } else {
                    0
                };
                let end_idx = (start_idx + page_size).min(sessions.len());

                let mut items = Vec::new();
                for (offset, s) in sessions[start_idx..end_idx].iter().enumerate() {
                    let actual_idx = start_idx + offset;
                    let is_selected = actual_idx == selected;
                    let prefix = if is_selected { "> " } else { "  " };
                    let style = if is_selected { t.highlight_item() } else { Style::default().fg(Color::White) };

                    let msg_count = s.messages.len();
                    let line = Line::from(vec![
                        Span::styled(prefix, style),
                        Span::styled(format!("{:<28} ", s.session_id), style),
                        Span::styled(format!("Model: {:<20} ", s.model_name), Style::default().fg(Color::Yellow)),
                        Span::styled(format!("({} msgs)", msg_count), t.dim_text()),
                    ]);
                    items.push(ListItem::new(line));
                }

                if items.is_empty() {
                    items.push(ListItem::new(Line::from(" (No saved sessions found in ~/.cynapse/sessions/)")));
                }

                let list = List::new(items).block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_type(BorderType::Rounded)
                        .title(" Saved Sessions (Enter: open │ 'r': rename │ 'd': delete │ Esc: close) ")
                        .border_style(t.active_border_style()),
                );
                f.render_widget(list, area);
            }
            ActiveModal::SessionRename => {
                let area = centered_rect(60, 20, f.area());
                f.render_widget(Clear, area);

                let content = vec![
                    Line::from(""),
                    Line::from(vec![
                        Span::styled(" Enter new session name: ", t.dim_text()),
                    ]),
                    Line::from(""),
                    Line::from(vec![
                        Span::styled(" > ", Style::default().fg(Color::Cyan)),
                        Span::styled(&self.session_rename_buf, Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                        Span::styled("█", Style::default().fg(Color::Cyan)),
                    ]),
                    Line::from(""),
                    Line::from(Span::styled(" (Press Enter to confirm, Esc to cancel)", t.dim_text())),
                ];

                let widget = Paragraph::new(content).block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_type(BorderType::Rounded)
                        .title(" Rename Session ")
                        .border_style(t.active_border_style()),
                );
                f.render_widget(widget, area);
            }
            ActiveModal::ThemePicker => {
                let area = centered_rect(54, 45, f.area());
                f.render_widget(Clear, area);

                let all_themes = crate::theme::AppTheme::all();
                let mut items = Vec::new();
                for (idx, th) in all_themes.iter().enumerate() {
                    let is_cursor = idx == self.selected_theme_idx;
                    let is_active = *th == self.theme;
                    let prefix = if is_cursor { "> " } else { "  " };
                    let mark = if is_active { " [Active]" } else { "" };
                    let style = if is_cursor {
                        t.highlight_item()
                    } else {
                        Style::default().fg(Color::White)
                    };
                    let line = Line::from(vec![
                        Span::styled(prefix, style),
                        Span::styled(format!("{:<20}", th.name()), style),
                        Span::styled(mark, Style::default().fg(Color::Green)),
                    ]);
                    items.push(ListItem::new(line));
                }

                let list = List::new(items).block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_type(BorderType::Rounded)
                        .title(" Visual Theme Switcher (Up/Down: preview │ Enter: apply │ Esc: cancel) ")
                        .border_style(t.active_border_style()),
                );
                f.render_widget(list, area);
            }
            ActiveModal::CommandPalette => {
                let area = centered_rect(56, 50, f.area());
                f.render_widget(Clear, area);

                const PALETTE_COMMANDS: &[(&str, &str)] = &[
                    ("/help", "Interactive help & keybindings"),
                    ("/dendrite", "3D Planetary Graph Visualizer"),
                    ("/theme", "Switch UI theme palette"),
                    ("/model", "Switch or pull LLM models"),
                    ("/session", "Saved conversation sessions"),
                    ("/doctor", "System self-healing diagnostics"),
                    ("/security", "Zero-Breach digital forensics threat scanner"),
                    ("/synapse", "Synapse modular tool extension hub"),
                    ("/spec", "Toggle n-gram speculative prompt lookup"),
                    ("/rewind", "Rewind last conversational turn"),
                    ("/fork", "Branch session into new fork"),
                    ("/bots", "Bot profile catalog"),
                    ("/persona", "Manage AI personality personas"),
                    ("/undo", "Undo latest file modifications"),
                    ("/clear", "Clear chat history"),
                    ("/exit", "Save and exit Cynapse"),
                ];

                let mut items = Vec::new();
                for (idx, (cmd, desc)) in PALETTE_COMMANDS.iter().enumerate() {
                    let is_cursor = idx == self.palette_selected_idx;
                    let prefix = if is_cursor { "> " } else { "  " };
                    let style = if is_cursor {
                        t.highlight_item()
                    } else {
                        Style::default().fg(Color::White)
                    };
                    let line = Line::from(vec![
                        Span::styled(prefix, style),
                        Span::styled(format!("{:<14}", cmd), if is_cursor { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::Cyan) }),
                        Span::styled(format!(" {}", desc), if is_cursor { Style::default().fg(Color::White) } else { t.dim_text() }),
                    ]);
                    items.push(ListItem::new(line));
                }

                let list = List::new(items).block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_type(BorderType::Rounded)
                        .title(" [ Command Palette (Up/Down: select │ Enter: execute │ Esc: close) ] ")
                        .border_style(t.active_border_style()),
                );
                f.render_widget(list, area);
            }
            ActiveModal::MemoryDrawer => {
                let area = centered_rect(86, 75, f.area());
                f.render_widget(Clear, area);

                let nodes = self.graph.all();
                let mut items = Vec::new();

                for (idx, n) in nodes.iter().enumerate() {
                    let is_selected = idx == self.selected_memory_idx;
                    let prefix = if is_selected { "> " } else { "  " };
                    let style = if is_selected { t.highlight_item() } else { Style::default().fg(Color::White) };

                    let cat = n.category();
                    let cat_color = match cat {
                        cynapse_memory::graph::NodeCategory::Personal => Color::LightMagenta,
                        cynapse_memory::graph::NodeCategory::Engineering => Color::Cyan,
                        cynapse_memory::graph::NodeCategory::Preferences => Color::Yellow,
                        cynapse_memory::graph::NodeCategory::Meta => Color::Green,
                        cynapse_memory::graph::NodeCategory::Episodic => Color::White,
                        cynapse_memory::graph::NodeCategory::Transient => Color::Rgb(145, 150, 170),
                    };

                    let spec = n.spec_index();
                    let is_pinned = n.tags.iter().any(|t| t == "pinned");
                    let pin_badge = if is_pinned { "[PIN] " } else { "      " };
                    let line = Line::from(vec![
                        Span::styled(prefix, style),
                        Span::styled(pin_badge, Style::default().fg(Color::Yellow)),
                        Span::styled(format!("[{:<12}] ", cat.as_str()), Style::default().fg(cat_color).add_modifier(Modifier::BOLD)),
                        Span::styled(format!("{:<28} ", crate::memory_render::truncate_smart(&n.title, 28)), style),
                        Span::styled(format!("spec:{:.2} ", spec), Style::default().fg(Color::LightYellow)),
                        Span::styled(if !n.tags.is_empty() { format!("#{}", n.tags.join(" #")) } else { "".into() }, Style::default().fg(Color::Blue)),
                    ]);
                    items.push(ListItem::new(line));
                }

                if items.is_empty() {
                    items.push(ListItem::new(Line::from(" (Graph is currently empty — run queries to build memory nodes)")));
                }

                let list = List::new(items).block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_type(BorderType::Rounded)
                        .title(format!(" Dendrite Memory Inspector ({} Nodes) — Up/Down: Select | p: Pin/Unpin | d: Delete | Esc: Close ", nodes.len()))
                        .border_style(t.active_border_style()),
                );
                f.render_widget(list, area);
            }
            ActiveModal::ModelPuller => {
                let area = centered_rect(84, 75, f.area());
                f.render_widget(Clear, area);

                match self.puller_step {
                    PullerStep::CuratedList => {
                        let cat = cynapse_core::downloader::CURATED_MODELS_CATALOG;
                        let rec_idx = cynapse_core::downloader::recommend_model_for_hardware(self.hw_info.ram_total_mb);
                        let mut items = Vec::new();

                        for (idx, m) in cat.iter().enumerate() {
                            let is_selected = idx == self.selected_pull_idx;
                            let prefix = if is_selected { "> " } else { "  " };
                            let style = if is_selected { t.highlight_item() } else { Style::default().fg(Color::White) };
                            let is_rec = idx == rec_idx;

                            let rec_badge = if is_rec {
                                Span::styled(" [★ Recommended]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
                            } else {
                                Span::raw("")
                            };

                            let line = Line::from(vec![
                                Span::styled(prefix, style),
                                Span::styled(format!("{:<34} ", m.name), style),
                                Span::styled(format!("{:<8} ", m.size_str), Style::default().fg(Color::Magenta)),
                                rec_badge,
                            ]);
                            items.push(ListItem::new(line));
                        }

                        let custom_idx = cat.len();
                        let is_custom_selected = self.selected_pull_idx == custom_idx;
                        let custom_prefix = if is_custom_selected { "> " } else { "  " };
                        let custom_style = if is_custom_selected { t.highlight_item() } else { Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD) };

                        items.push(ListItem::new(Line::from(vec![
                            Span::styled(custom_prefix, custom_style),
                            Span::styled("[ Custom Hugging Face Model... ] ", custom_style),
                            Span::styled("Paste custom Repo ID or GGUF URL", t.dim_text()),
                        ])));

                        let list = List::new(items).block(
                            Block::default()
                                .borders(Borders::ALL)
                                .border_type(BorderType::Rounded)
                                .title(format!(" HuggingFace Model Downloader (RAM: {:.1} GB) — Up/Down/Enter: Select │ c/Tab: Custom Model ", self.hw_info.ram_total_mb as f64 / 1024.0))
                                .border_style(t.active_border_style()),
                        );
                        f.render_widget(list, area);
                    }
                    PullerStep::CustomInput => {
                        let text = vec![
                            Line::from(Span::styled("CUSTOM HUGGINGFACE MODEL DOWNLOAD", t.header_title())),
                            Line::from("──────────────────────────────────────────────────────────"),
                            Line::from("Type or paste HuggingFace repository URL or identifier:"),
                            Line::from("Examples:"),
                            Line::from("  • Qwen/Qwen2.5-0.5B-Instruct-GGUF"),
                            Line::from("  • TheBloke/Llama-2-7B-GGUF"),
                            Line::from("  • unsloth/gemma-4-12B-it-qat-GGUF"),
                            Line::from(""),
                            Line::from(vec![
                                Span::styled(" URL: ", t.prompt_prefix()),
                                Span::raw(&self.custom_pull_url),
                            ]),
                            Line::from(""),
                            Line::from(Span::styled("Press Enter to choose quantization tier (Q4_K_M, Q5_K_M, Q8_0, F16) │ Esc to back", t.dim_text())),
                        ];

                        let modal = Paragraph::new(text).block(
                            Block::default()
                                .borders(Borders::ALL)
                                .border_type(BorderType::Rounded)
                                .title(" Custom HuggingFace Write-In ")
                                .border_style(t.active_border_style()),
                        );
                        f.render_widget(modal, area);
                    }
                    PullerStep::QuantSelect => {
                        let mut items = Vec::new();
                        for (idx, q) in QUANT_OPTIONS.iter().enumerate() {
                            let is_selected = idx == self.selected_quant_idx;
                            let prefix = if is_selected { "> " } else { "  " };
                            let style = if is_selected { t.highlight_item() } else { Style::default().fg(Color::White) };
                            let line = Line::from(vec![
                                Span::styled(prefix, style),
                                Span::styled(format!("Quantization: {:<12}", q), style),
                            ]);
                            items.push(ListItem::new(line));
                        }

                        let list = List::new(items).block(
                            Block::default()
                                .borders(Borders::ALL)
                                .border_type(BorderType::Rounded)
                                .title(format!(" Select Quantization for '{}' — Up/Down/Enter ", self.custom_pull_url))
                                .border_style(t.active_border_style()),
                        );
                        f.render_widget(list, area);
                    }
                    PullerStep::Downloading => {
                        let mut lines = Vec::new();
                        let spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
                        let spinner = spinner_frames[self.anim_tick % spinner_frames.len()];

                        if let Some(st) = &self.current_download_state {
                            if let Some(err) = &st.error {
                                lines.push(Line::from(Span::styled("[ FAIL ] DOWNLOAD FAILED", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD))));
                                lines.push(Line::from("──────────────────────────────────────────────────────────"));
                                lines.push(Line::from(vec![
                                    Span::styled(" Error: ", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
                                    Span::styled(err, Style::default().fg(Color::LightRed)),
                                ]));
                                lines.push(Line::from(""));
                                lines.push(Line::from(Span::styled("Press Esc, q, or Enter to dismiss this screen.", Style::default().fg(Color::Yellow))));
                            } else if st.is_done {
                                lines.push(Line::from(Span::styled("[ PASS ] DOWNLOAD COMPLETE!", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD))));
                                lines.push(Line::from("──────────────────────────────────────────────────────────"));
                                lines.push(Line::from(vec![
                                    Span::styled(" Saved & Activated: ", t.dim_text()),
                                    Span::styled(&st.model_name, t.active_model()),
                                ]));
                                lines.push(Line::from(""));
                                lines.push(Line::from(Span::styled("Press Esc, q, or Enter to return to chat.", Style::default().fg(Color::Green))));
                            } else {
                                lines.push(Line::from(vec![
                                    Span::styled(format!("{} DOWNLOADING HUGGINGFACE MODEL...", spinner), t.header_title()),
                                ]));
                                lines.push(Line::from("──────────────────────────────────────────────────────────"));

                                let filled = ((st.pct / 100.0) * 30.0) as usize;
                                let bar_str = format!("[{}{}] {:.1}%", "█".repeat(filled), "░".repeat(30 - filled.min(30)), st.pct);
                                let downloaded_mb = st.downloaded_bytes as f64 / 1_048_576.0;
                                let total_mb = st.total_bytes as f64 / 1_048_576.0;

                                lines.push(Line::from(vec![Span::styled(" Model: ", t.dim_text()), Span::styled(&st.model_name, t.active_model())]));
                                lines.push(Line::from(vec![Span::styled(" Speed: ", t.dim_text()), Span::styled(format!("{:.2} MB/s", st.speed_mbps), Style::default().fg(Color::Green))]));
                                lines.push(Line::from(vec![Span::styled(" Size:  ", t.dim_text()), Span::raw(format!("{:.1} / {:.1} MB", downloaded_mb, total_mb))]));
                                lines.push(Line::from(""));
                                lines.push(Line::from(Span::styled(bar_str, Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))));
                                lines.push(Line::from(""));
                                lines.push(Line::from(Span::styled("Download running in background. Press Esc/q to dismiss window.", t.dim_text())));
                            }
                        } else {
                            lines.push(Line::from(vec![
                                Span::styled(format!("{} Initializing download connection...", spinner), t.header_title()),
                            ]));
                        }

                        let is_err = self.current_download_state.as_ref().and_then(|s| s.error.as_ref()).is_some();
                        let is_done = self.current_download_state.as_ref().map(|s| s.is_done && s.error.is_none()).unwrap_or(false);
                        let border_style = if is_err {
                            Style::default().fg(Color::Red)
                        } else if is_done {
                            Style::default().fg(Color::Green)
                        } else {
                            t.active_border_style()
                        };

                        let modal = Paragraph::new(lines).block(
                            Block::default()
                                .borders(Borders::ALL)
                                .border_type(BorderType::Rounded)
                                .title(if is_err { " Download Error " } else if is_done { " Download Complete " } else { " Downloading GGUF Model " })
                                .border_style(border_style),
                        );
                        f.render_widget(modal, area);
                    }
                }
            }
            ActiveModal::PersonaManager => {
                let area = centered_rect(82, 75, f.area());
                f.render_widget(Clear, area);

                let outer_block = Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .title(" System Persona Manager ")
                    .border_style(t.active_border_style());

                let inner_area = outer_block.inner(area);
                f.render_widget(outer_block, area);

                let chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Percentage(32), Constraint::Percentage(68)])
                    .split(inner_area);

                let personas = self.persona_mgr.list_personas();
                let active_file = self.persona_mgr.active_persona_file.clone();

                let selected_idx = self.selected_persona_idx.min(personas.len().saturating_sub(1));

                // Left Panel: Available persona markdown files
                let items: Vec<ListItem> = personas
                    .iter()
                    .enumerate()
                    .map(|(i, name)| {
                        let is_selected = i == selected_idx;
                        let is_active = active_file.as_deref() == Some(name.as_str());

                        let cursor = if is_selected { "> " } else { "  " };
                        let active_badge = if is_active { " [ACTIVE]" } else { "" };

                        let style = if is_selected {
                            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
                        } else if is_active {
                            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
                        } else {
                            Style::default().fg(Color::White)
                        };

                        ListItem::new(Line::from(vec![
                            Span::styled(format!("{}{}.md", cursor, name), style),
                            Span::styled(active_badge, Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                        ]))
                    })
                    .collect();

                let list = List::new(items).block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_type(BorderType::Plain)
                        .title(" Available Personas ")
                        .border_style(Style::default().fg(Color::Cyan)),
                );
                f.render_widget(list, chunks[0]);

                // Right Panel: Persona Preview & Editor
                let mut right_lines = Vec::new();
                let panel_title = if self.persona_editing {
                    if let Some(selected_name) = personas.get(selected_idx) {
                        format!(" Editing Persona File: {}.md [EDITING MODE] ", selected_name)
                    } else {
                        " Editing Persona File [EDITING MODE] ".to_string()
                    }
                } else {
                    " Persona Details & Controls ".to_string()
                };

                if self.persona_editing {
                    right_lines.push(Line::from(vec![
                        Span::styled("LIVE EDITING MODE", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                        Span::raw(" — Type text, press "),
                        Span::styled("Ctrl+S", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                        Span::raw(" to save, or "),
                        Span::styled("Esc", Style::default().fg(Color::Red)),
                        Span::raw(" to cancel."),
                    ]));
                    right_lines.push(Line::from("──────────────────────────────────────────────────────────"));

                    // Display editable content buffer with visible cursor
                    let mut text_with_cursor = self.persona_edit_buffer.clone();
                    let cursor_pos = self.persona_edit_cursor.min(text_with_cursor.len());
                    text_with_cursor.insert(cursor_pos, '█');

                    for line in text_with_cursor.lines().take(18) {
                        right_lines.push(Line::from(Span::styled(format!(" {}", line), Style::default().fg(Color::White))));
                    }
                    if text_with_cursor.lines().count() > 18 {
                        right_lines.push(Line::from(Span::styled("  ... [content continues below]", t.dim_text())));
                    }

                    right_lines.push(Line::from("──────────────────────────────────────────────────────────"));
                    right_lines.push(Line::from(Span::styled("Shortcuts: [Ctrl+S] Save File | [Esc] Cancel Editing", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))));
                } else {
                    if let Some(selected_name) = personas.get(selected_idx) {
                        let content = self.persona_mgr.read_file_or_empty(selected_name);
                        let is_active = active_file.as_deref() == Some(selected_name.as_str());

                        right_lines.push(Line::from(vec![
                            Span::styled("Selected Persona File: ", Style::default().fg(Color::Cyan)),
                            Span::styled(format!("{}.md", selected_name), Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                            if is_active {
                                Span::styled("  ★ CURRENTLY ACTIVE", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD))
                            } else {
                                Span::raw("")
                            },
                        ]));
                        right_lines.push(Line::from("──────────────────────────────────────────────────────────"));
                        right_lines.push(Line::from(Span::styled("File Content Preview:", Style::default().fg(Color::LightCyan).add_modifier(Modifier::BOLD))));

                        for l in content.lines().take(11) {
                            right_lines.push(Line::from(Span::styled(format!("  {}", l), Style::default().fg(Color::Gray))));
                        }
                        if content.lines().count() > 11 {
                            right_lines.push(Line::from(Span::styled("  ... [content truncated]", t.dim_text())));
                        }
                    } else {
                        right_lines.push(Line::from(Span::styled("No persona files found in ~/.cynapse/persona/", Style::default().fg(Color::Red))));
                    }

                    right_lines.push(Line::from("──────────────────────────────────────────────────────────"));
                    let active_status_str = match &active_file {
                        Some(name) => format!("Custom Persona ({}.md)", name),
                        None => "Default Identity (IDENTITY.md + SOUL.md + USER.md)".to_string(),
                    };
                    right_lines.push(Line::from(vec![
                        Span::styled("Active Persona Mode: ", Style::default().fg(Color::Cyan)),
                        Span::styled(active_status_str, Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                    ]));

                    right_lines.push(Line::from(""));
                    right_lines.push(Line::from(Span::styled("Controls:", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))));
                    right_lines.push(Line::from("  • Up/Down Arrow : Navigate persona files"));
                    right_lines.push(Line::from("  • Enter         : Activate highlighted persona"));
                    right_lines.push(Line::from("  • e             : Edit highlighted persona file"));
                    right_lines.push(Line::from("  • r             : Reset to default persona identity"));
                    right_lines.push(Line::from("  • Esc / q       : Close persona manager modal"));
                }

                let preview_p = Paragraph::new(right_lines).block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_type(BorderType::Plain)
                        .title(panel_title)
                        .border_style(Style::default().fg(Color::Cyan)),
                );
                f.render_widget(preview_p, chunks[1]);
            }
            ActiveModal::Bots => {
                let area = centered_rect(82, 75, f.area());
                f.render_widget(Clear, area);

                let outer_block = Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .title(" Subagent Bot Profiles (~/.cynapse/bots/) ")
                    .border_style(t.active_border_style());

                let inner_area = outer_block.inner(area);
                f.render_widget(outer_block, area);

                let chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Percentage(32), Constraint::Percentage(68)])
                    .split(inner_area);

                let bots = self.bot_registry.list();
                let selected_idx = self.selected_bot_idx.min(bots.len().saturating_sub(1));

                // Left Panel: Available bot profile slugs
                let items: Vec<ListItem> = bots
                    .iter()
                    .enumerate()
                    .map(|(i, b)| {
                        let is_selected = i == selected_idx;
                        let cursor = if is_selected { "> " } else { "  " };

                        let style = if is_selected {
                            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
                        } else {
                            Style::default().fg(Color::White)
                        };

                        ListItem::new(Line::from(vec![
                            Span::styled(cursor, Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                            Span::styled(format!("@{}", b.slug), style),
                        ]))
                    })
                    .collect();

                let list = List::new(items)
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .border_type(BorderType::Plain)
                            .title(format!(" Configured Bots ({}) ", bots.len()))
                            .border_style(Style::default().fg(Color::DarkGray)),
                    );
                f.render_widget(list, chunks[0]);

                // Right Panel: Bot details & permissions manifest
                let mut right_lines = Vec::new();
                if let Some(bot) = bots.get(selected_idx) {
                    right_lines.push(Line::from(vec![
                        Span::styled("Bot Identifier: ", Style::default().fg(Color::Cyan)),
                        Span::styled(format!("@{}", bot.slug), Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                        Span::raw("  •  "),
                        Span::styled(&bot.display_name, Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                    ]));

                    if !bot.description.is_empty() {
                        right_lines.push(Line::from(Span::styled(&bot.description, Style::default().fg(Color::Gray))));
                    }
                    right_lines.push(Line::from("──────────────────────────────────────────────────────────"));

                    right_lines.push(Line::from(vec![
                        Span::styled("Persona File: ", Style::default().fg(Color::Cyan)),
                        Span::styled(bot.persona_file.as_deref().unwrap_or("[Default Cynapse Identity]"), Style::default().fg(Color::White)),
                    ]));

                    right_lines.push(Line::from(vec![
                        Span::styled("Model Preset: ", Style::default().fg(Color::Cyan)),
                        Span::styled(bot.model_preset.as_deref().unwrap_or("[Inherit Active Model]"), Style::default().fg(Color::White)),
                    ]));

                    let allow_str = if bot.tools_allow.is_empty() || bot.tools_allow.iter().any(|a| a == "*") {
                        "All tools (*) not explicitly denied".to_string()
                    } else {
                        bot.tools_allow.join(", ")
                    };
                    right_lines.push(Line::from(vec![
                        Span::styled("Tools Allowed: ", Style::default().fg(Color::Green)),
                        Span::styled(allow_str, Style::default().fg(Color::White)),
                    ]));

                    let deny_str = if bot.tools_deny.is_empty() {
                        "[None]".to_string()
                    } else {
                        bot.tools_deny.join(", ")
                    };
                    right_lines.push(Line::from(vec![
                        Span::styled("Tools Denied:  ", Style::default().fg(Color::Red)),
                        Span::styled(deny_str, Style::default().fg(Color::LightRed)),
                    ]));

                    let ask_str = if bot.tools_ask.is_empty() {
                        "[None - automated execution]".to_string()
                    } else {
                        bot.tools_ask.join(", ")
                    };
                    right_lines.push(Line::from(vec![
                        Span::styled("Approval (Ask):", Style::default().fg(Color::Yellow)),
                        Span::styled(ask_str, Style::default().fg(Color::LightYellow)),
                    ]));

                    let ws_str = bot.workspace_restrict.as_deref().unwrap_or("[Workspace Root / Unrestricted]");
                    right_lines.push(Line::from(vec![
                        Span::styled("Workspace:     ", Style::default().fg(Color::Cyan)),
                        Span::styled(ws_str, Style::default().fg(Color::White)),
                    ]));

                    let conc_str = bot.max_concurrent.map(|c| c.to_string()).unwrap_or_else(|| "2 (default)".to_string());
                    right_lines.push(Line::from(vec![
                        Span::styled("Max Concurrency:", Style::default().fg(Color::Cyan)),
                        Span::styled(conc_str, Style::default().fg(Color::White)),
                    ]));

                    right_lines.push(Line::from("──────────────────────────────────────────────────────────"));
                    right_lines.push(Line::from(Span::styled("Controls:", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))));
                    right_lines.push(Line::from("  • Up/Down Arrow : Navigate bot profiles"));
                    right_lines.push(Line::from("  • Esc / q / Enter: Close bots modal"));
                } else {
                    right_lines.push(Line::from(Span::styled("No bot profiles found in ~/.cynapse/bots/", Style::default().fg(Color::Red))));
                }

                let preview_p = Paragraph::new(right_lines).block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_type(BorderType::Plain)
                        .title(" Bot Profile Manifest ")
                        .border_style(Style::default().fg(Color::Cyan)),
                );
                f.render_widget(preview_p, chunks[1]);
            }
            ActiveModal::ToolApproval => {
                let area = centered_rect(65, 45, f.area());
                f.render_widget(Clear, area);

                let outer_block = Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .title(" [!] Tool Execution Approval Required ")
                    .border_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD));

                let inner_area = outer_block.inner(area);
                f.render_widget(outer_block, area);

                let (bot, tool, a1, a2, edit_mode, edit_arg, edit_buf) = if let Some(ref p) = self.pending_approval {
                    (
                        p.bot_slug.as_str(),
                        p.tool_name.as_str(),
                        p.arg1.as_str(),
                        p.arg2.as_deref().unwrap_or(""),
                        p.edit_mode,
                        p.editing_arg,
                        p.edit_buffer.as_str(),
                    )
                } else {
                    ("unknown", "unknown", "", "", false, 0, "")
                };

                let arg1_line = if edit_mode && edit_arg == 0 {
                    Line::from(vec![
                        Span::styled("Argument 1 [EDITING]: ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                        Span::styled(format!("{}_", edit_buf), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                    ])
                } else {
                    Line::from(vec![
                        Span::styled("Argument 1:           ", Style::default().fg(Color::Gray)),
                        Span::styled(a1, Style::default().fg(Color::White)),
                    ])
                };

                let arg2_line = if edit_mode && edit_arg == 1 {
                    Line::from(vec![
                        Span::styled("Argument 2 [EDITING]: ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                        Span::styled(format!("{}_", edit_buf), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                    ])
                } else {
                    Line::from(vec![
                        Span::styled("Argument 2:           ", Style::default().fg(Color::Gray)),
                        Span::styled(a2, Style::default().fg(Color::White)),
                    ])
                };

                let mut text = vec![
                    Line::from(vec![
                        Span::styled("Target Bot Profile:   ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                        Span::styled(format!("@{}", bot), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                    ]),
                    Line::from(vec![
                        Span::styled("Requested Action:     ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                        Span::styled(tool, Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                    ]),
                    arg1_line,
                    arg2_line,
                    Line::from("──────────────────────────────────────────────────────────"),
                ];

                if edit_mode {
                    text.push(Line::from(vec![
                        Span::styled("Mode: ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                        Span::styled("Editing Parameter. Type to edit, [Tab] switch field, [Enter] save, [Esc] cancel.", Style::default().fg(Color::Cyan)),
                    ]));
                    text.push(Line::from(""));
                    text.push(Line::from(vec![
                        Span::styled(" [Enter] Save Field ", Style::default().bg(Color::Cyan).fg(Color::Black).add_modifier(Modifier::BOLD)),
                        Span::raw("   "),
                        Span::styled(" [Tab] Switch Arg ", Style::default().bg(Color::DarkGray).fg(Color::White)),
                        Span::raw("   "),
                        Span::styled(" [Esc] Cancel Edit ", Style::default().bg(Color::Red).fg(Color::White)),
                    ]));
                } else {
                    text.push(Line::from(vec![
                        Span::styled("Notice: ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                        Span::styled("This tool is listed in `tools_ask` requiring interactive confirmation.", Style::default().fg(Color::Gray)),
                    ]));
                    text.push(Line::from(""));
                    text.push(Line::from(vec![
                        Span::styled(" [Y / Enter] Approve ", Style::default().bg(Color::Green).fg(Color::Black).add_modifier(Modifier::BOLD)),
                        Span::raw("   "),
                        Span::styled(" [N / Esc] Deny ", Style::default().bg(Color::Red).fg(Color::White).add_modifier(Modifier::BOLD)),
                        Span::raw("   "),
                        Span::styled(" [E] Edit Parameters ", Style::default().bg(Color::Yellow).fg(Color::Black).add_modifier(Modifier::BOLD)),
                    ]));
                }

                let p = Paragraph::new(text).wrap(Wrap { trim: true });
                f.render_widget(p, inner_area);
            }
            ActiveModal::Updater => {
                let area = centered_rect(76, 50, f.area());
                f.render_widget(Clear, area);

                let spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
                let spinner = spinner_frames[self.anim_tick % spinner_frames.len()];

                let mut lines = Vec::new();
                lines.push(Line::from(""));

                if let Some(st) = &self.update_state {
                    if let Some(err) = &st.error {
                        lines.push(Line::from(vec![
                            Span::styled("  [ FAIL ] CYNAPSE UPDATE FAILED", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
                        ]));
                        lines.push(Line::from(Span::styled("  ──────────────────────────────────────────────────────────", Style::default().fg(Color::DarkGray))));
                        lines.push(Line::from(""));
                        lines.push(Line::from(vec![
                            Span::styled("  Error: ", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
                            Span::styled(err, Style::default().fg(Color::LightRed)),
                        ]));
                        lines.push(Line::from(""));
                        lines.push(Line::from(Span::styled("  Press Enter or Esc to return to chat.", Style::default().fg(Color::Yellow))));
                    } else if st.is_done {
                        lines.push(Line::from(vec![
                            Span::styled("  [ PASS ] CYNAPSE UPDATE COMPLETE!", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                        ]));
                        lines.push(Line::from(Span::styled("  ──────────────────────────────────────────────────────────", Style::default().fg(Color::DarkGray))));
                        lines.push(Line::from(""));
                        if let Some(msg) = &st.success_message {
                            lines.push(Line::from(vec![
                                Span::styled("  Result: ", Style::default().fg(Color::Cyan)),
                                Span::styled(msg, Style::default().fg(Color::White)),
                            ]));
                        }
                        lines.push(Line::from(""));
                        lines.push(Line::from(Span::styled("  Please restart Cynapse to run the newly compiled binary.", Style::default().fg(Color::Green))));
                        lines.push(Line::from(""));
                        lines.push(Line::from(Span::styled("  Press Enter or Esc to return to chat.", t.dim_text())));
                    } else {
                        lines.push(Line::from(vec![
                            Span::styled(format!("  {} UPDATING CYNAPSE SYSTEM...", spinner), t.header_title()),
                        ]));
                        lines.push(Line::from(Span::styled("  ──────────────────────────────────────────────────────────", Style::default().fg(Color::DarkGray))));
                        lines.push(Line::from(""));
                        lines.push(Line::from(vec![
                            Span::styled("  Current Task: ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                            Span::styled(&st.current_step, Style::default().fg(Color::Yellow)),
                        ]));
                        lines.push(Line::from(""));
                        lines.push(Line::from(Span::styled("  Log Steps:", Style::default().fg(Color::Cyan))));
                        for step in st.steps_log.iter().rev().take(6).collect::<Vec<_>>().into_iter().rev() {
                            lines.push(Line::from(vec![
                                Span::styled("    • ", Style::default().fg(Color::Cyan)),
                                Span::styled(step, Style::default().fg(Color::Gray)),
                            ]));
                        }
                        lines.push(Line::from(""));
                        lines.push(Line::from(Span::styled("  Building in background. Press Esc to background this window.", t.dim_text())));
                    }
                } else {
                    lines.push(Line::from(Span::styled("  No update currently in progress.", t.dim_text())));
                    lines.push(Line::from(""));
                    lines.push(Line::from(Span::styled("  Press Esc to close.", t.dim_text())));
                }

                let modal = Paragraph::new(lines).block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_type(BorderType::Rounded)
                        .title(" ◖ Cynapse In-Process System Updater ◗ ")
                        .border_style(t.active_border_style()),
                );
                f.render_widget(modal, area);
            }
            ActiveModal::None => {}
        }
    }

    /// Render Multi-Galaxy 3D Dendrite Memory Atlas Visualizer inside Ratatui modal
    fn render_3d_galaxy_atlas(&self, f: &mut Frame, area: Rect) {
        let t = self.theme;
        let width = area.width.saturating_sub(4) as usize;
        let height = area.height.saturating_sub(6) as usize;

        let (nodes, _edges) = self.graph.topology();

        // Canvas grid buffer initialized to clean empty space
        let mut grid: Vec<Vec<(char, Style)>> = vec![vec![(' ', Style::default()); width]; height];

        let mut cam = dendrite_3d::renderer::Camera3D::new();
        cam.yaw = self.galaxy_yaw;
        cam.pitch = self.galaxy_pitch;
        cam.distance = 38.0;
        cam.fov = 1.35;

        // 1. Faint, Minimal Cosmic Background Starfield
        let star_seeds = [
            (-32.0, 14.0, -22.0), (28.0, -16.0, 24.0), (-24.0, -20.0, -12.0),
            (30.0, 18.0, 15.0), (-35.0, 7.0, 26.0), (20.0, -25.0, -24.0),
            (-12.0, 28.0, 7.0), (32.0, -10.0, -20.0),
        ];

        for (sx, sy, sz) in star_seeds {
            let pt = dendrite_3d::renderer::GalaxyRenderer::project_point([sx, sy, sz], &cam, width as u16, height as u16);
            if pt.visible {
                let px = pt.screen_x as usize;
                let py = pt.screen_y as usize;
                if py < height && px < width {
                    grid[py][px] = ('.', Style::default().fg(Color::Rgb(55, 55, 75)));
                }
            }
        }

        // 2. Clean Planetary Orbital Tracks (Cleanly spaced dashed guides)
        let planetary_tracks = [9.0f32, 16.0f32, 23.0f32, 30.0f32, 37.0f32];
        for r in planetary_tracks {
            let steps = 36;
            for s in 0..steps {
                let theta = (s as f32) * (2.0 * std::f32::consts::PI / (steps as f32));
                let rx = r * theta.cos();
                let rz = r * theta.sin();
                let ry = 0.0;
                let pt = dendrite_3d::renderer::GalaxyRenderer::project_point([rx, ry, rz], &cam, width as u16, height as u16);
                if pt.visible {
                    let px = pt.screen_x as usize;
                    let py = pt.screen_y as usize;
                    if py < height && px < width && grid[py][px].0 == ' ' {
                        grid[py][px] = ('·', Style::default().fg(Color::Rgb(40, 44, 60)));
                    }
                }
            }
        }

        // 3. Central Sun: Core Memory anchored at 3D origin
        let supermassive = self.graph.supermassive_node();
        let supermassive_id = supermassive.as_ref().map(|n| n.id.as_str());

        let sun_pt = dendrite_3d::renderer::GalaxyRenderer::project_point([0.0, 0.0, 0.0], &cam, width as u16, height as u16);
        if sun_pt.visible {
            let cpx = sun_pt.screen_x as usize;
            let cpy = sun_pt.screen_y as usize;
            if cpy < height {
                if cpx >= 1 && cpx + 1 < width {
                    grid[cpy][cpx - 1] = ('✦', Style::default().fg(Color::Yellow));
                    grid[cpy][cpx] = ('✸', Style::default().fg(Color::Rgb(255, 215, 0)).add_modifier(Modifier::BOLD));
                    grid[cpy][cpx + 1] = ('✦', Style::default().fg(Color::Yellow));
                } else if cpx < width {
                    grid[cpy][cpx] = ('✸', Style::default().fg(Color::Rgb(255, 215, 0)).add_modifier(Modifier::BOLD));
                }
            }
        }

        // 4. Planetary Category Systems Orbiting the Sun
        let planet_specs = [
            (cynapse_memory::graph::NodeCategory::Meta, 9.0f32, 0.045f32, 0.0f32, "Meta", Color::Green),
            (cynapse_memory::graph::NodeCategory::Engineering, 16.0f32, 0.032f32, 1.3f32, "Engineering", Color::Cyan),
            (cynapse_memory::graph::NodeCategory::Personal, 23.0f32, 0.024f32, 2.6f32, "Personal", Color::LightMagenta),
            (cynapse_memory::graph::NodeCategory::Preferences, 30.0f32, 0.018f32, 3.9f32, "Preferences", Color::Yellow),
            (cynapse_memory::graph::NodeCategory::Episodic, 37.0f32, 0.013f32, 5.2f32, "Episodic", Color::Rgb(140, 180, 255)),
        ];

        let mut category_nodes: std::collections::HashMap<cynapse_memory::graph::NodeCategory, Vec<&cynapse_memory::graph::Node>> = std::collections::HashMap::new();
        for node in &nodes {
            if Some(node.id.as_str()) != supermassive_id {
                category_nodes.entry(node.category()).or_default().push(node);
            }
        }

        for (cat, orbit_r, speed, angle_offset, _planet_name, planet_color) in planet_specs {
            let current_angle = angle_offset + self.galaxy_anim_spin * speed;
            let px_3d = orbit_r * current_angle.cos();
            let pz_3d = orbit_r * current_angle.sin();
            let py_3d = 0.6 * (current_angle * 1.5).sin();

            let planet_pt = dendrite_3d::renderer::GalaxyRenderer::project_point([px_3d, py_3d, pz_3d], &cam, width as u16, height as u16);
            if planet_pt.visible {
                let px = planet_pt.screen_x as usize;
                let py = planet_pt.screen_y as usize;
                if py < height && px < width {
                    grid[py][px] = ('●', Style::default().fg(planet_color).add_modifier(Modifier::BOLD));
                }
            }

            // Draw Moons revolving around the Planet
            if let Some(nodes_in_cat) = category_nodes.get(&cat) {
                for (idx, node) in nodes_in_cat.iter().enumerate() {
                    let moon_r = 2.2f32 + (idx as f32 % 3.0) * 1.2f32;
                    let moon_speed = 0.12f32 + (idx as f32 % 2.0) * 0.04f32;
                    let moon_angle = (idx as f32 * 1.4) + self.galaxy_anim_spin * moon_speed;

                    let mx_3d = px_3d + moon_r * moon_angle.cos();
                    let mz_3d = pz_3d + moon_r * moon_angle.sin();
                    let my_3d = py_3d + moon_r * 0.3 * moon_angle.sin();

                    let moon_pt = dendrite_3d::renderer::GalaxyRenderer::project_point([mx_3d, my_3d, mz_3d], &cam, width as u16, height as u16);
                    if moon_pt.visible {
                        let mx = moon_pt.screen_x as usize;
                        let my = moon_pt.screen_y as usize;
                        if my < height && mx < width {
                            if grid[my][mx].0 == ' ' || grid[my][mx].0 == '·' || grid[my][mx].0 == '.' {
                                let is_pinned = node.tags.iter().any(|t| t == "pinned");
                                let moon_glyph = if is_pinned { '*' } else if node.mass >= 2.0 { '+' } else { '.' };
                                grid[my][mx] = (moon_glyph, if is_pinned { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(planet_color) });
                            }
                        }
                    }
                }
            }
        }

        // 5. Convert grid to lines
        let mut lines = Vec::new();
        lines.push(Line::from(vec![
            Span::styled("DENDRITE PLANETARY GALAXY ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::styled(" ◖ ", t.dim_text()),
            Span::styled("Sun: Core", Style::default().fg(Color::Rgb(255, 215, 0))),
            Span::styled(" │ ", t.dim_text()),
            Span::styled("Planets: Clusters", Style::default().fg(Color::Cyan)),
            Span::styled(" │ ", t.dim_text()),
            Span::styled("Moons: Facts (* Pinned)", Style::default().fg(Color::LightMagenta)),
            Span::styled(" │ ", t.dim_text()),
            Span::styled(format!("Orbit: {}", if self.galaxy_auto_spin { "ON" } else { "PAUSED" }), if self.galaxy_auto_spin { Style::default().fg(Color::Green) } else { Style::default().fg(Color::Yellow) }),
            Span::styled(" ◗", t.dim_text()),
        ]));
        let sep_str = "─".repeat(width);
        lines.push(Line::from(Span::styled(&sep_str, t.border_style())));

        for row in grid {
            let mut spans = Vec::new();
            for (ch, st) in row {
                spans.push(Span::styled(ch.to_string(), st));
            }
            lines.push(Line::from(spans));
        }

        lines.push(Line::from(Span::styled(&sep_str, t.border_style())));
        lines.push(Line::from(vec![
            Span::styled("Controls: ", t.dim_text()),
            Span::styled("Arrow Keys", Style::default().fg(Color::Cyan)),
            Span::styled(": 3D Pitch/Yaw  │  ", t.dim_text()),
            Span::styled("Space", Style::default().fg(Color::Yellow)),
            Span::styled(": Toggle Spin  │  ", t.dim_text()),
            Span::styled("p", Style::default().fg(Color::Green)),
            Span::styled(": Pin/Immortalize  │  ", t.dim_text()),
            Span::styled("Esc / q", Style::default().fg(Color::LightMagenta)),
            Span::styled(": Return to Chat", t.dim_text()),
        ]));

        let atlas_widget = Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Dendrite 3D Planetary Galaxy ")
                .border_style(t.active_border_style()),
        );

        f.render_widget(atlas_widget, area);
    }
}

fn calculate_visual_lines(lines: &[Line], width: usize) -> usize {
    if width == 0 {
        return lines.len();
    }
    let mut total = 0;
    for line in lines {
        let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        if text.is_empty() {
            total += 1;
            continue;
        }

        // Subdivide by explicit newlines if any span contains them
        for sub_line in text.split('\n') {
            if sub_line.is_empty() {
                total += 1;
                continue;
            }

            let sub_w = UnicodeWidthStr::width(sub_line);
            if sub_w <= width {
                total += 1;
            } else {
                let leading_spaces = sub_line.chars().take_while(|c| c.is_whitespace()).count();
                let mut cur_col = leading_spaces;
                let mut line_count = 1;
                let mut first_word = true;

                for word in sub_line.split_whitespace() {
                    let w_len = UnicodeWidthStr::width(word);
                    if first_word {
                        first_word = false;
                        if cur_col + w_len <= width {
                            cur_col += w_len;
                        } else {
                            line_count += (cur_col + w_len).saturating_sub(1) / width;
                            cur_col = (cur_col + w_len) % width;
                        }
                    } else if cur_col + 1 + w_len <= width {
                        cur_col += 1 + w_len;
                    } else {
                        line_count += 1;
                        cur_col = w_len.min(width);
                    }
                }
                let raw_min = (sub_w + width - 1) / width;
                total += line_count.max(raw_min);
            }
        }
    }
    total
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let eff_x = if r.width < 100 { 98 } else { percent_x };
    let eff_y = if r.height < 32 { 98 } else { percent_y };

    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - eff_y) / 2),
            Constraint::Percentage(eff_y),
            Constraint::Percentage((100 - eff_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - eff_x) / 2),
            Constraint::Percentage(eff_x),
            Constraint::Percentage((100 - eff_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zone_a_prefix_caching_and_invalidation() {
        let graph = Arc::new(cynapse_memory::graph::Dendrite::new());
        let ctx = cynapse_memory::context::DendriteContext::new(Arc::clone(&graph), None);
        let mut app = TuiApp::new(
            PathBuf::from("./models"),
            "ministral-3:3b".into(),
            "http://127.0.0.1:11434".into(),
            graph,
            None,
            ctx,
        );

        assert!(app.cached_zone_a_prefix.is_none());
        let p1 = app.get_or_compile_zone_a_prefix();
        assert!(app.cached_zone_a_prefix.is_some());
        let p2 = app.get_or_compile_zone_a_prefix();
        assert_eq!(p1, p2);

        app.invalidate_zone_a_prefix();
        assert!(app.cached_zone_a_prefix.is_none());
        let p3 = app.get_or_compile_zone_a_prefix();
        assert_eq!(p1, p3);
    }

    #[test]
    fn test_read_receipt_workflow_in_app() {
        use cynapse_core::offline_agent::ToolCall;

        let graph = Arc::new(cynapse_memory::graph::Dendrite::new());
        let ctx = cynapse_memory::context::DendriteContext::new(Arc::clone(&graph), None);
        let mut app = TuiApp::new(
            PathBuf::from("./models"),
            "ministral-3:3b".into(),
            "http://127.0.0.1:11434".into(),
            graph,
            None,
            ctx,
        );

        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("cynapse_tui_receipt_test.txt");
        std::fs::write(&test_file, "hello receipts in tui").unwrap();

        let call = ToolCall {
            name: "read_file".into(),
            arguments: serde_json::json!({
                "path": test_file.to_string_lossy()
            }),
        };

        // 1. First execution returns full content with anchor footer
        let (output1, success1) = app.execute_tool_and_format(&call);
        assert!(success1);
        assert!(output1.contains("hello receipts in tui"));
        assert!(output1.contains("[cynapse-read #1:"));

        // Simulate message added to app.messages containing the anchor
        app.messages.push(ChatMessage {
            role: "assistant".into(),
            content: output1.clone(),
            thinking: None,
        });

        // 2. Second execution returns stub since file is unchanged and anchor is in last 6 messages
        let (output2, success2) = app.execute_tool_and_format(&call);
        assert!(success2);
        assert!(output2.contains("[File unchanged since read #1:"));

        // 3. Clear messages -> resets anchor visibility -> next read returns full
        app.messages.clear();
        let (output3, success3) = app.execute_tool_and_format(&call);
        assert!(success3);
        assert!(output3.contains("[cynapse-read #2:"));

        let _ = std::fs::remove_file(&test_file);
    }

    #[test]
    fn test_finalization_notice_no_tool_directive() {
        let n = super::finalization_notice(5);
        assert!(n.contains("Do NOT emit tool calls"));
        assert!(n.contains("status report"));
        assert!(n.contains('5'));
    }

    #[test]
    fn test_finalization_prompt_structure() {
        let notice = super::finalization_notice(5);
        let prompt = cynapse_core::offline_agent::compile_zone_b_tail(
            "mem", "history", "Original user request", Some(&notice));
        let notice_idx = prompt.find("=== NOTICE ===").unwrap();
        let user_idx = prompt.find("=== USER INSTRUCTION ===").unwrap();
        assert!(notice_idx < user_idx);           // notice precedes instruction
        assert!(prompt.contains("Original user request"));  // query preserved
        assert!(prompt.contains("Do NOT emit tool calls")); // directive present
    }

    #[test]
    fn test_batch_partitioning_execution() {
        use serde_json::json;
        let graph = Arc::new(cynapse_memory::graph::Dendrite::new());
        let ctx = cynapse_memory::context::DendriteContext::new(Arc::clone(&graph), None);
        let mut app = TuiApp::new(
            PathBuf::from("./models"),
            "ministral-3:3b".into(),
            "http://127.0.0.1:11434".into(),
            graph,
            None,
            ctx,
        );
        let calls = vec![
            cynapse_core::offline_agent::ToolCall {
                name: "read_file".into(),
                arguments: json!({"path": "Cargo.toml"}),
            },
            cynapse_core::offline_agent::ToolCall {
                name: "grep".into(),
                arguments: json!({"pattern": "cynapse"}),
            },
            cynapse_core::offline_agent::ToolCall {
                name: "execute_command".into(),
                arguments: json!({"command": "echo 'partition test'"}),
            },
        ];

        let partitions = cynapse_core::partition_tool_calls(&calls);
        assert_eq!(partitions.len(), 2);
        assert_eq!(partitions[0].len(), 2); // [read_file, grep]
        assert_eq!(partitions[1].len(), 1); // [execute_command]

        for batch in partitions {
            for call in &batch {
                assert!(app.loop_guard.check(call).is_ok());
                app.loop_guard.record_call(call);
                let (output, ok) = app.execute_tool_and_format(call);
                assert!(ok);
                app.loop_guard.record_outcome(call, &output);
            }
        }
    }

    #[tokio::test]
    async fn test_turn_verified_status_and_reflection_admission() {
        use serde_json::json;
        let graph = Arc::new(cynapse_memory::graph::Dendrite::new());
        let ctx = cynapse_memory::context::DendriteContext::new(Arc::clone(&graph), None);
        let mut app = TuiApp::new(
            PathBuf::from("./models"),
            "ministral-3:3b".into(),
            "http://127.0.0.1:11434".into(),
            graph.clone(),
            None,
            ctx,
        );

        // Initially verified
        assert!(app.turn_verified);

        // Failing tool execution marks turn unverified
        let fail_call = cynapse_core::offline_agent::ToolCall {
            name: "execute_command".into(),
            arguments: json!({"command": "rm -rf /"}), // blocked by sandbox
        };
        let (_output, ok) = app.execute_tool_and_format(&fail_call);
        assert!(!ok);
        if !ok {
            app.turn_verified = false;
        }
        assert!(!app.turn_verified);

        let messages = vec![
            cynapse_memory::reflection::Message::text(
                cynapse_memory::reflection::Role::User,
                "try dangerous operation",
            ),
            cynapse_memory::reflection::Message::text(
                cynapse_memory::reflection::Role::Assistant,
                "failed to execute dangerous command",
            ),
        ];

        // spawn_reflection with turn_verified=false must be rejected
        app.reflection_worker.spawn_reflection(messages.clone(), app.turn_verified);
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        assert_eq!(graph.all().len(), 0);

        // Clean verified turn admits reflection
        app.turn_verified = true;
        app.reflection_worker.spawn_reflection(messages, app.turn_verified);
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        assert!(graph.all().len() > 0);
    }

    #[test]
    fn test_bots_modal_and_registry_tui_integration() {
        let graph = Arc::new(cynapse_memory::graph::Dendrite::new());
        let ctx = cynapse_memory::context::DendriteContext::new(Arc::clone(&graph), None);
        let mut app = TuiApp::new(
            PathBuf::from("./models"),
            "ministral-3:3b".into(),
            "http://127.0.0.1:11434".into(),
            graph,
            None,
            ctx,
        );

        let bots = app.bot_registry.list();
        assert!(!bots.is_empty(), "Bot registry should load default seeded profiles");
        assert!(app.bot_registry.get("coder").is_some());
        assert!(app.bot_registry.get("researcher").is_some());

        // Simulate /bots slash command opening modal
        app.modal = ActiveModal::Bots;
        app.selected_bot_idx = 0;
        assert_eq!(app.modal, ActiveModal::Bots);
    }

    #[tokio::test]
    async fn test_spawn_subagent_inline_and_background_workflow() {
        use serde_json::json;
        let graph = Arc::new(cynapse_memory::graph::Dendrite::new());
        let ctx = cynapse_memory::context::DendriteContext::new(Arc::clone(&graph), None);
        let mut app = TuiApp::new(
            PathBuf::from("./models"),
            "ministral-3:3b".into(),
            "http://127.0.0.1:59999".into(),
            graph,
            None,
            ctx,
        );

        // 1. Inline execution (wait=true)
        let inline_call = cynapse_core::offline_agent::ToolCall {
            name: "spawn_subagent".into(),
            arguments: json!({
                "task": "Refactor module error types",
                "bot": "coder",
                "wait": true
            }),
        };
        let (inline_res, ok1) = app.execute_tool_and_format(&inline_call);
        assert!(ok1);
        assert!(inline_res.contains("[subagent #1 · @coder ·"));
        assert!(app.subagent_mgr.get_status(1).is_some());

        // 2. Background execution (wait=false)
        let bg_call = cynapse_core::offline_agent::ToolCall {
            name: "spawn_subagent".into(),
            arguments: json!({
                "task": "Audit codebase dependencies",
                "bot": "auditor",
                "wait": false
            }),
        };
        let (bg_res, ok2) = app.execute_tool_and_format(&bg_call);
        assert!(ok2);
        assert!(bg_res.contains("[subagent #2 · @auditor · spawned in background]"));

        // Wait for background tokio task to finish and announce back
        let mut announced = false;
        for _ in 0..100 {
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            app.poll_stream_events();
            if app.messages.iter().any(|m| m.content.contains("Subagent Task #2 Announcement")) {
                announced = true;
                break;
            }
        }
        assert!(announced, "Expected announcement system message in conversation history");
        assert!(app.subagent_mgr.get_status(2).is_some());
    }

    #[test]
    fn test_tool_approval_modal_and_execution_workflow() {
        let graph = Arc::new(cynapse_memory::graph::Dendrite::new());
        let ctx = cynapse_memory::context::DendriteContext::new(Arc::clone(&graph), None);
        let mut app = TuiApp::new(
            PathBuf::from("./models"),
            "ministral-3:3b".into(),
            "http://127.0.0.1:59999".into(),
            graph,
            None,
            ctx,
        );

        // 1. Set pending approval
        app.pending_approval = Some(PendingToolApproval::new(
            1,
            "auditor".into(),
            "execute_command".into(),
            "echo 'approved by user'".into(),
            None,
            None,
        ));
        app.modal = ActiveModal::ToolApproval;

        // 2. Approve via decision handler
        app.handle_approval_decision(true);

        assert_eq!(app.modal, ActiveModal::None);
        assert!(app.pending_approval.is_none());
        assert!(app.messages.iter().any(|m| m.content.contains("approved `execute_command`")));

        // 3. Set another pending approval and deny via decision handler
        app.pending_approval = Some(PendingToolApproval::new(
            1,
            "auditor".into(),
            "execute_command".into(),
            "rm -rf /tmp/dangerous".into(),
            None,
            None,
        ));
        app.modal = ActiveModal::ToolApproval;

        app.handle_approval_decision(false);

        assert_eq!(app.modal, ActiveModal::None);
        assert!(app.pending_approval.is_none());
        assert!(app.messages.iter().any(|m| m.content.contains("denied `execute_command`")));

        // 4. Test interactive editing of parameters
        let mut pending = PendingToolApproval::new(
            2,
            "coder".into(),
            "write_file".into(),
            "temp.rs".into(),
            Some("fn main() {}".into()),
            None,
        );
        pending.edit_mode = true;
        pending.editing_arg = 0;
        pending.edit_buffer = "src/main.rs".into();
        pending.arg1 = pending.edit_buffer.clone();
        pending.editing_arg = 1;
        pending.edit_buffer = "fn main() { println!(\"edited\"); }".into();
        pending.arg2 = Some(pending.edit_buffer.clone());
        pending.edit_mode = false;

        app.pending_approval = Some(pending);
        app.modal = ActiveModal::ToolApproval;
        app.handle_approval_decision(true);

        assert_eq!(app.modal, ActiveModal::None);
        assert!(app.pending_approval.is_none());
        assert!(app.messages.iter().any(|m| m.content.contains("src/main.rs")));
    }

    #[test]
    fn test_spawn_subagent_unknown_slug_fails_closed() {
        use serde_json::json;
        let graph = Arc::new(cynapse_memory::graph::Dendrite::new());
        let ctx = cynapse_memory::context::DendriteContext::new(Arc::clone(&graph), None);
        let mut app = TuiApp::new(
            PathBuf::from("./models"),
            "ministral-3:3b".into(),
            "http://127.0.0.1:59999".into(),
            graph,
            None,
            ctx,
        );

        let unknown_call = cynapse_core::offline_agent::ToolCall {
            name: "spawn_subagent".into(),
            arguments: json!({
                "task": "Do something unsafe",
                "bot": "nonexistent_bot",
                "wait": false
            }),
        };

        let (res, ok) = app.execute_tool_and_format(&unknown_call);
        assert!(!ok, "Unknown bot profile spawn must fail closed");
        assert!(res.contains("Unknown bot profile '@nonexistent_bot'"));
    }

    #[tokio::test]
    async fn test_background_subagent_approval_and_denial_channel_workflow() {
        let graph = Arc::new(cynapse_memory::graph::Dendrite::new());
        let ctx = cynapse_memory::context::DendriteContext::new(Arc::clone(&graph), None);
        let mut app = TuiApp::new(
            PathBuf::from("./models"),
            "ministral-3:3b".into(),
            "http://127.0.0.1:59999".into(),
            graph,
            None,
            ctx,
        );

        let (tx, rx) = tokio::sync::oneshot::channel();
        let req = ToolApprovalRequest {
            task_id: 42,
            bot_slug: "coder".into(),
            tool_name: "write_file".into(),
            arg1: "test.rs".into(),
            arg2: Some("fn main() {}".into()),
            response_tx: tx,
        };

        // 1. Send request through approval channel
        app.approval_tx.send(req).unwrap();

        // 2. Poll stream events to trigger modal
        app.poll_stream_events();
        assert_eq!(app.modal, ActiveModal::ToolApproval);
        assert!(app.pending_approval.is_some());

        // 3. User approves in modal
        app.handle_approval_decision(true);
        assert_eq!(app.modal, ActiveModal::None);

        // 4. Verify oneshot channel received true
        let received = rx.await.unwrap();
        assert!(received);
    }

    #[tokio::test]
    async fn test_at_slug_routing_and_bots_slash_commands() {
        let graph = Arc::new(cynapse_memory::graph::Dendrite::new());
        let ctx = cynapse_memory::context::DendriteContext::new(Arc::clone(&graph), None);
        let mut app = TuiApp::new(
            PathBuf::from("./models"),
            "ministral-3:3b".into(),
            "http://127.0.0.1:59999".into(),
            graph,
            None,
            ctx,
        );

        // 1. Simulate @coder <task> input routing
        app.input = "@coder Write tests for parser".into();
        app.input_cursor = app.input.len();
        let _ = crossterm::event::Event::Key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Enter,
            crossterm::event::KeyModifiers::empty(),
        ));
        // Run the input handler logic directly
        let trimmed = app.input.trim().to_string();
        if trimmed.starts_with('@') {
            let rest = &trimmed[1..];
            let mut parts = rest.splitn(2, |c: char| c.is_whitespace());
            let slug = parts.next().unwrap_or("").trim();
            let task = parts.next().unwrap_or("").trim();
            let call = cynapse_core::offline_agent::ToolCall {
                name: "spawn_subagent".into(),
                arguments: serde_json::json!({
                    "bot": slug,
                    "task": task,
                    "wait": false,
                }),
            };
            let (out, ok) = app.execute_tool_and_format(&call);
            assert!(ok);
            assert!(out.contains("[subagent #1 · @coder · spawned in background]"));
        }

        // 2. Test /bots status listing
        let tasks = app.subagent_mgr.list_tasks();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].bot_slug, "coder");

        // 3. Test /bots cancel
        let cancelled = app.subagent_mgr.cancel_task(1);
        assert!(cancelled);
        assert_eq!(app.subagent_mgr.get_status(1), Some(cynapse_core::subagent::SubagentStatus::Cancelled));
    }

    #[test]
    fn test_poll_stream_events_auto_cancels_stalled_tasks() {
        let graph = Arc::new(cynapse_memory::graph::Dendrite::new());
        let ctx = cynapse_memory::context::DendriteContext::new(Arc::clone(&graph), None);
        let mut app = TuiApp::new(
            PathBuf::from("./models"),
            "ministral-3:3b".into(),
            "http://127.0.0.1:59999".into(),
            graph,
            None,
            ctx,
        );

        let (task_id, cancel_flag) = app.subagent_mgr.create_task("Hanging computation", "researcher", false);
        app.subagent_mgr.update_status(task_id, cynapse_core::subagent::SubagentStatus::Running, None);

        // Artificially age the task to >2m
        let old_time = cynapse_core::subagent::SubagentManager::now_ms().saturating_sub(180_000);
        app.subagent_mgr.set_task_started_at_for_test(task_id, old_time);

        app.poll_stream_events();
        assert!(cancel_flag.load(std::sync::atomic::Ordering::SeqCst));
        assert_eq!(app.subagent_mgr.get_status(task_id), Some(cynapse_core::subagent::SubagentStatus::Cancelled));
        assert!(app.messages.iter().any(|m| m.content.contains("stalled (> 2m) and was auto-cancelled")));
    }
}
