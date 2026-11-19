use std::path::PathBuf;
use anyhow::Result;
use clap::{Parser, Subcommand};
use cynapse_core::pull_huggingface_model;
use cynapse_engine::route_model;
use cynapse_tui::TuiSession;

#[derive(Parser)]
#[command(name = "cynapse")]
#[command(about = "Pure Rust AI Agent System: 3-tier LLM engine, REPL TUI, atomic-agent local tools, and Dendrite graph memory.")]
struct Cli {
    /// Force full visual Ratatui TUI mode
    #[arg(long)]
    tui: bool,

    /// Force simple line-by-line CLI text mode
    #[arg(long)]
    cli: bool,

    /// Resume a past conversation session by ID
    #[arg(long)]
    resume: Option<String>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// List downloaded models in models directory
    List,
    /// Select and run a model by number or name
    Run {
        target: Option<String>,
    },
    /// Route model execution through Semantic Hardware Tier Router
    Route {
        model_path: Option<PathBuf>,
    },
    /// Pull model from HuggingFace
    Pull {
        url_or_repo: String,
    },
    /// Render visual 4-tier Dendrite memory overview & graph topology
    Memory,
    /// Run self-healing Cynapse Doctor system diagnostic & recovery
    Doctor {
        /// Enable automatic self-healing repair mode
        #[arg(long)]
        fix: bool,
    },
    /// Clean stale disk spillovers, reproduction scripts, and logs older than N days
    Clean {
        /// Max age in days before pruning files (default: 14)
        #[arg(long, default_value_t = 14)]
        days: u64,
    },
    /// Run standardized agent benchmark suite against local engine
    Bench {
        /// Target model to benchmark
        #[arg(long)]
        model: Option<String>,
    },
    /// Revert the latest file change or patch performed by the agent
    Undo,
}

/// Resolve models directory dynamically relative to workspace first, or user home folder.
fn resolve_models_dir() -> PathBuf {
    let local = PathBuf::from("./models");
    if local.is_dir() {
        return local;
    }
    if let Ok(cwd) = std::env::current_dir() {
        let cwd_models = cwd.join("models");
        if cwd_models.is_dir() {
            return cwd_models;
        }
    }
    if let Some(home) = dirs::home_dir() {
        let user_models = home.join(".cynapse").join("models");
        if user_models.is_dir() {
            return user_models;
        }
    }
    let fallback = PathBuf::from("models");
    let _ = std::fs::create_dir_all(&fallback);
    fallback
}

#[tokio::main]
async fn main() -> Result<()> {
    cynapse_engine::pin_rayon_to_fast_cores();
    let cli = Cli::parse();
    let models_dir = resolve_models_dir();
    std::fs::create_dir_all(&models_dir)?;

    let mut session = TuiSession::new(models_dir.clone());

    match cli.command {
        Some(Commands::List) => {
            session.list_models();
        }
        Some(Commands::Route { model_path }) => {
            let target = if let Some(ref p) = model_path {
                let p_str = p.to_string_lossy();
                if let Some((_, resolved_path)) = session.resolve_model_by_name_or_index(&p_str) {
                    resolved_path
                } else {
                    p.clone()
                }
            } else if let Some((_, resolved_path)) = session.resolve_model_by_name_or_index(&session.active_model_name) {
                resolved_path
            } else {
                models_dir.join("model.gguf")
            };
            let decision = route_model(&target, false);
            println!("======================================================================");
            println!("             CYNAPSE SEMANTIC HARDWARE & MODEL ROUTER              ");
            println!("======================================================================");
            println!("Target Model Path:  {}", target.display());
            println!("Model Disk Size:    {:.2} MB", decision.model_size_mb);
            println!("Available Host RAM: {} MB", decision.ram_available_mb);
            println!("Memory Needed:      {:.2} MB (Model + Dynamic Headroom Reserve)", decision.ram_needed_mb);
            println!("----------------------------------------------------------------------");
            println!("SELECTED ENGINE TIER: {}", decision.tier.label());
            println!("======================================================================");
        }
        Some(Commands::Pull { url_or_repo }) => {
            let target_path = pull_huggingface_model(&url_or_repo, &models_dir).await?;
            println!("✓ Model successfully downloaded to: {}", target_path.display());
        }
        Some(Commands::Memory) => {
            let db_path = if PathBuf::from("data/dendrite.db").exists() {
                PathBuf::from("data/dendrite.db")
            } else if let Some(home) = dirs::home_dir() {
                let p = home.join(".cynapse").join("dendrite.db");
                if p.exists() { p } else { PathBuf::from("data/dendrite.db") }
            } else {
                PathBuf::from("data/dendrite.db")
            };
            if db_path.exists() {
                if let Ok(store) = cynapse_memory::store::DendriteStore::open(&db_path) {
                    let _ = store.load_all(&session.graph);
                }
            }
            cynapse_tui::memory_render::render_dendrite_visualizer(&session.graph);
        }
        Some(Commands::Doctor { fix }) => {
            let db_path = if PathBuf::from("data/dendrite.db").exists() {
                PathBuf::from("data/dendrite.db")
            } else if let Some(home) = dirs::home_dir() {
                let p = home.join(".cynapse").join("dendrite.db");
                if p.exists() { p } else { PathBuf::from("data/dendrite.db") }
            } else {
                PathBuf::from("data/dendrite.db")
            };
            let doctor = cynapse_core::doctor::CynapseDoctor::new(models_dir.clone(), db_path.clone(), fix);
            let report = doctor.run_diagnostics();

            println!("======================================================================");
            println!("           CYNAPSE AGENT SYSTEM SELF-HEALING DOCTOR                ");
            println!("======================================================================");
            println!("Overall System Health Score: [ {}% ]", report.health_score);
            println!("Summary: {} Pass | {} Warning | {} Repaired | {} Failed", report.total_pass, report.total_warn, report.total_repaired, report.total_fail);
            println!("----------------------------------------------------------------------");
            for item in &report.items {
                println!("{}{} [{:<14}] {:<35} - {}", item.status.color_code(), item.status.badge(), item.subsystem, item.check_name, item.detail);
                if let Some(fix) = &item.fix_recommendation {
                    println!("    └─> Fix Recommendation: {}", fix);
                }
                print!("\x1b[0m"); // reset color
            }
            println!("======================================================================");
            if report.total_fail > 0 || report.total_warn > 0 {
                if !fix && std::io::IsTerminal::is_terminal(&std::io::stdin()) {
                    print!("\nInteractive Remediation: [R] Auto-remediate now | [I] Ignore | [D] View details? (r/i/d): ");
                    use std::io::Write;
                    let _ = std::io::stdout().flush();
                    let mut input = String::new();
                    if std::io::stdin().read_line(&mut input).is_ok() {
                        let choice = input.trim().to_lowercase();
                        if choice == "r" {
                            println!("\n[+] Launching automatic self-healing remediation...");
                            let fix_doctor = cynapse_core::doctor::CynapseDoctor::new(models_dir.clone(), db_path.clone(), true);
                            let fix_report = fix_doctor.run_diagnostics();
                            println!("Remediation complete. New Health Score: [ {}% ] (Repaired: {})", fix_report.health_score, fix_report.total_repaired);
                            if fix_report.total_fail == 0 {
                                return Ok(());
                            }
                        } else if choice == "d" {
                            println!("\n=== Diagnostic Deep Dive ===");
                            for item in &report.items {
                                if item.status != cynapse_core::doctor::DoctorStatus::Pass {
                                    println!("Subsystem: {}", item.subsystem);
                                    println!("Check: {}", item.check_name);
                                    println!("Status: {:?}", item.status);
                                    println!("Detail: {}", item.detail);
                                    if let Some(ref rec) = item.fix_recommendation {
                                        println!("Remediation Action: {}", rec);
                                    }
                                    println!("---");
                                }
                            }
                        }
                    }
                } else if !fix {
                    println!("Tip: Run 'cynapse doctor --fix' to execute automatic self-healing repairs.");
                }
            }
            if report.total_fail > 0 && !fix {
                std::process::exit(1);
            }
        }
        Some(Commands::Clean { days }) => {
            println!("Running Cynapse Workspace & Log Cleanup (Pruning files older than {} days)...", days);
            let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
            let cynapse_dir = home.join(".cynapse");
            let mut cleaned_files = 0;
            let mut freed_bytes: u64 = 0;
            let cutoff_secs = days * 86400;
            let now = std::time::SystemTime::now();

            let target_subdirs = ["spill", "repro", "logs"];
            for sub in target_subdirs {
                let dir = cynapse_dir.join(sub);
                if dir.exists() {
                    if let Ok(entries) = std::fs::read_dir(&dir) {
                        for entry in entries.flatten() {
                            let path = entry.path();
                            if path.is_file() {
                                if let Ok(meta) = path.metadata() {
                                    if let Ok(mtime) = meta.modified() {
                                        if let Ok(elapsed) = now.duration_since(mtime) {
                                            if elapsed.as_secs() >= cutoff_secs {
                                                let size = meta.len();
                                                if std::fs::remove_file(&path).is_ok() {
                                                    cleaned_files += 1;
                                                    freed_bytes += size;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            println!("✓ Cleanup complete: {} files pruned, {:.2} MB storage recovered.", cleaned_files, freed_bytes as f64 / (1024.0 * 1024.0));
        }
        Some(Commands::Bench { model }) => {
            let hw = cynapse_engine::probe_hardware_info();
            let pwr = cynapse_core::power::PowerThrottler::new().detect_power_source();
            let target_model = model.unwrap_or_else(|| {
                if !session.active_model_name.is_empty() && !session.active_model_name.starts_with('(') {
                    session.active_model_name.clone()
                } else {
                    "qwen2.5-0.5b-instruct-q4_k_m.gguf".to_string()
                }
            });
            let model_path = models_dir.join(&target_model);
            let tier = route_model(&model_path, false).tier;
            let start = std::time::Instant::now();
            let mut ttft_ms: Option<u128> = None;
            let mut token_pieces = 0usize;

            println!("Running live inference benchmark on '{}' ({})...", target_model, tier.label());
            let bench_res = cynapse_engine::query_model_stream(
                tier,
                &session.tier1_endpoint,
                &target_model,
                "Reply with a 25-word summary of local-first AI inference.",
                "Be concise and direct.",
                |_, piece| {
                    if !piece.is_empty() && ttft_ms.is_none() {
                        ttft_ms = Some(start.elapsed().as_millis());
                    }
                    token_pieces += 1;
                },
            ).await;

            match bench_res {
                Ok(stats) => {
                    let report = format!(
                        "======================================================================\n\
                                   CYNAPSE LIVE HARDWARE BENCHMARK REPORT\n\
                        ======================================================================\n\
                        Model              : {}\n\
                        Engine Tier        : {}\n\
                        CPU Cores          : {} cores\n\
                        Power Source       : {:?}\n\
                        TTFT               : {} ms\n\
                        Tokens Generated   : {} (streamed chunks: {})\n\
                        Elapsed Time       : {:.2} s\n\
                        Measured Speed     : {:.2} tok/s\n\
                        Available Host RAM : {:.2} GiB\n\
                        ======================================================================",
                        stats.model_name, tier.label(), hw.cpu_cores, pwr,
                        ttft_ms.unwrap_or(0), stats.tokens_generated, token_pieces,
                        stats.elapsed_sec, stats.tok_per_sec, stats.avail_ram_gb
                    );
                    println!("{}", report);
                    if let Some(home) = dirs::home_dir() {
                        let out = home.join(".cynapse").join("bench_report.txt");
                        if let Some(p) = out.parent() { let _ = std::fs::create_dir_all(p); }
                        if std::fs::write(&out, &report).is_ok() {
                            println!("Saved report: {}", out.display());
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Benchmark failed: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Some(Commands::Undo) => {
            println!("↩ Invoking Cynapse Agent File Undo Journal...");
            match cynapse_core::undo::UndoJournal::undo_latest() {
                Ok(msg) => println!("✓ {}", msg),
                Err(err) => {
                    eprintln!("✗ Undo failed: {}", err);
                    std::process::exit(1);
                }
            }
        }
        Some(Commands::Run { target }) => {
            if let Some(t) = target {
                if let Some((resolved_name, resolved_path)) = session.resolve_model_by_name_or_index(&t) {
                    session.active_model_name = resolved_name;
                    session.active_model_path = resolved_path;
                } else {
                    session.active_model_name = t;
                }
            }
            if cli.cli {
                session.run_cli_loop_with_resume(cli.resume.as_deref()).await?;
            } else {
                session.run_tui_app_with_resume(cli.resume.as_deref()).await?;
            }
        }
        None => {
            if cli.cli {
                session.run_cli_loop_with_resume(cli.resume.as_deref()).await?;
            } else {
                session.run_tui_app_with_resume(cli.resume.as_deref()).await?;
            }
        }
    }

    Ok(())
}
