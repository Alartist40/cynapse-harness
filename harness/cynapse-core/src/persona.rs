//! Persona Management System for Cynapse
//!
//! Manages markdown persona files (IDENTITY.md, SOUL.md, USER.md, SYSTEM.md)
//! in `~/.cynapse/persona/` or custom directories, dynamically constructing
//! non-generic, high-character system prompts for LLM queries.

use anyhow::Result;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct PersonaManager {
    pub persona_dir: PathBuf,
    pub active_persona_file: Option<String>,
}

impl PersonaManager {
    /// Initialize PersonaManager with target directory (defaults to ~/.cynapse/persona).
    pub fn new<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let persona_dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&persona_dir)?;

        let mgr = Self {
            persona_dir,
            active_persona_file: None,
        };

        mgr.init_defaults()?;
        Ok(mgr)
    }

    /// Default constructor pointing to ~/.cynapse/persona/
    pub fn default_dir() -> PathBuf {
        if let Ok(home) = std::env::var("HOME") {
            PathBuf::from(home).join(".cynapse").join("persona")
        } else {
            PathBuf::from("./persona")
        }
    }

    /// In-memory fallback constructor when filesystem persona directories cannot be created.
    pub fn in_memory() -> Self {
        Self {
            persona_dir: PathBuf::from("./persona"),
            active_persona_file: None,
        }
    }

    /// Seed default persona files if they do not exist.
    pub fn init_defaults(&self) -> Result<()> {
        let identity_path = self.persona_dir.join("IDENTITY.md");
        if !identity_path.exists() {
            let default_identity = r#"# Cynapse Core Identity
Name: Cynapse
Role: Autonomous Local-First Intelligent Companion
Tagline: Fast, private, in-process intelligence without dependencies.
"#;
            fs::write(&identity_path, default_identity)?;
        }

        let soul_path = self.persona_dir.join("SOUL.md");
        if !soul_path.exists() {
            let default_soul = r#"# Soul & Behavior Protocol
1. Lead with the direct answer or action immediately. No greetings, preamble, or "Great question!" openers.
2. Be concise, sharp, noble, and direct.
3. Use bullet points or numbered lists capped at 5 items for multi-step tasks.
4. Maintain a distinct, dignified personality — helpful, industrial, and loyal.
5. Format code blocks and technical responses with clean GitHub markdown.
"#;
            fs::write(&soul_path, default_soul)?;
        }

        let user_path = self.persona_dir.join("USER.md");
        if !user_path.exists() {
            let default_user = r#"# User Profile Context
Target: Local Power User / Developer
Mode: High Efficiency & Direct Execution
"#;
            fs::write(&user_path, default_user)?;
        }

        let memory_path = self.persona_dir.join("MEMORY.md");
        if !memory_path.exists() {
            let default_memory = r#"# Active Working Memory
- Mode: Direct local execution
- Knowledge Store: Dendrite SQLite FTS5 graph database
"#;
            fs::write(&memory_path, default_memory)?;
        }

        // Distilled specialist personas (adapted from agency-agents, compact for SLMs)
        let specialists = [
            (
                "SECURITY_AUDITOR.md",
                "# Security Auditor\n\
                 Role: Local-first defensive security & OWASP auditor.\n\
                 Rules:\n\
                 1. Check trust boundaries, input validation, and fail-closed error paths.\n\
                 2. Never recommend destructive commands; prefer read-only diagnostics first.\n\
                 3. Cite exact file:line evidence for every finding.\n",
            ),
            (
                "RUST_ENGINEER.md",
                "# Rust Systems Engineer\n\
                 Role: Lean pure-Rust systems architect.\n\
                 Rules:\n\
                 1. Keep modules small (<=100 lines where practical), zero bloat, zero unwrap() in prod.\n\
                 2. Reuse existing workspace crates before adding dependencies.\n\
                 3. Verify every change with cargo test.\n",
            ),
            (
                "GLIA.md",
                "# Glia Cybersecurity Specialist\n\
                 Role: Concise local-first defensive cybersecurity & hardware-rescue specialist.\n\
                 Rules:\n\
                 1. Lead with the direct diagnosis or safe command. Never prefix replies with 'Cynapse:' or 'Glia:'.\n\
                 2. Prefer non-destructive evidence preservation (ps, lsof, ss, sha256sum, ufw/nft) first.\n\
                 3. Refuse destructive commands and provide safe containment in <=5 crisp steps.\n",
            ),
            (
                "REALITY_CHECKER.md",
                "# Reality Checker\n\
                 Role: Adversarial QA & evidence verifier.\n\
                 Rules:\n\
                 1. Never accept 'should work' — require observed command output and exit codes.\n\
                 2. Check edge cases (empty inputs, OOM/limits, UTF-8 boundaries).\n\
                 3. Report exact scope and any unverified gaps.\n",
            ),
        ];
        for (filename, body) in specialists {
            let path = self.persona_dir.join(filename);
            if !path.exists() {
                fs::write(&path, body)?;
            }
        }

        Ok(())
    }

    /// Read persona file content or empty string if absent.
    pub fn read_file_or_empty(&self, name: &str) -> String {
        let path = if name.ends_with(".md") {
            self.persona_dir.join(name)
        } else {
            self.persona_dir.join(format!("{}.md", name))
        };

        fs::read_to_string(path).unwrap_or_default()
    }

    /// Save or overwrite a persona file.
    pub fn write_file(&self, name: &str, content: &str) -> Result<()> {
        let file_name = if name.ends_with(".md") {
            name.to_string()
        } else {
            format!("{}.md", name)
        };
        let file_path = self.persona_dir.join(file_name);
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(file_path, content)?;
        Ok(())
    }

    /// List all available `.md` persona files in the persona directory.
    pub fn list_personas(&self) -> Vec<String> {
        let mut list = Vec::new();
        if let Ok(entries) = fs::read_dir(&self.persona_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("md") {
                    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                        list.push(stem.to_string());
                    }
                }
            }
        }
        list.sort();
        list
    }

    /// Set active persona file override (e.g., "paraclea" or "custom").
    pub fn set_active_persona(&mut self, name: Option<String>) {
        self.active_persona_file = name;
    }

    /// Build dynamic comprehensive system prompt combining all persona files.
    pub fn build_system_prompt(&self) -> String {
        if let Some(ref active) = self.active_persona_file {
            let custom_content = self.read_file_or_empty(active);
            if !custom_content.trim().is_empty() {
                return custom_content;
            }
        }

        let identity = self.read_file_or_empty("IDENTITY.md");
        let soul = self.read_file_or_empty("SOUL.md");
        let user = self.read_file_or_empty("USER.md");
        let memory = self.read_file_or_empty("MEMORY.md");

        let mut parts = Vec::new();
        if !identity.trim().is_empty() {
            parts.push(format!("=== IDENTITY ===\n{}", identity.trim()));
        }
        if !soul.trim().is_empty() {
            parts.push(format!("=== SOUL & BEHAVIOR PROTOCOL ===\n{}", soul.trim()));
        }
        if !user.trim().is_empty() {
            parts.push(format!("=== USER PROFILE ===\n{}", user.trim()));
        }
        if !memory.trim().is_empty() {
            parts.push(format!("=== ACTIVE WORKING MEMORY ===\n{}", memory.trim()));
        }

        if parts.is_empty() {
            "You are Cynapse — a fast, private, local-first AI companion. Be direct and concise.".to_string()
        } else {
            parts.join("\n\n")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_persona_manager_defaults_and_prompt_build() {
        let temp_dir = TempDir::new().unwrap();
        let mgr = PersonaManager::new(temp_dir.path()).unwrap();

        let personas = mgr.list_personas();
        assert!(personas.contains(&"IDENTITY".to_string()));
        assert!(personas.contains(&"SOUL".to_string()));
        assert!(personas.contains(&"USER".to_string()));
        assert!(personas.contains(&"MEMORY".to_string()));

        let prompt = mgr.build_system_prompt();
        assert!(prompt.contains("Cynapse Core Identity"));
        assert!(prompt.contains("Soul & Behavior Protocol"));
        assert!(prompt.contains("Active Working Memory"));
    }

    #[test]
    fn test_custom_persona_override() {
        let temp_dir = TempDir::new().unwrap();
        let mut mgr = PersonaManager::new(temp_dir.path()).unwrap();

        mgr.write_file("custom_helper", "You are a custom assistant.").unwrap();
        mgr.set_active_persona(Some("custom_helper".to_string()));

        let prompt = mgr.build_system_prompt();
        assert_eq!(prompt, "You are a custom assistant.");
    }
}
