//! Bot Profiles & Registry for Cynapse Agency Capabilities
//!
//! Provides first-class named bot configurations, permissions scoping (allow/deny/ask),
//! workspace confinement, and persistent TOML profile management in `~/.cynapse/bots/`.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Configuration and permission specification for an autonomous bot profile.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BotProfile {
    /// Unique lowercase slug identifier (e.g. "coder", "researcher").
    pub slug: String,
    /// Human-friendly display title.
    pub display_name: String,
    /// Optional summary of role and responsibilities.
    #[serde(default)]
    pub description: String,
    /// Optional specific persona markdown file (relative to persona directory).
    #[serde(default)]
    pub persona_file: Option<String>,
    /// Optional specific model preset name or endpoint.
    #[serde(default)]
    pub model_preset: Option<String>,
    /// Allowed tool names. Empty or `["*"]` grants all tools not denied.
    #[serde(default)]
    pub tools_allow: Vec<String>,
    /// Denied tool names. Deny strictly overrides allow.
    #[serde(default)]
    pub tools_deny: Vec<String>,
    /// Tools that require interactive user approval before execution.
    #[serde(default)]
    pub tools_ask: Vec<String>,
    /// Optional directory path prefix confining file system operations.
    #[serde(default)]
    pub workspace_restrict: Option<String>,
    /// Maximum concurrent background subagents for this profile.
    #[serde(default)]
    pub max_concurrent: Option<usize>,
}

impl Default for BotProfile {
    fn default() -> Self {
        Self {
            slug: "default".into(),
            display_name: "Default Assistant".into(),
            description: "General-purpose autonomous companion".into(),
            persona_file: None,
            model_preset: None,
            tools_allow: vec!["*".into()],
            tools_deny: Vec::new(),
            tools_ask: Vec::new(),
            workspace_restrict: None,
            max_concurrent: Some(2),
        }
    }
}

impl BotProfile {
    /// Evaluates if a given tool is permitted by this bot profile.
    ///
    /// Precedence rule: `tools_deny` > `tools_allow`.
    /// If denied: returns false.
    /// If allow list is empty or contains "*": returns true.
    /// If allow list contains tool: returns true.
    /// Otherwise: returns false (fail-closed).
    pub fn is_tool_allowed(&self, tool_name: &str) -> bool {
        if self.tools_deny.iter().any(|d| d == "*" || d == tool_name) {
            return false;
        }

        if self.tools_allow.is_empty() || self.tools_allow.iter().any(|a| a == "*") {
            return true;
        }

        self.tools_allow.iter().any(|a| a == tool_name)
    }

    /// Evaluates if a given tool requires user approval before execution.
    pub fn requires_approval(&self, tool_name: &str) -> bool {
        self.tools_ask.iter().any(|a| a == "*" || a == tool_name)
    }

    /// Evaluates whether a target path is confined within `workspace_restrict` if configured.
    ///
    /// If `workspace_restrict` is None, returns true.
    /// Resolves canonical/absolute representations before prefix matching.
    pub fn is_path_in_workspace<P: AsRef<Path>>(&self, target_path: P) -> bool {
        let restriction = match &self.workspace_restrict {
            Some(w) => w,
            None => return true,
        };

        let raw_restrict = if restriction.starts_with('~') {
            if let Some(home) = dirs::home_dir() {
                home.join(restriction.trim_start_matches('~').trim_start_matches('/'))
            } else {
                PathBuf::from(restriction)
            }
        } else {
            PathBuf::from(restriction)
        };

        let norm_restrict = if raw_restrict.is_absolute() {
            raw_restrict
        } else {
            std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")).join(raw_restrict)
        };
        let canonical_restrict = norm_restrict.canonicalize().unwrap_or(norm_restrict);

        let target = target_path.as_ref();
        let raw_target = if target.to_string_lossy().starts_with('~') {
            if let Some(home) = dirs::home_dir() {
                home.join(target.to_string_lossy().trim_start_matches('~').trim_start_matches('/'))
            } else {
                target.to_path_buf()
            }
        } else {
            target.to_path_buf()
        };

        let norm_target = if raw_target.is_absolute() {
            raw_target
        } else {
            std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")).join(raw_target)
        };
        let canonical_target = norm_target.canonicalize().unwrap_or(norm_target);

        canonical_target.starts_with(&canonical_restrict)
    }

    /// Validates profile syntax, slug format, and tool references.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        if self.slug.trim().is_empty() {
            errors.push("Bot slug cannot be empty".into());
        } else if !self.slug.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
            errors.push(format!("Bot slug '{}' must only contain alphanumeric characters, '-', or '_'", self.slug));
        }

        if self.display_name.trim().is_empty() {
            errors.push("Display name cannot be empty".into());
        }

        // Warn on unknown tool identifiers (fail-closed at dispatch)
        let known_tools: Vec<&'static str> = crate::list_tools().into_iter().map(|t| t.name).collect();
        for t in &self.tools_allow {
            if t != "*" && !known_tools.contains(&t.as_str()) {
                eprintln!("Warning: Bot profile '{}' references unknown tool '{}' in tools_allow (will fail-closed)", self.slug, t);
            }
        }
        for t in &self.tools_deny {
            if t != "*" && !known_tools.contains(&t.as_str()) {
                eprintln!("Warning: Bot profile '{}' references unknown tool '{}' in tools_deny", self.slug, t);
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// Registry managing storage, discovery, and retrieval of bot profiles.
#[derive(Debug, Clone)]
pub struct BotRegistry {
    pub base_dir: PathBuf,
    profiles: HashMap<String, BotProfile>,
}

impl BotRegistry {
    /// Default directory at `~/.cynapse/bots/` or `./bots`.
    pub fn default_dir() -> PathBuf {
        if let Ok(home) = std::env::var("HOME") {
            PathBuf::from(home).join(".cynapse").join("bots")
        } else {
            PathBuf::from("./bots")
        }
    }

    /// Creates a new BotRegistry pointing to the given directory.
    pub fn new<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let base_dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&base_dir)
            .with_context(|| format!("Failed to create bot registry directory: {}", base_dir.display()))?;

        let mut reg = Self {
            base_dir,
            profiles: HashMap::new(),
        };

        reg.init_defaults()?;
        reg.load_all()?;
        Ok(reg)
    }

    /// Creates an empty in-memory registry without disk side-effects.
    pub fn empty() -> Self {
        Self {
            base_dir: PathBuf::from("./bots"),
            profiles: HashMap::new(),
        }
    }

    /// Seed default profile templates if directory is empty.
    pub fn init_defaults(&self) -> Result<()> {
        let coder_path = self.base_dir.join("coder.toml");
        if !coder_path.exists() {
            let coder = BotProfile {
                slug: "coder".into(),
                display_name: "Code Specialist".into(),
                description: "Code inspection, implementation, and safe workspace modifications.".into(),
                persona_file: None,
                model_preset: None,
                tools_allow: vec!["read_file".into(), "grep".into(), "write_file".into(), "security_forensic_scan".into()],
                tools_deny: Vec::new(),
                tools_ask: vec!["write_file".into()],
                workspace_restrict: None,
                max_concurrent: Some(2),
            };
            let content = toml::to_string_pretty(&coder)?;
            fs::write(&coder_path, content)?;
        }

        let researcher_path = self.base_dir.join("researcher.toml");
        if !researcher_path.exists() {
            let researcher = BotProfile {
                slug: "researcher".into(),
                display_name: "Research Analyst".into(),
                description: "Read-only workspace scanning, indexing, and information synthesis.".into(),
                persona_file: None,
                model_preset: None,
                tools_allow: vec!["read_file".into(), "grep".into()],
                tools_deny: vec!["write_file".into(), "execute_command".into()],
                tools_ask: Vec::new(),
                workspace_restrict: None,
                max_concurrent: Some(2),
            };
            let content = toml::to_string_pretty(&researcher)?;
            fs::write(&researcher_path, content)?;
        }

        let auditor_path = self.base_dir.join("auditor.toml");
        if !auditor_path.exists() {
            let auditor = BotProfile {
                slug: "auditor".into(),
                display_name: "Security & Test Auditor".into(),
                description: "Codebase security inspection, test runner, and vulnerability detection.".into(),
                persona_file: None,
                model_preset: None,
                tools_allow: vec!["read_file".into(), "grep".into(), "execute_command".into(), "security_forensic_scan".into()],
                tools_deny: vec!["write_file".into()],
                tools_ask: vec!["execute_command".into()],
                workspace_restrict: None,
                max_concurrent: Some(1),
            };
            let content = toml::to_string_pretty(&auditor)?;
            fs::write(&auditor_path, content)?;
        }

        let needle_path = self.base_dir.join("needle.toml");
        if !needle_path.exists() {
            let needle = BotProfile {
                slug: "needle".into(),
                display_name: "Needle Edge Specialist".into(),
                description: "Ultra-fast deterministic tool caller and JSON extractor with zero hallucination.".into(),
                persona_file: None,
                model_preset: Some("needle3".into()),
                tools_allow: vec!["read_file".into(), "grep".into(), "write_file".into(), "security_forensic_scan".into()],
                tools_deny: vec!["execute_command".into()],
                tools_ask: Vec::new(),
                workspace_restrict: None,
                max_concurrent: Some(4),
            };
            let content = toml::to_string_pretty(&needle)?;
            fs::write(&needle_path, content)?;
        }

        let fast_path = self.base_dir.join("fast.toml");
        if !fast_path.exists() {
            let fast = BotProfile {
                slug: "fast".into(),
                display_name: "Fast Dispatcher".into(),
                description: "High-throughput subagent for rapid parallel read and indexing tasks.".into(),
                persona_file: None,
                model_preset: Some("needle3".into()),
                tools_allow: vec!["read_file".into(), "grep".into()],
                tools_deny: vec!["execute_command".into(), "write_file".into()],
                tools_ask: Vec::new(),
                workspace_restrict: None,
                max_concurrent: Some(4),
            };
            let content = toml::to_string_pretty(&fast)?;
            fs::write(&fast_path, content)?;
        }

        Ok(())
    }

    /// Reloads all `*.toml` bot profiles from the registry directory.
    pub fn load_all(&mut self) -> Result<()> {
        self.profiles.clear();

        if !self.base_dir.exists() {
            return Ok(());
        }

        for entry in fs::read_dir(&self.base_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("toml") {
                let content = fs::read_to_string(&path)?;
                match toml::from_str::<BotProfile>(&content) {
                    Ok(profile) => {
                        if let Err(errs) = profile.validate() {
                            eprintln!("Warning: Invalid bot profile in {}: {:?}", path.display(), errs);
                            continue;
                        }
                        if self.profiles.contains_key(&profile.slug) {
                            eprintln!("Warning: Duplicate bot slug '{}' in {}; overwriting previous profile", profile.slug, path.display());
                        }
                        self.profiles.insert(profile.slug.clone(), profile);
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to parse bot profile {}: {}", path.display(), e);
                    }
                }
            }
        }

        Ok(())
    }

    /// Get bot profile by slug.
    pub fn get(&self, slug: &str) -> Option<&BotProfile> {
        self.profiles.get(slug)
    }

    /// List all bot profiles sorted alphabetically by slug.
    pub fn list(&self) -> Vec<&BotProfile> {
        let mut list: Vec<&BotProfile> = self.profiles.values().collect();
        list.sort_by_key(|p| &p.slug);
        list
    }

    /// Insert or update a bot profile and save to disk.
    pub fn save_profile(&mut self, profile: BotProfile) -> Result<()> {
        if let Err(errs) = profile.validate() {
            bail!("Cannot save invalid profile: {:?}", errs);
        }

        let file_path = self.base_dir.join(format!("{}.toml", profile.slug));
        let content = toml::to_string_pretty(&profile)?;
        fs::write(&file_path, content)?;
        self.profiles.insert(profile.slug.clone(), profile);
        Ok(())
    }

    /// Delete a bot profile by slug.
    pub fn delete_profile(&mut self, slug: &str) -> Result<bool> {
        let file_path = self.base_dir.join(format!("{}.toml", slug));
        if file_path.exists() {
            fs::remove_file(file_path)?;
        }
        Ok(self.profiles.remove(slug).is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bot_profile_toml_roundtrip() {
        let profile = BotProfile {
            slug: "test-bot".into(),
            display_name: "Test Bot".into(),
            description: "A test bot profile".into(),
            persona_file: Some("IDENTITY.md".into()),
            model_preset: Some("Tier 1 Fast".into()),
            tools_allow: vec!["read_file".into(), "grep".into()],
            tools_deny: vec!["write_file".into()],
            tools_ask: vec!["execute_command".into()],
            workspace_restrict: Some("/tmp/workspace".into()),
            max_concurrent: Some(3),
        };

        let toml_str = toml::to_string_pretty(&profile).unwrap();
        let parsed: BotProfile = toml::from_str(&toml_str).unwrap();
        assert_eq!(profile, parsed);
    }

    #[test]
    fn test_bot_profile_permissions_precedence() {
        let profile = BotProfile {
            slug: "sec-bot".into(),
            display_name: "Security Bot".into(),
            description: "".into(),
            persona_file: None,
            model_preset: None,
            tools_allow: vec!["read_file".into(), "write_file".into(), "grep".into()],
            tools_deny: vec!["write_file".into()], // deny overrides allow
            tools_ask: vec!["read_file".into()],
            workspace_restrict: None,
            max_concurrent: None,
        };

        assert!(profile.is_tool_allowed("read_file"));
        assert!(profile.is_tool_allowed("grep"));
        assert!(!profile.is_tool_allowed("write_file")); // denied
        assert!(!profile.is_tool_allowed("execute_command")); // not in allow list
        assert!(profile.requires_approval("read_file"));
        assert!(!profile.requires_approval("grep"));
    }

    #[test]
    fn test_bot_registry_defaults_and_loading() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut registry = BotRegistry::new(temp_dir.path()).unwrap();

        let list = registry.list();
        assert_eq!(list.len(), 5);
        assert!(registry.get("coder").is_some());
        assert!(registry.get("researcher").is_some());
        assert!(registry.get("auditor").is_some());
        assert!(registry.get("needle").is_some());
        assert!(registry.get("fast").is_some());

        // Test save new custom profile
        let custom = BotProfile {
            slug: "reviewer".into(),
            display_name: "Code Reviewer".into(),
            description: "Autonomous PR reviewer".into(),
            persona_file: None,
            model_preset: None,
            tools_allow: vec!["read_file".into()],
            tools_deny: vec![],
            tools_ask: vec![],
            workspace_restrict: None,
            max_concurrent: Some(1),
        };

        registry.save_profile(custom).unwrap();
        assert_eq!(registry.list().len(), 6);
        assert_eq!(registry.get("reviewer").unwrap().display_name, "Code Reviewer");
    }

    #[test]
    fn test_bot_profile_slug_validation() {
        let mut profile = BotProfile::default();
        profile.slug = "invalid slug with spaces".into();
        assert!(profile.validate().is_err());

        profile.slug = "valid-slug_123".into();
        assert!(profile.validate().is_ok());
    }

    #[test]
    fn test_bot_profile_workspace_restriction() {
        let temp = tempfile::tempdir().unwrap();
        let ws_dir = temp.path().join("sandbox");
        fs::create_dir_all(&ws_dir).unwrap();
        let file_inside = ws_dir.join("code.rs");
        fs::write(&file_inside, "fn main() {}").unwrap();
        let file_outside = temp.path().join("secret.txt");
        fs::write(&file_outside, "secret").unwrap();

        let profile = BotProfile {
            slug: "sandboxed".into(),
            display_name: "Sandboxed Bot".into(),
            description: "".into(),
            persona_file: None,
            model_preset: None,
            tools_allow: vec!["*".into()],
            tools_deny: vec![],
            tools_ask: vec![],
            workspace_restrict: Some(ws_dir.to_string_lossy().to_string()),
            max_concurrent: None,
        };

        assert!(profile.is_path_in_workspace(&file_inside));
        assert!(!profile.is_path_in_workspace(&file_outside));
        assert!(!profile.is_path_in_workspace("/etc/passwd"));
    }
}
