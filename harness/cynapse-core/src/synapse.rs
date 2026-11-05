//! Modular Synapse Extension & MCP Tool Hub for CYNAPSE.
//!
//! In biological neural systems, synapses allow neurons to dynamically connect
//! to transmit specialized neurotransmitters. In CYNAPSE, Synapse extensions
//! allow users to download and plug in modular tools (e.g. model fine-tuning,
//! web crawlers, database connectors) into the local agent without bloating
//! the core binary.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapseManifest {
    pub name: String,
    pub title: String,
    pub version: String,
    pub author: Option<String>,
    pub description: String,
    pub enabled: Option<bool>,
    #[serde(default)]
    pub tools: Vec<SynapseToolDef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapseToolDef {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub command: String,
    #[serde(default = "default_transport")]
    pub transport: String,
    #[serde(default = "default_concurrency_safe")]
    pub concurrency_safe: bool,
    pub input_schema: Option<serde_json::Value>,
}

/// Lifecycle hook signatures for Synapse extensions (DeepSeek Harness / Cordis paradigm).
pub type BeforeToolHook = Arc<dyn Fn(&str, &serde_json::Value) -> Result<()> + Send + Sync>;
pub type AfterToolHook = Arc<dyn Fn(&str, &str) -> Result<String> + Send + Sync>;

/// Execution pipeline lifecycle hooks container.
#[derive(Default)]
pub struct SynapseLifecycleHooks {
    pub before_tool: Vec<BeforeToolHook>,
    pub after_tool: Vec<AfterToolHook>,
}

fn default_transport() -> String {
    "stdio".to_string()
}

fn default_concurrency_safe() -> bool {
    false
}

/// Global registry of dynamically loaded Synapse extensions.
pub struct SynapseRegistry {
    synapses_dir: PathBuf,
    manifests: Mutex<HashMap<String, SynapseManifest>>,
    hooks: Mutex<SynapseLifecycleHooks>,
}

static REGISTRY: OnceLock<Arc<SynapseRegistry>> = OnceLock::new();

impl SynapseRegistry {
    pub fn global() -> Arc<Self> {
        REGISTRY
            .get_or_init(|| {
                let dir = dirs::home_dir()
                    .map(|h| h.join(".cynapse").join("synapses"))
                    .unwrap_or_else(|| PathBuf::from("./synapses"));
                let _ = fs::create_dir_all(&dir);
                let reg = Self {
                    synapses_dir: dir,
                    manifests: Mutex::new(HashMap::new()),
                    hooks: Mutex::new(SynapseLifecycleHooks::default()),
                };
                reg.reload();
                Arc::new(reg)
            })
            .clone()
    }

    /// Registers a before-tool-execution lifecycle hook.
    pub fn register_before_tool<F>(&self, hook: F)
    where
        F: Fn(&str, &serde_json::Value) -> Result<()> + Send + Sync + 'static,
    {
        if let Ok(mut h) = self.hooks.lock() {
            h.before_tool.push(Arc::new(hook));
        }
    }

    /// Registers an after-tool-execution lifecycle hook.
    pub fn register_after_tool<F>(&self, hook: F)
    where
        F: Fn(&str, &str) -> Result<String> + Send + Sync + 'static,
    {
        if let Ok(mut h) = self.hooks.lock() {
            h.after_tool.push(Arc::new(hook));
        }
    }

    /// Reloads all installed Synapse manifests from ~/.cynapse/synapses/
    pub fn reload(&self) {
        let mut map = self.manifests.lock().unwrap_or_else(|e| e.into_inner());
        map.clear();

        if let Ok(entries) = fs::read_dir(&self.synapses_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    let manifest_file = path.join("synapse.toml");
                    let json_manifest = path.join("synapse.json");

                    if manifest_file.exists() {
                        if let Ok(content) = fs::read_to_string(&manifest_file) {
                            if let Ok(m) = toml::from_str::<SynapseManifest>(&content) {
                                map.insert(m.name.clone(), m);
                            }
                        }
                    } else if json_manifest.exists() {
                        if let Ok(content) = fs::read_to_string(&json_manifest) {
                            if let Ok(m) = serde_json::from_str::<SynapseManifest>(&content) {
                                map.insert(m.name.clone(), m);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Lists all installed Synapse manifests.
    pub fn list_synapses(&self) -> Vec<SynapseManifest> {
        let map = self.manifests.lock().unwrap_or_else(|e| e.into_inner());
        map.values().cloned().collect()
    }

    /// Returns a specific manifest by name.
    pub fn get_synapse(&self, name: &str) -> Option<SynapseManifest> {
        let map = self.manifests.lock().unwrap_or_else(|e| e.into_inner());
        map.get(name).cloned()
    }

    /// Toggles enabling/disabling a Synapse extension.
    pub fn set_enabled(&self, name: &str, enabled: bool) -> Result<bool> {
        let mut map = self.manifests.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(m) = map.get_mut(name) {
            m.enabled = Some(enabled);
            let manifest_path = self.synapses_dir.join(name).join("synapse.toml");
            if manifest_path.exists() {
                let toml_str = toml::to_string_pretty(m)?;
                fs::write(manifest_path, toml_str)?;
            }
            Ok(true)
        } else {
            bail!("Synapse extension '{}' not found", name)
        }
    }

    /// Returns all registered tools across active Synapse extensions.
    pub fn active_tools(&self) -> Vec<(String, SynapseToolDef)> {
        let map = self.manifests.lock().unwrap_or_else(|e| e.into_inner());
        let mut out = Vec::new();
        for (syn_name, m) in map.iter() {
            if m.enabled.unwrap_or(true) {
                for t in &m.tools {
                    out.push((syn_name.clone(), t.clone()));
                }
            }
        }
        out
    }

    /// Executes a tool provided by a Synapse extension.
    pub fn execute_tool(
        &self,
        tool_name: &str,
        arguments: &serde_json::Value,
    ) -> Result<String> {
        let active = self.active_tools();
        let (syn_name, tool_def) = match active.iter().find(|(_, t)| t.name == tool_name) {
            Some(found) => found,
            None => bail!("Synapse tool '{}' not found or extension disabled", tool_name),
        };

        let synapse_path = self.synapses_dir.join(syn_name);
        let cmd_str = &tool_def.command;
        if cmd_str.trim().is_empty() {
            return Ok(format!("Tool '{}' executed successfully with arguments: {}", tool_name, arguments));
        }

        let arg_json = serde_json::to_string(arguments)?;
        let mut child = Command::new("bash")
            .arg("-c")
            .arg(cmd_str)
            .current_dir(&synapse_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("Failed to spawn synapse command: {}", cmd_str))?;

        if let Some(mut stdin) = child.stdin.take() {
            use std::io::Write;
            let _ = stdin.write_all(arg_json.as_bytes());
        }

        // Lifecycle hook: before_tool_exec
        if let Ok(hooks) = self.hooks.lock() {
            for hook in &hooks.before_tool {
                hook(tool_name, arguments)?;
            }
        }

        let output = child.wait_with_output().context("Waiting for synapse tool process")?;
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();

        let raw_result = if output.status.success() {
            if stdout.is_empty() && !stderr.is_empty() {
                stderr
            } else {
                stdout
            }
        } else {
            bail!("Synapse tool exited with error ({}):\n{}", output.status, stderr)
        };

        // Lifecycle hook: after_tool_exec
        let mut final_result = raw_result;
        if let Ok(hooks) = self.hooks.lock() {
            for hook in &hooks.after_tool {
                final_result = hook(tool_name, &final_result)?;
            }
        }

        Ok(final_result)
    }
}
