//! In-process Git & Release updater for Cynapse.
//!
//! Provides non-blocking remote version/commit polling and one-command `/update` execution.

use std::path::Path;
use std::process::{Command, Stdio};
use anyhow::{bail, Context, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdateInfo {
    pub current_commit: String,
    pub remote_commit: String,
    pub branch: String,
    pub remote_url: String,
    pub has_update: bool,
}

impl UpdateInfo {
    pub fn summary(&self) -> String {
        if self.has_update {
            format!(
                "Update available on '{}'! Current: {} -> Remote: {}. Run /update to install.",
                self.branch,
                &self.current_commit[..7.min(self.current_commit.len())],
                &self.remote_commit[..7.min(self.remote_commit.len())]
            )
        } else {
            format!(
                "Up to date on '{}' ({})",
                self.branch,
                &self.current_commit[..7.min(self.current_commit.len())]
            )
        }
    }
}

/// Checks remote git repository for updates without modifying local working directory.
pub fn check_for_updates(repo_dir: &Path) -> Result<Option<UpdateInfo>> {
    // 1. Get current branch
    let branch_out = Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .current_dir(repo_dir)
        .output()
        .context("Failed to query git branch")?;
    if !branch_out.status.success() {
        return Ok(None);
    }
    let branch = String::from_utf8_lossy(&branch_out.stdout).trim().to_string();

    // 2. Get local HEAD commit
    let local_head_out = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(repo_dir)
        .output()
        .context("Failed to query local HEAD")?;
    if !local_head_out.status.success() {
        return Ok(None);
    }
    let current_commit = String::from_utf8_lossy(&local_head_out.stdout).trim().to_string();

    // 3. Get remote URL
    let remote_url_out = Command::new("git")
        .args(["remote", "get-url", "origin"])
        .current_dir(repo_dir)
        .output()
        .context("Failed to query git remote")?;
    if !remote_url_out.status.success() {
        return Ok(None);
    }
    let remote_url = String::from_utf8_lossy(&remote_url_out.stdout).trim().to_string();

    // 4. Query remote HEAD using ls-remote with piped stdout
    let remote_head_out = Command::new("git")
        .args(["ls-remote", "origin", &format!("refs/heads/{}", branch)])
        .current_dir(repo_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output();

    let remote_commit = match remote_head_out {
        Ok(out) if out.status.success() => {
            let text = String::from_utf8_lossy(&out.stdout);
            text.split_whitespace().next().unwrap_or("").to_string()
        }
        _ => return Ok(None),
    };

    if remote_commit.is_empty() {
        return Ok(None);
    }

    let has_update = if current_commit == remote_commit {
        false
    } else {
        // If remote_commit is an ancestor of HEAD, local is ahead of remote, so no update needed
        let is_ancestor = Command::new("git")
            .args(["merge-base", "--is-ancestor", &remote_commit, "HEAD"])
            .current_dir(repo_dir)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        !is_ancestor
    };

    Ok(Some(UpdateInfo {
        current_commit,
        remote_commit,
        branch,
        remote_url,
        has_update,
    }))
}

/// Executes full update sequence: git pull, cargo build --release, and binary verification.
pub fn execute_update<F>(repo_dir: &Path, mut progress: F) -> Result<String>
where
    F: FnMut(&str),
{
    progress("[1/3] Checking repository status...");
    let status_out = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(repo_dir)
        .output()
        .context("Failed to check git status")?;

    let dirty = !status_out.stdout.is_empty();
    if dirty {
        progress("[1/3] Stashing local changes to prevent merge conflicts...");
        let _ = Command::new("git")
            .args(["stash", "save", "cynapse-auto-update"])
            .current_dir(repo_dir)
            .output();
    }

    // Get current branch
    let branch_out = Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .current_dir(repo_dir)
        .output()
        .context("Failed to query git branch")?;
    let branch = String::from_utf8_lossy(&branch_out.stdout).trim().to_string();
    let branch = if branch.is_empty() { "main".to_string() } else { branch };

    progress(&format!("[2/3] Fetching remote branch '{}' from origin...", branch));
    let _ = Command::new("git")
        .args(["fetch", "origin", &branch])
        .current_dir(repo_dir)
        .output();

    let remote_ref = format!("origin/{}", branch);
    let is_ancestor = Command::new("git")
        .args(["merge-base", "--is-ancestor", &remote_ref, "HEAD"])
        .current_dir(repo_dir)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    if is_ancestor {
        progress(&format!("Local '{}' is already up to date with {}.", branch, remote_ref));
    } else {
        progress(&format!("[2/3] Fast-forwarding local '{}' from {}...", branch, remote_ref));
        let pull_out = Command::new("git")
            .args(["pull", "--ff-only", "origin", &branch])
            .current_dir(repo_dir)
            .output()
            .context("Failed to run git pull")?;

        if !pull_out.status.success() {
            let err = String::from_utf8_lossy(&pull_out.stderr);
            if dirty {
                let _ = Command::new("git")
                    .args(["stash", "pop"])
                    .current_dir(repo_dir)
                    .output();
            }
            bail!("Git pull failed: {}", err.trim());
        }
    }

    if dirty {
        progress("Restoring stashed changes...");
        let _ = Command::new("git")
            .args(["stash", "pop"])
            .current_dir(repo_dir)
            .output();
    }

    progress("[3/3] Compiling optimized release binary (cargo build --release --bin cynapse)...");
    let build_out = Command::new("cargo")
        .args(["build", "--release", "--bin", "cynapse"])
        .current_dir(repo_dir)
        .output()
        .context("Failed to run cargo build")?;

    if !build_out.status.success() {
        let err = String::from_utf8_lossy(&build_out.stderr);
        bail!("Compilation failed: {}", err.trim());
    }

    let bin_path = repo_dir.join("target").join("release").join("cynapse");
    if !bin_path.exists() {
        bail!("Build completed but binary not found at {:?}", bin_path);
    }

    let head_out = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(repo_dir)
        .output()
        .ok();
    let head_hash = head_out
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "latest".into());

    Ok(format!("Cynapse release binary successfully built at ./target/release/cynapse (commit {}). Restart Cynapse to run updated binary.", head_hash))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_info_summary() {
        let info = UpdateInfo {
            current_commit: "abc12345".into(),
            remote_commit: "def67890".into(),
            branch: "main".into(),
            remote_url: "https://github.com/Alartist40/cynapse-harness.git".into(),
            has_update: true,
        };
        assert!(info.summary().contains("Update available"));
        assert!(info.summary().contains("/update"));

        let current = UpdateInfo {
            current_commit: "abc12345".into(),
            remote_commit: "abc12345".into(),
            branch: "main".into(),
            remote_url: "https://github.com/Alartist40/cynapse-harness.git".into(),
            has_update: false,
        };
        assert!(current.summary().contains("Up to date"));
    }
}
