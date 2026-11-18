//! Lightweight Spec-Driven Development (SDD) scaffolder inspired by `github/spec-kit`.
//! Generates compact `SPEC.md` and `GATES.md` files so small local models never guess.

use anyhow::Result;
use std::fs;
use std::path::{Path, PathBuf};

/// Scaffold `SPEC.md` and starter `GATES.md` (if absent) for a feature or bugfix.
pub fn scaffold_spec(workspace: &Path, goal: &str) -> Result<(PathBuf, PathBuf)> {
    let clean_goal = if goal.trim().is_empty() {
        "Describe the feature or fix here"
    } else {
        goal.trim()
    };

    let spec_path = workspace.join("SPEC.md");
    let spec_content = format!(
        "# Specification: {goal}\n\n\
         ## 1. Goal\n{goal}\n\n\
         ## 2. Scope\n\
         - **In-Scope:** Minimal changes needed to satisfy the goal.\n\
         - **Out-of-Scope:** Unrelated refactors, extra dependencies, or unrequested features.\n\n\
         ## 3. Plan & Owned Files\n\
         - `OWNS:` List exact files to create/edit before coding.\n\n\
         ## 4. Acceptance Criteria\n\
         - [ ] Implementation matches spec.\n\
         - [ ] `GATES.md` checks pass with observed output.\n",
        goal = clean_goal
    );
    fs::write(&spec_path, spec_content)?;

    let gates_path = workspace.join("GATES.md");
    if !gates_path.exists() {
        let starter_gates = format!(
            "# Acceptance Gates: {}\n\n\
             ## GATE_BUILD\n\
             CHECK: cargo check --workspace\n\
             EXPECT: Finished\n",
            clean_goal
        );
        fs::write(&gates_path, starter_gates)?;
    }

    Ok((spec_path, gates_path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaffolds_spec_and_starter_gates() {
        let dir = std::env::temp_dir().join(format!("cynapse_spec_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let (spec, gates) = scaffold_spec(&dir, "Add RTK filter").unwrap();
        assert!(spec.exists());
        assert!(gates.exists());
        let s = fs::read_to_string(&spec).unwrap();
        assert!(s.contains("# Specification: Add RTK filter"));
        assert!(s.contains("In-Scope:"));

        let _ = fs::remove_dir_all(&dir);
    }
}
