//! Failure-Replay Harvester for student-learning post-training (`/soup harvest`).
//! Extracts only failed/weak eval cases into `data/glia_failures.jsonl` for Teacher diagnosis.

use crate::cyber_eval::CyberEvalSummary;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureRecord {
    pub id: String,
    pub model_name: String,
    pub prompt: String,
    pub student_output: String,
    pub score: f32,
    pub failure_reason: String,
}

/// Filter an evaluation summary down to only failing/weak responses and write JSONL.
pub fn harvest_failures(summary: &CyberEvalSummary, out_jsonl: &Path, min_score: f32) -> Result<Vec<FailureRecord>> {
    let failures: Vec<FailureRecord> = summary
        .grades
        .iter()
        .filter(|g| !g.passed || g.score < min_score)
        .map(|g| FailureRecord {
            id: g.id.clone(),
            model_name: summary.model_name.clone(),
            prompt: g.prompt.clone(),
            student_output: g.response.clone(),
            score: g.score,
            failure_reason: g
                .failure_reason
                .clone()
                .unwrap_or_else(|| format!("score {:.2} < {:.2}", g.score, min_score)),
        })
        .collect();

    if let Some(parent) = out_jsonl.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(out_jsonl)?;
    for rec in &failures {
        writeln!(file, "{}", serde_json::to_string(rec)?)?;
    }
    Ok(failures)
}

/// Format a compact, harness-agnostic Teacher diagnosis prompt for harvested failures.
pub fn format_teacher_diagnosis_prompt(failures: &[FailureRecord]) -> String {
    if failures.is_empty() {
        return "All cybersecurity challenges passed — no weak points to replay.".to_string();
    }
    let mut out = String::from(
        "You are a senior defensive cybersecurity teacher. Below are ONLY the failed responses from a 1.7B student model. \
         For each failure, write a concise, harness-agnostic, safe diagnostic & remediation answer (standard Linux/OS tools only):\n\n",
    );
    for (i, f) in failures.iter().take(5).enumerate() {
        out.push_str(&format!(
            "### Failure {} ({})\nPrompt: {}\nReason: {}\nStudent: {}\n\n",
            i + 1,
            f.id,
            f.prompt,
            f.failure_reason,
            f.student_output.chars().take(300).collect::<String>()
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cyber_eval::CyberGrade;

    #[test]
    fn harvests_only_failing_cases_to_jsonl() {
        let summary = CyberEvalSummary {
            model_name: "Glia-1.7B.gguf".into(),
            total: 2,
            passed_count: 1,
            mean_score: 0.5,
            grades: vec![
                CyberGrade {
                    id: "pass_case".into(),
                    prompt: "p1".into(),
                    passed: true,
                    score: 1.0,
                    failure_reason: None,
                    response: "good".into(),
                    gen_tps: 28.0,
                },
                CyberGrade {
                    id: "fail_case".into(),
                    prompt: "p2".into(),
                    passed: false,
                    score: 0.0,
                    failure_reason: Some("missing defensive command".into()),
                    response: "bad".into(),
                    gen_tps: 28.0,
                },
            ],
        };
        let path = std::env::temp_dir().join(format!("glia_failures_{}.jsonl", std::process::id()));
        let harvested = harvest_failures(&summary, &path, 0.75).unwrap();
        assert_eq!(harvested.len(), 1);
        assert_eq!(harvested[0].id, "fail_case");
        let _ = fs::remove_file(&path);
    }
}
