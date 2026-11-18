//! Lightweight, pure-Rust RTK (Rust Token Killer) output compactor for `execute_command`.
//! Compresses noisy terminal output for small-context local LLMs without extra prompt tokens.

const FAST_PATH_MAX_CHARS: usize = 1200;
const FAST_PATH_MAX_LINES: usize = 20;
const HARD_CHAR_CAP: usize = 3500;

/// Compact terminal output transparently. Short outputs bypass filtering with zero overhead.
pub fn compact_command_output(command: &str, raw: &str) -> String {
    if raw.len() <= FAST_PATH_MAX_CHARS && raw.lines().count() <= FAST_PATH_MAX_LINES {
        return raw.to_string();
    }

    let cleaned = strip_ansi_and_cr(raw);
    let base_cmd = detect_base_command(command);

    let filtered = match base_cmd.as_str() {
        "cargo" | "pytest" | "npm" | "pnpm" | "yarn" | "ctest" | "go" => filter_build_test(&cleaned),
        "git" => filter_git(command, &cleaned),
        "ls" | "find" | "rg" | "grep" | "ps" | "tree" | "du" | "df" => cap_lines(&cleaned, 35),
        _ => cleaned,
    };

    let final_out = truncate_head_tail(&filtered, HARD_CHAR_CAP);
    if final_out.len() < raw.len() {
        format!(
            "{}\n[RTK: compacted {} -> {} chars]",
            final_out.trim_end(),
            raw.len(),
            final_out.len()
        )
    } else {
        final_out
    }
}

fn detect_base_command(command: &str) -> String {
    for tok in command.split_whitespace() {
        if tok.contains('=') || tok == "sudo" || tok == "env" {
            continue;
        }
        return tok.rsplit('/').next().unwrap_or(tok).to_lowercase();
    }
    String::new()
}

/// Strip ANSI CSI escapes and collapse `\r` spinner/progress overwrites in a single pass.
fn strip_ansi_and_cr(input: &str) -> String {
    let normalized = input.replace("\r\n", "\n");
    let mut out = String::with_capacity(normalized.len());
    let mut blank_run = 0usize;

    for raw_line in normalized.lines() {
        let latest = raw_line.rsplit('\r').next().unwrap_or(raw_line);
        let stripped = strip_ansi_line(latest);
        let trimmed_end = stripped.trim_end();
        if trimmed_end.is_empty() {
            blank_run += 1;
            if blank_run <= 1 {
                out.push('\n');
            }
        } else {
            blank_run = 0;
            out.push_str(trimmed_end);
            out.push('\n');
        }
    }
    out
}

fn strip_ansi_line(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            if chars.peek() == Some(&'[') {
                chars.next();
                for esc_c in chars.by_ref() {
                    if ('@'..='~').contains(&esc_c) {
                        break;
                    }
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn filter_build_test(text: &str) -> String {
    let mut kept = Vec::new();
    let mut omitted = 0usize;

    for line in text.lines() {
        let lower = line.to_lowercase();
        let is_signal = lower.contains("error")
            || lower.contains("warning:")
            || lower.contains("failed")
            || lower.contains("fail ")
            || lower.contains("panicked")
            || lower.contains("finished")
            || lower.contains("test result:")
            || lower.contains("traceback")
            || lower.contains("exception")
            || lower.contains("assert")
            || line.trim_start().starts_with("-->")
            || line.trim_start().starts_with('|');

        let is_noise = lower.trim_start().starts_with("compiling ")
            || lower.trim_start().starts_with("downloading ")
            || lower.trim_start().starts_with("downloaded ")
            || lower.trim_start().starts_with("fresh ")
            || lower.ends_with("... ok");

        if is_signal || !is_noise {
            kept.push(line);
        } else {
            omitted += 1;
        }
    }

    if omitted > 0 {
        kept.push("");
        let mut out = kept.join("\n");
        out.push_str(&format!("\n[... {} build/passing lines omitted]", omitted));
        out
    } else {
        kept.join("\n")
    }
}

fn filter_git(command: &str, text: &str) -> String {
    if command.contains("diff") {
        let mut kept = Vec::new();
        let mut file_hunk_lines = 0usize;
        let mut omitted = 0usize;
        for line in text.lines() {
            if line.starts_with("diff --git") || line.starts_with("+++") || line.starts_with("---") || line.starts_with("@@") {
                file_hunk_lines = 0;
                kept.push(line);
            } else if file_hunk_lines < 40 {
                file_hunk_lines += 1;
                kept.push(line);
            } else {
                omitted += 1;
            }
        }
        let mut out = kept.join("\n");
        if omitted > 0 {
            out.push_str(&format!("\n[... {} diff lines omitted]", omitted));
        }
        out
    } else {
        cap_lines(text, 40)
    }
}

fn cap_lines(text: &str, max_lines: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() <= max_lines {
        return text.to_string();
    }
    let omitted = lines.len() - max_lines;
    let mut out = lines[..max_lines].join("\n");
    out.push_str(&format!(
        "\n[... {} more lines omitted; narrow path or grep to inspect]",
        omitted
    ));
    out
}

fn truncate_head_tail(text: &str, max_chars: usize) -> String {
    let total_chars = text.chars().count();
    if total_chars <= max_chars {
        return text.to_string();
    }
    let head_chars = (max_chars * 60) / 100;
    let tail_chars = max_chars - head_chars;

    let head_end = text
        .char_indices()
        .nth(head_chars)
        .map(|(idx, _)| idx)
        .unwrap_or(text.len());
    let tail_start = text
        .char_indices()
        .nth(total_chars.saturating_sub(tail_chars))
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    let omitted = total_chars.saturating_sub(head_chars + tail_chars);
    format!(
        "{}\n\n[... {} chars omitted ...]\n\n{}",
        &text[..head_end],
        omitted,
        &text[tail_start..]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_path_preserves_short_output() {
        let short = "Exit Status: 0\nhello world\n";
        assert_eq!(compact_command_output("echo hello", short), short);
    }

    #[test]
    fn strips_ansi_and_caps_listing() {
        let mut noisy = String::new();
        for i in 0..80 {
            noisy.push_str(&format!("\x1b[32mfile_{:02}.rs\x1b[0m\n", i));
        }
        let compacted = compact_command_output("ls -la", &noisy);
        assert!(!compacted.contains("\x1b[32m"));
        assert!(compacted.contains("more lines omitted"));
        assert!(compacted.len() < noisy.len());
    }

    #[test]
    fn filters_cargo_compile_noise_but_keeps_errors() {
        let mut log = String::new();
        for i in 0..60 {
            log.push_str(&format!("   Compiling crate_{} v0.1.0\n", i));
        }
        log.push_str("error[E0425]: cannot find value `x` in this scope\n");
        log.push_str("test result: FAILED. 0 passed; 1 failed\n");
        let compacted = compact_command_output("cargo test", &log);
        assert!(compacted.contains("error[E0425]"));
        assert!(compacted.contains("test result: FAILED"));
        assert!(compacted.contains("build/passing lines omitted"));
    }
}
