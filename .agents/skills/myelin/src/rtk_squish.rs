// Pure stdlib Rust terminal output squisher (rtk-inspired, zero external deps).
use std::env;
use std::io::{self, Read};
use std::process::{self, Command, Stdio};

fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            if chars.peek() == Some(&'[') {
                chars.next();
                while let Some(&nc) = chars.peek() {
                    chars.next();
                    if ('@'..='~').contains(&nc) {
                        break;
                    }
                }
            }
        } else if c != '\r' {
            out.push(c);
        }
    }
    out
}

fn is_signal(line: &str) -> bool {
    let lower = line.to_lowercase();
    const KEYWORDS: &[&str] = &[
        "error", "fail", "exception", "traceback", "panic", "fatal", "warn", "diff --git", "@@",
    ];
    if KEYWORDS.iter().any(|k| lower.contains(k)) {
        return true;
    }
    // Match file:line style anchors
    let bytes = line.as_bytes();
    for i in 0..bytes.len().saturating_sub(1) {
        if bytes[i] == b':' && bytes[i + 1].is_ascii_digit() {
            return true;
        }
    }
    false
}

fn squish(text: &str, max_lines: usize) -> String {
    let mut deduped: Vec<(String, usize)> = Vec::new();
    for raw in text.lines() {
        let clean = strip_ansi(raw).trim_end().to_string();
        if clean.trim().is_empty() {
            continue;
        }
        if let Some((last, count)) = deduped.last_mut() {
            if *last == clean {
                *count += 1;
                continue;
            }
        }
        deduped.push((clean, 1));
    }

    let lines: Vec<String> = deduped
        .into_iter()
        .map(|(l, c)| if c > 1 { format!("{l} [x{c}]") } else { l })
        .collect();

    if lines.len() <= max_lines {
        return lines.join("\n");
    }

    let head = &lines[..15];
    let tail = &lines[lines.len() - 25..];
    let mid: Vec<String> = lines[15..lines.len() - 25]
        .iter()
        .filter(|l| is_signal(l))
        .take(20)
        .cloned()
        .collect();
    let omitted = lines.len() - head.len() - mid.len() - tail.len();

    let mut out: Vec<String> = Vec::new();
    out.extend_from_slice(head);
    if omitted > 0 {
        out.push(format!("... [{omitted} noise lines squished] ..."));
    }
    out.extend(mid);
    out.extend_from_slice(tail);
    out.join("\n")
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("-h")
        || args.first().map(String::as_str) == Some("--help")
    {
        println!("Usage: rtk-squish <command...>  OR  <cmd> | rtk-squish");
        return;
    }

    if args.is_empty() {
        let mut buf = String::new();
        let _ = io::stdin().read_to_string(&mut buf);
        println!("{}", squish(&buf, 60));
        return;
    }

    let output = Command::new(&args[0])
        .args(&args[1..])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();

    match output {
        Ok(res) => {
            let mut combined = String::from_utf8_lossy(&res.stdout).to_string();
            if !res.stderr.is_empty() {
                if !combined.is_empty() && !combined.ends_with('\n') {
                    combined.push('\n');
                }
                combined.push_str(&String::from_utf8_lossy(&res.stderr));
            }
            let squished = squish(&combined, 60);
            if !squished.is_empty() {
                println!("{squished}");
            }
            process::exit(res.status.code().unwrap_or(1));
        }
        Err(e) => {
            eprintln!("rtk-squish failed to run '{}': {e}", args[0]);
            process::exit(127);
        }
    }
}
