// Pure stdlib Rust skill/MCP security scanner (SkillSpector-inspired, zero deps).
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

const SKIP_DIRS: &[&str] = &[".git", "node_modules", "target", "dist", "build", ".venv", "venv"];
const EXTS: &[&str] = &["md", "sh", "py", "js", "ts", "rs", "json", "yaml", "yml", "toml"];

const RULES: &[(&str, &str, &[&str])] = &[
    ("PROMPT_INJECT", "HIGH", &[
        "ignore previous instructions", "disregard previous", "override system prompt",
        "do not tell the user", "hide this from the user", "bypass safety",
    ]),
    ("OBFUSCATION_EXEC", "HIGH", &[
        "base64 -d", "base64 --decode", "| sh", "| bash", "eval(", "exec(", "atob(", "rm -rf /",
    ]),
    ("SECRET_ACCESS", "WARN", &[
        "aws_secret", "github_token", "openai_api_key", "anthropic_api_key",
        "id_rsa", ".ssh/", "/etc/shadow", "os.environ", "process.env",
    ]),
    ("NET_EXFIL", "WARN", &[
        "curl ", "wget ", "nc -", "ncat ", "requests.post", "urllib.request",
    ]),
];

fn collect(path: &Path, files: &mut Vec<PathBuf>) {
    if path.is_file() {
        files.push(path.to_path_buf());
        return;
    }
    let Ok(entries) = fs::read_dir(path) else { return };
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|x| x.path())).collect();
    paths.sort();
    for p in paths {
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if p.is_dir() {
            if !SKIP_DIRS.contains(&name) {
                collect(&p, files);
            }
        } else if EXTS.contains(&p.extension().and_then(|e| e.to_str()).unwrap_or("")) {
            files.push(p);
        }
    }
}

fn scan_file(path: &Path, root: &Path, hits: &mut usize) {
    let Ok(content) = fs::read_to_string(path) else { return };
    let rel = path.strip_prefix(root).unwrap_or(path).display();
    for (idx, line) in content.lines().enumerate() {
        let lower = line.to_lowercase();
        for (cat, sev, patterns) in RULES {
            if let Some(pat) = patterns.iter().find(|p| lower.contains(**p)) {
                let snippet: String = line.trim().chars().take(80).collect();
                println!("{rel}:{}: [{sev}:{cat}] '{pat}' -> {snippet}", idx + 1);
                *hits += 1;
            }
        }
    }
}

fn main() {
    let target = env::args().nth(1).unwrap_or_else(|| ".".to_string());
    if target == "-h" || target == "--help" {
        println!("Usage: skill-scan <dir_or_file>");
        return;
    }
    let root = Path::new(&target);
    let mut files = Vec::new();
    collect(root, &mut files);
    let mut hits = 0usize;
    for f in &files {
        scan_file(f, root, &mut hits);
    }
    if hits == 0 {
        println!("CLEAN: {} files scanned.", files.len());
    } else {
        println!("FOUND: {hits} issue(s) across {} files.", files.len());
        process::exit(2);
    }
}
