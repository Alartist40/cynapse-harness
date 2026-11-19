// Pure stdlib Rust knowledge-graph indexer (graphify-inspired, zero external deps).
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

const SKIP_DIRS: &[&str] = &[
    ".git", "node_modules", "target", "dist", "build", "__pycache__", ".venv", "venv", ".next",
];
const EXTS: &[&str] = &["rs", "py", "js", "ts", "jsx", "tsx", "go", "md", "sh"];
const INDEX_FILE: &str = ".myelin-graph.tsv";

fn should_skip(name: &str) -> bool {
    SKIP_DIRS.contains(&name)
}

fn walk(dir: &Path, root: &Path, out: &mut Vec<(String, usize, Vec<(String, String, usize)>)>) {
    let Ok(entries) = fs::read_dir(dir) else { return };
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|x| x.path())).collect();
    paths.sort();

    for path in paths {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if path.is_dir() {
            if !should_skip(name) {
                walk(&path, root, out);
            }
            continue;
        }
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !EXTS.contains(&ext) {
            continue;
        }
        let Ok(content) = fs::read_to_string(&path) else { continue };
        let rel = path.strip_prefix(root).unwrap_or(&path).to_string_lossy().to_string();
        let lines: Vec<&str> = content.lines().collect();
        let syms = extract_symbols(ext, &lines);
        out.push((rel, lines.len(), syms));
    }
}

fn extract_symbols(ext: &str, lines: &[&str]) -> Vec<(String, String, usize)> {
    let mut syms = Vec::new();
    for (idx, raw) in lines.iter().enumerate() {
        let line_no = idx + 1;
        let t = raw.trim();
        if ext == "md" {
            if t.starts_with('#') {
                let level = t.chars().take_while(|&c| c == '#').count();
                if level <= 3 {
                    let title = t[level..].trim().replace('\t', " ");
                    if !title.is_empty() {
                        syms.push((format!("h{level}"), title, line_no));
                    }
                }
            }
            continue;
        }
        let prefixes = [
            ("fn", "pub fn "), ("fn", "fn "), ("fn", "pub async fn "), ("fn", "async fn "),
            ("struct", "pub struct "), ("struct", "struct "), ("enum", "pub enum "), ("enum", "enum "),
            ("trait", "pub trait "), ("trait", "trait "), ("impl", "impl "),
            ("fn", "def "), ("fn", "async def "), ("class", "class "),
            ("fn", "export function "), ("fn", "export async function "), ("fn", "function "),
            ("class", "export class "), ("type", "export interface "), ("type", "export type "),
            ("fn", "func "), ("type", "type "),
        ];
        for (kind, pat) in prefixes {
            if let Some(rest) = t.strip_prefix(pat) {
                let ident: String = rest
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
                    .collect();
                if !ident.is_empty() {
                    syms.push((kind.to_string(), ident, line_no));
                }
                break;
            }
        }
    }
    syms
}

fn build(root: &Path) -> String {
    let mut files = Vec::new();
    walk(root, root, &mut files);
    let mut tsv = String::new();
    for (rel, line_count, syms) in &files {
        if syms.is_empty() {
            tsv.push_str(&format!("{rel}\t{line_count}\tfile\t-\t0\n"));
        } else {
            for (kind, name, lno) in syms {
                tsv.push_str(&format!("{rel}\t{line_count}\t{kind}\t{name}\t{lno}\n"));
            }
        }
    }
    let idx_path = root.join(INDEX_FILE);
    if let Err(e) = fs::write(&idx_path, &tsv) {
        eprintln!("Failed to write {}: {e}", idx_path.display());
        process::exit(1);
    }
    println!("Indexed {} files -> {}", files.len(), idx_path.display());
    tsv
}

fn load_or_build(root: &Path) -> String {
    let idx_path = root.join(INDEX_FILE);
    fs::read_to_string(&idx_path).unwrap_or_else(|_| build(root))
}

fn cmd_map(root: &Path) {
    let tsv = load_or_build(root);
    let mut current_file = String::new();
    let mut current_lines = String::new();
    let mut syms: Vec<String> = Vec::new();

    let flush = |f: &str, l: &str, s: &[String]| {
        if f.is_empty() { return; }
        let preview: Vec<&str> = s.iter().take(8).map(String::as_str).collect();
        let extra = if s.len() > 8 { format!(" (+{})", s.len() - 8) } else { String::new() };
        println!("{f} ({l}L) -> [{}{extra}]", preview.join(", "));
    };

    for row in tsv.lines() {
        let cols: Vec<&str> = row.split('\t').collect();
        if cols.len() < 5 { continue; }
        if cols[0] != current_file {
            flush(&current_file, &current_lines, &syms);
            current_file = cols[0].to_string();
            current_lines = cols[1].to_string();
            syms.clear();
        }
        if cols[2] != "file" {
            syms.push(format!("{}:{}", cols[3], cols[4]));
        }
    }
    flush(&current_file, &current_lines, &syms);
}

fn cmd_query(root: &Path, term: &str) {
    let tsv = load_or_build(root);
    let q = term.to_lowercase();
    let mut hits = 0usize;
    let mut last_file = String::new();
    for row in tsv.lines() {
        let cols: Vec<&str> = row.split('\t').collect();
        if cols.len() < 5 { continue; }
        if cols[0].to_lowercase().contains(&q) || cols[3].to_lowercase().contains(&q) {
            if cols[0] != last_file {
                println!("{} ({}L)", cols[0], cols[1]);
                last_file = cols[0].to_string();
                hits += 1;
            }
            if cols[2] != "file" {
                println!("  {} {} @ L{}", cols[2], cols[3], cols[4]);
            }
        }
    }
    if hits == 0 {
        println!("No graph matches for '{term}'.");
    }
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        println!("Usage: graph-index <build|map|query> [term] [root_dir]");
        return;
    }
    match args[0].as_str() {
        "build" => {
            let root = Path::new(args.get(1).map(String::as_str).unwrap_or("."));
            build(root);
        }
        "map" => {
            let root = Path::new(args.get(1).map(String::as_str).unwrap_or("."));
            cmd_map(root);
        }
        "query" if args.len() >= 2 => {
            let root = Path::new(args.get(2).map(String::as_str).unwrap_or("."));
            cmd_query(root, &args[1]);
        }
        _ => {
            eprintln!("Invalid arguments. Run with --help.");
            process::exit(1);
        }
    }
}
