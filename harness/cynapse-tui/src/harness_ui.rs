//! Modular harness UI components: Tool Pills, /todo Strip, Context Drawer, & Fuzzy @ Picker.

use crate::theme::AppTheme;
use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Clear, Paragraph, Wrap},
    Frame,
};
use std::path::Path;

/// Fast bounded workspace file discovery for `@` fuzzy picker (skips target/.git/llama.cpp).
pub fn fuzzy_file_matches(query: &str, max_items: usize) -> Vec<String> {
    let q = query.to_lowercase();
    let mut out = Vec::new();
    fn walk(dir: &Path, q: &str, out: &mut Vec<String>, depth: usize) {
        if depth > 4 || out.len() >= 120 {
            return;
        }
        let Ok(rd) = std::fs::read_dir(dir) else { return; };
        for entry in rd.flatten() {
            let p = entry.path();
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if name.starts_with('.') || matches!(name, "target" | "node_modules" | "llama.cpp" | "models") {
                continue;
            }
            if p.is_dir() {
                walk(&p, q, out, depth + 1);
            } else {
                let s = p.to_string_lossy().trim_start_matches("./").to_string();
                if q.is_empty() || s.to_lowercase().contains(q) {
                    out.push(s);
                }
            }
        }
    }
    walk(Path::new("."), &q, &mut out, 0);
    out.sort_by_key(|s| (s.len(), s.clone()));
    out.truncate(max_items);
    out
}

/// Collapse verbose tool result blocks into a 1-line status pill unless `expanded` (`Ctrl+O`).
pub fn format_tool_summary(content: &str, expanded: bool) -> String {
    if expanded || (!content.contains("Exit Status:") && !content.contains("Tool `") && content.lines().count() <= 4) {
        return content.to_string();
    }
    let lines = content.lines().count();
    let first = content.lines().next().unwrap_or("").chars().take(72).collect::<String>();
    let ok = !content.contains("Exit Status: exit status: 1") && !content.contains("Error") && !content.contains("failed");
    let badge = if ok { "✓" } else { "✗" };
    format!("[{} {} ({} lines) — Ctrl+O to expand]", badge, first, lines)
}

/// Render pinned `/todo` checklist strip above input bar.
pub fn render_todo_strip(f: &mut Frame, area: Rect, todos: &[(bool, String)], t: AppTheme) {
    let mut items = Vec::new();
    for (idx, (done, text)) in todos.iter().take(3).enumerate() {
        let mark = if *done { "[✓]" } else { "[●]" };
        let style = if *done {
            Style::default().fg(Color::Green)
        } else {
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
        };
        items.push(Line::from(vec![
            Span::styled(format!(" {} {}. ", mark, idx + 1), style),
            Span::styled(text.chars().take(70).collect::<String>(), t.user_text()),
        ]));
    }
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Plain)
        .title(" /todo Plan ")
        .border_style(Style::default().fg(Color::Cyan))
        .style(Style::default().bg(t.card_bg()));
    f.render_widget(Paragraph::new(items).block(block), area);
}

/// Render side-by-side Context & Tool Drawer (`Ctrl+D`).
pub fn render_context_drawer(
    f: &mut Frame,
    area: Rect,
    git_status: &[String],
    last_tool: &str,
    queue_len: usize,
    t: AppTheme,
) {
    let mut lines = vec![
        Line::from(Span::styled(" MODIFIED FILES (git)", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))),
    ];
    if git_status.is_empty() {
        lines.push(Line::from(Span::styled("  (clean workspace)", t.dim_text())));
    } else {
        for g in git_status.iter().take(6) {
            lines.push(Line::from(Span::styled(format!("  {}", g), Style::default().fg(Color::Yellow))));
        }
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        format!(" QUEUE (/queue): {} pending", queue_len),
        Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(" LATEST TOOL OUTPUT", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD))));
    for l in last_tool.lines().take(10) {
        lines.push(Line::from(Span::styled(format!("  {}", l.chars().take(30).collect::<String>()), t.user_text())));
    }
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .title(" Context Drawer [Ctrl+D] ")
        .border_style(Style::default().fg(Color::Cyan))
        .style(Style::default().bg(t.card_bg()));
    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }).block(block), area);
}

/// Render `@` fuzzy file picker popup directly above the prompt input box.
pub fn render_at_picker(f: &mut Frame, input_area: Rect, matches: &[String], selected: usize, t: AppTheme) {
    if matches.is_empty() || input_area.y < 8 {
        return;
    }
    let h = (matches.len() as u16 + 2).min(9);
    let w = input_area.width.min(64);
    let area = Rect::new(input_area.x + 1, input_area.y.saturating_sub(h), w, h);
    f.render_widget(Clear, area);
    let lines: Vec<Line> = matches
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let style = if i == selected {
                Style::default().bg(Color::Blue).fg(Color::White).add_modifier(Modifier::BOLD)
            } else {
                t.user_text()
            };
            Line::from(Span::styled(format!(" @{}", m), style))
        })
        .collect();
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .title(" Files (@ fuzzy — Tab/Enter) ")
        .border_style(Style::default().fg(Color::LightCyan))
        .style(Style::default().bg(t.card_bg()));
    f.render_widget(Paragraph::new(lines).block(block), area);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_summary_collapses_verbose_output() {
        let raw = "Exit Status: exit status: 0\nline1\nline2\nline3\nline4\nline5";
        let collapsed = format_tool_summary(raw, false);
        assert!(collapsed.starts_with("[✓ Exit Status:"));
        assert_eq!(format_tool_summary(raw, true), raw);
    }
}
