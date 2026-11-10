//! Cynapse TUI Visual Theme Manager — Inspired by jcode.
//!
//! Provides color palettes and styling presets for TUI widgets, headers, user prompt,
//! assistant response, thinking blocks, and modals.

use ratatui::style::{Color, Modifier, Style};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AppTheme {
    DarkSlate,
    Cyberpunk,
    AmberCRT,
    EmeraldMatrix,
    VioletSynth,
    Nord,
    Gruvbox,
    Catppuccin,
    OledBlack,
}

impl AppTheme {
    pub const ALL: &'static [AppTheme] = &[
        AppTheme::DarkSlate,
        AppTheme::Cyberpunk,
        AppTheme::AmberCRT,
        AppTheme::EmeraldMatrix,
        AppTheme::VioletSynth,
        AppTheme::Nord,
        AppTheme::Gruvbox,
        AppTheme::Catppuccin,
        AppTheme::OledBlack,
    ];

    pub fn all() -> &'static [AppTheme] {
        Self::ALL
    }

    pub fn name(&self) -> &'static str {
        match self {
            AppTheme::DarkSlate => "Dark Slate",
            AppTheme::Cyberpunk => "Cyberpunk Neon",
            AppTheme::AmberCRT => "Amber CRT",
            AppTheme::EmeraldMatrix => "Emerald Matrix",
            AppTheme::VioletSynth => "Tokyo Night",
            AppTheme::Nord => "Nord Frost",
            AppTheme::Gruvbox => "Gruvbox Retro",
            AppTheme::Catppuccin => "Catppuccin Mocha",
            AppTheme::OledBlack => "OLED Monochrome",
        }
    }

    pub fn next(&self) -> Self {
        let all = Self::ALL;
        let pos = all.iter().position(|t| t == self).unwrap_or(0);
        all[(pos + 1) % all.len()]
    }

    // Header Styles
    pub fn header_title(&self) -> Style {
        match self {
            AppTheme::DarkSlate => Style::default().fg(Color::Rgb(80, 210, 240)).add_modifier(Modifier::BOLD),
            AppTheme::Cyberpunk => Style::default().fg(Color::Rgb(255, 60, 180)).add_modifier(Modifier::BOLD),
            AppTheme::AmberCRT => Style::default().fg(Color::Rgb(255, 191, 0)).add_modifier(Modifier::BOLD),
            AppTheme::EmeraldMatrix => Style::default().fg(Color::Rgb(50, 255, 100)).add_modifier(Modifier::BOLD),
            AppTheme::VioletSynth => Style::default().fg(Color::Rgb(187, 154, 247)).add_modifier(Modifier::BOLD),
            AppTheme::Nord => Style::default().fg(Color::Rgb(136, 192, 208)).add_modifier(Modifier::BOLD),
            AppTheme::Gruvbox => Style::default().fg(Color::Rgb(254, 128, 25)).add_modifier(Modifier::BOLD),
            AppTheme::Catppuccin => Style::default().fg(Color::Rgb(203, 166, 247)).add_modifier(Modifier::BOLD),
            AppTheme::OledBlack => Style::default().fg(Color::Rgb(255, 255, 255)).add_modifier(Modifier::BOLD),
        }
    }

    pub fn active_model(&self) -> Style {
        match self {
            AppTheme::DarkSlate => Style::default().fg(Color::Rgb(255, 215, 100)).add_modifier(Modifier::BOLD),
            AppTheme::Cyberpunk => Style::default().fg(Color::Rgb(255, 240, 120)).add_modifier(Modifier::BOLD),
            AppTheme::AmberCRT => Style::default().fg(Color::Rgb(255, 215, 0)).add_modifier(Modifier::BOLD),
            AppTheme::EmeraldMatrix => Style::default().fg(Color::Rgb(140, 255, 160)).add_modifier(Modifier::BOLD),
            AppTheme::VioletSynth => Style::default().fg(Color::Rgb(224, 175, 104)).add_modifier(Modifier::BOLD),
            AppTheme::Nord => Style::default().fg(Color::Rgb(235, 203, 139)).add_modifier(Modifier::BOLD),
            AppTheme::Gruvbox => Style::default().fg(Color::Rgb(250, 189, 47)).add_modifier(Modifier::BOLD),
            AppTheme::Catppuccin => Style::default().fg(Color::Rgb(249, 226, 175)).add_modifier(Modifier::BOLD),
            AppTheme::OledBlack => Style::default().fg(Color::Rgb(220, 220, 220)).add_modifier(Modifier::BOLD),
        }
    }

    pub fn border_style(&self) -> Style {
        match self {
            AppTheme::DarkSlate => Style::default().fg(Color::Rgb(90, 100, 120)),
            AppTheme::Cyberpunk => Style::default().fg(Color::Rgb(140, 60, 160)),
            AppTheme::AmberCRT => Style::default().fg(Color::Rgb(160, 110, 30)),
            AppTheme::EmeraldMatrix => Style::default().fg(Color::Rgb(40, 130, 60)),
            AppTheme::VioletSynth => Style::default().fg(Color::Rgb(86, 95, 137)),
            AppTheme::Nord => Style::default().fg(Color::Rgb(76, 86, 106)),
            AppTheme::Gruvbox => Style::default().fg(Color::Rgb(102, 92, 84)),
            AppTheme::Catppuccin => Style::default().fg(Color::Rgb(88, 91, 112)),
            AppTheme::OledBlack => Style::default().fg(Color::Rgb(70, 70, 70)),
        }
    }

    pub fn active_border_style(&self) -> Style {
        match self {
            AppTheme::DarkSlate => Style::default().fg(Color::Rgb(80, 210, 240)),
            AppTheme::Cyberpunk => Style::default().fg(Color::Rgb(255, 80, 220)),
            AppTheme::AmberCRT => Style::default().fg(Color::Rgb(255, 191, 0)),
            AppTheme::EmeraldMatrix => Style::default().fg(Color::Rgb(0, 255, 120)),
            AppTheme::VioletSynth => Style::default().fg(Color::Rgb(122, 162, 247)),
            AppTheme::Nord => Style::default().fg(Color::Rgb(143, 188, 187)),
            AppTheme::Gruvbox => Style::default().fg(Color::Rgb(214, 93, 14)),
            AppTheme::Catppuccin => Style::default().fg(Color::Rgb(203, 166, 247)),
            AppTheme::OledBlack => Style::default().fg(Color::Rgb(255, 255, 255)),
        }
    }

    // Role Headers & Text
    pub fn user_header(&self) -> Style {
        match self {
            AppTheme::DarkSlate => Style::default().fg(Color::Rgb(138, 180, 248)).add_modifier(Modifier::BOLD),
            AppTheme::Cyberpunk => Style::default().fg(Color::Rgb(0, 240, 255)).add_modifier(Modifier::BOLD),
            AppTheme::AmberCRT => Style::default().fg(Color::Rgb(255, 215, 0)).add_modifier(Modifier::BOLD),
            AppTheme::EmeraldMatrix => Style::default().fg(Color::Rgb(100, 255, 140)).add_modifier(Modifier::BOLD),
            AppTheme::VioletSynth => Style::default().fg(Color::Rgb(122, 162, 247)).add_modifier(Modifier::BOLD),
            AppTheme::Nord => Style::default().fg(Color::Rgb(129, 161, 193)).add_modifier(Modifier::BOLD),
            AppTheme::Gruvbox => Style::default().fg(Color::Rgb(131, 165, 152)).add_modifier(Modifier::BOLD),
            AppTheme::Catppuccin => Style::default().fg(Color::Rgb(137, 180, 250)).add_modifier(Modifier::BOLD),
            AppTheme::OledBlack => Style::default().fg(Color::Rgb(200, 220, 255)).add_modifier(Modifier::BOLD),
        }
    }

    pub fn user_text(&self) -> Style {
        match self {
            AppTheme::DarkSlate => Style::default().fg(Color::Rgb(245, 245, 255)),
            AppTheme::Cyberpunk => Style::default().fg(Color::Rgb(230, 250, 255)),
            AppTheme::AmberCRT => Style::default().fg(Color::Rgb(255, 240, 200)),
            AppTheme::EmeraldMatrix => Style::default().fg(Color::Rgb(220, 255, 220)),
            AppTheme::VioletSynth => Style::default().fg(Color::Rgb(192, 202, 245)),
            AppTheme::Nord => Style::default().fg(Color::Rgb(236, 239, 244)),
            AppTheme::Gruvbox => Style::default().fg(Color::Rgb(235, 219, 178)),
            AppTheme::Catppuccin => Style::default().fg(Color::Rgb(205, 214, 244)),
            AppTheme::OledBlack => Style::default().fg(Color::Rgb(255, 255, 255)),
        }
    }

    pub fn assistant_header(&self) -> Style {
        match self {
            AppTheme::DarkSlate => Style::default().fg(Color::Rgb(129, 199, 132)).add_modifier(Modifier::BOLD),
            AppTheme::Cyberpunk => Style::default().fg(Color::Rgb(80, 255, 130)).add_modifier(Modifier::BOLD),
            AppTheme::AmberCRT => Style::default().fg(Color::Rgb(255, 180, 0)).add_modifier(Modifier::BOLD),
            AppTheme::EmeraldMatrix => Style::default().fg(Color::Rgb(50, 240, 90)).add_modifier(Modifier::BOLD),
            AppTheme::VioletSynth => Style::default().fg(Color::Rgb(158, 206, 106)).add_modifier(Modifier::BOLD),
            AppTheme::Nord => Style::default().fg(Color::Rgb(163, 190, 140)).add_modifier(Modifier::BOLD),
            AppTheme::Gruvbox => Style::default().fg(Color::Rgb(184, 187, 38)).add_modifier(Modifier::BOLD),
            AppTheme::Catppuccin => Style::default().fg(Color::Rgb(166, 227, 161)).add_modifier(Modifier::BOLD),
            AppTheme::OledBlack => Style::default().fg(Color::Rgb(160, 240, 160)).add_modifier(Modifier::BOLD),
        }
    }

    pub fn assistant_text(&self) -> Style {
        match self {
            AppTheme::DarkSlate => Style::default().fg(Color::Rgb(225, 228, 238)),
            AppTheme::Cyberpunk => Style::default().fg(Color::Rgb(240, 245, 255)),
            AppTheme::AmberCRT => Style::default().fg(Color::Rgb(255, 225, 160)),
            AppTheme::EmeraldMatrix => Style::default().fg(Color::Rgb(200, 255, 200)),
            AppTheme::VioletSynth => Style::default().fg(Color::Rgb(207, 216, 235)),
            AppTheme::Nord => Style::default().fg(Color::Rgb(229, 233, 240)),
            AppTheme::Gruvbox => Style::default().fg(Color::Rgb(213, 196, 161)),
            AppTheme::Catppuccin => Style::default().fg(Color::Rgb(186, 194, 222)),
            AppTheme::OledBlack => Style::default().fg(Color::Rgb(240, 240, 240)),
        }
    }

    pub fn thinking_header(&self) -> Style {
        match self {
            AppTheme::DarkSlate => Style::default().fg(Color::Rgb(186, 139, 255)).add_modifier(Modifier::ITALIC),
            AppTheme::Cyberpunk => Style::default().fg(Color::Rgb(255, 80, 180)).add_modifier(Modifier::ITALIC),
            AppTheme::AmberCRT => Style::default().fg(Color::Rgb(230, 150, 20)).add_modifier(Modifier::ITALIC),
            AppTheme::EmeraldMatrix => Style::default().fg(Color::Rgb(80, 200, 120)).add_modifier(Modifier::ITALIC),
            AppTheme::VioletSynth => Style::default().fg(Color::Rgb(187, 154, 247)).add_modifier(Modifier::ITALIC),
            AppTheme::Nord => Style::default().fg(Color::Rgb(180, 142, 173)).add_modifier(Modifier::ITALIC),
            AppTheme::Gruvbox => Style::default().fg(Color::Rgb(211, 134, 155)).add_modifier(Modifier::ITALIC),
            AppTheme::Catppuccin => Style::default().fg(Color::Rgb(245, 194, 231)).add_modifier(Modifier::ITALIC),
            AppTheme::OledBlack => Style::default().fg(Color::Rgb(150, 150, 150)).add_modifier(Modifier::ITALIC),
        }
    }

    pub fn thinking_text(&self) -> Style {
        match self {
            AppTheme::DarkSlate => Style::default().fg(Color::Rgb(175, 160, 210)),
            AppTheme::Cyberpunk => Style::default().fg(Color::Rgb(255, 140, 210)),
            AppTheme::AmberCRT => Style::default().fg(Color::Rgb(210, 140, 30)),
            AppTheme::EmeraldMatrix => Style::default().fg(Color::Rgb(100, 190, 110)),
            AppTheme::VioletSynth => Style::default().fg(Color::Rgb(157, 140, 200)),
            AppTheme::Nord => Style::default().fg(Color::Rgb(150, 155, 175)),
            AppTheme::Gruvbox => Style::default().fg(Color::Rgb(168, 153, 132)),
            AppTheme::Catppuccin => Style::default().fg(Color::Rgb(166, 173, 200)),
            AppTheme::OledBlack => Style::default().fg(Color::Rgb(130, 130, 130)),
        }
    }

    pub fn system_text(&self) -> Style {
        match self {
            AppTheme::DarkSlate => Style::default().fg(Color::Rgb(255, 200, 100)),
            AppTheme::Cyberpunk => Style::default().fg(Color::Rgb(255, 220, 80)),
            AppTheme::AmberCRT => Style::default().fg(Color::Rgb(255, 200, 50)),
            AppTheme::EmeraldMatrix => Style::default().fg(Color::Rgb(160, 240, 120)),
            AppTheme::VioletSynth => Style::default().fg(Color::Rgb(224, 175, 104)),
            AppTheme::Nord => Style::default().fg(Color::Rgb(235, 203, 139)),
            AppTheme::Gruvbox => Style::default().fg(Color::Rgb(250, 189, 47)),
            AppTheme::Catppuccin => Style::default().fg(Color::Rgb(250, 179, 135)),
            AppTheme::OledBlack => Style::default().fg(Color::Rgb(220, 200, 120)),
        }
    }

    pub fn error_text(&self) -> Style {
        Style::default().fg(Color::Rgb(255, 100, 100)).add_modifier(Modifier::BOLD)
    }

    // Input Bar
    pub fn prompt_prefix(&self) -> Style {
        match self {
            AppTheme::DarkSlate => Style::default().fg(Color::Rgb(80, 210, 240)).add_modifier(Modifier::BOLD),
            AppTheme::Cyberpunk => Style::default().fg(Color::Rgb(255, 60, 180)).add_modifier(Modifier::BOLD),
            AppTheme::AmberCRT => Style::default().fg(Color::Rgb(255, 191, 0)).add_modifier(Modifier::BOLD),
            AppTheme::EmeraldMatrix => Style::default().fg(Color::Rgb(50, 255, 100)).add_modifier(Modifier::BOLD),
            AppTheme::VioletSynth => Style::default().fg(Color::Rgb(187, 154, 247)).add_modifier(Modifier::BOLD),
            AppTheme::Nord => Style::default().fg(Color::Rgb(136, 192, 208)).add_modifier(Modifier::BOLD),
            AppTheme::Gruvbox => Style::default().fg(Color::Rgb(254, 128, 25)).add_modifier(Modifier::BOLD),
            AppTheme::Catppuccin => Style::default().fg(Color::Rgb(203, 166, 247)).add_modifier(Modifier::BOLD),
            AppTheme::OledBlack => Style::default().fg(Color::Rgb(255, 255, 255)).add_modifier(Modifier::BOLD),
        }
    }

    // High-Contrast Autocomplete & List Selection Highlight
    pub fn highlight_item(&self) -> Style {
        match self {
            AppTheme::DarkSlate => Style::default().fg(Color::White).bg(Color::Rgb(40, 75, 140)).add_modifier(Modifier::BOLD),
            AppTheme::Cyberpunk => Style::default().fg(Color::White).bg(Color::Rgb(140, 30, 130)).add_modifier(Modifier::BOLD),
            AppTheme::AmberCRT => Style::default().fg(Color::Black).bg(Color::Rgb(255, 191, 0)).add_modifier(Modifier::BOLD),
            AppTheme::EmeraldMatrix => Style::default().fg(Color::Black).bg(Color::Rgb(50, 255, 100)).add_modifier(Modifier::BOLD),
            AppTheme::VioletSynth => Style::default().fg(Color::White).bg(Color::Rgb(104, 80, 170)).add_modifier(Modifier::BOLD),
            AppTheme::Nord => Style::default().fg(Color::Black).bg(Color::Rgb(136, 192, 208)).add_modifier(Modifier::BOLD),
            AppTheme::Gruvbox => Style::default().fg(Color::Black).bg(Color::Rgb(250, 189, 47)).add_modifier(Modifier::BOLD),
            AppTheme::Catppuccin => Style::default().fg(Color::Black).bg(Color::Rgb(203, 166, 247)).add_modifier(Modifier::BOLD),
            AppTheme::OledBlack => Style::default().fg(Color::Black).bg(Color::Rgb(240, 240, 240)).add_modifier(Modifier::BOLD),
        }
    }

    /// Solid background color for modal dialogs to prevent transparent terminal bleeding
    pub fn modal_bg(&self) -> Color {
        match self {
            AppTheme::DarkSlate => Color::Rgb(15, 18, 26),
            AppTheme::Cyberpunk => Color::Rgb(20, 10, 25),
            AppTheme::AmberCRT => Color::Rgb(18, 14, 5),
            AppTheme::EmeraldMatrix => Color::Rgb(5, 20, 10),
            AppTheme::VioletSynth => Color::Rgb(26, 27, 38),
            AppTheme::Nord => Color::Rgb(46, 52, 64),
            AppTheme::Gruvbox => Color::Rgb(40, 40, 40),
            AppTheme::Catppuccin => Color::Rgb(30, 30, 46),
            AppTheme::OledBlack => Color::Rgb(10, 10, 10),
        }
    }

    /// Solid elevated card surface background color for inner dashboard panels
    pub fn card_bg(&self) -> Color {
        match self {
            AppTheme::DarkSlate => Color::Rgb(22, 27, 38),
            AppTheme::Cyberpunk => Color::Rgb(32, 16, 42),
            AppTheme::AmberCRT => Color::Rgb(28, 22, 8),
            AppTheme::EmeraldMatrix => Color::Rgb(8, 28, 15),
            AppTheme::VioletSynth => Color::Rgb(36, 40, 59),
            AppTheme::Nord => Color::Rgb(59, 66, 82),
            AppTheme::Gruvbox => Color::Rgb(50, 48, 47),
            AppTheme::Catppuccin => Color::Rgb(49, 50, 68),
            AppTheme::OledBlack => Color::Rgb(20, 20, 20),
        }
    }

    // Secondary / Dim Text (High contrast, crisp & readable on all terminals)
    pub fn dim_text(&self) -> Style {
        match self {
            AppTheme::DarkSlate => Style::default().fg(Color::Rgb(180, 195, 215)),
            AppTheme::Cyberpunk => Style::default().fg(Color::Rgb(215, 195, 235)),
            AppTheme::AmberCRT => Style::default().fg(Color::Rgb(240, 195, 110)),
            AppTheme::EmeraldMatrix => Style::default().fg(Color::Rgb(165, 230, 175)),
            AppTheme::VioletSynth => Style::default().fg(Color::Rgb(180, 185, 220)),
            AppTheme::Nord => Style::default().fg(Color::Rgb(185, 195, 215)),
            AppTheme::Gruvbox => Style::default().fg(Color::Rgb(215, 195, 165)),
            AppTheme::Catppuccin => Style::default().fg(Color::Rgb(195, 200, 225)),
            AppTheme::OledBlack => Style::default().fg(Color::Rgb(200, 200, 200)),
        }
    }

    /// Theme-adaptive character styling for ASCII art
    pub fn ascii_char_style(&self, ch: char) -> Style {
        match self {
            AppTheme::DarkSlate => match ch {
                '@' | '%' => Style::default().fg(Color::Rgb(100, 230, 255)).add_modifier(Modifier::BOLD),
                '#' | '*' => Style::default().fg(Color::Rgb(130, 180, 255)),
                '+' | '=' => Style::default().fg(Color::Rgb(80, 190, 210)),
                '-' | ':' => Style::default().fg(Color::Rgb(100, 130, 160)),
                '.' => Style::default().fg(Color::Rgb(70, 90, 115)),
                _ => Style::default().fg(Color::Rgb(60, 75, 95)),
            },
            AppTheme::Cyberpunk => match ch {
                '@' | '%' => Style::default().fg(Color::Rgb(255, 60, 180)).add_modifier(Modifier::BOLD),
                '#' | '*' => Style::default().fg(Color::Rgb(255, 110, 220)),
                '+' | '=' => Style::default().fg(Color::Rgb(0, 240, 255)),
                '-' | ':' => Style::default().fg(Color::Rgb(170, 80, 220)),
                '.' => Style::default().fg(Color::Rgb(100, 50, 130)),
                _ => Style::default().fg(Color::Rgb(70, 35, 90)),
            },
            AppTheme::AmberCRT => match ch {
                '@' | '%' => Style::default().fg(Color::Rgb(255, 220, 60)).add_modifier(Modifier::BOLD),
                '#' | '*' => Style::default().fg(Color::Rgb(255, 191, 0)),
                '+' | '=' => Style::default().fg(Color::Rgb(255, 150, 0)),
                '-' | ':' => Style::default().fg(Color::Rgb(190, 110, 0)),
                '.' => Style::default().fg(Color::Rgb(110, 70, 10)),
                _ => Style::default().fg(Color::Rgb(70, 45, 10)),
            },
            AppTheme::EmeraldMatrix => match ch {
                '@' | '%' => Style::default().fg(Color::Rgb(80, 255, 130)).add_modifier(Modifier::BOLD),
                '#' | '*' => Style::default().fg(Color::Rgb(0, 255, 100)),
                '+' | '=' => Style::default().fg(Color::Rgb(30, 200, 80)),
                '-' | ':' => Style::default().fg(Color::Rgb(20, 130, 50)),
                '.' => Style::default().fg(Color::Rgb(15, 80, 30)),
                _ => Style::default().fg(Color::Rgb(10, 50, 20)),
            },
            AppTheme::VioletSynth => match ch {
                '@' | '%' => Style::default().fg(Color::Rgb(210, 160, 255)).add_modifier(Modifier::BOLD),
                '#' | '*' => Style::default().fg(Color::Rgb(187, 154, 247)),
                '+' | '=' => Style::default().fg(Color::Rgb(122, 162, 247)),
                '-' | ':' => Style::default().fg(Color::Rgb(100, 110, 160)),
                '.' => Style::default().fg(Color::Rgb(65, 75, 110)),
                _ => Style::default().fg(Color::Rgb(45, 50, 75)),
            },
            AppTheme::Nord => match ch {
                '@' | '%' => Style::default().fg(Color::Rgb(143, 188, 187)).add_modifier(Modifier::BOLD),
                '#' | '*' => Style::default().fg(Color::Rgb(136, 192, 208)),
                '+' | '=' => Style::default().fg(Color::Rgb(129, 161, 193)),
                '-' | ':' => Style::default().fg(Color::Rgb(94, 129, 172)),
                '.' => Style::default().fg(Color::Rgb(76, 86, 106)),
                _ => Style::default().fg(Color::Rgb(59, 66, 82)),
            },
            AppTheme::Gruvbox => match ch {
                '@' | '%' => Style::default().fg(Color::Rgb(254, 128, 25)).add_modifier(Modifier::BOLD),
                '#' | '*' => Style::default().fg(Color::Rgb(250, 189, 47)),
                '+' | '=' => Style::default().fg(Color::Rgb(184, 187, 38)),
                '-' | ':' => Style::default().fg(Color::Rgb(142, 192, 124)),
                '.' => Style::default().fg(Color::Rgb(102, 92, 84)),
                _ => Style::default().fg(Color::Rgb(60, 56, 54)),
            },
            AppTheme::Catppuccin => match ch {
                '@' | '%' => Style::default().fg(Color::Rgb(203, 166, 247)).add_modifier(Modifier::BOLD),
                '#' | '*' => Style::default().fg(Color::Rgb(245, 194, 231)),
                '+' | '=' => Style::default().fg(Color::Rgb(137, 180, 250)),
                '-' | ':' => Style::default().fg(Color::Rgb(116, 199, 236)),
                '.' => Style::default().fg(Color::Rgb(88, 91, 112)),
                _ => Style::default().fg(Color::Rgb(49, 50, 68)),
            },
            AppTheme::OledBlack => match ch {
                '@' | '%' => Style::default().fg(Color::Rgb(255, 255, 255)).add_modifier(Modifier::BOLD),
                '#' | '*' => Style::default().fg(Color::Rgb(200, 200, 200)),
                '+' | '=' => Style::default().fg(Color::Rgb(160, 160, 160)),
                '-' | ':' => Style::default().fg(Color::Rgb(120, 120, 120)),
                '.' => Style::default().fg(Color::Rgb(80, 80, 80)),
                _ => Style::default().fg(Color::Rgb(40, 40, 40)),
            },
        }
    }
}

pub fn settings_path() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".cynapse").join("settings.json"))
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct UserSettings {
    theme: Option<AppTheme>,
}

/// Loads the saved theme from ~/.cynapse/settings.json, defaulting to AmberCRT.
pub fn load_theme() -> AppTheme {
    if let Some(path) = settings_path() {
        if path.exists() {
            if let Ok(content) = std::fs::read_to_string(&path) {
                if let Ok(settings) = serde_json::from_str::<UserSettings>(&content) {
                    if let Some(t) = settings.theme {
                        return t;
                    }
                }
            }
        }
    }
    AppTheme::AmberCRT
}

/// Persists the selected theme to ~/.cynapse/settings.json.
pub fn save_theme(theme: AppTheme) {
    if let Some(path) = settings_path() {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let mut settings = if path.exists() {
            std::fs::read_to_string(&path)
                .ok()
                .and_then(|c| serde_json::from_str::<UserSettings>(&c).ok())
                .unwrap_or_default()
        } else {
            UserSettings::default()
        };
        settings.theme = Some(theme);
        if let Ok(json) = serde_json::to_string_pretty(&settings) {
            let _ = std::fs::write(&path, json);
        }
    }
}
