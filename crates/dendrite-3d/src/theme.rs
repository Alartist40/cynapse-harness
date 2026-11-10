//! Galaxy theme abstraction and preset color definitions.

use ratatui::style::{Color, Modifier, Style};

/// Trait defining the color palette and typography styles for Galaxy visualization.
pub trait GalaxyTheme {
    fn primary(&self) -> Color;
    fn secondary(&self) -> Color;
    fn accent(&self) -> Color;
    fn text(&self) -> Color;

    fn border_focused(&self) -> Style {
        Style::default().fg(self.primary())
    }

    fn border_normal(&self) -> Style {
        Style::default().fg(Color::Rgb(60, 60, 80))
    }

    fn header_title(&self) -> Style {
        Style::default()
            .fg(self.primary())
            .add_modifier(Modifier::BOLD)
    }

    fn header_badge(&self) -> Style {
        Style::default()
            .fg(self.secondary())
            .add_modifier(Modifier::BOLD)
    }
}

/// Built-in preset themes for standalone use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresetTheme {
    RoyalGold,
    MonasteryAmber,
    CyberCyan,
    EmeraldMint,
    CelestialIndigo,
}

impl GalaxyTheme for PresetTheme {
    fn primary(&self) -> Color {
        match self {
            PresetTheme::RoyalGold => Color::Rgb(255, 215, 0),
            PresetTheme::MonasteryAmber => Color::Rgb(255, 191, 0),
            PresetTheme::CyberCyan => Color::Rgb(0, 229, 255),
            PresetTheme::EmeraldMint => Color::Rgb(0, 255, 127),
            PresetTheme::CelestialIndigo => Color::Rgb(100, 181, 246),
        }
    }

    fn secondary(&self) -> Color {
        match self {
            PresetTheme::RoyalGold => Color::Rgb(177, 74, 237),
            PresetTheme::MonasteryAmber => Color::Rgb(212, 163, 115),
            PresetTheme::CyberCyan => Color::Rgb(255, 0, 127),
            PresetTheme::EmeraldMint => Color::Rgb(0, 150, 70),
            PresetTheme::CelestialIndigo => Color::Rgb(255, 215, 0),
        }
    }

    fn accent(&self) -> Color {
        match self {
            PresetTheme::RoyalGold => Color::Rgb(255, 140, 0),
            PresetTheme::MonasteryAmber => Color::Rgb(220, 100, 60),
            PresetTheme::CyberCyan => Color::Rgb(180, 0, 255),
            PresetTheme::EmeraldMint => Color::Rgb(0, 210, 230),
            PresetTheme::CelestialIndigo => Color::Rgb(186, 104, 200),
        }
    }

    fn text(&self) -> Color {
        match self {
            PresetTheme::RoyalGold => Color::Rgb(240, 235, 255),
            PresetTheme::MonasteryAmber => Color::Rgb(255, 245, 220),
            PresetTheme::CyberCyan => Color::Rgb(220, 245, 255),
            PresetTheme::EmeraldMint => Color::Rgb(220, 255, 220),
            PresetTheme::CelestialIndigo => Color::Rgb(230, 240, 255),
        }
    }
}
