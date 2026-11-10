//! Database schema trait and data transfer structures for generic celestial mapping.

#[derive(Debug, Clone, PartialEq)]
pub struct Category {
    pub id: String,
    pub name: String,
    pub code: String,
}

impl Category {
    pub fn new(id: impl Into<String>, name: impl Into<String>, code: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            code: code.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Item {
    pub id: String,
    pub name: String,
    pub code: String,
    pub is_flagship: bool,
}

impl Item {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        code: impl Into<String>,
        is_flagship: bool,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            code: code.into(),
            is_flagship,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Deck {
    pub id: String,
    pub name: String,
    pub code: String,
}

impl Deck {
    pub fn new(id: impl Into<String>, name: impl Into<String>, code: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            code: code.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DustConfig {
    pub count: usize,
    pub scatter: f32,
}

impl Default for DustConfig {
    fn default() -> Self {
        Self {
            count: 80,
            scatter: 1.5,
        }
    }
}

/// Generic Database / Knowledge Graph abstraction for mapping to a 3D celestial galaxy.
pub trait DatabaseSchema {
    /// Identifier for the central star (Sun).
    fn root_id(&self) -> &str {
        "sun_root"
    }

    /// Display name of the central star.
    fn root_name(&self) -> &str;

    /// Primary category systems orbiting the sun as planets (e.g. Languages, Topics, Modules).
    fn categories(&self) -> Vec<Category>;

    /// Items/translations orbiting a specific parent category planet as moons.
    fn items_for_category(&self, category_id: &str) -> Vec<Item>;

    /// Outer peripheral decks (e.g. Knowledge Categories, Plugins, Libraries).
    fn outer_decks(&self) -> Vec<Deck> {
        Vec::new()
    }

    /// Number of ring particles orbiting each outer deck.
    fn particles_for_deck(&self, _deck_id: &str) -> usize {
        8
    }

    /// Central core dust cloud configuration.
    fn core_dust_config(&self) -> DustConfig {
        DustConfig::default()
    }

    /// Number of ambient inter-orbit dust particles.
    fn ambient_dust_count(&self) -> usize {
        50
    }
}
