//! Dendrite-3D — Pure Rust Ratatui 3D Celestial Galaxy Visualization & Hierarchical Orbit Engine for Cynapse Memory.

pub mod builder;
pub mod physics;
pub mod renderer;
pub mod schema;
pub mod theme;

// Primary re-exports
pub use builder::GalaxyBuilder;
pub use physics::{
    fibonacci_sphere, pseudo_scatter, EntityType, GalaxyNode, GalaxySystem, NodeSubtype,
};
pub use renderer::{Camera3D, GalaxyRenderer, ProjectedPoint};
pub use schema::{Category, DatabaseSchema, Deck, DustConfig, Item};
pub use theme::{GalaxyTheme, PresetTheme};

#[cfg(test)]
mod tests {
    use super::*;

    struct MockSchema;

    impl DatabaseSchema for MockSchema {
        fn root_name(&self) -> &str {
            "Test Knowledge Star"
        }

        fn categories(&self) -> Vec<Category> {
            vec![
                Category::new("cat_a", "Category Alpha", "ALP"),
                Category::new("cat_b", "Category Beta", "BET"),
            ]
        }

        fn items_for_category(&self, category_id: &str) -> Vec<Item> {
            match category_id {
                "cat_a" => vec![
                    Item::new("item_a1", "Item A1", "A1", true),
                    Item::new("item_a2", "Item A2", "A2", false),
                ],
                _ => vec![],
            }
        }

        fn outer_decks(&self) -> Vec<Deck> {
            vec![Deck::new("deck_1", "Outer Deck 1", "D1")]
        }
    }

    #[test]
    fn test_galaxy_hierarchical_orbit() {
        let mut system = GalaxySystem::new();
        let sun = GalaxyNode::new_sun("sun", "The Word");
        let planet = GalaxyNode::new_planet("earth", "Earth", "ENG", 10.0, 1.0, 0.0, 0.0, 0.85);
        let moon = GalaxyNode::new_moon("moon", "Luna", "LUN", "earth", 2.0, 2.0, 0.0, 0.5, true);

        system.add_node(sun);
        system.add_node(planet);
        system.add_node(moon);

        // Initial positions at time 0
        system.update(0.0);
        assert_eq!(system.nodes[0].pos, [0.0, 0.0, 0.0]);
        assert!((system.nodes[1].pos[0] - 10.0).abs() < 1e-4);
        assert!((system.nodes[2].pos[0] - 12.0).abs() < 1e-4);

        // Step simulation forward by PI/2
        system.update(std::f32::consts::PI / 2.0);
        assert_eq!(system.nodes[0].pos, [0.0, 0.0, 0.0]);
        assert!(system.nodes[1].pos[0].abs() < 0.1);
        assert!((system.nodes[1].pos[2] - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_galaxy_projection_depth() {
        let cam = Camera3D::new();
        let center = GalaxyRenderer::project_point([0.0, 0.0, 0.0], &cam, 100, 50);
        assert!(center.visible);
        assert_eq!(center.screen_x, 50);
        assert_eq!(center.screen_y, 25);
        assert!(center.depth > 0.0);

        let behind = GalaxyRenderer::project_point([0.0, 0.0, -100.0], &cam, 100, 50);
        assert!(!behind.visible || behind.depth < 2.0 || behind.screen_x < 0 || behind.screen_x >= 100);
    }

    #[test]
    fn test_galaxy_builder_from_schema() {
        let schema = MockSchema;
        let system = GalaxyBuilder::build(&schema);

        assert!(!system.nodes.is_empty());
        assert_eq!(system.nodes[0].entity_type, EntityType::Sun);
        assert_eq!(system.nodes[0].name, "Test Knowledge Star");

        let planets: Vec<_> = system.nodes.iter().filter(|n| n.entity_type == EntityType::Planet).collect();
        assert_eq!(planets.len(), 2);

        let moons: Vec<_> = system.nodes.iter().filter(|n| n.entity_type == EntityType::Moon).collect();
        assert_eq!(moons.len(), 2);

        let decks: Vec<_> = system.nodes.iter().filter(|n| n.sub_type == NodeSubtype::CategoryDeck).collect();
        assert_eq!(decks.len(), 1);
    }

    #[test]
    fn test_preset_theme_palette() {
        for theme in [
            PresetTheme::RoyalGold,
            PresetTheme::MonasteryAmber,
            PresetTheme::CyberCyan,
            PresetTheme::EmeraldMint,
            PresetTheme::CelestialIndigo,
        ] {
            let _ = theme.primary();
            let _ = theme.secondary();
            let _ = theme.accent();
            let _ = theme.text();
            let _ = theme.header_title();
            let _ = theme.border_focused();
        }
    }

    #[test]
    fn test_ascii_hierarchy() {
        let sun = GalaxyNode::new_sun("sun", "The Word");
        let planet = GalaxyNode::new_planet("book", "Genesis", "GEN", 10.0, 1.0, 0.0, 0.0, 0.85);
        let moon = GalaxyNode::new_moon("chap", "Chapter 1", "CH1", "book", 2.0, 2.0, 0.0, 0.5, true);
        let verse = GalaxyNode::new_asteroid("verse", "Verse 1", NodeSubtype::CategoryRingDust, Some("chap"), 1.0, 1.0, 0.0, 0.4);

        assert_eq!(sun.glyph, '@');
        assert_eq!(planet.glyph, 'o');
        assert_eq!(moon.glyph, '*');
        assert_eq!(verse.glyph, '.');
    }
}
