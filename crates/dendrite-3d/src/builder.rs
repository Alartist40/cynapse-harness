//! Generic GalaxyBuilder constructing a GalaxySystem from any DatabaseSchema.

use super::physics::{pseudo_scatter, GalaxyNode, GalaxySystem, NodeSubtype};
use super::schema::DatabaseSchema;

pub struct GalaxyBuilder;

impl GalaxyBuilder {
    /// Build and initialize a 3D celestial GalaxySystem from a DatabaseSchema.
    pub fn build(schema: &dyn DatabaseSchema) -> GalaxySystem {
        let mut system = GalaxySystem::new();

        // 1. Central Core (Tier 0 Sun)
        let root_id = schema.root_id();
        system.add_node(GalaxyNode::new_sun(root_id, schema.root_name()));

        // 2. Primary Categories as Planets (Tier 1)
        let categories = schema.categories();
        let num_cats = categories.len().max(1);

        for (c_idx, cat) in categories.iter().enumerate() {
            let base_radius = 3.8 + (c_idx as f32 % 4.0) * 0.45;
            let orbit_speed = 0.078 + ((c_idx as f32) * 0.007) % 0.035;
            let phase = (c_idx as f32 / num_cats as f32) * 2.0 * std::f32::consts::PI;
            let inclination = ((c_idx as f32 * 0.8).sin()) * 0.22;

            system.add_node(GalaxyNode::new_planet(
                &cat.id,
                &cat.name,
                &cat.code,
                base_radius,
                orbit_speed,
                phase,
                inclination,
                0.88,
            ));

            // 3. Category Items as Moons (Tier 2)
            let items = schema.items_for_category(&cat.id);
            let num_items = items.len().max(1);

            for (t_idx, item) in items.iter().enumerate() {
                let m_radius = 0.7 + (t_idx as f32) * 0.22;
                let m_speed = 0.185 + (t_idx as f32) * 0.032;
                let m_phase = (t_idx as f32 / num_items as f32) * 2.0 * std::f32::consts::PI;
                let brightness = if item.is_flagship { 0.72 } else { 0.48 };

                system.add_node(GalaxyNode::new_moon(
                    &item.id,
                    &item.name,
                    &item.code,
                    &cat.id,
                    m_radius,
                    m_speed,
                    m_phase,
                    brightness,
                    item.is_flagship,
                ));
            }
        }

        // 4. Central Core Dust Cloud
        let dust_cfg = schema.core_dust_config();
        let core_glyphs = ['+', '∘', '·', '✦', '*'];

        for g_idx in 0..dust_cfg.count {
            let seed = 0xCAFE_BABE_0000 + (g_idx as u64) * 97 + 13;
            let offset = pseudo_scatter(seed, dust_cfg.scatter);
            let brightness = 0.28 + (g_idx as f32 % 5.0) * 0.048;
            let node_id = format!("core_dust_{}", g_idx);
            let node_name = format!("Core Dust #{}", g_idx + 1);

            let mut node = GalaxyNode::new_asteroid(
                &node_id,
                &node_name,
                NodeSubtype::CoreDust,
                None,
                (offset[0] * offset[0] + offset[2] * offset[2]).sqrt().max(0.3),
                0.042 + (g_idx as f32 % 4.0) * 0.008,
                (g_idx as f32 / dust_cfg.count.max(1) as f32) * 2.0 * std::f32::consts::PI,
                brightness,
            );
            node.pos = offset;
            node.glyph = core_glyphs[g_idx % core_glyphs.len()];
            system.add_node(node);
        }

        // 4b. Ambient Inter-Orbit Dust
        let amb_count = schema.ambient_dust_count();
        let amb_glyphs = ['+', '∘', '·'];

        for d_idx in 0..amb_count {
            let seed = 0xDEAD_BEEF_0000 + (d_idx as u64) * 53 + 7;
            let angle = (d_idx as f32 / amb_count.max(1) as f32) * 2.0 * std::f32::consts::PI;
            let radius = 2.0 + (d_idx as f32 % 6.0) * 0.7;
            let y_jitter = pseudo_scatter(seed, 0.4);
            let pos = [
                radius * angle.cos(),
                y_jitter[1],
                radius * angle.sin(),
            ];
            let brightness = 0.22 + (d_idx as f32 % 4.0) * 0.04;
            let mut node = GalaxyNode::new_asteroid(
                &format!("ambient_dust_{}", d_idx),
                &format!("Ambient Dust #{}", d_idx + 1),
                NodeSubtype::CoreDust,
                None,
                radius,
                0.03 + (d_idx as f32 % 5.0) * 0.006,
                angle,
                brightness,
            );
            node.pos = pos;
            node.glyph = amb_glyphs[d_idx % amb_glyphs.len()];
            system.add_node(node);
        }

        // 5. Outer Knowledge Decks & Ring Clusters (Tier 3)
        let outer_decks = schema.outer_decks();
        let num_decks = outer_decks.len().max(1);
        let ring_glyphs = ['+', '∘', '·'];

        for (c_idx, deck) in outer_decks.iter().enumerate() {
            let cat_radius = 8.0 + (c_idx as f32) * 0.7;
            let cat_speed = 0.038 + (c_idx as f32) * 0.005;
            let cat_phase = (c_idx as f32 / num_decks as f32) * 2.0 * std::f32::consts::PI;

            system.add_node(GalaxyNode::new_asteroid(
                &deck.id,
                &deck.name,
                NodeSubtype::CategoryDeck,
                None,
                cat_radius,
                cat_speed,
                cat_phase,
                0.62,
            ));

            let num_particles = schema.particles_for_deck(&deck.id);
            for p_idx in 0..num_particles {
                let p_id = format!("ring_{}_{}", deck.id, p_idx);
                let p_name = format!("{} Cluster #{}", deck.code, p_idx + 1);
                let p_radius = 0.8 + (p_idx as f32) * 0.18;
                let p_speed = 0.14 + (p_idx as f32) * 0.018;
                let p_phase = (p_idx as f32 / num_particles.max(1) as f32) * 2.0 * std::f32::consts::PI;

                let mut p_node = GalaxyNode::new_asteroid(
                    &p_id,
                    &p_name,
                    NodeSubtype::CategoryRingDust,
                    Some(&deck.id),
                    p_radius,
                    p_speed,
                    p_phase,
                    0.40,
                );
                p_node.glyph = ring_glyphs[p_idx % ring_glyphs.len()];
                system.add_node(p_node);
            }
        }

        // Initialize positions
        system.update(0.0);
        system
    }
}
