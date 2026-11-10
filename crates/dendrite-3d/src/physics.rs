//! 3D Hierarchical coordinate simulation and Fibonacci distribution for Mazzaroth Galaxy.
//!
//! Architecture:
//! - Tier 0: Central Core (The Primary Anchor) at [0, 0, 0]
//! - Tier 1: Primary Categories (Planets) orbiting the Central Core
//! - Tier 2: Category Items (Moons / Version Rings) orbiting their parent Planet
//! - Tier 3: Knowledge Decks (Outer Anchors & Book Particle Rings) orbiting the perimeter

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityType {
    Sun,       // Core Center
    Planet,    // Primary Category Anchors
    Moon,      // Sub-items orbiting their Category Planet
    Asteroid,  // Outer Knowledge Decks and Chapter Dust Rings
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeSubtype {
    SunWord,
    LanguagePlanet,
    FlagshipTranslation,
    SecondaryTranslation,
    CategoryDeck,
    CategoryRingDust,
    CoreDust,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GalaxyNode {
    pub id: String,
    pub name: String,
    pub short_code: String,
    pub entity_type: EntityType,
    pub sub_type: NodeSubtype,
    pub parent_id: Option<String>,
    pub base_radius: f32,
    pub orbit_speed: f32,
    pub phase: f32,
    pub inclination: f32,
    pub pos: [f32; 3],   // Current 3D world position
    pub brightness: f32, // 0.18 .. 1.0
    pub glyph: char,
    pub is_flagship: bool,
}

impl GalaxyNode {
    pub fn new_sun(id: &str, name: &str) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            short_code: "ROOT".to_string(),
            entity_type: EntityType::Sun,
            sub_type: NodeSubtype::SunWord,
            parent_id: None,
            base_radius: 0.0,
            orbit_speed: 0.0,
            phase: 0.0,
            inclination: 0.0,
            pos: [0.0, 0.0, 0.0],
            brightness: 1.0,
            glyph: '@',
            is_flagship: true,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_planet(
        id: &str,
        name: &str,
        short_code: &str,
        base_radius: f32,
        orbit_speed: f32,
        phase: f32,
        inclination: f32,
        brightness: f32,
    ) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            short_code: short_code.to_string(),
            entity_type: EntityType::Planet,
            sub_type: NodeSubtype::LanguagePlanet,
            parent_id: Some("sun_root".to_string()),
            base_radius,
            orbit_speed,
            phase,
            inclination,
            pos: [0.0, 0.0, 0.0],
            brightness: brightness.clamp(0.1, 1.0),
            glyph: 'o',
            is_flagship: true,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_moon(
        id: &str,
        name: &str,
        short_code: &str,
        parent_id: &str,
        base_radius: f32,
        orbit_speed: f32,
        phase: f32,
        brightness: f32,
        is_flagship: bool,
    ) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            short_code: short_code.to_string(),
            entity_type: EntityType::Moon,
            sub_type: if is_flagship {
                NodeSubtype::FlagshipTranslation
            } else {
                NodeSubtype::SecondaryTranslation
            },
            parent_id: Some(parent_id.to_string()),
            base_radius,
            orbit_speed,
            phase,
            inclination: 0.18,
            pos: [0.0, 0.0, 0.0],
            brightness: brightness.clamp(0.1, 1.0),
            glyph: '*',
            is_flagship,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_asteroid(
        id: &str,
        name: &str,
        sub_type: NodeSubtype,
        parent_id: Option<&str>,
        base_radius: f32,
        orbit_speed: f32,
        phase: f32,
        brightness: f32,
    ) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            short_code: "AST".to_string(),
            entity_type: EntityType::Asteroid,
            sub_type,
            parent_id: parent_id.map(|p| p.to_string()),
            base_radius,
            orbit_speed,
            phase,
            inclination: 0.12,
            pos: [0.0, 0.0, 0.0],
            brightness: brightness.clamp(0.1, 1.0),
            glyph: match sub_type {
                NodeSubtype::CategoryDeck => 'o',
                _ => '.',
            },
            is_flagship: false,
        }
    }
}

/// Generate evenly spaced anchor angles using a Fibonacci spiral (Golden Angle).
pub fn fibonacci_sphere(count: usize, radius: f32) -> Vec<[f32; 3]> {
    let golden = std::f32::consts::PI * (1.0 + 5.0_f32.sqrt());
    (0..count)
        .map(|i| {
            let ph = (1.0 - 2.0 * (i as f32 + 0.5) / count.max(1) as f32)
                .clamp(-1.0, 1.0)
                .acos();
            let th = golden * (i as f32);
            [
                th.cos() * ph.sin() * radius,
                ph.cos() * radius,
                th.sin() * ph.sin() * radius,
            ]
        })
        .collect()
}

/// Deterministic pseudo-random 3D scatter offset for data points around an anchor.
pub fn pseudo_scatter(seed: u64, scatter: f32) -> [f32; 3] {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let r1 = ((s >> 32) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0;
    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let r2 = ((s >> 32) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0;
    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let r3 = ((s >> 32) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0;
    [r1 * scatter, r2 * scatter * 0.6, r3 * scatter]
}

#[derive(Debug, Clone, Default)]
pub struct GalaxySystem {
    pub nodes: Vec<GalaxyNode>,
    pub time: f32,
    pub sim_paused: bool,
}

impl GalaxySystem {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            time: 0.0,
            sim_paused: false,
        }
    }

    pub fn add_node(&mut self, node: GalaxyNode) {
        self.nodes.push(node);
    }

    /// Step hierarchical orbital simulation forward by dt seconds.
    pub fn update(&mut self, dt: f32) {
        if self.sim_paused {
            return;
        }
        self.time += dt;

        // Pass 1: Update Central Sun, Primary Language Planets, and Outer Primary Categories
        let mut parent_positions: std::collections::HashMap<String, [f32; 3]> =
            std::collections::HashMap::with_capacity(64);

        for node in &mut self.nodes {
            match node.entity_type {
                EntityType::Sun => {
                    node.pos = [0.0, 0.0, 0.0];
                    parent_positions.insert(node.id.clone(), [0.0, 0.0, 0.0]);
                }
                EntityType::Planet => {
                    let angle = node.phase + self.time * node.orbit_speed;
                    let x = node.base_radius * angle.cos();
                    let z = node.base_radius * angle.sin();
                    let y = (node.base_radius * 0.18) * (angle * 1.5).sin() * node.inclination.sin();
                    let pos = [x, y, z];
                    node.pos = pos;
                    parent_positions.insert(node.id.clone(), pos);
                }
                EntityType::Asteroid if node.parent_id.is_none() => {
                    let angle = node.phase + self.time * node.orbit_speed;
                    let x = node.base_radius * angle.cos();
                    let z = node.base_radius * angle.sin();
                    let y = (node.base_radius * 0.12) * (angle * 1.2).sin() * node.inclination.sin();
                    let pos = [x, y, z];
                    node.pos = pos;
                    parent_positions.insert(node.id.clone(), pos);
                }
                _ => {}
            }
        }

        // Pass 2: Update Translation Moons and Category Particle Rings relative to their parent position
        for node in &mut self.nodes {
            if node.entity_type == EntityType::Moon
                || (node.entity_type == EntityType::Asteroid && node.parent_id.is_some())
            {
                let parent_pos = node
                    .parent_id
                    .as_deref()
                    .and_then(|pid| parent_positions.get(pid).copied())
                    .unwrap_or([0.0, 0.0, 0.0]);

                let angle = node.phase + self.time * node.orbit_speed;
                let rel_x = node.base_radius * angle.cos();
                let rel_z = node.base_radius * angle.sin();
                let rel_y = 0.22 * (angle * 2.0).sin() * node.inclination;

                node.pos = [
                    parent_pos[0] + rel_x,
                    parent_pos[1] + rel_y,
                    parent_pos[2] + rel_z,
                ];
            }
        }
    }
}
