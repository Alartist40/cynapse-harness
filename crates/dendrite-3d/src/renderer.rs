//! 3D Perspective Projection and Distance-Alpha Renderer for Mazzaroth Galaxy.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};

use super::physics::{EntityType, GalaxyNode, GalaxySystem, NodeSubtype};
use super::theme::GalaxyTheme;

#[derive(Debug, Clone)]
pub struct Camera3D {
    pub yaw: f32,       // Horizontal rotation in radians
    pub pitch: f32,     // Vertical rotation in radians
    pub distance: f32,  // Distance from origin (Perspective focal reference)
    pub fov: f32,       // Field of view scale
    pub auto_spin: bool,
}

impl Default for Camera3D {
    fn default() -> Self {
        Self::new()
    }
}

impl Camera3D {
    pub fn new() -> Self {
        Self {
            yaw: 0.5,
            pitch: 0.35,
            distance: 32.0,
            fov: 1.45,
            auto_spin: true,
        }
    }

    pub fn rotate_yaw(&mut self, delta: f32) {
        self.yaw = (self.yaw + delta) % (2.0 * std::f32::consts::PI);
    }

    pub fn rotate_pitch(&mut self, delta: f32) {
        let max_pitch = 1.45;
        self.pitch = (self.pitch + delta).clamp(-max_pitch, max_pitch);
    }

    pub fn zoom(&mut self, delta: f32) {
        self.distance = (self.distance + delta).clamp(12.0, 90.0);
    }

    pub fn reset(&mut self) {
        self.yaw = 0.5;
        self.pitch = 0.35;
        self.distance = 32.0;
        self.fov = 1.45;
    }
}

pub struct ProjectedPoint {
    pub screen_x: i32,
    pub screen_y: i32,
    pub depth: f32,
    pub f: f32, // Perspective scale factor
    pub visible: bool,
}

pub struct GalaxyRenderer;

fn color_to_rgb(c: Color) -> (u8, u8, u8) {
    match c {
        Color::Rgb(r, g, b) => (r, g, b),
        _ => (240, 236, 227),
    }
}

impl GalaxyRenderer {
    /// Project a 3D world coordinate to 2D screen coordinate using camera rotation,
    /// perspective foreshortening, and horizontal stretch XS=1.62 for galaxy disk shape.
    pub fn project_point(
        point: [f32; 3],
        cam: &Camera3D,
        width: u16,
        height: u16,
    ) -> ProjectedPoint {
        let x = point[0];
        let y = point[1];
        let z = point[2];

        // 1. Rotate around Y (Yaw)
        let cos_y = cam.yaw.cos();
        let sin_y = cam.yaw.sin();
        let x1 = x * cos_y + z * sin_y;
        let z1 = -x * sin_y + z * cos_y;
        let y1 = y;

        // 2. Rotate around X (Pitch)
        let cos_p = cam.pitch.cos();
        let sin_p = cam.pitch.sin();
        let y2 = y1 * cos_p - z1 * sin_p;
        let z2 = y1 * sin_p + z1 * cos_p;
        let x2 = x1;

        // 3. Camera distance along Z
        let cam_z = z2 + cam.distance;
        if cam_z <= 1.0 {
            return ProjectedPoint {
                screen_x: -1,
                screen_y: -1,
                depth: cam_z,
                f: 0.0,
                visible: false,
            };
        }

        // Perspective factor (normalized around origin)
        let f = cam.distance / cam_z;

        // 4. Perspective projection accounting for terminal character 2:1 aspect ratio
        // and horizontal stretch XS=1.62 to shape the spherical distribution into a wide galaxy disk
        let half_w = width as f32 / 2.0;
        let half_h = height as f32 / 2.0;
        let aspect = 2.0; // Character height vs width compensation
        let xs = 1.62;   // Golden ratio horizontal galaxy disk stretch factor

        let proj_x = (x2 / cam_z) * cam.fov * aspect * xs * half_w;
        let proj_y = (y2 / cam_z) * cam.fov * half_h;

        let screen_x = (half_w + proj_x).round() as i32;
        let screen_y = (half_h - proj_y).round() as i32;

        let visible = screen_x >= 0
            && screen_x < width as i32
            && screen_y >= 0
            && screen_y < height as i32;

        ProjectedPoint {
            screen_x,
            screen_y,
            depth: cam_z,
            f,
            visible,
        }
    }

    /// Render the galaxy system with theme-aware colors, distance-based alpha fading, and front-hemisphere gating.
    pub fn render_galaxy(
        system: &GalaxySystem,
        cam: &Camera3D,
        selected_node_id: Option<&str>,
        area: Rect,
        buf: &mut Buffer,
        theme: &dyn GalaxyTheme,
    ) {
        if area.width < 10 || area.height < 5 {
            return;
        }

        let width = area.width;
        let height = area.height;

        let c_primary = color_to_rgb(theme.primary());
        let c_secondary = color_to_rgb(theme.secondary());
        let c_accent = color_to_rgb(theme.accent());
        let c_text = color_to_rgb(theme.text());

        // 1. Background Cosmic Starfield with Parallax (Dense 350-star field throughout galaxy volume)
        let star_anchors = super::physics::fibonacci_sphere(350, 18.0);
        let star_glyphs = ['+', '∘', '·', '✦', '*', '+', '∘'];

        for (i, &anchor) in star_anchors.iter().enumerate() {
            let offset = super::physics::pseudo_scatter(0x51A8_0000 + (i as u64) * 31, 2.0);
            let star_pos = [
                anchor[0] + offset[0],
                anchor[1] + offset[1] * 0.5,
                anchor[2] + offset[2],
            ];
            let glyph = star_glyphs[i % star_glyphs.len()];
            let brightness = 0.18 + ((i as f32 * 0.17).sin().abs() * 0.34);

            let proj = Self::project_point(star_pos, cam, width, height);
            if proj.visible {
                let alpha = ((proj.f - 0.35) * 1.2).clamp(0.18, 1.0) * brightness;
                let r = (c_text.0 as f32 * alpha).min(255.0) as u8;
                let g = (c_text.1 as f32 * alpha).min(255.0) as u8;
                let b = (c_text.2 as f32 * alpha).min(255.0) as u8;

                let gx = area.left() + proj.screen_x as u16;
                let gy = area.top() + proj.screen_y as u16;
                if let Some(cell) = buf.cell_mut((gx, gy)) {
                    cell.set_char(glyph)
                        .set_style(Style::default().fg(Color::Rgb(r, g, b)));
                }
            }
        }

        // 2. Project and Collect Celestial Nodes
        struct RenderNode<'a> {
            node: &'a GalaxyNode,
            proj: ProjectedPoint,
            is_selected: bool,
        }

        let mut render_nodes = Vec::new();

        for node in &system.nodes {
            let proj = Self::project_point(node.pos, cam, width, height);
            if !proj.visible {
                continue;
            }

            let is_selected = selected_node_id.map(|id| id == node.id).unwrap_or(false);
            render_nodes.push(RenderNode {
                node,
                proj,
                is_selected,
            });
        }

        // Sort back-to-front (Painter's Algorithm by depth descending)
        render_nodes.sort_by(|a, b| {
            b.proj.depth.partial_cmp(&a.proj.depth).unwrap_or(std::cmp::Ordering::Equal)
        });

        // 3. Render Nodes with Theme-Aware Color Mapping & Distance-Based Alpha Fading
        for rn in &render_nodes {
            let gx = area.left() + rn.proj.screen_x as u16;
            let gy = area.top() + rn.proj.screen_y as u16;

            // Calibrated alpha formula with high-visibility floor: alpha = max(0.18, (f - 0.35) * 1.2) * brightness
            let alpha = ((rn.proj.f - 0.35) * 1.2).clamp(0.18, 1.0) * rn.node.brightness;

            let (r, g, b) = if rn.is_selected {
                (255, 255, 255)
            } else {
                match rn.node.sub_type {
                    NodeSubtype::SunWord => {
                        let eff = alpha.max(0.70);
                        (
                            (c_primary.0 as f32 * eff).min(255.0) as u8,
                            (c_primary.1 as f32 * eff).min(255.0) as u8,
                            (c_primary.2 as f32 * eff).min(255.0) as u8,
                        )
                    }
                    NodeSubtype::LanguagePlanet => {
                        let eff = alpha.max(0.45);
                        (
                            (c_secondary.0 as f32 * eff).min(255.0) as u8,
                            (c_secondary.1 as f32 * eff).min(255.0) as u8,
                            (c_secondary.2 as f32 * eff).min(255.0) as u8,
                        )
                    }
                    NodeSubtype::FlagshipTranslation => {
                        let eff = alpha.max(0.35);
                        (
                            (c_primary.0 as f32 * eff).min(255.0) as u8,
                            (c_primary.1 as f32 * eff).min(255.0) as u8,
                            (c_primary.2 as f32 * eff).min(255.0) as u8,
                        )
                    }
                    NodeSubtype::SecondaryTranslation => {
                        let eff = alpha.max(0.30);
                        (
                            (c_secondary.0 as f32 * eff).min(255.0) as u8,
                            (c_secondary.1 as f32 * eff).min(255.0) as u8,
                            (c_secondary.2 as f32 * eff).min(255.0) as u8,
                        )
                    }
                    NodeSubtype::CategoryDeck => {
                        let eff = alpha.max(0.40);
                        (
                            (c_accent.0 as f32 * eff).min(255.0) as u8,
                            (c_accent.1 as f32 * eff).min(255.0) as u8,
                            (c_accent.2 as f32 * eff).min(255.0) as u8,
                        )
                    }
                    NodeSubtype::CategoryRingDust => {
                        let eff = alpha.max(0.25);
                        (
                            (c_accent.0 as f32 * eff).min(255.0) as u8,
                            (c_accent.1 as f32 * eff).min(255.0) as u8,
                            (c_accent.2 as f32 * eff).min(255.0) as u8,
                        )
                    }
                    NodeSubtype::CoreDust => {
                        let eff = alpha.max(0.22);
                        (
                            (c_primary.0 as f32 * eff).min(255.0) as u8,
                            (c_primary.1 as f32 * eff).min(255.0) as u8,
                            (c_primary.2 as f32 * eff).min(255.0) as u8,
                        )
                    }
                }
            };

            let mut style = Style::default().fg(Color::Rgb(r, g, b));

            if rn.is_selected {
                style = style.add_modifier(Modifier::BOLD | Modifier::REVERSED);
            } else if rn.node.entity_type == EntityType::Sun || rn.node.is_flagship {
                style = style.add_modifier(Modifier::BOLD);
            }

            // Tri-glyph Core for Central Sun (@ at center)
            if rn.node.entity_type == EntityType::Sun {
                if gx > area.left() {
                    if let Some(left_cell) = buf.cell_mut((gx - 1, gy)) {
                        left_cell.set_char('[').set_style(
                            Style::default().fg(theme.primary()).add_modifier(Modifier::BOLD),
                        );
                    }
                }
                if let Some(cell) = buf.cell_mut((gx, gy)) {
                    cell.set_char('@').set_style(
                        Style::default().fg(Color::Rgb(255, 255, 255)).add_modifier(Modifier::BOLD),
                    );
                }
                if gx + 1 < area.right() {
                    if let Some(right_cell) = buf.cell_mut((gx + 1, gy)) {
                        right_cell.set_char(']').set_style(
                            Style::default().fg(theme.primary()).add_modifier(Modifier::BOLD),
                        );
                    }
                }
            } else if let Some(cell) = buf.cell_mut((gx, gy)) {
                cell.set_char(rn.node.glyph).set_style(style);
            }
        }

        // 4. Render Topic & Category Labels (Front-Hemisphere Gated)
        for rn in &render_nodes {
            let gx = area.left() + rn.proj.screen_x as u16;
            let gy = area.top() + rn.proj.screen_y as u16;

            // Only show labels for:
            // - Selected node (always)
            // - Sun when in front hemisphere (f > 0.65)
            // - Primary Category planets when front-facing (f > 0.88)
            let show_label = rn.is_selected
                || (rn.node.entity_type == EntityType::Sun && rn.proj.f > 0.65)
                || (rn.node.entity_type == EntityType::Planet && rn.proj.f > 0.88);

            if show_label {
                let label = if rn.is_selected {
                    format!(" [ {} ] ", rn.node.name)
                } else if rn.node.entity_type == EntityType::Sun {
                    format!(" ✦ {} ", rn.node.name)
                } else {
                    format!(" {}", rn.node.short_code)
                };

                let label_chars: Vec<char> = label.chars().collect();
                let label_start_x = if rn.node.entity_type == EntityType::Sun { gx + 2 } else { gx + 1 };

                for (c_idx, &ch) in label_chars.iter().enumerate() {
                    let lx = label_start_x + c_idx as u16;
                    if lx < area.right() {
                        let l_style = if rn.is_selected {
                            Style::default()
                                .fg(Color::Black)
                                .bg(theme.primary())
                                .add_modifier(Modifier::BOLD)
                        } else if rn.node.entity_type == EntityType::Sun {
                            Style::default()
                                .fg(theme.primary())
                                .add_modifier(Modifier::BOLD)
                        } else if rn.node.entity_type == EntityType::Planet {
                            Style::default()
                                .fg(theme.secondary())
                                .add_modifier(Modifier::BOLD)
                        } else {
                            let l_alpha = ((rn.proj.f - 0.35) * 1.2).clamp(0.18, 1.0);
                            let lr = (c_text.0 as f32 * l_alpha).min(255.0) as u8;
                            let lg = (c_text.1 as f32 * l_alpha).min(255.0) as u8;
                            let lb = (c_text.2 as f32 * l_alpha).min(255.0) as u8;
                            Style::default().fg(Color::Rgb(lr, lg, lb))
                        };

                        if let Some(cell) = buf.cell_mut((lx, gy)) {
                            cell.set_char(ch).set_style(l_style);
                        }
                    }
                }
            }
        }
    }
}
