//! Brainstem-to-Cortex 2x4 Braille Sub-Pixel Neural Atlas for Dendrite-3D.
//!
//! Inspired by MindCorpus: renders memories as a coloured neural point-cloud
//! growing from a central Brainstem/Spine into cortical category hubs, with
//! frustum culling, subtle synaptic firing sparks, semantic zoom, and temporal
//! growth scrubbing.

use crate::renderer::{Camera3D, GalaxyRenderer};
use ratatui::style::{Color, Modifier, Style};

// Ratatui 2x4 Braille bitmask offsets per (sub_x % 2, sub_y % 4):
const BRAILLE_BITS: [[u8; 4]; 2] = [
    [0x01, 0x02, 0x04, 0x40], // left column: dots 1, 2, 3, 7
    [0x08, 0x10, 0x20, 0x80], // right column: dots 4, 5, 6, 8
];

pub struct BrailleCanvas {
    pub width: usize,
    pub height: usize,
    cells: Vec<u8>,
    depths: Vec<f32>,
    colors: Vec<Color>,
    bold: Vec<bool>,
}

impl BrailleCanvas {
    pub fn new(width: usize, height: usize) -> Self {
        let len = width.saturating_mul(height);
        Self {
            width,
            height,
            cells: vec![0; len],
            depths: vec![f32::INFINITY; len],
            colors: vec![Color::Reset; len],
            bold: vec![false; len],
        }
    }

    #[inline]
    pub fn px_width(&self) -> i32 {
        (self.width * 2) as i32
    }

    #[inline]
    pub fn px_height(&self) -> i32 {
        (self.height * 4) as i32
    }

    /// Plot a sub-pixel dot with depth-buffer & frustum culling.
    pub fn plot_subpixel(&mut self, px: i32, py: i32, depth: f32, color: Color, is_bold: bool) {
        if px < 0 || py < 0 || px >= self.px_width() || py >= self.px_height() || depth <= 1.0 {
            return;
        }
        let cx = (px as usize) / 2;
        let cy = (py as usize) / 4;
        let idx = cy * self.width + cx;
        let bit = BRAILLE_BITS[(px as usize) % 2][(py as usize) % 4];
        self.cells[idx] |= bit;
        if depth <= self.depths[idx] {
            self.depths[idx] = depth;
            self.colors[idx] = color;
            self.bold[idx] = is_bold;
        }
    }

    /// Plot a 2x2 sub-pixel Soma hub for stronger visual anchoring.
    pub fn plot_soma(&mut self, px: i32, py: i32, depth: f32, color: Color) {
        for dx in 0..=1 {
            for dy in 0..=1 {
                self.plot_subpixel(px + dx, py + dy, depth, color, true);
            }
        }
    }

    /// Bresenham sub-pixel line for brainstem spine and synaptic axons.
    pub fn draw_line(&mut self, mut x0: i32, mut y0: i32, x1: i32, y1: i32, depth: f32, color: Color) {
        let dx = (x1 - x0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let dy = -(y1 - y0).abs();
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        let mut steps = 0usize;
        let max_steps = (dx + (-dy)).min(240) as usize;
        while steps <= max_steps {
            if steps % 2 == 0 {
                self.plot_subpixel(x0, y0, depth, color, false);
            }
            if x0 == x1 && y0 == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x0 += sx;
            }
            if e2 <= dx {
                err += dx;
                y0 += sy;
            }
            steps += 1;
        }
    }

    pub fn to_rows(&self) -> Vec<Vec<(char, Style)>> {
        let mut rows = Vec::with_capacity(self.height);
        for cy in 0..self.height {
            let mut row = Vec::with_capacity(self.width);
            for cx in 0..self.width {
                let idx = cy * self.width + cx;
                let mask = self.cells[idx];
                if mask == 0 {
                    row.push((' ', Style::default()));
                } else {
                    let ch = char::from_u32(0x2800 + mask as u32).unwrap_or('·');
                    let mut st = Style::default().fg(self.colors[idx]);
                    if self.bold[idx] {
                        st = st.add_modifier(Modifier::BOLD);
                    }
                    row.push((ch, st));
                }
            }
            rows.push(row);
        }
        rows
    }
}

#[derive(Clone, Debug)]
pub struct NeuralPoint {
    pub cluster_idx: usize,
    pub mass: f32,
    pub pinned: bool,
    pub created_at: i64,
    pub color: Color,
}

/// Render Brainstem-to-Cortex 2x4 Braille point-cloud with semantic zoom & temporal scrubbing.
pub fn render_neural_point_cloud(
    width: usize,
    height: usize,
    cam: &Camera3D,
    anim_time: f32,
    time_pct: f32,
    points: &[NeuralPoint],
) -> Vec<Vec<(char, Style)>> {
    let mut canvas = BrailleCanvas::new(width, height);
    let pw = canvas.px_width() as u16;
    let ph = canvas.px_height() as u16;
    if pw < 4 || ph < 4 {
        return canvas.to_rows();
    }

    // 1. Vertical Brainstem / Spine (5 Soma hubs rising along Y)
    let soma_specs: [([f32; 3], Color); 5] = [
        ([0.0, -14.0, 0.0], Color::Green),               // Meta / Core Stem
        ([4.5, -6.0, 3.0], Color::Cyan),                 // Engineering
        ([-4.5, 1.0, 3.5], Color::LightMagenta),         // Personal
        ([4.0, 8.0, -3.5], Color::Yellow),               // Preferences
        ([-3.5, 13.5, -2.5], Color::Rgb(140, 180, 255)), // Episodic Cortex
    ];

    let stem_base = GalaxyRenderer::project_point([0.0, -19.0, 0.0], cam, pw, ph);
    let mut soma_pts = Vec::with_capacity(soma_specs.len());
    for (pos, color) in soma_specs {
        let p = GalaxyRenderer::project_point(pos, cam, pw, ph);
        soma_pts.push((pos, p, color));
    }

    // Draw Brainstem spine segments
    let spine_color = Color::Rgb(75, 95, 130);
    if stem_base.visible {
        canvas.plot_soma(stem_base.screen_x, stem_base.screen_y, stem_base.depth, Color::Rgb(255, 215, 0));
    }
    let mut prev_proj = &stem_base;
    for (_, proj, color) in &soma_pts {
        if prev_proj.visible && proj.visible {
            canvas.draw_line(
                prev_proj.screen_x,
                prev_proj.screen_y,
                proj.screen_x,
                proj.screen_y,
                proj.depth + 0.5,
                spine_color,
            );
        }
        if proj.visible {
            canvas.plot_soma(proj.screen_x, proj.screen_y, proj.depth, *color);
        }
        prev_proj = proj;
    }

    // Semantic Zoom: when zoomed far out (> 42.0), show Soma hubs & brainstem only.
    if cam.distance > 42.0 || points.is_empty() {
        return canvas.to_rows();
    }

    // 2. Temporal Growth Scrubber: sort by created_at and keep prefix up to time_pct.
    let mut sorted: Vec<(usize, &NeuralPoint)> = points.iter().enumerate().collect();
    sorted.sort_by_key(|(_, p)| p.created_at);
    let visible_count = ((sorted.len() as f32) * time_pct.clamp(0.05, 1.0)).ceil() as usize;

    // 3. Cortical Point-Cloud + Subtle Synaptic Firing Spark
    for (rank, (orig_idx, pt)) in sorted.into_iter().take(visible_count).enumerate() {
        let (soma_pos, soma_proj, _) = &soma_pts[pt.cluster_idx % soma_pts.len()];
        let golden = (orig_idx as f32) * 2.399_963;
        let spread = 3.2 + (rank as f32 % 5.0) * 1.15;
        let sway = 0.25 * (anim_time * 0.7 + golden).sin();
        let node_3d = [
            soma_pos[0] + spread * golden.cos() + sway,
            soma_pos[1] + ((golden * 1.7).sin() * 2.8),
            soma_pos[2] + spread * golden.sin(),
        ];

        let proj = GalaxyRenderer::project_point(node_3d, cam, pw, ph);
        if !proj.visible || proj.depth <= 1.0 {
            continue;
        }

        // Draw subtle synaptic axon + crisp single traveling spark for pinned/heavy nodes
        if soma_proj.visible && (pt.pinned || pt.mass >= 2.0) {
            canvas.draw_line(
                soma_proj.screen_x,
                soma_proj.screen_y,
                proj.screen_x,
                proj.screen_y,
                proj.depth + 0.4,
                Color::Rgb(58, 72, 98),
            );
            let pulse_t = (anim_time * 0.45 + (orig_idx as f32) * 0.37).fract();
            let sx = soma_proj.screen_x as f32 + (proj.screen_x - soma_proj.screen_x) as f32 * pulse_t;
            let sy = soma_proj.screen_y as f32 + (proj.screen_y - soma_proj.screen_y) as f32 * pulse_t;
            canvas.plot_subpixel(sx as i32, sy as i32, proj.depth - 0.1, Color::Rgb(255, 200, 90), true);
        }

        let node_color = if pt.pinned { Color::Rgb(255, 215, 0) } else { pt.color };
        if pt.pinned || pt.mass >= 2.0 {
            canvas.plot_soma(proj.screen_x, proj.screen_y, proj.depth, node_color);
        } else {
            canvas.plot_subpixel(proj.screen_x, proj.screen_y, proj.depth, node_color, false);
        }
    }

    canvas.to_rows()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn braille_canvas_renders_subpixel_and_culls_behind_camera() {
        let mut canvas = BrailleCanvas::new(10, 5);
        canvas.plot_subpixel(0, 0, 5.0, Color::Cyan, true);
        canvas.plot_subpixel(-1, 0, 5.0, Color::Red, false);
        canvas.plot_subpixel(2, 0, 0.5, Color::Red, false); // behind near plane
        let rows = canvas.to_rows();
        assert_eq!(rows[0][0].0, '⠁');
        assert_eq!(rows[0][1].0, ' ');
    }

    #[test]
    fn neural_point_cloud_handles_5000_nodes_and_semantic_zoom() {
        let points: Vec<NeuralPoint> = (0..5000)
            .map(|i| NeuralPoint {
                cluster_idx: i % 5,
                mass: if i % 20 == 0 { 2.5 } else { 1.0 },
                pinned: i % 50 == 0,
                created_at: i as i64,
                color: Color::Cyan,
            })
            .collect();
        let mut cam = Camera3D::new();
        cam.distance = 32.0;
        let zoomed_in = render_neural_point_cloud(80, 24, &cam, 1.0, 1.0, &points);
        assert_eq!(zoomed_in.len(), 24);
        assert_eq!(zoomed_in[0].len(), 80);

        cam.distance = 55.0; // zoomed out -> soma hubs only
        let zoomed_out = render_neural_point_cloud(80, 24, &cam, 1.0, 1.0, &points);
        assert_eq!(zoomed_out.len(), 24);
    }
}
