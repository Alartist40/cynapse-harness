//! Looped Transformer Weight-Reuse Architecture inspired by `thinkingmachines/Inkling-2-Mini`.
//! Re-executes a shared middle layer range `num_loops` times to gain effective depth
//! without increasing model RAM or disk footprint on edge SBCs.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoopDepthConfig {
    pub start_layer: usize,
    pub end_layer: usize,
    pub num_loops: usize,
    pub residual_scale: f32,
}

impl Default for LoopDepthConfig {
    fn default() -> Self {
        Self {
            start_layer: 6,
            end_layer: 18,
            num_loops: 2,
            residual_scale: 0.5,
        }
    }
}

/// Execute a recurrent middle-block loop `cfg.num_loops` times over hidden state `x`.
pub fn apply_looped_layers<F>(
    x: &mut [f32],
    cfg: LoopDepthConfig,
    mut step_layer: F,
) where
    F: FnMut(usize, &mut [f32]),
{
    let loops = cfg.num_loops.max(1);
    for _ in 0..loops {
        for layer_idx in cfg.start_layer..cfg.end_layer {
            let prev: Vec<f32> = x.to_vec();
            step_layer(layer_idx, x);
            if cfg.residual_scale != 1.0 {
                for (xi, pi) in x.iter_mut().zip(prev.iter()) {
                    *xi = *pi + (*xi - *pi) * cfg.residual_scale;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn looped_block_reuses_layers_without_extra_weights() {
        let mut hidden = vec![1.0f32, 2.0f32];
        let cfg = LoopDepthConfig {
            start_layer: 2,
            end_layer: 4,
            num_loops: 3,
            residual_scale: 1.0,
        };
        let mut calls = 0usize;
        apply_looped_layers(&mut hidden, cfg, |_layer, h| {
            calls += 1;
            h[0] += 1.0;
        });
        assert_eq!(calls, 6); // 2 layers * 3 loops
        assert!((hidden[0] - 7.0).abs() < 1e-5);
    }
}
