//! Streaming native Rust forward pass for Ornith safetensors.
//!
//! Architecture: AirLLM-style layer streaming.
//!   - Embedding: read ONE row (8KB BF16) from disk, not the whole 2GB table.
//!   - Per layer: read that layer's ~13 weight tensors from disk, convert
//!     BF16→f32, compute, DISCARD the weights. Peak RAM = one layer (~400MB).
//!   - lm_head: compute logits by reading one row at a time OR read the
//!     whole lm_head (same size as embed — 2GB) in chunks.
//!   - No global cache. No HashMap. No cloning.
//!
//! This is what beats AirLLM: Rust + direct file I/O + BLAS-like matmul,
//! with peak RAM = one layer's weights (~400MB for 9B), not the whole model.

use crate::bpe_tokenizer::BpeTokenizer;
use crate::cache::deltanet_state::DeltaNetStateCache;
use crate::model::tensor::Tensor;
use crate::ornith_config::OrnithConfig;
use crate::safetensors_loader::{Shards, WeightProvider};
use crate::gguf_provider::{self, GGUFWeightProvider};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Maximum number of sequence positions the streaming caches may hold.
///
/// `kv_cache` appends one K/V row per token, so without a cap a long
/// generation grows until OOM. Overflow is REFUSED (fail-loud), never
/// evicted: attention requires every prior K/V, so dropping entries would
/// silently corrupt the output. `deltanet_cache` holds one fixed-size state
/// matrix and one fixed-size conv buffer per layer (bounded by layer count);
/// it is gated by the same [`check_seq_len`] at sequence entry points.
pub const MAX_SEQ_CACHE: usize = 4096;

/// Whether this file's forward math implements `arch`.
///
/// The allow-list is deliberately a single variant, because the forward pass
/// hard-codes the ornith hybrid's conventions:
///   * `attention_forward` reads `self_attn.q_proj.weight` as `[2h, h]`
///     (first h = Q, second h = output gate, applied as
///     `attn_out *= sigmoid(gate)`) — the gated-attention layout the
///     ornith/qwen35moe converter emits for `blk.N.attn_q.weight`;
///   * RoPE is GLM-style split-pair with `rotary_dim = head_dim * 0.25` and
///     `rope_theta = 10000000` hard-coded (config's `rope_theta` is ignored);
///   * linear-attention layers require the `blk.N.ssm_*` DeltaNet tensors
///     (`gguf_provider::LINEAR_ATTN_MAP`).
/// `Qwen36` is the `ModelArchitecture::detect` arm for
/// `general.architecture ∈ {"ornith", "qwen35moe", "qwen36"}` — the only
/// family this math was written for (the project's own ornith GGUF reports
/// `qwen35moe`, see `src/bin/inspect_ornith.rs`). Anything else
/// (Llama/Qwen2/Gemma/Qwen35/GlmDsa/…) either trips
/// `Tensor::from_vec`'s shape panic on the `[2h, h]` q_proj split or runs
/// with the wrong RoPE/gate conventions and emits gibberish — so it is
/// refused before any weight is read.
pub fn streaming_arch_supported(arch: crate::model::arch::ModelArchitecture) -> bool {
    matches!(arch, crate::model::arch::ModelArchitecture::Qwen36)
}

/// The exact refusal message for an unsupported `general.architecture`.
pub fn arch_refusal_msg(arch: &str) -> String {
    format!("native engine does not support architecture '{arch}' — use Tier 1 (llama-server/Ollama) instead")
}

/// Fail-loud sequence-length guard for the streaming caches.
///
/// `len` is the number of positions the cache would hold after this step.
/// Returns `Err` past [`MAX_SEQ_CACHE`] instead of evicting.
pub fn check_seq_len(len: usize) -> Result<(), String> {
    if len > MAX_SEQ_CACHE {
        return Err(format!(
            "native engine context limit reached ({MAX_SEQ_CACHE} tokens) — shorten the prompt or use Tier 1"
        ));
    }
    Ok(())
}

/// Append one token's K/V row to a layer's KV cache, refusing (fail-loud)
/// once that layer already holds [`MAX_SEQ_CACHE`] positions. Returns the
/// new sequence length. On `Err` nothing is appended and nothing is evicted.
fn kv_cache_append(
    cache: &mut HashMap<usize, (Vec<f32>, Vec<f32>)>,
    layer_idx: usize,
    k: &[f32],
    v: &[f32],
) -> Result<usize, String> {
    if k.len() != v.len() {
        return Err(format!("kv cache row length mismatch (k={} v={})", k.len(), v.len()));
    }
    let kv_dim = k.len();
    if kv_dim == 0 {
        return Err("kv cache row is empty".to_string());
    }
    let existing = cache.get(&layer_idx).map(|e| e.0.len()).unwrap_or(0);
    let new_seq = existing / kv_dim + 1;
    check_seq_len(new_seq)?;
    let entry = cache.entry(layer_idx).or_insert_with(|| (Vec::new(), Vec::new()));
    entry.0.extend_from_slice(k);
    entry.1.extend_from_slice(v);
    Ok(new_seq)
}

/// Pre-transposed and pre-structured weights for a single layer, eliminating
/// redundant matrix clones and per-token heap churn during layer streaming.
pub struct PreparedStreamingLayer {
    pub input_layernorm: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
    pub mlp_gate_t: Tensor,
    pub mlp_up_t: Tensor,
    pub mlp_down_t: Tensor,

    // Linear attention
    pub linear_qkv_t: Option<Tensor>,
    pub linear_conv1d: Option<Vec<f32>>,
    pub linear_a_t: Option<Tensor>,
    pub linear_a_log: Option<Vec<f32>>,
    pub linear_dt_bias: Option<Vec<f32>>,
    pub linear_b_t: Option<Tensor>,
    pub linear_norm: Option<Vec<f32>>,
    pub linear_z_t: Option<Tensor>,
    pub linear_out_t: Option<Tensor>,

    // Full attention
    pub attn_q_t: Option<Tensor>,
    pub attn_k_t: Option<Tensor>,
    pub attn_v_t: Option<Tensor>,
    pub attn_o_t: Option<Tensor>,
    pub attn_q_norm: Option<Vec<f32>>,
    pub attn_k_norm: Option<Vec<f32>>,
}

impl PreparedStreamingLayer {
    pub fn from_map(
        mut w: HashMap<String, Vec<f32>>,
        layer_type: &str,
        h: usize,
        inter: usize,
        n_qk: usize,
        n_v: usize,
        d_k: usize,
        d_v: usize,
        n_heads: usize,
        head_dim: usize,
        n_kv: usize,
    ) -> Result<Self, String> {
        let input_layernorm = w.remove("input_layernorm.weight").ok_or("missing input_layernorm")?;
        let post_attention_layernorm = w.remove("post_attention_layernorm.weight").ok_or("missing post_attention_layernorm")?;

        let gate_raw = w.remove("mlp.gate_proj.weight").ok_or("missing mlp.gate_proj.weight")?;
        let up_raw = w.remove("mlp.up_proj.weight").ok_or("missing mlp.up_proj.weight")?;
        let down_raw = w.remove("mlp.down_proj.weight").ok_or("missing mlp.down_proj.weight")?;

        let mlp_gate_t = Tensor::from_vec(gate_raw, vec![inter, h]).transpose();
        let mlp_up_t = Tensor::from_vec(up_raw, vec![inter, h]).transpose();
        let mlp_down_t = Tensor::from_vec(down_raw, vec![h, inter]).transpose();

        let conv_dim = n_qk * d_k + n_qk * d_k + n_v * d_v;
        let v_total = n_v * d_v;

        let (mut linear_qkv_t, mut linear_conv1d, mut linear_a_t, mut linear_a_log, mut linear_dt_bias, mut linear_b_t, mut linear_norm, mut linear_z_t, mut linear_out_t) =
            (None, None, None, None, None, None, None, None, None);
        let (mut attn_q_t, mut attn_k_t, mut attn_v_t, mut attn_o_t, mut attn_q_norm, mut attn_k_norm) =
            (None, None, None, None, None, None);

        if layer_type == "linear_attention" {
            let qkv = w.remove("linear_attn.in_proj_qkv.weight").ok_or("missing in_proj_qkv")?;
            linear_qkv_t = Some(Tensor::from_vec(qkv, vec![conv_dim, h]).transpose());
            linear_conv1d = Some(w.remove("linear_attn.conv1d.weight").ok_or("missing conv1d")?);
            let a = w.remove("linear_attn.in_proj_a.weight").ok_or("missing in_proj_a")?;
            linear_a_t = Some(Tensor::from_vec(a, vec![n_v, h]).transpose());
            linear_a_log = Some(w.remove("linear_attn.A_log").ok_or("missing A_log")?);
            linear_dt_bias = Some(w.remove("linear_attn.dt_bias").ok_or("missing dt_bias")?);
            let b = w.remove("linear_attn.in_proj_b.weight").ok_or("missing in_proj_b")?;
            linear_b_t = Some(Tensor::from_vec(b, vec![n_v, h]).transpose());
            linear_norm = Some(w.remove("linear_attn.norm.weight").ok_or("missing norm.weight")?);
            let z = w.remove("linear_attn.in_proj_z.weight").ok_or("missing in_proj_z")?;
            linear_z_t = Some(Tensor::from_vec(z, vec![v_total, h]).transpose());
            let out = w.remove("linear_attn.out_proj.weight").ok_or("missing out_proj")?;
            linear_out_t = Some(Tensor::from_vec(out, vec![h, v_total]).transpose());
        } else {
            let q = w.remove("self_attn.q_proj.weight").ok_or("missing q_proj")?;
            attn_q_t = Some(Tensor::from_vec(q, vec![2 * h, h]).transpose());
            let k = w.remove("self_attn.k_proj.weight").ok_or("missing k_proj")?;
            attn_k_t = Some(Tensor::from_vec(k, vec![n_kv * head_dim, h]).transpose());
            let v = w.remove("self_attn.v_proj.weight").ok_or("missing v_proj")?;
            attn_v_t = Some(Tensor::from_vec(v, vec![n_kv * head_dim, h]).transpose());
            let o = w.remove("self_attn.o_proj.weight").ok_or("missing o_proj")?;
            attn_o_t = Some(Tensor::from_vec(o, vec![h, n_heads * head_dim]).transpose());
            attn_q_norm = w.remove("self_attn.q_norm.weight");
            attn_k_norm = w.remove("self_attn.k_norm.weight");
        }

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            mlp_gate_t,
            mlp_up_t,
            mlp_down_t,
            linear_qkv_t,
            linear_conv1d,
            linear_a_t,
            linear_a_log,
            linear_dt_bias,
            linear_b_t,
            linear_norm,
            linear_z_t,
            linear_out_t,
            attn_q_t,
            attn_k_t,
            attn_v_t,
            attn_o_t,
            attn_q_norm,
            attn_k_norm,
        })
    }
}

pub struct StreamingOrnith {
    pub cfg: OrnithConfig,
    pub weights: Arc<dyn WeightProvider>,
    pub tok: BpeTokenizer,
    /// Per-layer DeltaNet state matrices (accumulated across prompt tokens)
    pub deltanet_cache: DeltaNetStateCache,
    /// Per-layer KV cache for full_attention layers: layer_idx -> (keys, values)
    /// keys/values: flat Vec<f32> of shape [seq_len, n_kv * head_dim] each
    pub kv_cache: HashMap<usize, (Vec<f32>, Vec<f32>)>,
    /// Optional layer progress callback (layer_idx, total_layers)
    pub layer_callback: Option<Arc<dyn Fn(usize, usize) + Send + Sync>>,
}

impl StreamingOrnith {
    /// Open a model from a directory containing config.json, tokenizer.json,
    /// and .safetensors file(s).
    pub fn open(dir: &Path) -> Result<Self, String> {
        let cfg = OrnithConfig::load(dir.join("config.json").to_str().unwrap())?;
        let shards = Shards::open_dir(dir)?;
        let tok = BpeTokenizer::load(dir.join("tokenizer.json").to_str().unwrap())?;
        Ok(Self { cfg, weights: Arc::new(shards), tok, deltanet_cache: DeltaNetStateCache::new(), kv_cache: HashMap::new(), layer_callback: None })
    }

    /// Open a model from a GGUF file.
    ///
    /// `gguf_path` — path to the .gguf file.
    /// `tokenizer_path` — path to tokenizer.json (can be in the same directory).
    ///
    /// Refuses (before any weight is read) architectures whose forward math
    /// this file does not implement — see [`streaming_arch_supported`].
    pub fn open_gguf(gguf_path: &str, tokenizer_path: &str) -> Result<Self, String> {
        use crate::model::gguf::GGUFile;
        let gguf = GGUFile::open(gguf_path)
            .map_err(|e| format!("open GGUF: {e}"))?;
        // Architecture refusal first: never proceed to math on an arch this
        // forward pass does not implement (GLM split-pair RoPE + 0.25 head_dim
        // rotary + Q/gate q_proj split are ornith-hybrid specific).
        let arch = crate::model::arch::ModelArchitecture::detect(&gguf);
        if !streaming_arch_supported(arch) {
            return Err(arch_refusal_msg(arch.name()));
        }
        let cfg = gguf_provider::extract_ornith_config(&gguf)?;
        let provider = GGUFWeightProvider::from_gguf(gguf)
            .map_err(|e| format!("GGUF provider: {e}"))?;
        let tok = BpeTokenizer::load(tokenizer_path)?;
        Ok(Self { cfg, weights: Arc::new(provider), tok, deltanet_cache: DeltaNetStateCache::new(), kv_cache: HashMap::new(), layer_callback: None })
    }

    /// Forward pass for ONE token. Returns logits (vocab_size entries).
    ///
    /// `token_id`: the input token.
    /// Forward a multi-token sequence, processing all tokens layer by layer.
    /// Each layer's weights are loaded ONCE from disk and shared across all tokens.
    /// This avoids re-loading weights for every token.
    pub fn forward_sequence(&mut self, tokens: &[i32]) -> Result<Vec<f32>, String> {
        let h = self.cfg.hidden_size;
        let num_layers = self.cfg.num_hidden_layers;
        let layer_types: Vec<String> = self.cfg.layer_types.clone();
        let rms_eps = self.cfg.rms_norm_eps;
        let vocab_size = self.cfg.vocab_size;
        let seq_len = tokens.len();
        // Sequence cap: refuse before any weight I/O (fail-loud, no eviction).
        check_seq_len(seq_len)?;

        // 1. Embeddings for all tokens
        let embed_name = "model.language_model.embed_tokens.weight";
        let mut hidden_states: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        for &tid in tokens {
            let embed = self.weights.read_tensor_slice_f32(embed_name, tid as usize * h, h)?;
            hidden_states.push(embed);
        }

        // 2. Process each layer: load weights ONCE via double-buffered prefetch, run all tokens
        let mut prefetcher = LayerPrefetcher::new(
            Arc::clone(&self.weights),
            self.cfg.clone(),
            layer_types.clone(),
            num_layers,
        );

        for layer_idx in 0..num_layers {
            if let Some(cb) = &self.layer_callback {
                cb(layer_idx, num_layers);
            }
            let layer_type = layer_types
                .get(layer_idx)
                .map(|s| s.as_str())
                .unwrap_or("linear_attention");

            let layer = prefetcher.next_layer(layer_idx)?;

            for pos in 0..seq_len {
                let attn_out = if layer_type == "linear_attention" {
                    self.deltanet_forward_layer(&layer, &hidden_states[pos], layer_idx)?
                } else {
                    self.attention_forward_layer(&layer, &hidden_states[pos], layer_idx, pos)?
                };

                let mut new_hidden = hidden_states[pos].clone();
                for i in 0..h { new_hidden[i] += attn_out[i]; }

                let normed = rms_norm(&new_hidden, &layer.post_attention_layernorm, rms_eps);
                let mlp_out = self.mlp_forward_layer(&layer, &normed)?;

                for i in 0..h { new_hidden[i] += mlp_out[i]; }
                hidden_states[pos] = new_hidden;
            }
        }

        // 3. Final norm on last token
        let final_w = self.weights.read_tensor_f32("model.language_model.norm.weight")?;
        let last_hidden = rms_norm(&hidden_states[seq_len - 1], &final_w, rms_eps);

        // 4. LM head: logits = hidden @ lm_head.T
        let lm_head_name = "lm_head.weight";
        let mut logits = vec![0.0f32; vocab_size];
        let chunk_size = 1024;
        for chunk_start in (0..vocab_size).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(vocab_size);
            let n_rows = chunk_end - chunk_start;
            let chunk = self.weights.read_tensor_slice_f32(lm_head_name, chunk_start * h, n_rows * h)?;
            let chunk_t = Tensor::from_vec(chunk, vec![n_rows, h]);
            let hidden_t = Tensor::from_vec(last_hidden.clone(), vec![1, h]);
            let logits_chunk = hidden_t.matmul(&chunk_t.transpose());
            logits[chunk_start..chunk_end].copy_from_slice(&logits_chunk.data);
        }

        Ok(logits)
    }

    /// Forward ONE token with zero weight cloning.
    pub fn forward_one_token(
        &mut self,
        token_id: i32,
        pos: usize,
    ) -> Result<Vec<f32>, String> {
        check_seq_len(pos + 1)?;
        let h = self.cfg.hidden_size;
        let num_layers = self.cfg.num_hidden_layers;
        let layer_types: Vec<String> = self.cfg.layer_types.clone();
        let rms_eps = self.cfg.rms_norm_eps;
        let vocab_size = self.cfg.vocab_size;

        let embed_name = "model.language_model.embed_tokens.weight";
        let mut hidden = self.weights.read_tensor_slice_f32(embed_name, token_id as usize * h, h)?;

        // Double-buffered asynchronous layer prefetching
        let mut prefetcher = LayerPrefetcher::new(
            Arc::clone(&self.weights),
            self.cfg.clone(),
            layer_types.clone(),
            num_layers,
        );

        for layer_idx in 0..num_layers {
            if let Some(cb) = &self.layer_callback {
                cb(layer_idx, num_layers);
            }
            let layer_type = layer_types
                .get(layer_idx)
                .map(|s| s.as_str())
                .unwrap_or("linear_attention");

            let layer = prefetcher.next_layer(layer_idx)?;

            let residual = hidden.clone();
            let attn_out = if layer_type == "linear_attention" {
                self.deltanet_forward_layer(&layer, &hidden, layer_idx)?
            } else {
                self.attention_forward_layer(&layer, &hidden, layer_idx, pos)?
            };

            for i in 0..h { hidden[i] = residual[i] + attn_out[i]; }

            let normed = rms_norm(&hidden, &layer.post_attention_layernorm, rms_eps);
            let mlp_out = self.mlp_forward_layer(&layer, &normed)?;

            for i in 0..h { hidden[i] += mlp_out[i]; }
        }

        let final_w = self.weights.read_tensor_f32("model.language_model.norm.weight")?;
        hidden = rms_norm(&hidden, &final_w, rms_eps);

        let lm_head_name = "lm_head.weight";
        let mut logits = vec![0.0f32; vocab_size];
        let chunk_size = 1024;
        for chunk_start in (0..vocab_size).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(vocab_size);
            let n_rows = chunk_end - chunk_start;
            let chunk = self.weights.read_tensor_slice_f32(lm_head_name, chunk_start * h, n_rows * h)?;
            let chunk_t = Tensor::from_vec(chunk, vec![n_rows, h]);
            let hidden_t = Tensor::from_vec(hidden.clone(), vec![1, h]);
            let logits_chunk = hidden_t.matmul(&chunk_t.transpose());
            logits[chunk_start..chunk_end].copy_from_slice(&logits_chunk.data);
        }

        Ok(logits)
    }

    /// Forward a sequence in dynamic chunks based on memory budget.
    pub fn forward_sequence_chunked(
        &mut self,
        tokens: &[i32],
        chunk_size: usize,
    ) -> Result<Vec<f32>, String> {
        if tokens.is_empty() {
            return Err("Cannot forward empty token sequence".into());
        }
        let chunk_size = chunk_size.max(64);
        if tokens.len() <= chunk_size {
            return self.forward_sequence(tokens);
        }
        let mut last_logits = Vec::new();
        for chunk in tokens.chunks(chunk_size) {
            last_logits = self.forward_sequence(chunk)?;
        }
        Ok(last_logits)
    }

    /// Set an optional callback triggered before each layer forward pass (layer_idx, total_layers).
    pub fn set_layer_callback<F>(&mut self, cb: F)
    where
        F: Fn(usize, usize) + Send + Sync + 'static,
    {
        self.layer_callback = Some(Arc::new(cb));
    }

    /// Load and prepare layer weights into pre-transposed tensors with zero inner-loop cloning.
    #[allow(dead_code)]
    fn load_prepared_layer(
        &self,
        layer_idx: usize,
        layer_type: &str,
    ) -> Result<PreparedStreamingLayer, String> {
        load_prepared_layer_static(&*self.weights, &self.cfg, layer_idx, layer_type)
    }

    /// Load all weights for ONE layer from disk. ~13 tensors, ~400MB for 9B.
    /// These are returned as f32 and will be DROPPED after the layer computes.
    #[allow(dead_code)]
    fn load_layer_weights(
        &self,
        layer_idx: usize,
        layer_type: &str,
    ) -> Result<HashMap<String, Vec<f32>>, String> {
        let pfx = format!("model.language_model.layers.{layer_idx}.");

        let names: Vec<&str> = match layer_type {
            "linear_attention" => vec![
                "input_layernorm.weight",
                "linear_attn.in_proj_qkv.weight",
                "linear_attn.in_proj_a.weight",
                "linear_attn.in_proj_b.weight",
                "linear_attn.in_proj_z.weight",
                "linear_attn.conv1d.weight",
                "linear_attn.A_log",
                "linear_attn.dt_bias",
                "linear_attn.norm.weight",
                "linear_attn.out_proj.weight",
                "post_attention_layernorm.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            ],
            "full_attention" => vec![
                "input_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
                "post_attention_layernorm.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            ],
            _ => return Err(format!("unknown layer type: {layer_type}")),
        };

        self.weights.load_layer_weights(layer_idx, layer_type, &names, &pfx)
    }
}

/// Static helper to load and prepare layer weights without borrowing `StreamingOrnith`.
fn load_prepared_layer_static(
    weights: &dyn WeightProvider,
    cfg: &OrnithConfig,
    layer_idx: usize,
    layer_type: &str,
) -> Result<PreparedStreamingLayer, String> {
    let pfx = format!("model.language_model.layers.{layer_idx}.");
    let names: Vec<&str> = match layer_type {
        "linear_attention" => vec![
            "input_layernorm.weight",
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.in_proj_a.weight",
            "linear_attn.in_proj_b.weight",
            "linear_attn.in_proj_z.weight",
            "linear_attn.conv1d.weight",
            "linear_attn.A_log",
            "linear_attn.dt_bias",
            "linear_attn.norm.weight",
            "linear_attn.out_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ],
        "full_attention" => vec![
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ],
        _ => return Err(format!("unknown layer type: {layer_type}")),
    };

    let raw = weights.load_layer_weights(layer_idx, layer_type, &names, &pfx)?;
    PreparedStreamingLayer::from_map(
        raw,
        layer_type,
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.linear_num_key_heads,
        cfg.linear_num_value_heads,
        cfg.linear_key_head_dim,
        cfg.linear_value_head_dim,
        cfg.num_attention_heads,
        cfg.head_dim,
        cfg.num_key_value_heads,
    )
}

/// Double-buffered asynchronous layer prefetcher.
/// Overlaps background NVMe disk reads & weight prep of Layer L+1 with Layer L compute.
struct LayerPrefetcher {
    weights: Arc<dyn WeightProvider>,
    cfg: OrnithConfig,
    layer_types: Vec<String>,
    num_layers: usize,
    next_slot: Option<std::thread::JoinHandle<Result<PreparedStreamingLayer, String>>>,
}

impl LayerPrefetcher {
    fn new(weights: Arc<dyn WeightProvider>, cfg: OrnithConfig, layer_types: Vec<String>, num_layers: usize) -> Self {
        let mut prefetcher = Self {
            weights,
            cfg,
            layer_types,
            num_layers,
            next_slot: None,
        };
        if num_layers > 0 {
            prefetcher.spawn_prefetch(0);
        }
        prefetcher
    }

    fn spawn_prefetch(&mut self, layer_idx: usize) {
        let w = Arc::clone(&self.weights);
        let c = self.cfg.clone();
        let lt = self.layer_types.get(layer_idx).cloned().unwrap_or_else(|| "linear_attention".into());
        let handle = std::thread::spawn(move || {
            load_prepared_layer_static(&*w, &c, layer_idx, &lt)
        });
        self.next_slot = Some(handle);
    }

    fn next_layer(&mut self, layer_idx: usize) -> Result<PreparedStreamingLayer, String> {
        let layer_res = match self.next_slot.take() {
            Some(handle) => handle.join().map_err(|e| format!("Prefetch thread join failed: {:?}", e))?,
            None => {
                let lt = self.layer_types.get(layer_idx).map(|s| s.as_str()).unwrap_or("linear_attention");
                load_prepared_layer_static(&*self.weights, &self.cfg, layer_idx, lt)
            }
        };

        let next_idx = layer_idx + 1;
        if next_idx < self.num_layers {
            self.spawn_prefetch(next_idx);
        }

        layer_res
    }
}

impl StreamingOrnith {

    /// DeltaNet (linear attention) forward with pre-structured zero-clone layer weights.
    fn deltanet_forward_layer(
        &mut self,
        layer: &PreparedStreamingLayer,
        hidden: &[f32],
        layer_idx: usize,
    ) -> Result<Vec<f32>, String> {
        let h = self.cfg.hidden_size;
        let rms_eps = self.cfg.rms_norm_eps;

        // Config for linear attention heads
        let n_qk = self.cfg.linear_num_key_heads;   // 16
        let n_v = self.cfg.linear_num_value_heads;   // 32
        let d_k = self.cfg.linear_key_head_dim;     // 128
        let d_v = self.cfg.linear_value_head_dim;   // 128
        let conv_k = self.cfg.linear_conv_kernel_dim; // 4
        let conv_dim = n_qk * d_k + n_qk * d_k + n_v * d_v; // 8192

        // 1. Input norm
        let normed = rms_norm(hidden, &layer.input_layernorm, rms_eps);
        // 2. QKV projection: [1, h] @ [conv_dim, h]^T = [1, conv_dim]
        let qkv_t = layer.linear_qkv_t.as_ref().ok_or("missing linear_qkv_t")?;
        let hidden_t = Tensor::from_vec(normed, vec![1, h]);
        let qkv_proj = hidden_t.matmul(qkv_t);
        // qkv_proj.data: [conv_dim] = [Q(2048) | K(2048) | V(4096)]

        // 3. Conv1d
        let conv_w = layer.linear_conv1d.as_ref().ok_or("missing conv1d")?;
        let cbuf = self.deltanet_cache.get_conv_buf_mut(layer_idx, conv_dim, conv_k);
        for c in 0..conv_dim {
            let base = c * conv_k;
            for k in 0..conv_k - 1 {
                cbuf[base + k] = cbuf[base + k + 1];
            }
            cbuf[base + conv_k - 1] = qkv_proj.data[c];
        }
        let mut conv_out = vec![0.0f32; conv_dim];
        for c in 0..conv_dim {
            let mut sum = 0.0f32;
            let base = c * conv_k;
            for k in 0..conv_k {
                sum += conv_w[base + k] * cbuf[base + k];
            }
            conv_out[c] = sum;
        }
        // SiLU after conv
        for v in conv_out.iter_mut() {
            *v = *v / (1.0 + (-*v).exp());
        }
        let q_total = n_qk * d_k;   // 2048
        let k_total = n_qk * d_k;   // 2048
        let v_total = n_v * d_v;    // 4096
        let q_data = &conv_out[..q_total];
        let k_data = &conv_out[q_total..q_total + k_total];
        let v_data = &conv_out[q_total + k_total..];

        // 5. L2-normalize Q and K (per-head)
        let mut q = q_data.to_vec();
        let mut k = k_data.to_vec();
        for head in 0..n_qk {
            let base = head * d_k;
            let mut norm_sq = 0.0f32;
            for d in 0..d_k { norm_sq += q[base + d] * q[base + d]; }
            let norm = norm_sq.sqrt().max(1e-6);
            for d in 0..d_k { q[base + d] /= norm; }

            let mut norm_sq = 0.0f32;
            for d in 0..d_k { norm_sq += k[base + d] * k[base + d]; }
            let norm = norm_sq.sqrt().max(1e-6);
            for d in 0..d_k { k[base + d] /= norm; }
        }
        let scale = 1.0f32 / (d_k as f32).sqrt();
        for v in q.iter_mut() { *v *= scale; }

        // 6. Compute decay rates
        let a_t = layer.linear_a_t.as_ref().ok_or("missing linear_a_t")?;
        let alpha = hidden_t.matmul(a_t);

        let a_log = layer.linear_a_log.as_ref().ok_or("missing A_log")?;
        let dt_bias = layer.linear_dt_bias.as_ref().ok_or("missing dt_bias")?;

        let mut decay = vec![0.0f32; n_v];
        for head in 0..n_v {
            let alpha_val = alpha.data[head];
            let dt_val = dt_bias.get(head).copied().unwrap_or(0.0);
            let a_log_val = a_log.get(head).copied().unwrap_or(0.0);
            let a = -a_log_val.exp();
            let dt = softplus(alpha_val + dt_val);
            decay[head] = (dt * a).exp();
        }
        // 7. Compute beta gates
        let b_t = layer.linear_b_t.as_ref().ok_or("missing linear_b_t")?;
        let beta_logits = hidden_t.matmul(b_t);
        let beta: Vec<f32> = beta_logits.data.iter().map(|&v| sigmoid(v)).collect();

        // 8. Delta rule state update + output
        let mut output = vec![0.0f32; v_total];

        if self.deltanet_cache.get(layer_idx).is_none() {
            self.deltanet_cache.init_layer(layer_idx, n_v, d_v, d_k);
        }

        for h_v in 0..n_v {
            let h_qk = h_v % n_qk;
            let q_h = &q[h_qk * d_k..(h_qk + 1) * d_k];
            let k_h = &k[h_qk * d_k..(h_qk + 1) * d_k];
            let v_h = &v_data[h_v * d_v..(h_v + 1) * d_v];
            {
                let decay_h = decay[h_v];
                let beta_h = beta[h_v];

                let state = self.deltanet_cache.get_mut(layer_idx).unwrap();
                let state_stride = h_v * d_v * d_k;
                for i in 0..d_v {
                    for j in 0..d_k {
                        let idx = state_stride + i * d_k + j;
                        state[idx] = decay_h * state[idx];
                    }
                }

                let mut v_pred = vec![0.0f32; d_v];
                for i in 0..d_v {
                    let mut sum = 0.0f32;
                    for j in 0..d_k {
                        sum += state[state_stride + i * d_k + j] * k_h[j];
                    }
                    v_pred[i] = sum;
                }

                for i in 0..d_v {
                    let delta = v_h[i] - v_pred[i];
                    for j in 0..d_k {
                        let idx = state_stride + i * d_k + j;
                        state[idx] = state[idx] + beta_h * delta * k_h[j];
                    }
                }

                for i in 0..d_v {
                    let mut sum = 0.0f32;
                    for j in 0..d_k {
                        sum += state[state_stride + i * d_k + j] * q_h[j];
                    }
                    output[h_v * d_v + i] = sum;
                }
            }
        }

        // 9. Per-head RMSNorm using linear_attn.norm.weight
        let norm_weight = layer.linear_norm.as_ref().ok_or("missing norm.weight")?;
        for head in 0..n_v {
            let base = head * d_v;
            let mut sq_sum = 0.0f32;
            for d in 0..d_v { sq_sum += output[base + d] * output[base + d]; }
            let rms = (sq_sum / d_v as f32 + rms_eps).sqrt();
            for d in 0..d_v {
                output[base + d] = (output[base + d] / rms) * norm_weight.get(d).copied().unwrap_or(1.0);
            }
        }
        // 10. Z-gate
        let z_t = layer.linear_z_t.as_ref().ok_or("missing linear_z_t")?;
        let z = hidden_t.matmul(z_t);
        for i in 0..output.len() {
            let z_val = z.data[i];
            let silu_z = z_val * (1.0 / (1.0 + (-z_val).exp()));
            output[i] *= silu_z;
        }

        // 11. Output projection
        let o_t = layer.linear_out_t.as_ref().ok_or("missing linear_out_t")?;
        let out_t = Tensor::from_vec(output, vec![1, v_total]);
        let result = out_t.matmul(o_t);
        Ok(result.data)
    }

    /// DeltaNet forward compatibility shim.
    #[allow(dead_code)]
    fn deltanet_forward(
        &mut self,
        w: &HashMap<String, Vec<f32>>,
        hidden: &[f32],
        layer_idx: usize,
    ) -> Result<Vec<f32>, String> {
        let prep = PreparedStreamingLayer::from_map(
            w.clone(),
            "linear_attention",
            self.cfg.hidden_size,
            self.cfg.intermediate_size,
            self.cfg.linear_num_key_heads,
            self.cfg.linear_num_value_heads,
            self.cfg.linear_key_head_dim,
            self.cfg.linear_value_head_dim,
            self.cfg.num_attention_heads,
            self.cfg.head_dim,
            self.cfg.num_key_value_heads,
        )?;
        self.deltanet_forward_layer(&prep, hidden, layer_idx)
    }

    /// Standard attention forward with pre-structured zero-clone layer weights.
    fn attention_forward_layer(
        &mut self,
        layer: &PreparedStreamingLayer,
        hidden: &[f32],
        layer_idx: usize,
        pos: usize,
    ) -> Result<Vec<f32>, String> {
        let h = self.cfg.hidden_size;
        let n_heads = self.cfg.num_attention_heads;
        let head_dim = self.cfg.head_dim;
        let n_kv = self.cfg.num_key_value_heads;
        let rms_eps = self.cfg.rms_norm_eps;

        let normed = rms_norm(hidden, &layer.input_layernorm, rms_eps);
        let hidden_t = Tensor::from_vec(normed, vec![1, h]);

        let q_t = layer.attn_q_t.as_ref().ok_or("missing attn_q_t")?;
        let k_t = layer.attn_k_t.as_ref().ok_or("missing attn_k_t")?;
        let v_t = layer.attn_v_t.as_ref().ok_or("missing attn_v_t")?;
        let o_t = layer.attn_o_t.as_ref().ok_or("missing attn_o_t")?;

        let q_all = hidden_t.matmul(q_t);
        let mut q = q_all.data[..h].to_vec();
        let gate = &q_all.data[h..];
        let mut k = hidden_t.matmul(k_t).data;
        let v = hidden_t.matmul(v_t).data;

        if let Some(ref qnw) = layer.attn_q_norm {
            for head in 0..n_heads {
                let base = head * head_dim;
                let mut sq = 0.0f32;
                for d in 0..head_dim { sq += q[base + d] * q[base + d]; }
                let r = (sq / head_dim as f32 + rms_eps).sqrt();
                for d in 0..head_dim { q[base + d] = q[base + d] / r * qnw[d]; }
            }
        }
        if let Some(ref knw) = layer.attn_k_norm {
            for head in 0..n_kv {
                let base = head * head_dim;
                let mut sq = 0.0f32;
                for d in 0..head_dim { sq += k[base + d] * k[base + d]; }
                let r = (sq / head_dim as f32 + rms_eps).sqrt();
                for d in 0..head_dim { k[base + d] = k[base + d] / r * knw[d]; }
            }
        }

        let rotary_dim = (self.cfg.head_dim as f32 * 0.25) as usize;
        let rope_theta = 10000000f32;
        let half_rotary = rotary_dim / 2;
        for head in 0..n_heads {
            let base = head * head_dim;
            for i in 0..half_rotary {
                let i0 = base + i;
                let i1 = base + i + half_rotary;
                let angle = pos as f32 / rope_theta.powf(2.0 * i as f32 / rotary_dim as f32);
                let (sin, cos) = angle.sin_cos();
                let x0 = q[i0];
                let x1 = q[i1];
                q[i0] = x0 * cos - x1 * sin;
                q[i1] = x0 * sin + x1 * cos;
            }
        }
        for head in 0..n_kv {
            let base = head * head_dim;
            for i in 0..half_rotary {
                let i0 = base + i;
                let i1 = base + i + half_rotary;
                let angle = pos as f32 / rope_theta.powf(2.0 * i as f32 / rotary_dim as f32);
                let (sin, cos) = angle.sin_cos();
                let x0 = k[i0];
                let x1 = k[i1];
                k[i0] = x0 * cos - x1 * sin;
                k[i1] = x0 * sin + x1 * cos;
            }
        }

        let seq_len = kv_cache_append(&mut self.kv_cache, layer_idx, &k, &v)?;
        let kv_entry = self.kv_cache.get(&layer_idx).ok_or("kv cache entry missing after append")?;
        let cached_k = &kv_entry.0;
        let cached_v = &kv_entry.1;

        let mut attn_out = vec![0.0f32; h];
        for head in 0..n_heads {
            let kv_head = head / (n_heads / n_kv);
            let q_base = head * head_dim;
            let mut max_score = f32::NEG_INFINITY;
            for t in 0..seq_len {
                let k_base = t * n_kv * head_dim + kv_head * head_dim;
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q[q_base + d] * cached_k[k_base + d];
                }
                score /= (head_dim as f32).sqrt();
                if score > max_score { max_score = score; }
            }
            let mut sum_exp = 0.0f32;
            let mut weighted_v = vec![0.0f32; head_dim];
            for t in 0..seq_len {
                let k_base = t * n_kv * head_dim + kv_head * head_dim;
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q[q_base + d] * cached_k[k_base + d];
                }
                score = (score / (head_dim as f32).sqrt() - max_score).exp();
                sum_exp += score;
                let v_base = t * n_kv * head_dim + kv_head * head_dim;
                for d in 0..head_dim {
                    weighted_v[d] += score * cached_v[v_base + d];
                }
            }
            let inv_sum = 1.0 / sum_exp;
            for d in 0..head_dim {
                attn_out[q_base + d] = weighted_v[d] * inv_sum;
            }
        }

        for i in 0..h {
            attn_out[i] *= 1.0 / (1.0 + (-gate[i]).exp());
        }

        let out_t = Tensor::from_vec(attn_out, vec![1, h]);
        let result = out_t.matmul(o_t);
        Ok(result.data)
    }

    /// Attention forward compatibility shim.
    #[allow(dead_code)]
    fn attention_forward(
        &mut self,
        w: &HashMap<String, Vec<f32>>,
        hidden: &[f32],
        layer_idx: usize,
        pos: usize,
    ) -> Result<Vec<f32>, String> {
        let prep = PreparedStreamingLayer::from_map(
            w.clone(),
            "full_attention",
            self.cfg.hidden_size,
            self.cfg.intermediate_size,
            self.cfg.linear_num_key_heads,
            self.cfg.linear_num_value_heads,
            self.cfg.linear_key_head_dim,
            self.cfg.linear_value_head_dim,
            self.cfg.num_attention_heads,
            self.cfg.head_dim,
            self.cfg.num_key_value_heads,
        )?;
        self.attention_forward_layer(&prep, hidden, layer_idx, pos)
    }

    /// MLP forward with pre-structured zero-clone layer weights.
    fn mlp_forward_layer(
        &self,
        layer: &PreparedStreamingLayer,
        hidden: &[f32],
    ) -> Result<Vec<f32>, String> {
        let h = self.cfg.hidden_size;
        let inter = self.cfg.intermediate_size;

        let hidden_t = Tensor::from_vec(hidden.to_vec(), vec![1, h]);
        let gate_t = hidden_t.matmul(&layer.mlp_gate_t);
        let up_t = hidden_t.matmul(&layer.mlp_up_t);

        let mut hidden_inter = vec![0.0f32; inter];
        for i in 0..inter {
            let g = gate_t.data[i];
            let u = up_t.data[i];
            hidden_inter[i] = (g / (1.0 + (-g).exp())) * u;
        }

        let inter_t = Tensor::from_vec(hidden_inter, vec![1, inter]);
        let result = inter_t.matmul(&layer.mlp_down_t);
        Ok(result.data)
    }

    /// MLP forward compatibility shim.
    #[allow(dead_code)]
    fn mlp_forward(
        &self,
        w: &HashMap<String, Vec<f32>>,
        hidden: &[f32],
    ) -> Result<Vec<f32>, String> {
        let h = self.cfg.hidden_size;
        let inter = self.cfg.intermediate_size;

        let gate = w.get("mlp.gate_proj.weight").ok_or("missing gate")?;
        let up = w.get("mlp.up_proj.weight").ok_or("missing up")?;
        let down = w.get("mlp.down_proj.weight").ok_or("missing down")?;

        let hidden_t = Tensor::from_vec(hidden.to_vec(), vec![1, h]);
        let gate_t = hidden_t.matmul(&Tensor::from_vec(gate.clone(), vec![inter, h]).transpose());
        let up_t = hidden_t.matmul(&Tensor::from_vec(up.clone(), vec![inter, h]).transpose());

        let mut hidden_inter = vec![0.0f32; inter];
        for i in 0..inter {
            let g = gate_t.data[i];
            let u = up_t.data[i];
            hidden_inter[i] = (g / (1.0 + (-g).exp())) * u;
        }

        let inter_t = Tensor::from_vec(hidden_inter, vec![1, inter]);
        let result = inter_t.matmul(&Tensor::from_vec(down.clone(), vec![h, inter]).transpose());
        Ok(result.data)
    }

    /// Greedy argmax sampling
    pub fn argmax(logits: &[f32]) -> usize {
        let mut best = 0;
        let mut best_v = logits[0];
        for (i, &v) in logits.iter().enumerate().skip(1) {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        best
    }

    /// Generate text: single-token autoregressive loop.
    ///
    /// KV cache note: `kv_cache` DOES exist (one K/V row set per
    /// full-attention layer, appended once per token in `attention_forward`)
    /// and is now capped at `MAX_SEQ_CACHE` — overflowing it returns Err
    /// (fail-loud, never eviction: attention needs every prior K/V).
    /// `deltanet_cache` holds one fixed-size state matrix + conv buffer per
    /// layer, gated by the same `check_seq_len` cap.
    ///
    /// Generate text from a raw prompt (no chat wrapping).
    /// Sampling uses the process-wide config from `inference::sampler`
    /// (`set_default_sampling` temperature/top_p, `set_default_top_k` /
    /// env `LEAFCUTTER_TOP_K` for top-k); temperature 0.0 stays greedy argmax.
    /// `stop_tokens` — generation stops when any of these token IDs is produced.
    pub fn generate_with_stop(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        stop_tokens: &[i32],
    ) -> Result<String, String> {
        let mut ids = self.tok.encode(prompt, 1024);
        if ids.is_empty() {
            return Ok(String::new());
        }
        let (temperature, top_p) = crate::inference::sampler::default_sampling();
        let top_k = crate::inference::sampler::resolve_top_k();

        // Layer-major prompt prefill: evaluates all prompt tokens layer-by-layer
        // (loading each layer once from disk) with zero-clone tensor arithmetic.
        let mut logits = if ids.len() > 1 {
            self.forward_sequence(&ids)?
        } else {
            let last = *ids.last().unwrap() as i32;
            self.forward_one_token(last, 0)?
        };

        for _ in 0..max_tokens {
            let next = crate::inference::sampler::sample_top_p_top_k(&logits, temperature, top_p, top_k) as i32;
            if stop_tokens.contains(&next) {
                break;
            }
            ids.push(next);
            let text = self.tok.decode(&[next]);
            print!("{text}");
            std::io::Write::flush(&mut std::io::stdout()).ok();

            let pos = ids.len() - 1;
            logits = self.forward_one_token(next, pos)?;
        }
        Ok(self.tok.decode(&ids))
    }

    /// Streaming generation with callback for each token piece (AirLLM-style layer streaming output).
    pub fn generate_stream_with<F>(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        stop_tokens: &[i32],
        mut on_token: F,
    ) -> Result<String, String>
    where
        F: FnMut(&str),
    {
        let mut ids = self.tok.encode(prompt, 1024);
        if ids.is_empty() {
            return Ok(String::new());
        }
        let (temperature, top_p) = crate::inference::sampler::default_sampling();
        let top_k = crate::inference::sampler::resolve_top_k();

        // Layer-major prompt prefill
        let mut logits = if ids.len() > 1 {
            self.forward_sequence(&ids)?
        } else {
            let last = *ids.last().unwrap() as i32;
            self.forward_one_token(last, 0)?
        };

        for _ in 0..max_tokens {
            let next = crate::inference::sampler::sample_top_p_top_k(&logits, temperature, top_p, top_k) as i32;
            if stop_tokens.contains(&next) {
                break;
            }
            ids.push(next);
            let text = self.tok.decode(&[next]);
            on_token(&text);

            let pos = ids.len() - 1;
            logits = self.forward_one_token(next, pos)?;
        }
        Ok(self.tok.decode(&ids))
    }

    /// Chat convenience: wraps input in the Ornith chat template and
    /// stops at `<|im_end|>` (looked up dynamically from the tokenizer).
    pub fn chat(&mut self, user_input: &str, max_tokens: usize) -> Result<String, String> {
        let system = "You are Ornith, an open-source agentic coding assistant. Think step by step in a reasoning block, then act. Use the provided tools when they help. Be concise, correct, and direct: write working code and explain only what is non-obvious.";
        let prompt = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n",
            system, user_input
        );
        let stop_im_end = self.tok.id_of("<|im_end|>");
        let stop_eot = self.tok.id_of("<|endoftext|>");
        let stops: Vec<i32> = [stop_im_end, stop_eot].into_iter().filter(|&id| id >= 0).collect();
        self.generate_with_stop(&prompt, max_tokens, &stops)
    }

    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String, String> {
        self.generate_with_stop(prompt, max_tokens, &[])
    }
}

/// RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
/// NOTE: converter bakes gamma as (1 + gamma) into GGUF, so weight multiplies directly.
fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    x.iter()
        .zip(weight.iter())
        .map(|(&xi, &wi)| xi * inv_rms * wi)
        .collect()
}

#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0f32 / (1.0f32 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::arch::ModelArchitecture;

    #[test]
    fn test_streaming_arch_supported_ornith_hybrid() {
        // The forward math implements exactly the ornith hybrid
        // (general.architecture: "ornith" | "qwen35moe" | "qwen36").
        assert!(streaming_arch_supported(ModelArchitecture::Qwen36));
    }

    #[test]
    fn test_streaming_arch_unsupported_foreign_archs() {
        // Q/gate q_proj split + GLM split-pair RoPE (0.25 head_dim) do not
        // fit these — they must be refused, not silently gibberish.
        assert!(!streaming_arch_supported(ModelArchitecture::Llama));
        assert!(!streaming_arch_supported(ModelArchitecture::Qwen2));
        assert!(!streaming_arch_supported(ModelArchitecture::Gemma));
        assert!(!streaming_arch_supported(ModelArchitecture::Qwen35));
        assert!(!streaming_arch_supported(ModelArchitecture::GlmDsa));
        assert!(!streaming_arch_supported(ModelArchitecture::Unknown));
    }

    #[test]
    fn test_arch_refusal_msg_format() {
        let msg = arch_refusal_msg("Llama");
        assert_eq!(
            msg,
            "native engine does not support architecture 'Llama' — use Tier 1 (llama-server/Ollama) instead"
        );
        assert!(msg.starts_with("native engine does not support architecture '"));
        assert!(msg.ends_with("' — use Tier 1 (llama-server/Ollama) instead"));
    }

    #[test]
    fn test_open_gguf_refuses_unsupported_arch_before_any_load() {
        use std::io::Write;
        // Minimal GGUF: magic, version, 0 tensors, one metadata kv
        // (general.architecture = "llama").
        let path = std::env::temp_dir().join("streaming_ornith_arch_refusal.gguf");
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&0x46554747u32.to_le_bytes()); // "GGUF"
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor count
        buf.extend_from_slice(&1u64.to_le_bytes()); // metadata count
        let key = b"general.architecture";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        buf.extend_from_slice(&8u32.to_le_bytes()); // GGUFValue::String
        let val = b"llama";
        buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
        buf.extend_from_slice(val);
        let mut f = std::fs::File::create(&path).expect("create temp gguf");
        f.write_all(&buf).expect("write temp gguf");
        drop(f);

        // The tokenizer path deliberately does not exist: the ONLY way to get
        // this exact error is if the arch refusal fired before any config,
        // provider, tokenizer or weight work.
        let err = match StreamingOrnith::open_gguf(
            path.to_str().unwrap(),
            "/nonexistent/tokenizer.json",
        ) {
            Ok(_) => panic!("unsupported arch must be refused before load succeeds"),
            Err(e) => e,
        };
        assert_eq!(err, arch_refusal_msg("Llama"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_check_seq_len_at_cap_ok_past_cap_errors() {
        assert!(check_seq_len(0).is_ok());
        assert!(check_seq_len(MAX_SEQ_CACHE).is_ok());
        let err = check_seq_len(MAX_SEQ_CACHE + 1).unwrap_err();
        assert_eq!(
            err,
            "native engine context limit reached (4096 tokens) — shorten the prompt or use Tier 1"
        );
    }

    #[test]
    fn test_kv_cache_append_refuses_growth_past_cap() {
        let mut cache: HashMap<usize, (Vec<f32>, Vec<f32>)> = HashMap::new();
        let k = [0.5f32; 4];
        let v = [0.25f32; 4];

        for i in 0..MAX_SEQ_CACHE {
            let seq = kv_cache_append(&mut cache, 0, &k, &v).expect("within cap must append");
            assert_eq!(seq, i + 1);
        }

        let rows_before = cache.get(&0).map(|e| e.0.len()).unwrap_or(0);
        let err = kv_cache_append(&mut cache, 0, &k, &v).unwrap_err();
        assert!(err.contains("context limit reached"), "unexpected error: {err}");
        // Fail-loud: the refused insert must not have grown the cache
        // (and no earlier row was evicted).
        let rows_after = cache.get(&0).map(|e| e.0.len()).unwrap_or(0);
        assert_eq!(rows_after, rows_before);
        assert_eq!(rows_before, MAX_SEQ_CACHE * k.len());
        assert_eq!(cache.get(&0).unwrap().1.len(), MAX_SEQ_CACHE * v.len());
    }
}
