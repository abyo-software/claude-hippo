//! Pure-Rust local prediction-loss backend, gated by `--features candle`.
//!
//! Productionization of the v0.4 D-spike (`examples/candle_spike.rs`):
//! loads a HF safetensors causal LM into memory, runs per-position forward
//! passes to compute the negative log-likelihood of each token given its
//! prefix, and returns `mean_nll / loss_scale` as a surprise score in
//! `[0, 1]`. Mirrors the OpenAI-compat `/v1/completions echo +
//! max_tokens=0 + logprobs` math but without the HTTP round-trip or the
//! external service dependency.
//!
//! # When to use this vs the HTTP backend
//!
//! - **`ExternalPredictionLossBackend`** (always-on): great if you
//!   already run vLLM / llama.cpp / a hosted completions endpoint. No
//!   GPU required in the claude-hippo process. Network latency is the
//!   floor.
//! - **`CandleLocalPredictionLoss`** (this module, `--features candle`):
//!   no external service. Single static binary serves the model in-
//!   process. CPU works but is slow (~3-4 s / sentence on Qwen2.5-0.5B);
//!   `--features candle-cuda` pushes it to ~50-100 ms / sentence on a
//!   16GB consumer GPU once cuDNN is installed.
//!
//! # Design notes
//!
//! - Single Qwen2-family model for v0.5 (validated in the spike). Other
//!   architectures (Phi-3, Llama-3) need their own `forward + KV cache
//!   reset` glue; tracked for v0.6.
//! - `predict_loss` calls `model.forward` once per position because the
//!   default Qwen2 `forward` returns logits for the LAST position only.
//!   That's O(N) forward passes for an N-token text. v0.6 will explore
//!   patching the Qwen2 model to expose all-position logits in one pass.
//! - The model + tokenizer are loaded eagerly at construction so the
//!   first scoring call doesn't pay download/load latency. Construction
//!   blocks the calling thread; do it before serving requests.
//! - Holds a `parking_lot::Mutex<ModelForCausalLM>` because forward is
//!   `&mut self` (KV cache mutation), and `MemoryServer` shares the
//!   backend through `Arc<dyn PredictionLossBackend>`. Lock contention
//!   only matters under concurrent scoring; in practice MCP stdio is
//!   single-client so this is fine.

use super::PredictionLossBackend;
use crate::{HippoError, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::{Config as Qwen2Config, ModelForCausalLM};
use hf_hub::api::sync::Api;
use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Default model id. Small (0.5B), fast to download (~1 GB), acceptable
/// surprise gradient (validated in v0.4 spike: cliché → 1.20 NLL,
/// specific decision → 4.96 NLL).
pub const DEFAULT_CANDLE_MODEL_ID: &str = "Qwen/Qwen2.5-0.5B";

/// Default cross-entropy scale (nats / token) for mapping mean NLL to
/// `[0, 1]`. Mirrors the external HTTP backend's default.
pub const DEFAULT_LOSS_SCALE: f32 = 6.0;

/// Build-time configuration.
#[derive(Debug, Clone)]
pub struct CandleLocalConfig {
    /// HuggingFace model id (e.g. `"Qwen/Qwen2.5-0.5B"`). Must be a
    /// Qwen2-family model in v0.5; other architectures land in v0.6.
    pub model_id: String,
    /// Override the HF cache directory. `None` lets `hf-hub` pick the
    /// default (`~/.cache/huggingface/`).
    pub cache_dir: Option<PathBuf>,
    /// Cross-entropy scale (nats / token) for mapping mean NLL to [0,1].
    pub loss_scale: f32,
    /// `true` = try CUDA first, fall back to CPU. `false` = CPU only.
    /// Has no effect unless the binary was built with
    /// `--features candle-cuda` (CPU-only build ignores this).
    pub use_gpu: bool,
}

impl Default for CandleLocalConfig {
    fn default() -> Self {
        Self {
            model_id: DEFAULT_CANDLE_MODEL_ID.into(),
            cache_dir: None,
            loss_scale: DEFAULT_LOSS_SCALE,
            use_gpu: true,
        }
    }
}

impl CandleLocalConfig {
    pub fn validate(&self) -> Result<()> {
        if self.model_id.is_empty() {
            return Err(HippoError::Config(
                "candle_local model_id is empty".into(),
            ));
        }
        if !self.loss_scale.is_finite() || self.loss_scale <= 0.0 {
            return Err(HippoError::Config(format!(
                "candle_local loss_scale must be > 0 and finite, got {}",
                self.loss_scale
            )));
        }
        Ok(())
    }
}

pub struct CandleLocalPredictionLoss {
    cfg: CandleLocalConfig,
    device: Device,
    model: Mutex<ModelForCausalLM>,
    tokenizer: Arc<Tokenizer>,
}

impl CandleLocalPredictionLoss {
    /// Eagerly load model + tokenizer. Blocks until everything is in
    /// memory. Failures (network, OOM, unsupported architecture) bubble
    /// up immediately so `hippo serve` doesn't claim it's running with
    /// candle when it's actually broken.
    pub fn new(cfg: CandleLocalConfig) -> Result<Self> {
        cfg.validate()?;

        let device = pick_device(cfg.use_gpu);
        tracing::info!(
            model = cfg.model_id.as_str(),
            ?device,
            loss_scale = cfg.loss_scale,
            "loading candle-rs prediction-loss backend"
        );

        let api =
            Api::new().map_err(|e| HippoError::Config(format!("hf-hub api: {e}")))?;
        let repo = api.model(cfg.model_id.clone());

        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| HippoError::Config(format!("download tokenizer.json: {e}")))?;
        let config_path = repo
            .get("config.json")
            .map_err(|e| HippoError::Config(format!("download config.json: {e}")))?;
        let weights_path = repo
            .get("model.safetensors")
            .map_err(|e| HippoError::Config(format!("download model.safetensors: {e}")))?;

        let raw = std::fs::read(&config_path)
            .map_err(|e| HippoError::Config(format!("read config.json: {e}")))?;
        let qwen_config: Qwen2Config = serde_json::from_slice(&raw)
            .map_err(|e| HippoError::Config(format!("parse config.json (v0.5 supports Qwen2 family only): {e}")))?;

        let dtype = match device {
            Device::Cuda(_) => DType::BF16,
            _ => DType::F32,
        };
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device)
                .map_err(|e| HippoError::Config(format!("mmap safetensors: {e}")))?
        };
        let model = ModelForCausalLM::new(&qwen_config, vb)
            .map_err(|e| HippoError::Config(format!("build model: {e}")))?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| HippoError::Config(format!("load tokenizer: {e}")))?;

        Ok(Self {
            cfg,
            device,
            model: Mutex::new(model),
            tokenizer: Arc::new(tokenizer),
        })
    }

    pub fn config(&self) -> &CandleLocalConfig {
        &self.cfg
    }

    /// Per-position forward + log-softmax + gather, mirroring the spike.
    /// Returns mean negative log-likelihood (in nats / token).
    fn mean_nll(&self, content: &str) -> Result<f32> {
        if content.trim().is_empty() {
            return Ok(0.0);
        }
        let enc = self
            .tokenizer
            .encode(content, true)
            .map_err(|e| HippoError::Embedding(format!("candle_local encode: {e}")))?;
        let ids: Vec<u32> = enc.get_ids().to_vec();
        if ids.len() < 2 {
            // Single token: no position to score. Treat as neutral.
            return Ok(0.5 * self.cfg.loss_scale);
        }
        let mut model = self.model.lock();
        let mut sum = 0.0_f64;
        let mut count = 0_u32;
        for i in 0..(ids.len() - 1) {
            // Reset KV cache so each prefix forward starts from offset=0
            // without the previous call's stale rotary positions.
            model.clear_kv_cache();
            let prefix: Vec<u32> = ids[..=i].to_vec();
            let input = Tensor::new(prefix.as_slice(), &self.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| HippoError::Embedding(format!("candle_local input: {e}")))?;
            let logits = model
                .forward(&input, 0)
                .map_err(|e| HippoError::Embedding(format!("candle_local forward: {e}")))?;
            let logits = logits
                .to_dtype(DType::F32)
                .map_err(|e| HippoError::Embedding(format!("candle_local dtype: {e}")))?;
            let lp = candle_nn::ops::log_softmax(&logits, candle_core::D::Minus1)
                .map_err(|e| HippoError::Embedding(format!("candle_local log_softmax: {e}")))?;
            let lp = lp
                .flatten_all()
                .map_err(|e| HippoError::Embedding(format!("candle_local flatten: {e}")))?;
            let target = ids[i + 1] as usize;
            let token_lp: f32 = lp
                .get(target)
                .and_then(|t| t.to_scalar::<f32>())
                .map_err(|e| HippoError::Embedding(format!("candle_local gather: {e}")))?;
            sum += token_lp as f64;
            count += 1;
        }
        let mean_lp = (sum / count as f64) as f32;
        Ok(-mean_lp)
    }
}

impl PredictionLossBackend for CandleLocalPredictionLoss {
    fn predict_loss(&self, content: &str) -> Result<f32> {
        let nll = self.mean_nll(content)?;
        Ok((nll / self.cfg.loss_scale).clamp(0.0, 1.0))
    }
}

/// Pick CUDA if requested AND compiled in AND probe succeeds; else CPU.
fn pick_device(use_gpu: bool) -> Device {
    if use_gpu {
        #[cfg(feature = "candle-cuda")]
        if let Ok(d) = Device::new_cuda(0) {
            return d;
        }
    }
    Device::Cpu
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_rejects_empty_model_id() {
        let c = CandleLocalConfig {
            model_id: String::new(),
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_rejects_bad_loss_scale() {
        for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let c = CandleLocalConfig {
                loss_scale: bad,
                ..Default::default()
            };
            assert!(c.validate().is_err(), "loss_scale {bad} should fail");
        }
    }

    #[test]
    fn defaults_validate() {
        assert!(CandleLocalConfig::default().validate().is_ok());
    }

    /// Real model load + scoring. Downloads ~1 GB on first run; gated
    /// `#[ignore]` so CI doesn't pay it. Run with
    /// `cargo test --features candle -- --ignored candle_local_smoke`.
    #[test]
    #[ignore = "downloads ~1GB Qwen2.5-0.5B; CPU inference takes ~5s"]
    fn candle_local_smoke() {
        let backend = CandleLocalPredictionLoss::new(CandleLocalConfig {
            use_gpu: false, // deterministic, no cuDNN required
            ..Default::default()
        })
        .expect("backend init");

        let cliche = backend
            .predict_loss("the quick brown fox jumps over the lazy dog")
            .expect("score cliche");
        let specific = backend
            .predict_loss(
                "After auditing 47k OpenTelemetry spans we picked OTLP over Jaeger \
                 because of native TLS 1.3 support",
            )
            .expect("score specific");
        // Specific must score strictly higher than cliché — same gradient
        // the v0.4 D-spike captured.
        assert!(
            specific > cliche,
            "expected specific ({specific:.3}) > cliché ({cliche:.3})"
        );
        assert!((0.0..=1.0).contains(&cliche));
        assert!((0.0..=1.0).contains(&specific));
    }
}
