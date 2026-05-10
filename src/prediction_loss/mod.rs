//! Prediction-loss backend — fills `SurpriseComponents.prediction_loss`.
//!
//! Why: v0.1 / v0.2 left `prediction_loss = None` and re-distributed the
//! `w_prediction = 0.3` weight onto outlier + engagement. That works but it
//! means the surprise score has no "did the LLM expect this content?"
//! signal — only "is this embedding novel" + "is the content long" + "did
//! the user mark it important".
//!
//! v0.3 adds an optional pluggable backend that scores raw content via an
//! external LLM and returns a value in `[0, 1]` where 1.0 is "highly
//! surprising / unpredictable" and 0.0 is "completely expected". When the
//! backend is wired in (`--prediction-loss-backend openai-compat`), the
//! `MemoryServer.remember` flow populates `prediction_loss = Some(value)`
//! and the surprise score uses the full 4-component formula instead of the
//! redistributed 3-component fallback.
//!
//! # Why "content alone" scoring (not "content given context")
//!
//! The richest signal would be `loss(content | recent_chat_history)` — i.e.
//! how surprising is this memory given everything we just said. But that
//! requires:
//! - Tokenizing or sending the chat history each call (cost + latency)
//! - A backend that supports `echo=true, max_tokens=0` on a multi-segment
//!   prompt and lets us recover per-segment logprobs
//!
//! For v0.3 we score `content` in isolation. The interpretation is
//! "how unlikely is this string under the LLM's prior?" — a generic
//! observation like `"todo: fix bug"` will score low; a specific decision
//! like `"after auditing 47k spans we picked OTLP over Jaeger because of
//! TLS 1.3 support"` will score high. v0.4 will revisit context-aware
//! scoring once a Rust-native local LLM (candle-rs port of abyo-llm-probe)
//! is on hand.
//!
//! # Wire compatibility
//!
//! Backends are expected to implement the OpenAI legacy `/v1/completions`
//! endpoint with these features:
//! - `echo: true` — return the prompt with logprobs filled in
//! - `max_tokens: 0` — no generation, just scoring
//! - `logprobs: 1` — per-token log-probability
//!
//! That feature set is supported by **vLLM** (`/v1/completions`),
//! **llama.cpp** (`/completion`, `n_predict: 0`, OpenAI-compat shim),
//! **Ollama** (via OpenAI-compat shim, depends on version), and the legacy
//! OpenAI `/v1/completions` endpoint where still available. OpenAI's chat
//! completions endpoint does NOT expose prompt logprobs, so it cannot
//! drive this backend.

use crate::Result;

pub mod external;
#[cfg(feature = "candle")]
pub mod candle_local;

pub use external::{
    ExternalPredictionLossBackend, ExternalPredictionLossConfig, DEFAULT_LOSS_SCALE,
};
#[cfg(feature = "candle")]
pub use candle_local::{
    CandleLocalConfig, CandleLocalPredictionLoss, DEFAULT_CANDLE_MODEL_ID,
    DEFAULT_LOSS_SCALE as CANDLE_DEFAULT_LOSS_SCALE,
};

/// Scores the surprise of arbitrary content via an LLM. Sync trait — like
/// [`crate::embeddings::Embedder`] — because [`crate::server`] calls into
/// it from inside async MCP handlers without `.await`.
pub trait PredictionLossBackend: Send + Sync {
    /// Returns `surprise ∈ [0, 1]`. Higher = more surprising / less
    /// predictable. Backends should compute mean negative log-likelihood
    /// of the content's tokens and squash to `[0, 1]` so the value can
    /// drop directly into `SurpriseComponents.prediction_loss`.
    fn predict_loss(&self, content: &str) -> Result<f32>;
}

/// CLI selector for prediction-loss backend.
///
/// `CandleLocal` is only available when compiled with `--features candle`
/// (CPU) or `--features candle-cuda` (GPU). The enum variant is always
/// declared so the parser surface is stable; selecting it without the
/// feature returns an actionable error at construction time rather than
/// silently falling back.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PredictionLossBackendKind {
    /// No backend wired. `prediction_loss` stays `None`; surprise score
    /// re-distributes `w_prediction` over outlier + engagement (v0.2
    /// behavior).
    #[default]
    None,
    /// OpenAI-compatible `/v1/completions` legacy endpoint with
    /// `echo + max_tokens=0 + logprobs`. Works with vLLM, llama.cpp,
    /// Ollama (via shim), legacy OpenAI completions.
    OpenAiCompat,
    /// Pure-Rust local backend via candle-rs (v0.5, gated `--features
    /// candle`). Loads a HuggingFace model in-process and computes mean
    /// NLL per token without a network round-trip.
    CandleLocal,
}

impl PredictionLossBackendKind {
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "none" | "off" | "disabled" => Ok(Self::None),
            "openai-compat" | "openai" | "vllm" | "llamacpp" | "llama-cpp" => {
                Ok(Self::OpenAiCompat)
            }
            "candle-local" | "candle" | "local" => Ok(Self::CandleLocal),
            other => Err(format!(
                "unknown prediction-loss backend: {other:?} \
                 (expected: none, openai-compat, candle-local)"
            )),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::OpenAiCompat => "openai-compat",
            Self::CandleLocal => "candle-local",
        }
    }
}

/// Mock backend for tests / determinism. Scores content via a hash so the
/// value is deterministic but varies across inputs. Not a real loss.
pub struct MockPredictionLoss;

impl PredictionLossBackend for MockPredictionLoss {
    fn predict_loss(&self, content: &str) -> Result<f32> {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(content.as_bytes());
        let seed = h.finalize();
        // Map first byte to [0, 1].
        Ok(seed[0] as f32 / 255.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_kind_parse_canonical() {
        assert_eq!(
            PredictionLossBackendKind::parse("none").unwrap(),
            PredictionLossBackendKind::None
        );
        assert_eq!(
            PredictionLossBackendKind::parse("openai-compat").unwrap(),
            PredictionLossBackendKind::OpenAiCompat
        );
    }

    #[test]
    fn backend_kind_parse_aliases() {
        assert_eq!(
            PredictionLossBackendKind::parse("vllm").unwrap(),
            PredictionLossBackendKind::OpenAiCompat
        );
        assert_eq!(
            PredictionLossBackendKind::parse("OFF").unwrap(),
            PredictionLossBackendKind::None
        );
        assert_eq!(
            PredictionLossBackendKind::parse("candle-local").unwrap(),
            PredictionLossBackendKind::CandleLocal
        );
        assert_eq!(
            PredictionLossBackendKind::parse("candle").unwrap(),
            PredictionLossBackendKind::CandleLocal
        );
        assert_eq!(
            PredictionLossBackendKind::parse("LOCAL").unwrap(),
            PredictionLossBackendKind::CandleLocal
        );
    }

    #[test]
    fn backend_kind_parse_rejects_unknown() {
        assert!(PredictionLossBackendKind::parse("gpt-7").is_err());
    }

    #[test]
    fn backend_kind_default_is_none() {
        assert_eq!(
            PredictionLossBackendKind::default(),
            PredictionLossBackendKind::None
        );
    }

    #[test]
    fn mock_returns_deterministic_in_range() {
        let m = MockPredictionLoss;
        let a = m.predict_loss("alpha").unwrap();
        let b = m.predict_loss("alpha").unwrap();
        assert_eq!(a, b);
        assert!((0.0..=1.0).contains(&a));
        let c = m.predict_loss("bravo").unwrap();
        assert!(a != c, "different inputs must produce different scores");
    }
}
