//! Embedding layer.
//!
//! Two backends are wired in v0.3:
//! - [`local::FastEmbedder`] — fastembed (ONNX) on-host inference. Default.
//!   RSS ~150 MB after model load. Fully offline once cached.
//! - [`external::ExternalEmbedder`] — OpenAI-compatible HTTP `/v1/embeddings`
//!   endpoint. RSS ~25 MB (no model in process). Network required.
//!
//! Both produce L2-normalized 384-dim vectors so KNN cosine math, the
//! `EMBEDDING_DIM = 384` constant, and DB swap with mcp-memory-service-rs all
//! continue to hold regardless of backend.

use crate::Result;

pub mod external;
pub mod local;

pub use external::{ExternalEmbedder, ExternalEmbeddingConfig};
pub use local::{EmbeddingModelKind, FastEmbedder};

/// 384 次元の embedding を生成する trait。テスト時に mock 差し替え可。
pub trait Embedder: Send + Sync {
    fn embed_one(&self, text: &str) -> Result<Vec<f32>>;
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
}

/// CLI / config 用 backend 切替。`FastEmbedder` か `ExternalEmbedder` のどちらを
/// build するか決めるだけのタグ。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EmbeddingBackendKind {
    /// fastembed (ONNX) ローカル推論。
    #[default]
    Local,
    /// OpenAI 互換 HTTP API。
    External,
}

impl EmbeddingBackendKind {
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "local" | "fastembed" | "onnx" => Ok(Self::Local),
            "external" | "http" | "openai" | "openai-compat" => Ok(Self::External),
            other => Err(format!(
                "unknown embedding backend: {other:?} (expected: local, external)"
            )),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::External => "external",
        }
    }
}

pub fn default_cache_dir() -> std::path::PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("claude-hippo")
        .join("models")
}

// ---------- Mock embedder (tests / bench) ----------

/// Mock implementation: テスト・bench 用。content の hash を expand して
/// deterministic な L2 normalized vector を返す。
pub struct MockEmbedder;

impl MockEmbedder {
    pub fn new() -> Self {
        Self
    }

    fn deterministic_vec(text: &str) -> Vec<f32> {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(text.as_bytes());
        let seed = h.finalize();
        let mut v = vec![0.0_f32; crate::EMBEDDING_DIM];
        for (i, b) in (0..crate::EMBEDDING_DIM).zip(seed.iter().cycle()) {
            v[i] = (*b as f32 / 127.5) - 1.0;
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        for x in v.iter_mut() {
            *x /= norm;
        }
        v
    }
}

impl Default for MockEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

impl Embedder for MockEmbedder {
    fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        Ok(Self::deterministic_vec(text))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| Self::deterministic_vec(t)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EMBEDDING_DIM;

    #[test]
    fn mock_dims_and_l2_norm() {
        let m = MockEmbedder::new();
        let v = m.embed_one("hello").unwrap();
        assert_eq!(v.len(), EMBEDDING_DIM);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "norm = {norm}");
    }

    #[test]
    fn mock_deterministic() {
        let m = MockEmbedder::new();
        let a = m.embed_one("same input").unwrap();
        let b = m.embed_one("same input").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn mock_different_inputs_differ() {
        let m = MockEmbedder::new();
        let a = m.embed_one("alpha").unwrap();
        let b = m.embed_one("bravo").unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn mock_batch_matches_individual() {
        let m = MockEmbedder::new();
        let batch = m.embed_batch(&["x", "y"]).unwrap();
        let single_x = m.embed_one("x").unwrap();
        let single_y = m.embed_one("y").unwrap();
        assert_eq!(batch[0], single_x);
        assert_eq!(batch[1], single_y);
    }

    #[test]
    fn backend_kind_parse_canonical() {
        assert_eq!(
            EmbeddingBackendKind::parse("local").unwrap(),
            EmbeddingBackendKind::Local
        );
        assert_eq!(
            EmbeddingBackendKind::parse("external").unwrap(),
            EmbeddingBackendKind::External
        );
    }

    #[test]
    fn backend_kind_parse_aliases() {
        assert_eq!(
            EmbeddingBackendKind::parse("openai").unwrap(),
            EmbeddingBackendKind::External
        );
        assert_eq!(
            EmbeddingBackendKind::parse("ONNX").unwrap(),
            EmbeddingBackendKind::Local
        );
    }

    #[test]
    fn backend_kind_default_is_local() {
        assert_eq!(EmbeddingBackendKind::default(), EmbeddingBackendKind::Local);
    }
}
