//! Embedding layer — fastembed wrapper.
//!
//! Default model: `all-MiniLM-L6-v2` (384 dim) — mcp-memory-service-rs と
//! 同一 vector space を使うため、DB を swap しても retrieval semantics が
//! 保たれる。
//!
//! Backends:
//! - `local` (default): `fastembed` で ONNX を CPU 実行。RSS ~175MB。
//! - `external`: OpenAI 互換 API に POST。RSS <30MB だがネットワーク必須。
//!
//! Lazy load: model は initial `embed` 呼び出し時に load される。`serve`
//! cold-start を 23MB 程度に抑える。

use crate::{HippoError, Result, EMBEDDING_DIM};
use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::Arc;

/// 384 次元の embedding を生成する trait。テスト時に mock 差し替え可。
pub trait Embedder: Send + Sync {
    fn embed_one(&self, text: &str) -> Result<Vec<f32>>;
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
}

/// Fast-embed local (ONNX) backend。lazy load (initial `embed` で model load)。
pub struct FastEmbedder {
    inner: Arc<Mutex<Option<fastembed::TextEmbedding>>>,
    cache_dir: PathBuf,
}

impl FastEmbedder {
    /// 新規インスタンス作成。**model は load しない**(lazy load)。
    /// `cache_dir` に model file を保管する。
    pub fn new(cache_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&cache_dir).map_err(|e| {
            HippoError::Config(format!("create cache_dir {}: {e}", cache_dir.display()))
        })?;
        Ok(Self {
            inner: Arc::new(Mutex::new(None)),
            cache_dir,
        })
    }

    /// HIPPO_MODEL_CACHE env または ~/.cache/claude-hippo/models/ を使う。
    pub fn from_env() -> Result<Self> {
        let cache_dir = std::env::var("HIPPO_MODEL_CACHE")
            .ok()
            .map(PathBuf::from)
            .unwrap_or_else(default_cache_dir);
        Self::new(cache_dir)
    }

    /// model が load 済か判定。テスト・bench 用。
    pub fn is_loaded(&self) -> bool {
        self.inner.lock().is_some()
    }

    fn ensure_loaded(&self) -> Result<()> {
        let mut guard = self.inner.lock();
        if guard.is_some() {
            return Ok(());
        }
        let opts = fastembed::TextInitOptions::new(fastembed::EmbeddingModel::AllMiniLML6V2)
            .with_cache_dir(self.cache_dir.clone())
            .with_show_download_progress(false);
        let model = fastembed::TextEmbedding::try_new(opts)
            .map_err(|e| HippoError::Embedding(format!("load AllMiniLML6V2: {e}")))?;
        *guard = Some(model);
        Ok(())
    }
}

impl Embedder for FastEmbedder {
    fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let v = self.embed_batch(&[text])?;
        v.into_iter()
            .next()
            .ok_or_else(|| HippoError::Embedding("empty result".into()))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        self.ensure_loaded()?;
        let mut guard = self.inner.lock();
        let model = guard.as_mut().expect("ensure_loaded set this Some");
        let owned: Vec<String> = texts.iter().map(|s| (*s).to_string()).collect();
        let result = model
            .embed(owned, None)
            .map_err(|e| HippoError::Embedding(format!("embed: {e}")))?;
        // L2 正規化を保証 (fastembed は normalize 済の model が多いが、念のため)。
        let normalized: Vec<Vec<f32>> = result
            .into_iter()
            .map(|v| {
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
                let mut out = v;
                for x in out.iter_mut() {
                    *x /= norm;
                }
                if out.len() != EMBEDDING_DIM {
                    return Err(HippoError::Embedding(format!(
                        "model returned dim {} (expected {EMBEDDING_DIM})",
                        out.len()
                    )));
                }
                Ok(out)
            })
            .collect::<Result<_>>()?;
        Ok(normalized)
    }
}

pub fn default_cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("claude-hippo")
        .join("models")
}

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
        let mut v = vec![0.0_f32; EMBEDDING_DIM];
        // seed (32 bytes) を repeat して 384 dim 埋める。
        for (i, b) in (0..EMBEDDING_DIM).zip(seed.iter().cycle()) {
            // -1.0..=1.0 にマップ
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
    fn fast_embedder_lazy_load() {
        let tmp = tempfile::tempdir().unwrap();
        let e = FastEmbedder::new(tmp.path().to_path_buf()).unwrap();
        // model はまだ load されていない
        assert!(!e.is_loaded());
    }
}
