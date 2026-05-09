//! Local embedding backend — fastembed (ONNX runtime).
//!
//! Default model: `all-MiniLM-L6-v2` (384 dim) — mcp-memory-service-rs と
//! 同一 vector space を使うため、DB を swap しても retrieval semantics が
//! 保たれる。
//!
//! Lazy load: model は initial `embed` 呼び出し時に load される。`serve`
//! cold-start を 23 MB 程度に抑える。

use super::Embedder;
use crate::{HippoError, Result, EMBEDDING_DIM};
use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::Arc;

/// 384 次元を保つ ONNX モデル。`--embedding-model` から選択。
///
/// **DB swap 互換性のため 384 dim 縛りは絶対**: mcp-memory-service-rs は
/// 384 dim 固定で、別 dim のモデルを混ぜると memory_embeddings の vec0 列と
/// 整合しない (storage::SCHEMA_SQL は FLOAT[384])。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EmbeddingModelKind {
    /// `sentence-transformers/all-MiniLM-L6-v2` (default).
    /// 384 dim, ~23 MB ONNX, mcp-memory-service-rs と同 vector space。
    #[default]
    MiniLmL6V2,
    /// `Qdrant/bge-small-en-v1.5-onnx-Q` (量子化変種)。
    /// 384 dim、~33 MB に軽量化、cold-start の DL コストが小さい。
    /// **vector space は MiniLM と互換性なし**: 既存 DB を切替時はリビルド必須。
    BgeSmallEnV15Q,
}

impl EmbeddingModelKind {
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "minilm-l6-v2" | "all-minilm-l6-v2" | "minilm" => Ok(Self::MiniLmL6V2),
            "bge-small-en-v15-q" | "bge-small-en-v1.5-q" | "bge-q" => Ok(Self::BgeSmallEnV15Q),
            other => Err(format!(
                "unknown embedding model: {other:?} (expected: minilm-l6-v2, bge-small-en-v15-q)"
            )),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::MiniLmL6V2 => "minilm-l6-v2",
            Self::BgeSmallEnV15Q => "bge-small-en-v15-q",
        }
    }

    pub fn fastembed_model(&self) -> fastembed::EmbeddingModel {
        match self {
            Self::MiniLmL6V2 => fastembed::EmbeddingModel::AllMiniLML6V2,
            Self::BgeSmallEnV15Q => fastembed::EmbeddingModel::BGESmallENV15Q,
        }
    }
}

/// Fast-embed local (ONNX) backend。lazy load (initial `embed` で model load)。
pub struct FastEmbedder {
    inner: Arc<Mutex<Option<fastembed::TextEmbedding>>>,
    cache_dir: PathBuf,
    model: EmbeddingModelKind,
}

impl FastEmbedder {
    /// 新規インスタンス作成 (default model)。**model は load しない**(lazy load)。
    pub fn new(cache_dir: PathBuf) -> Result<Self> {
        Self::new_with_model(cache_dir, EmbeddingModelKind::default())
    }

    /// model 指定版。
    pub fn new_with_model(cache_dir: PathBuf, model: EmbeddingModelKind) -> Result<Self> {
        std::fs::create_dir_all(&cache_dir).map_err(|e| {
            HippoError::Config(format!("create cache_dir {}: {e}", cache_dir.display()))
        })?;
        Ok(Self {
            inner: Arc::new(Mutex::new(None)),
            cache_dir,
            model,
        })
    }

    /// HIPPO_MODEL_CACHE env または ~/.cache/claude-hippo/models/ を使う。
    pub fn from_env() -> Result<Self> {
        let cache_dir = std::env::var("HIPPO_MODEL_CACHE")
            .ok()
            .map(PathBuf::from)
            .unwrap_or_else(super::default_cache_dir);
        Self::new(cache_dir)
    }

    /// model kind getter。
    pub fn model_kind(&self) -> EmbeddingModelKind {
        self.model
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
        let model_id = self.model.fastembed_model();
        let opts = fastembed::TextInitOptions::new(model_id)
            .with_cache_dir(self.cache_dir.clone())
            .with_show_download_progress(false);
        let model = fastembed::TextEmbedding::try_new(opts)
            .map_err(|e| HippoError::Embedding(format!("load {}: {e}", self.model.as_str())))?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_embedder_lazy_load() {
        let tmp = tempfile::tempdir().unwrap();
        let e = FastEmbedder::new(tmp.path().to_path_buf()).unwrap();
        // model はまだ load されていない
        assert!(!e.is_loaded());
    }

    #[test]
    fn embedding_model_parse_canonical() {
        assert_eq!(
            EmbeddingModelKind::parse("minilm-l6-v2").unwrap(),
            EmbeddingModelKind::MiniLmL6V2
        );
        assert_eq!(
            EmbeddingModelKind::parse("bge-small-en-v15-q").unwrap(),
            EmbeddingModelKind::BgeSmallEnV15Q
        );
    }

    #[test]
    fn embedding_model_parse_aliases() {
        assert_eq!(
            EmbeddingModelKind::parse("MiniLM").unwrap(),
            EmbeddingModelKind::MiniLmL6V2
        );
        assert_eq!(
            EmbeddingModelKind::parse("All-MiniLM-L6-v2").unwrap(),
            EmbeddingModelKind::MiniLmL6V2
        );
        assert_eq!(
            EmbeddingModelKind::parse("bge-small-en-v1.5-q").unwrap(),
            EmbeddingModelKind::BgeSmallEnV15Q
        );
        assert_eq!(
            EmbeddingModelKind::parse("BGE-Q").unwrap(),
            EmbeddingModelKind::BgeSmallEnV15Q
        );
    }

    #[test]
    fn embedding_model_parse_rejects_unknown() {
        assert!(EmbeddingModelKind::parse("gpt-4").is_err());
        assert!(EmbeddingModelKind::parse("").is_err());
    }

    #[test]
    fn embedding_model_default_is_minilm() {
        assert_eq!(
            EmbeddingModelKind::default(),
            EmbeddingModelKind::MiniLmL6V2
        );
    }

    #[test]
    fn fast_embedder_carries_model_kind() {
        let tmp = tempfile::tempdir().unwrap();
        let e = FastEmbedder::new_with_model(
            tmp.path().to_path_buf(),
            EmbeddingModelKind::BgeSmallEnV15Q,
        )
        .unwrap();
        assert_eq!(e.model_kind(), EmbeddingModelKind::BgeSmallEnV15Q);
    }
}
