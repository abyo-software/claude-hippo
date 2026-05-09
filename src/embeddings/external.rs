//! External embedding backend — OpenAI-compatible HTTP `/v1/embeddings`.
//!
//! Drop-in replacement for [`crate::embeddings::FastEmbedder`] that delegates
//! inference to a remote (or local) HTTP service. Usable with OpenAI,
//! Azure OpenAI, Ollama, vLLM, llama.cpp, LM Studio, HuggingFace TEI,
//! Together, OpenRouter — anything that exposes a POST `/v1/embeddings`
//! returning `data[].embedding`.
//!
//! # Why
//! Local fastembed pulls in `ort` + ONNX model = ~150 MB RSS. External
//! shrinks the in-process footprint to ~25 MB (sqlite-vec + reqwest + rustls),
//! at the cost of network dependency.
//!
//! # Compat invariants
//! - L2-normalize the response (KNN cosine math depends on it).
//! - Reject any response whose dim != `EMBEDDING_DIM` (DB schema is
//!   `FLOAT[384]` — silent dim drift would corrupt the vector index).
//! - Send `dimensions: 384` in the request so OpenAI v3 (which supports
//!   per-call dim) returns 384. Other providers ignore the field; if they
//!   return something other than 384, we fail loud rather than truncate.
//!
//! # Async-from-sync bridging
//! The [`super::Embedder`] trait is sync because [`crate::server`] calls
//! into it from inside `async` MCP handlers without `.await`. Reqwest's
//! async `Client` cannot be `.await`ed from a sync function, and the
//! blocking `Client` panics if invoked from inside a Tokio runtime. We
//! reconcile by holding an async `Client` and using
//! [`tokio::task::block_in_place`] + [`tokio::runtime::Handle::block_on`]
//! when called from within a Tokio context (the production path), and
//! falling back to spinning a single-threaded runtime for sync-only test
//! contexts.

use super::Embedder;
use crate::{HippoError, Result, EMBEDDING_DIM};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Build-time configuration for [`ExternalEmbedder`]. All fields are user
/// supplied; defaults live in CLI parsing.
#[derive(Debug, Clone)]
pub struct ExternalEmbeddingConfig {
    /// Full URL of the embeddings endpoint, e.g.
    /// `https://api.openai.com/v1/embeddings` or
    /// `http://localhost:11434/v1/embeddings`.
    pub url: String,
    /// Model name passed in the request body (`"model"` field).
    pub model: String,
    /// Expected output dim. Must equal [`EMBEDDING_DIM`] (384) — anything
    /// else breaks DB schema compatibility and is rejected at construction.
    pub dim: usize,
    /// API key (already resolved from env). Empty string disables the
    /// `Authorization: Bearer …` header (use for keyless local backends
    /// like Ollama).
    pub api_key: String,
    /// Per-request timeout.
    pub timeout: Duration,
    /// Max texts per HTTP request. Larger requests get split into chunks
    /// and concatenated.
    pub batch_size: usize,
    /// Max retries on 429 / 5xx / network errors. Each retry waits
    /// `200 * 2^attempt` ms (capped at 5 s). 4xx other than 429 is fatal.
    pub max_retries: u32,
}

impl ExternalEmbeddingConfig {
    pub fn validate(&self) -> Result<()> {
        if self.url.is_empty() {
            return Err(HippoError::Config("external embedding url is empty".into()));
        }
        if self.model.is_empty() {
            return Err(HippoError::Config(
                "external embedding model name is empty".into(),
            ));
        }
        if self.dim != EMBEDDING_DIM {
            return Err(HippoError::Config(format!(
                "external embedding dim {} != schema-required {} (DB swap compat — \
                 mcp-memory-service-rs uses FLOAT[384])",
                self.dim, EMBEDDING_DIM
            )));
        }
        if self.batch_size == 0 {
            return Err(HippoError::Config(
                "external embedding batch_size must be ≥ 1".into(),
            ));
        }
        if !(self.url.starts_with("http://") || self.url.starts_with("https://")) {
            return Err(HippoError::Config(format!(
                "external embedding url must start with http:// or https://: got {:?}",
                self.url
            )));
        }
        Ok(())
    }
}

pub struct ExternalEmbedder {
    cfg: ExternalEmbeddingConfig,
    client: reqwest::Client,
    headers: HeaderMap,
}

impl ExternalEmbedder {
    pub fn new(cfg: ExternalEmbeddingConfig) -> Result<Self> {
        cfg.validate()?;
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if !cfg.api_key.is_empty() {
            let bearer = format!("Bearer {}", cfg.api_key);
            let mut v = HeaderValue::from_str(&bearer)
                .map_err(|e| HippoError::Config(format!("invalid api_key for header: {e}")))?;
            v.set_sensitive(true);
            headers.insert(AUTHORIZATION, v);
        }
        // Identify ourselves so backend logs can attribute traffic.
        headers.insert(
            HeaderName::from_static("user-agent"),
            HeaderValue::from_static(concat!("claude-hippo/", env!("CARGO_PKG_VERSION"))),
        );

        let client = reqwest::Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| HippoError::Config(format!("reqwest client build: {e}")))?;
        Ok(Self {
            cfg,
            client,
            headers,
        })
    }

    pub fn config(&self) -> &ExternalEmbeddingConfig {
        &self.cfg
    }

    async fn send_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let body = EmbeddingsRequest {
            model: &self.cfg.model,
            input: texts,
            encoding_format: "float",
            dimensions: self.cfg.dim as u32,
        };

        let mut attempt: u32 = 0;
        loop {
            let resp_result = self
                .client
                .post(&self.cfg.url)
                .headers(self.headers.clone())
                .json(&body)
                .send()
                .await;

            let resp = match resp_result {
                Ok(r) => r,
                Err(e) => {
                    if attempt >= self.cfg.max_retries {
                        return Err(HippoError::Embedding(format!(
                            "external embeddings: network error after {} retries: {e}",
                            attempt
                        )));
                    }
                    tokio::time::sleep(backoff_delay(attempt)).await;
                    attempt += 1;
                    continue;
                }
            };

            let status = resp.status();
            if status.is_success() {
                let parsed: EmbeddingsResponse = resp.json().await.map_err(|e| {
                    HippoError::Embedding(format!("external embeddings: bad JSON body: {e}"))
                })?;
                return self.normalize_response(parsed);
            }

            // 429 + 5xx → retry; other 4xx → fatal.
            let retriable = status.as_u16() == 429 || (500..600).contains(&status.as_u16());
            let body_text = resp.text().await.unwrap_or_default();
            if !retriable || attempt >= self.cfg.max_retries {
                return Err(classify_http_error(status, body_text, &self.cfg));
            }
            tokio::time::sleep(backoff_delay(attempt)).await;
            attempt += 1;
        }
    }

    fn normalize_response(&self, parsed: EmbeddingsResponse) -> Result<Vec<Vec<f32>>> {
        // Reorder by `index` so caller-side ordering survives upstream re-ordering.
        let mut data = parsed.data;
        data.sort_by_key(|d| d.index);

        let mut out = Vec::with_capacity(data.len());
        for d in data {
            if d.embedding.len() != self.cfg.dim {
                return Err(HippoError::Embedding(format!(
                    "external embeddings: model {:?} returned dim {} (expected {} for DB \
                     schema FLOAT[384] — reject rather than silently truncate)",
                    self.cfg.model,
                    d.embedding.len(),
                    self.cfg.dim
                )));
            }
            let mut v = d.embedding;
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            for x in v.iter_mut() {
                *x /= norm;
            }
            out.push(v);
        }
        Ok(out)
    }
}

impl Embedder for ExternalEmbedder {
    fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let v = self.embed_batch(&[text])?;
        v.into_iter()
            .next()
            .ok_or_else(|| HippoError::Embedding("external embeddings: empty response".into()))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let owned: Vec<String> = texts.iter().map(|s| (*s).to_string()).collect();

        let mut out = Vec::with_capacity(owned.len());
        for chunk in owned.chunks(self.cfg.batch_size) {
            let chunk_owned = chunk.to_vec();
            let part = run_async_in_sync(self.send_batch(&chunk_owned))?;
            out.extend(part);
        }
        Ok(out)
    }
}

// ---------- HTTP shapes (OpenAI-compatible) ----------

#[derive(Debug, Serialize)]
struct EmbeddingsRequest<'a> {
    model: &'a str,
    input: &'a [String],
    encoding_format: &'static str,
    dimensions: u32,
}

#[derive(Debug, Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingDatum>,
    #[allow(dead_code)]
    model: Option<String>,
    #[allow(dead_code)]
    object: Option<String>,
    #[allow(dead_code)]
    usage: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingDatum {
    embedding: Vec<f32>,
    #[serde(default)]
    index: usize,
    #[allow(dead_code)]
    object: Option<String>,
}

// ---------- helpers ----------

fn backoff_delay(attempt: u32) -> Duration {
    let base_ms: u64 = 200_u64.saturating_mul(1_u64 << attempt.min(5));
    Duration::from_millis(base_ms.min(5_000))
}

fn classify_http_error(
    status: reqwest::StatusCode,
    body: String,
    cfg: &ExternalEmbeddingConfig,
) -> HippoError {
    let body_preview = body.chars().take(400).collect::<String>();
    let kind = match status.as_u16() {
        401 => "auth: API key invalid or missing",
        403 => "auth: API key rejected for this model",
        404 => "endpoint not found (URL or model name wrong)",
        429 => "rate limited (gave up after retries)",
        s if (500..600).contains(&s) => "upstream 5xx (gave up after retries)",
        _ => "unexpected HTTP error",
    };
    HippoError::Embedding(format!(
        "external embeddings: {kind} — status={} url={} model={} body={:?}",
        status, cfg.url, cfg.model, body_preview
    ))
}

/// Bridge sync→async. Production path is "called from inside an async MCP
/// handler" → use the existing runtime via `block_in_place + handle.block_on`.
/// Sync test path: spin up a temporary current-thread runtime.
fn run_async_in_sync<F, T>(fut: F) -> Result<T>
where
    F: std::future::Future<Output = Result<T>> + Send,
    T: Send,
{
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        // We're inside Tokio; block_in_place lets us call block_on without
        // deadlocking the runtime (requires the multi-thread runtime, which
        // is what `serve` uses).
        match handle.runtime_flavor() {
            tokio::runtime::RuntimeFlavor::MultiThread => {
                tokio::task::block_in_place(|| handle.block_on(fut))
            }
            // Current-thread runtime can't block_on from within itself.
            // Spawn a fresh thread with its own runtime to host the future.
            _ => std::thread::scope(|s| {
                s.spawn(|| {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .map_err(|e| HippoError::Embedding(format!("tokio runtime: {e}")))?;
                    rt.block_on(fut)
                })
                .join()
                .map_err(|_| HippoError::Embedding("embedding worker panicked".into()))?
            }),
        }
    } else {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| HippoError::Embedding(format!("tokio runtime: {e}")))?;
        rt.block_on(fut)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_with(url: String) -> ExternalEmbeddingConfig {
        ExternalEmbeddingConfig {
            url,
            model: "text-embedding-3-small".into(),
            dim: EMBEDDING_DIM,
            api_key: "sk-test".into(),
            timeout: Duration::from_secs(2),
            batch_size: 4,
            max_retries: 2,
        }
    }

    #[test]
    fn validate_rejects_wrong_dim() {
        let mut cfg = cfg_with("https://example.com/v1/embeddings".into());
        cfg.dim = 768;
        let err = cfg.validate().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("768"), "got {msg}");
        assert!(msg.contains("384"), "got {msg}");
    }

    #[test]
    fn validate_rejects_empty_url() {
        let cfg = cfg_with(String::new());
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_non_http_url() {
        let cfg = cfg_with("file:///etc/passwd".into());
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_batch() {
        let mut cfg = cfg_with("https://example.com".into());
        cfg.batch_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn new_builds_when_valid() {
        let cfg = cfg_with("https://example.com/v1/embeddings".into());
        let e = ExternalEmbedder::new(cfg).expect("build");
        assert_eq!(e.config().model, "text-embedding-3-small");
    }

    #[test]
    fn backoff_grows_then_caps() {
        let d0 = backoff_delay(0);
        let d3 = backoff_delay(3);
        let d8 = backoff_delay(8);
        assert!(d3 > d0);
        assert!(d8.as_millis() <= 5_000);
    }
}
