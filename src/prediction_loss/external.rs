//! OpenAI-compatible legacy `/v1/completions` prediction-loss backend.
//!
//! Sends `{ prompt: content, max_tokens: 0, echo: true, logprobs: 1 }`,
//! reads `choices[0].logprobs.token_logprobs[]`, and returns
//! `surprise = clamp(mean_nll / scale, 0, 1)` where `mean_nll = -mean(non_null_logprobs)`.
//!
//! `scale` defaults to 6.0 nats per token: typical English text on a
//! decent LLM hits 2–4 nats / token; specialized / rare sequences reach
//! 5–10. Cap at 6 puts most natural content in the lower half of the
//! `[0, 1]` range and reserves the upper half for genuinely surprising
//! material — exactly what the surprise score wants.
//!
//! See the module docstring (`super::mod`) for the rationale on choosing
//! the legacy endpoint over chat/completions and the supported backends.

use super::PredictionLossBackend;
use crate::{HippoError, Result};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Default cross-entropy scale (nats / token) used to squash mean NLL into
/// `[0, 1]`. Tunable via [`ExternalPredictionLossConfig::loss_scale`].
pub const DEFAULT_LOSS_SCALE: f32 = 6.0;

#[derive(Debug, Clone)]
pub struct ExternalPredictionLossConfig {
    /// `/v1/completions` endpoint. Examples:
    /// `http://localhost:8000/v1/completions` (vLLM),
    /// `http://localhost:8080/completion` (llama.cpp native — not OpenAI-compat shape),
    /// `http://localhost:11434/v1/completions` (Ollama with shim).
    pub url: String,
    /// Model id to score against. `gpt-3.5-turbo-instruct` for legacy
    /// OpenAI; vLLM and llama.cpp use whatever model id was loaded.
    pub model: String,
    /// Already-resolved API key. Empty string skips Authorization header
    /// (use for keyless local backends).
    pub api_key: String,
    /// Per-request timeout.
    pub timeout: Duration,
    /// Retries on 429 / 5xx / network. 4xx (other than 429) → fatal.
    pub max_retries: u32,
    /// Cross-entropy scale (nats / token) for mapping mean NLL to [0,1].
    /// Default [`DEFAULT_LOSS_SCALE`] = 6.0.
    pub loss_scale: f32,
}

impl ExternalPredictionLossConfig {
    pub fn validate(&self) -> Result<()> {
        if self.url.is_empty() {
            return Err(HippoError::Config("prediction-loss url is empty".into()));
        }
        if !(self.url.starts_with("http://") || self.url.starts_with("https://")) {
            return Err(HippoError::Config(format!(
                "prediction-loss url must start with http:// or https://: got {:?}",
                self.url
            )));
        }
        if self.model.is_empty() {
            return Err(HippoError::Config(
                "prediction-loss model name is empty".into(),
            ));
        }
        if !self.loss_scale.is_finite() || self.loss_scale <= 0.0 {
            return Err(HippoError::Config(format!(
                "prediction-loss loss_scale must be > 0 and finite, got {}",
                self.loss_scale
            )));
        }
        Ok(())
    }
}

pub struct ExternalPredictionLossBackend {
    cfg: ExternalPredictionLossConfig,
    client: reqwest::Client,
    headers: HeaderMap,
}

impl ExternalPredictionLossBackend {
    pub fn new(cfg: ExternalPredictionLossConfig) -> Result<Self> {
        cfg.validate()?;
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if !cfg.api_key.is_empty() {
            let bearer = format!("Bearer {}", cfg.api_key);
            let mut v = HeaderValue::from_str(&bearer)
                .map_err(|e| HippoError::Config(format!("invalid prediction-loss api_key: {e}")))?;
            v.set_sensitive(true);
            headers.insert(AUTHORIZATION, v);
        }
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

    pub fn config(&self) -> &ExternalPredictionLossConfig {
        &self.cfg
    }

    async fn score(&self, content: &str) -> Result<f32> {
        if content.trim().is_empty() {
            return Ok(0.0);
        }
        let body = CompletionsRequest {
            model: &self.cfg.model,
            prompt: content,
            max_tokens: 0,
            echo: true,
            logprobs: 1,
            temperature: 0.0,
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
                            "prediction-loss: network error after {} retries: {e}",
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
                let parsed: CompletionsResponse = resp.json().await.map_err(|e| {
                    HippoError::Embedding(format!("prediction-loss: bad JSON body: {e}"))
                })?;
                return self.compute_surprise(parsed);
            }

            let retriable = status.as_u16() == 429 || (500..600).contains(&status.as_u16());
            let body_text = resp.text().await.unwrap_or_default();
            if !retriable || attempt >= self.cfg.max_retries {
                return Err(classify_http_error(status, body_text, &self.cfg));
            }
            tokio::time::sleep(backoff_delay(attempt)).await;
            attempt += 1;
        }
    }

    fn compute_surprise(&self, parsed: CompletionsResponse) -> Result<f32> {
        let choice =
            parsed.choices.into_iter().next().ok_or_else(|| {
                HippoError::Embedding("prediction-loss: empty choices array".into())
            })?;
        let logprobs = choice.logprobs.ok_or_else(|| {
            HippoError::Embedding(
                "prediction-loss: response has no logprobs field — backend may not support \
                 echo+max_tokens=0+logprobs (need vLLM, llama.cpp, or legacy OpenAI completions)"
                    .into(),
            )
        })?;
        // First token's logprob is null (no preceding context). Skip nulls
        // and average the rest.
        let mut count = 0_u32;
        let mut sum = 0.0_f32;
        for lp in logprobs.token_logprobs.iter().flatten() {
            sum += *lp;
            count += 1;
        }
        if count == 0 {
            // Single-token content; treat as neutrally surprising.
            return Ok(0.5);
        }
        let mean_logprob = sum / count as f32;
        let mean_nll = -mean_logprob;
        let scaled = (mean_nll / self.cfg.loss_scale).clamp(0.0, 1.0);
        Ok(scaled)
    }
}

impl PredictionLossBackend for ExternalPredictionLossBackend {
    fn predict_loss(&self, content: &str) -> Result<f32> {
        // Reuse the embeddings module's sync→async bridge so we don't
        // duplicate the block_in_place logic.
        run_async_in_sync(self.score(content))
    }
}

// ---------- HTTP shapes ----------

#[derive(Debug, Serialize)]
struct CompletionsRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    max_tokens: u32,
    echo: bool,
    logprobs: u32,
    temperature: f32,
}

#[derive(Debug, Deserialize)]
struct CompletionsResponse {
    choices: Vec<Choice>,
    #[allow(dead_code)]
    model: Option<String>,
    #[allow(dead_code)]
    usage: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    #[allow(dead_code)]
    text: Option<String>,
    logprobs: Option<Logprobs>,
    #[allow(dead_code)]
    index: Option<u32>,
    #[allow(dead_code)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Logprobs {
    /// First entry is `null` (no preceding context to predict from).
    token_logprobs: Vec<Option<f32>>,
    #[allow(dead_code)]
    tokens: Option<Vec<String>>,
    #[allow(dead_code)]
    text_offset: Option<Vec<u32>>,
}

// ---------- helpers ----------

fn backoff_delay(attempt: u32) -> Duration {
    let base_ms: u64 = 200_u64.saturating_mul(1_u64 << attempt.min(5));
    Duration::from_millis(base_ms.min(5_000))
}

fn classify_http_error(
    status: reqwest::StatusCode,
    body: String,
    cfg: &ExternalPredictionLossConfig,
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
        "prediction-loss: {kind} — status={} url={} model={} body={:?}",
        status, cfg.url, cfg.model, body_preview
    ))
}

/// Sync→async bridge identical to `embeddings::external::run_async_in_sync`.
/// Duplicated rather than re-exported to keep the modules independent.
fn run_async_in_sync<F, T>(fut: F) -> Result<T>
where
    F: std::future::Future<Output = Result<T>> + Send,
    T: Send,
{
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        match handle.runtime_flavor() {
            tokio::runtime::RuntimeFlavor::MultiThread => {
                tokio::task::block_in_place(|| handle.block_on(fut))
            }
            _ => std::thread::scope(|s| {
                s.spawn(|| {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .map_err(|e| HippoError::Embedding(format!("tokio runtime: {e}")))?;
                    rt.block_on(fut)
                })
                .join()
                .map_err(|_| HippoError::Embedding("prediction-loss worker panicked".into()))?
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

    fn cfg() -> ExternalPredictionLossConfig {
        ExternalPredictionLossConfig {
            url: "https://example.com/v1/completions".into(),
            model: "gpt-3.5-turbo-instruct".into(),
            api_key: "sk-test".into(),
            timeout: Duration::from_secs(2),
            max_retries: 1,
            loss_scale: DEFAULT_LOSS_SCALE,
        }
    }

    #[test]
    fn validate_rejects_empty_url() {
        let mut c = cfg();
        c.url = String::new();
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_rejects_non_http() {
        let mut c = cfg();
        c.url = "ws://example.com".into();
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_rejects_bad_loss_scale() {
        let mut c = cfg();
        c.loss_scale = 0.0;
        assert!(c.validate().is_err());
        c.loss_scale = -1.0;
        assert!(c.validate().is_err());
        c.loss_scale = f32::NAN;
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_accepts_canonical() {
        assert!(cfg().validate().is_ok());
    }

    #[test]
    fn new_builds() {
        let _ = ExternalPredictionLossBackend::new(cfg()).unwrap();
    }

    #[test]
    fn backoff_caps() {
        let d8 = backoff_delay(8);
        assert!(d8.as_millis() <= 5_000);
    }
}
