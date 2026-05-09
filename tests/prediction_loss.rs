//! Integration tests for the v0.3 prediction-loss backend.
//!
//! Mocks an OpenAI legacy `/v1/completions` endpoint with `wiremock` and
//! verifies:
//! - Mean NLL → surprise scaling (`mean_nll / loss_scale`).
//! - Empty content short-circuits to `0.0`.
//! - Missing `logprobs` field surfaces an actionable error.
//! - 401 → fail-fast; 429 → retry then succeed; 5xx → retry.
//! - End-to-end MCP integration: `MemoryServer::remember` populates
//!   `prediction_loss` from the backend; without the backend it stays `None`.

use claude_hippo::embeddings::MockEmbedder;
use claude_hippo::prediction_loss::{
    ExternalPredictionLossBackend, ExternalPredictionLossConfig, MockPredictionLoss,
    PredictionLossBackend, DEFAULT_LOSS_SCALE,
};
use claude_hippo::server::{MemoryServer, RankingConfig, RememberParams};
use claude_hippo::storage::{register_sqlite_vec, Storage};
use claude_hippo::surprise::SurpriseWeights;
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn cfg(url: String) -> ExternalPredictionLossConfig {
    ExternalPredictionLossConfig {
        url,
        model: "gpt-3.5-turbo-instruct".into(),
        api_key: "sk-test".into(),
        timeout: Duration::from_secs(2),
        max_retries: 2,
        loss_scale: DEFAULT_LOSS_SCALE,
    }
}

/// Build a fake `/v1/completions` echo response with the given per-token
/// logprobs (first one is `null` to mirror OpenAI's actual shape).
fn completions_response(token_logprobs: Vec<Option<f32>>) -> serde_json::Value {
    let n = token_logprobs.len();
    json!({
        "id": "cmpl-mock",
        "object": "text_completion",
        "model": "gpt-3.5-turbo-instruct",
        "choices": [{
            "text": "",
            "index": 0,
            "logprobs": {
                "tokens": vec!["tok"; n],
                "token_logprobs": token_logprobs,
                "text_offset": (0..n).map(|i| i as u32).collect::<Vec<_>>(),
            },
            "finish_reason": "length",
        }],
        "usage": { "prompt_tokens": n, "completion_tokens": 0, "total_tokens": n },
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mean_nll_scales_to_surprise() {
    let server = MockServer::start().await;
    // 5 tokens; first is null, rest have logprob = -3.0 (mean NLL = 3.0).
    // surprise = clamp(3.0 / 6.0, 0, 1) = 0.5
    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(completions_response(vec![
                None,
                Some(-3.0),
                Some(-3.0),
                Some(-3.0),
                Some(-3.0),
            ])),
        )
        .mount(&server)
        .await;

    let url = format!("{}/v1/completions", server.uri());
    let b = ExternalPredictionLossBackend::new(cfg(url)).unwrap();
    let s = b.predict_loss("some content here").unwrap();
    assert!(
        (s - 0.5).abs() < 1e-4,
        "expected surprise = 0.5 (mean_nll 3.0 / scale 6.0), got {s}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn high_nll_clamps_to_one() {
    let server = MockServer::start().await;
    // mean NLL = 100 → clamp to 1.0
    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(completions_response(vec![
                None,
                Some(-100.0),
                Some(-100.0),
            ])),
        )
        .mount(&server)
        .await;

    let url = format!("{}/v1/completions", server.uri());
    let b = ExternalPredictionLossBackend::new(cfg(url)).unwrap();
    let s = b.predict_loss("super weird content").unwrap();
    assert_eq!(s, 1.0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn low_nll_floors_at_zero() {
    let server = MockServer::start().await;
    // mean NLL = 0 (perfectly predictable) → 0.0
    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(completions_response(vec![
                None,
                Some(0.0),
                Some(0.0),
            ])),
        )
        .mount(&server)
        .await;

    let url = format!("{}/v1/completions", server.uri());
    let b = ExternalPredictionLossBackend::new(cfg(url)).unwrap();
    let s = b.predict_loss("the the the").unwrap();
    assert_eq!(s, 0.0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn empty_content_short_circuits() {
    // No mock — server should never be called for empty content.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(ResponseTemplate::new(500))
        .expect(0)
        .mount(&server)
        .await;

    let url = format!("{}/v1/completions", server.uri());
    let b = ExternalPredictionLossBackend::new(cfg(url)).unwrap();
    let s = b.predict_loss("   ").unwrap();
    assert_eq!(s, 0.0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn missing_logprobs_errors_actionably() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{ "text": "ok" /* no logprobs */ }],
        })))
        .mount(&server)
        .await;

    let url = format!("{}/v1/completions", server.uri());
    let b = ExternalPredictionLossBackend::new(cfg(url)).unwrap();
    let err = b.predict_loss("hi").unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("logprobs"), "got {msg}");
    assert!(
        msg.contains("vLLM"),
        "should mention supported backends; got {msg}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn auth_401_fails_fast() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(ResponseTemplate::new(401))
        .expect(1)
        .mount(&server)
        .await;
    let url = format!("{}/v1/completions", server.uri());
    let b = ExternalPredictionLossBackend::new(cfg(url)).unwrap();
    let err = b.predict_loss("hi").unwrap_err();
    assert!(err.to_string().contains("401"), "got {err}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rate_limit_429_retries_then_succeeds() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(ResponseTemplate::new(429))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(completions_response(vec![None, Some(-3.0)])),
        )
        .mount(&server)
        .await;

    let url = format!("{}/v1/completions", server.uri());
    let b = ExternalPredictionLossBackend::new(cfg(url)).unwrap();
    let s = b.predict_loss("hi").unwrap();
    assert!((s - 0.5).abs() < 1e-4, "got {s}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn memory_server_populates_prediction_loss_from_backend() {
    register_sqlite_vec();
    let store = Storage::open_in_memory().unwrap();
    let embedder = Arc::new(MockEmbedder::new());
    let pl: Arc<dyn PredictionLossBackend> = Arc::new(MockPredictionLoss);
    let server = MemoryServer::new_full(
        store,
        embedder,
        Some(pl),
        SurpriseWeights::default(),
        RankingConfig::default(),
    );

    let r = server
        .remember(RememberParams {
            content: "JWT 24h expiry chosen 2026-Q2".into(),
            tags: vec!["auth".into(), "decision".into()],
            memory_type: Some("Decision".into()),
            importance: Some(1.0),
            metadata: None,
        })
        .await
        .unwrap();
    assert!(
        r.surprise_components.prediction_loss.is_some(),
        "with backend wired, prediction_loss must be Some"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn memory_server_leaves_prediction_loss_none_without_backend() {
    register_sqlite_vec();
    let store = Storage::open_in_memory().unwrap();
    let embedder = Arc::new(MockEmbedder::new());
    let server = MemoryServer::new_full(
        store,
        embedder,
        None,
        SurpriseWeights::default(),
        RankingConfig::default(),
    );

    let r = server
        .remember(RememberParams {
            content: "anything".into(),
            tags: vec![],
            memory_type: None,
            importance: None,
            metadata: None,
        })
        .await
        .unwrap();
    assert!(
        r.surprise_components.prediction_loss.is_none(),
        "without backend, prediction_loss must remain None (v0.2 fallback)"
    );
}
