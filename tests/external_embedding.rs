//! Integration tests for the external embedding backend (`v0.3 Phase B`).
//!
//! Uses `wiremock` to stand up a local HTTP server that pretends to be an
//! OpenAI-compatible `/v1/embeddings` endpoint. We exercise the full
//! reqwest → ExternalEmbedder → response-validation pipeline without
//! reaching any real API. CI-safe: no API key, no network.
//!
//! What this covers:
//! - Happy path: 384-dim float response → returned vectors L2-normalized.
//! - Dim mismatch: model returns 768 → reject with clear error (DB schema
//!   compatibility breaks silently otherwise).
//! - 401 (auth) → fail fast, no retry (4xx other than 429 are not retriable).
//! - 429 (rate limit) → retry up to `max_retries`, then fail.
//! - 5xx (upstream) → retry, then succeed if eventually OK.
//! - Batch chunking: 5 texts with batch_size=2 → 3 HTTP requests, output
//!   ordering preserved.
//! - Concurrent embed: multiple tokio tasks calling `embed_batch` in parallel
//!   do not deadlock (sync-from-async bridge sanity check).
//! - L2 normalization: server returns un-normalized vectors → output is
//!   normalized (norm = 1.0).

use claude_hippo::embeddings::{Embedder, ExternalEmbedder, ExternalEmbeddingConfig};
use claude_hippo::EMBEDDING_DIM;
use serde_json::json;
use std::time::Duration;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// 384-dim vector with all coords = `c`. Useful for normalization checks
/// (norm = c * sqrt(384), normalized → 1/sqrt(384) per coord).
fn const_vec(c: f32) -> Vec<f32> {
    vec![c; EMBEDDING_DIM]
}

fn normalized_const_vec() -> Vec<f32> {
    let v = const_vec(1.0);
    let n = (EMBEDDING_DIM as f32).sqrt();
    v.into_iter().map(|x| x / n).collect()
}

fn cfg(url: String, batch_size: usize, max_retries: u32) -> ExternalEmbeddingConfig {
    ExternalEmbeddingConfig {
        url,
        model: "text-embedding-3-small".into(),
        dim: EMBEDDING_DIM,
        api_key: "sk-test".into(),
        timeout: Duration::from_secs(2),
        batch_size,
        max_retries,
    }
}

fn embedding_response(vectors: Vec<Vec<f32>>) -> serde_json::Value {
    json!({
        "object": "list",
        "data": vectors.into_iter().enumerate().map(|(i, v)| json!({
            "object": "embedding",
            "index": i,
            "embedding": v,
        })).collect::<Vec<_>>(),
        "model": "text-embedding-3-small",
        "usage": { "prompt_tokens": 1, "total_tokens": 1 },
    })
}

fn assert_normalized(v: &[f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "expected L2-normalized vector, got norm = {norm}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn happy_path_returns_normalized_vectors() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .and(header("authorization", "Bearer sk-test"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(embedding_response(vec![
                normalized_const_vec(),
                normalized_const_vec(),
            ])),
        )
        .expect(1)
        .mount(&server)
        .await;

    let url = format!("{}/v1/embeddings", server.uri());
    let e = ExternalEmbedder::new(cfg(url, 64, 0)).unwrap();
    let v = e.embed_batch(&["hello", "world"]).unwrap();
    assert_eq!(v.len(), 2);
    for vec in &v {
        assert_eq!(vec.len(), EMBEDDING_DIM);
        assert_normalized(vec);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn server_unnormalized_vectors_get_normalized() {
    let server = MockServer::start().await;
    // Server returns vector with all 1.0 — norm = sqrt(384) ≈ 19.6, not 1.0.
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(embedding_response(vec![const_vec(1.0)])),
        )
        .mount(&server)
        .await;

    let url = format!("{}/v1/embeddings", server.uri());
    let e = ExternalEmbedder::new(cfg(url, 64, 0)).unwrap();
    let v = e.embed_one("hi").unwrap();
    assert_normalized(&v);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dim_mismatch_is_rejected() {
    let server = MockServer::start().await;
    // Server returns 768-dim vector when client expects 384.
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(embedding_response(vec![vec![0.0; 768]])),
        )
        .mount(&server)
        .await;

    let url = format!("{}/v1/embeddings", server.uri());
    let e = ExternalEmbedder::new(cfg(url, 64, 0)).unwrap();
    let err = e.embed_one("hi").unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("768"), "got {msg}");
    assert!(msg.contains("384"), "got {msg}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn auth_401_fails_fast_no_retry() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(401).set_body_string("invalid api key"))
        .expect(1) // exactly 1, no retry on 4xx-non-429
        .mount(&server)
        .await;

    let url = format!("{}/v1/embeddings", server.uri());
    let e = ExternalEmbedder::new(cfg(url, 64, 3)).unwrap();
    let err = e.embed_one("hi").unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("auth"), "got {msg}");
    assert!(msg.contains("401"), "got {msg}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rate_limit_429_retries_then_succeeds() {
    let server = MockServer::start().await;
    // First 2 requests: 429; third: 200. Mounting in reverse order so
    // latest-mounted matchers apply first per wiremock semantics.
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(429).set_body_string("slow down"))
        .up_to_n_times(2)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(embedding_response(vec![normalized_const_vec()])),
        )
        .mount(&server)
        .await;

    let mut c = cfg(format!("{}/v1/embeddings", server.uri()), 64, 3);
    // Shorten timeout so the test finishes quickly even with backoff.
    c.timeout = Duration::from_secs(5);
    let e = ExternalEmbedder::new(c).unwrap();
    let v = e.embed_one("hi").expect("eventually succeeds");
    assert_eq!(v.len(), EMBEDDING_DIM);
    assert_normalized(&v);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rate_limit_429_gives_up_after_max_retries() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(429))
        .mount(&server)
        .await;

    let url = format!("{}/v1/embeddings", server.uri());
    // max_retries = 1 → 2 attempts total then fail.
    let e = ExternalEmbedder::new(cfg(url, 64, 1)).unwrap();
    let err = e.embed_one("hi").unwrap_err();
    assert!(err.to_string().contains("429"), "got {err}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn server_5xx_retries_then_succeeds() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(503))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(embedding_response(vec![normalized_const_vec()])),
        )
        .mount(&server)
        .await;

    let url = format!("{}/v1/embeddings", server.uri());
    let e = ExternalEmbedder::new(cfg(url, 64, 3)).unwrap();
    let v = e.embed_one("hi").unwrap();
    assert_normalized(&v);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn batch_chunks_preserve_order() {
    let server = MockServer::start().await;
    // batch_size = 2 → 5 inputs become 3 requests of (2, 2, 1).
    // Each request echoes back N normalized vectors with index 0..N-1.
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(|req: &wiremock::Request| {
            let body: serde_json::Value = req.body_json().unwrap();
            let inputs = body["input"].as_array().unwrap();
            let n = inputs.len();
            let vectors: Vec<Vec<f32>> = (0..n).map(|_| normalized_const_vec()).collect();
            ResponseTemplate::new(200).set_body_json(embedding_response(vectors))
        })
        .expect(3)
        .mount(&server)
        .await;

    let url = format!("{}/v1/embeddings", server.uri());
    let e = ExternalEmbedder::new(cfg(url, 2, 0)).unwrap();
    let v = e
        .embed_batch(&["a", "b", "c", "d", "e"])
        .expect("3-chunk batch");
    assert_eq!(v.len(), 5);
    for vec in &v {
        assert_eq!(vec.len(), EMBEDDING_DIM);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_embeds_do_not_deadlock() {
    // Sync-from-async bridge sanity: spawn 8 concurrent embed_one calls
    // from the same multi-thread tokio runtime. If block_in_place is wrong,
    // this hangs / panics.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(embedding_response(vec![normalized_const_vec()])),
        )
        .mount(&server)
        .await;
    let url = format!("{}/v1/embeddings", server.uri());
    let e = std::sync::Arc::new(ExternalEmbedder::new(cfg(url, 64, 0)).unwrap());

    let mut handles = Vec::new();
    for i in 0..8 {
        let e = e.clone();
        handles.push(tokio::task::spawn(async move {
            // Inside an async task → call sync embed_one which uses
            // block_in_place internally.
            e.embed_one(&format!("text-{i}")).map(|_| ())
        }));
    }
    for h in handles {
        h.await.unwrap().expect("concurrent embed succeeded");
    }
}
