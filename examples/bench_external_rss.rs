//! Self-contained RSS measurement for the v0.3 external embedding backend.
//!
//! Why: the local fastembed backend reports RSS via `hippo bench`, but
//! `--embedding-backend external` needs an HTTP server to talk to, which
//! makes the standard `hippo bench` flow harder to run in isolation. This
//! example spins up wiremock in-process so the measurement is reproducible
//! without depending on Ollama / OpenAI / TEI being installed.
//!
//! Goal (per docs/EXTERNAL_EMBEDDING.md): peak RSS < 30 MB. If the actual
//! number is higher, document it honestly rather than papering over.
//!
//! Run:
//! ```sh
//! cargo run --release --example bench_external_rss
//! ```

use claude_hippo::embeddings::{Embedder, ExternalEmbedder, ExternalEmbeddingConfig};
use claude_hippo::server::{MemoryServer, RankingConfig, RecallParams, RememberParams};
use claude_hippo::storage::{register_sqlite_vec, Storage};
use claude_hippo::surprise::SurpriseWeights;
use claude_hippo::EMBEDDING_DIM;
use serde_json::json;
use std::sync::Arc;
use std::time::{Duration, Instant};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const N: usize = 100;

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> anyhow::Result<()> {
    println!("claude-hippo external-backend RSS bench\n");

    let baseline_rss = read_self_rss_kb().unwrap_or(0);
    println!(
        "baseline RSS (after main start): {:.1} MB",
        baseline_rss as f64 / 1024.0
    );

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(|req: &wiremock::Request| {
            let body: serde_json::Value = req.body_json().unwrap();
            let inputs = body["input"].as_array().unwrap();
            // Echo back N normalized 384-dim vectors (deterministic per index).
            let vectors: Vec<Vec<f32>> = inputs
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    let mut v = vec![0.0_f32; EMBEDDING_DIM];
                    v[i % EMBEDDING_DIM] = 1.0;
                    v
                })
                .collect();
            ResponseTemplate::new(200).set_body_json(json!({
                "object": "list",
                "data": vectors.into_iter().enumerate().map(|(i, v)| json!({
                    "object": "embedding",
                    "index": i,
                    "embedding": v,
                })).collect::<Vec<_>>(),
                "model": "mock",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }))
        })
        .mount(&server)
        .await;

    let cfg = ExternalEmbeddingConfig {
        url: format!("{}/v1/embeddings", server.uri()),
        model: "mock".into(),
        dim: EMBEDDING_DIM,
        api_key: "sk-test".into(),
        timeout: Duration::from_secs(2),
        batch_size: 8,
        max_retries: 0,
    };
    let embedder: Arc<dyn Embedder> = Arc::new(ExternalEmbedder::new(cfg)?);

    register_sqlite_vec();
    let mut tmp_db = std::env::temp_dir();
    tmp_db.push(format!("claude-hippo-ext-rss-{}.db", std::process::id()));
    let _ = std::fs::remove_file(&tmp_db);
    let store = Storage::open(&tmp_db)?;

    let mem_server = MemoryServer::new_with_config(
        store,
        embedder,
        SurpriseWeights::default(),
        RankingConfig::default(),
    );

    // Warmup: 1 embed (touches reqwest, rustls, sqlite-vec).
    let warmup0 = Instant::now();
    let _ = mem_server
        .remember(RememberParams {
            content: "warmup".into(),
            tags: vec![],
            memory_type: None,
            importance: None,
            metadata: None,
        })
        .await
        .map_err(|e| anyhow::anyhow!("warmup: {:?}", e))?;
    let warmup_rss = read_self_rss_kb().unwrap_or(0);
    println!(
        "after warmup ({:?}): {:.1} MB",
        warmup0.elapsed(),
        warmup_rss as f64 / 1024.0
    );

    // Store N
    let t1 = Instant::now();
    for i in 0..N {
        let _ = mem_server
            .remember(RememberParams {
                content: format!("bench external memory {i}: timing harness"),
                tags: vec!["bench".into(), format!("i{}", i % 10)],
                memory_type: Some("Observation".into()),
                importance: Some(0.5),
                metadata: None,
            })
            .await
            .map_err(|e| anyhow::anyhow!("store err: {:?}", e))?;
    }
    let store_total = t1.elapsed();

    // Retrieve N
    let t2 = Instant::now();
    for _ in 0..N {
        let _ = mem_server
            .recall(RecallParams {
                query: "timing harness memory".into(),
                limit: 5,
                no_surprise_boost: false,
                oversample_factor: None,
                mode: None,
                seed_id: None,
            })
            .await
            .map_err(|e| anyhow::anyhow!("retrieve err: {:?}", e))?;
    }
    let retrieve_total = t2.elapsed();

    let final_rss = read_self_rss_kb().unwrap_or(0);
    let peak_rss = read_self_peak_rss_kb().unwrap_or(final_rss);

    println!("\n--- results ---");
    println!("backend          : external (wiremock in-process)");
    println!(
        "store    x{N:<5}: total={store_total:?}  ({:.1} ms/op)",
        store_total.as_secs_f64() * 1000.0 / N as f64
    );
    println!(
        "retrieve x{N:<5}: total={retrieve_total:?}  ({:.1} ms/op)",
        retrieve_total.as_secs_f64() * 1000.0 / N as f64
    );
    println!("final RSS        : {:.1} MB", final_rss as f64 / 1024.0);
    println!("peak  RSS (VmHWM): {:.1} MB", peak_rss as f64 / 1024.0);
    println!(
        "vs target <30 MB : {}",
        if peak_rss <= 30 * 1024 {
            "MET"
        } else {
            "OVER (honest disclosure required in docs)"
        }
    );
    println!("\nnote: this measures the in-process RSS of *both* the wiremock test");
    println!("server and the claude-hippo client in the same process. A standalone");
    println!("`hippo serve --embedding-backend external` against a remote URL will");
    println!("be lower (the API server's memory is in a separate process).");

    let _ = std::fs::remove_file(&tmp_db);
    Ok(())
}

fn read_self_rss_kb() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            return rest
                .split_whitespace()
                .next()
                .and_then(|n| n.parse::<u64>().ok());
        }
    }
    None
}

fn read_self_peak_rss_kb() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            return rest
                .split_whitespace()
                .next()
                .and_then(|n| n.parse::<u64>().ok());
        }
    }
    None
}
