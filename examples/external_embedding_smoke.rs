//! Manual smoke test for the v0.3 external embedding backend.
//!
//! This is **not** a CI test — it talks to a real HTTP service. Pick a
//! backend via env vars and run with `cargo run --example
//! external_embedding_smoke`.
//!
//! # Backends and how to run them
//!
//! ## OpenAI
//! ```sh
//! export OPENAI_API_KEY=sk-...
//! cargo run --example external_embedding_smoke -- openai
//! ```
//! Hits `https://api.openai.com/v1/embeddings` with
//! `text-embedding-3-small` and `dimensions=384`.
//!
//! ## Ollama (local, no API key)
//! ```sh
//! ollama pull nomic-embed-text   # or any embedding-capable model
//! cargo run --example external_embedding_smoke -- ollama
//! ```
//! Hits `http://localhost:11434/v1/embeddings`. The model must produce
//! 384-dim vectors or the test will fail loud (which is the point: any
//! dim drift breaks DB schema compatibility with mcp-memory-service-rs).
//! Most off-the-shelf Ollama embedding models produce 768 dim — you'll
//! need a 384-dim model (e.g. `all-minilm:l6-v2` if available, or run
//! a local proxy that projects down to 384).
//!
//! ## HuggingFace Text-Embeddings-Inference (TEI)
//! ```sh
//! docker run --rm -p 8080:80 ghcr.io/huggingface/text-embeddings-inference:cpu-1.6 \
//!   --model-id sentence-transformers/all-MiniLM-L6-v2
//! cargo run --example external_embedding_smoke -- tei
//! ```
//! Hits `http://localhost:8080/embed` (note: TEI uses `/embed` not
//! `/v1/embeddings`; pass `HF_TEI_URL=http://localhost:8080/embed`).
//!
//! ## Custom
//! Set the env vars manually and pass any first arg:
//! ```sh
//! HIPPO_EXTERNAL_EMBEDDING_URL=https://my-proxy/v1/embeddings \
//! HIPPO_EXTERNAL_EMBEDDING_MODEL=my-model \
//! HIPPO_EXTERNAL_EMBEDDING_API_KEY_ENV=MY_KEY \
//!     cargo run --example external_embedding_smoke -- custom
//! ```

use claude_hippo::embeddings::{Embedder, ExternalEmbedder, ExternalEmbeddingConfig};
use claude_hippo::EMBEDDING_DIM;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = std::env::args().nth(1).unwrap_or_else(|| "openai".into());

    let cfg = match backend.as_str() {
        "openai" => ExternalEmbeddingConfig {
            url: "https://api.openai.com/v1/embeddings".into(),
            model: "text-embedding-3-small".into(),
            dim: EMBEDDING_DIM,
            api_key: env_or_die("OPENAI_API_KEY")?,
            timeout: Duration::from_secs(15),
            batch_size: 8,
            max_retries: 2,
        },
        "ollama" => ExternalEmbeddingConfig {
            url: std::env::var("OLLAMA_URL")
                .unwrap_or_else(|_| "http://localhost:11434/v1/embeddings".into()),
            model: std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "nomic-embed-text".into()),
            dim: EMBEDDING_DIM,
            api_key: String::new(), // keyless local
            timeout: Duration::from_secs(30),
            batch_size: 8,
            max_retries: 1,
        },
        "tei" => ExternalEmbeddingConfig {
            url: std::env::var("HF_TEI_URL").unwrap_or_else(|_| {
                // TEI's OpenAI-compat shim:
                "http://localhost:8080/v1/embeddings".into()
            }),
            model: std::env::var("HF_TEI_MODEL")
                .unwrap_or_else(|_| "sentence-transformers/all-MiniLM-L6-v2".into()),
            dim: EMBEDDING_DIM,
            api_key: String::new(),
            timeout: Duration::from_secs(30),
            batch_size: 8,
            max_retries: 1,
        },
        // Anything else: read from the canonical CLI env vars.
        _ => {
            let key_env = std::env::var("HIPPO_EXTERNAL_EMBEDDING_API_KEY_ENV")
                .unwrap_or_else(|_| "OPENAI_API_KEY".into());
            ExternalEmbeddingConfig {
                url: env_or_die("HIPPO_EXTERNAL_EMBEDDING_URL")?,
                model: env_or_die("HIPPO_EXTERNAL_EMBEDDING_MODEL")?,
                dim: EMBEDDING_DIM,
                api_key: if key_env.eq_ignore_ascii_case("none") {
                    String::new()
                } else {
                    std::env::var(&key_env).unwrap_or_default()
                },
                timeout: Duration::from_secs(15),
                batch_size: 8,
                max_retries: 2,
            }
        }
    };

    println!("backend  : {backend}");
    println!("url      : {}", cfg.url);
    println!("model    : {}", cfg.model);
    println!(
        "api_key  : {}",
        if cfg.api_key.is_empty() {
            "(none)"
        } else {
            "(set)"
        }
    );

    let e = ExternalEmbedder::new(cfg)?;
    let texts = [
        "The mitochondrion is the powerhouse of the cell.",
        "Rust's ownership model prevents data races at compile time.",
        "claude-hippo is a surprise-aware memory MCP server.",
    ];

    let t0 = Instant::now();
    let vs = e.embed_batch(&texts)?;
    let dt = t0.elapsed();

    println!("\n---");
    println!(
        "embedded {} texts in {:?} ({:.1} ms/text)",
        vs.len(),
        dt,
        dt.as_secs_f64() * 1000.0 / vs.len() as f64
    );
    for (i, v) in vs.iter().enumerate() {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!(
            "  [{i}] dim={} norm={:.4} first 4 = {:?}",
            v.len(),
            norm,
            &v[..4]
        );
    }

    // Pairwise cosine sims as a quick sanity check (similar topics should
    // have higher cosine than unrelated topics).
    println!("\npairwise cosine sims:");
    for (i, vi) in vs.iter().enumerate() {
        for (j, vj) in vs.iter().enumerate().skip(i + 1) {
            let cos: f32 = vi.iter().zip(vj.iter()).map(|(a, b)| a * b).sum();
            println!("  ({i},{j}) = {cos:.4}");
        }
    }

    Ok(())
}

fn env_or_die(name: &str) -> Result<String, Box<dyn std::error::Error>> {
    std::env::var(name).map_err(|_| format!("env var {name} is required").into())
}
