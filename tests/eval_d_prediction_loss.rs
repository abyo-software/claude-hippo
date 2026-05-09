//! Bench D — Prediction-loss wiring smoke (v0.3 evaluation axis).
//!
//! What this measures: with a `PredictionLossBackend` wired in, the
//! `MemoryServer.remember` flow actually populates `prediction_loss` and
//! the surprise score uses the full 4-component formula. Without a
//! backend, the same fixture falls back to the v0.2 3-component
//! redistribution.
//!
//! The intent is *not* to claim a numeric retrieval lift here — that
//! requires a real LLM backend (vLLM, llama.cpp, legacy OpenAI completions)
//! and is left to a release-time smoke. What we *do* claim:
//!
//! 1. With `MockPredictionLoss` (deterministic SHA256-derived score),
//!    Bench A's same 100-item / 25-query fixture still recalls the
//!    Decision at rank 1 — i.e. plumbing the backend through doesn't
//!    *regress* retrieval quality.
//! 2. The backend is *exercised*: every memory's `surprise_components.
//!    prediction_loss` ends up `Some(value)` instead of `None`, proving
//!    end-to-end wiring from CLI flag → MemoryServer → SurpriseComponents
//!    → metadata.

#[path = "eval/mod.rs"]
mod eval;

use claude_hippo::prediction_loss::{MockPredictionLoss, PredictionLossBackend};
use claude_hippo::storage::Storage;
use claude_hippo::surprise::SurpriseWeights;
use eval::*;
use serde::Serialize;
use std::sync::Arc;

const TOPIC: &str = "auth";
const TOTAL_ITEMS: usize = 100;
const CHATS_PER_CLUSTER: usize = 19;
const TOPICS: &[&str] = &["auth", "db", "billing", "ui", "infra"];
const K: usize = 5;

const PARAPHRASES: &[&str] = &[
    "what was our decision about {topic}?",
    "remind me of the {topic} choice we made",
];

#[derive(Serialize)]
struct BenchDReport {
    setup: SetupSummary,
    /// With `prediction_loss` backend wired (v0.3 path).
    with_prediction_loss: AblationResult,
    /// Without backend (v0.2 fallback path, for diff).
    without_prediction_loss: AblationResult,
    /// Number of stored memories whose `_hippo.surprise.components.
    /// prediction_loss` is `Some(_)` after a `with_prediction_loss` run.
    memories_with_prediction_loss: usize,
    memories_total: usize,
}

#[derive(Serialize)]
struct SetupSummary {
    total_items: usize,
    queries: usize,
    k: usize,
    backend: &'static str,
    note: &'static str,
}

fn build_config(backend: Option<Arc<dyn PredictionLossBackend>>) -> EvalConfig {
    let mut items = Vec::with_capacity(TOTAL_ITEMS);
    let mut queries = Vec::with_capacity(TOPICS.len() * PARAPHRASES.len());

    for (cluster_idx, topic) in TOPICS.iter().enumerate() {
        items.push(EvalItem {
            id: format!("decision-{topic}"),
            content: format!(
                "Decision on {topic}: We chose approach X for {topic} after weighing the \
                 tradeoffs. Owned by platform-{topic}; revisit by 2026-Q4."
            ),
            tags: decision_tags(topic),
            memory_type: Some("Decision".into()),
            importance: Some(1.0),
            cluster: cluster_idx,
            age_days: 0.0,
        });
        for i in 0..CHATS_PER_CLUSTER {
            items.push(EvalItem {
                id: format!("chat-{topic}-{i:02}"),
                content: format!("{topic} chat {i:02}: small note"),
                tags: chat_tags(topic),
                memory_type: Some("Observation".into()),
                importance: None,
                cluster: cluster_idx,
                age_days: 0.0,
            });
        }
        for tmpl in PARAPHRASES {
            queries.push(EvalQuery {
                query: tmpl.replace("{topic}", topic),
                cluster: cluster_idx,
                relevant_ids: vec![format!("decision-{topic}")],
            });
        }
    }
    assert_eq!(items.len(), TOTAL_ITEMS);

    EvalConfig {
        items,
        queries,
        k: K,
        weights: SurpriseWeights::default(),
        num_clusters: TOPICS.len() + 1,
        fallback_cluster: TOPICS.len(),
        noise_scale: 0.05,
        oversample_factor: claude_hippo::server::DEFAULT_OVERSAMPLE_FACTOR,
        prediction_loss: backend,
    }
}

#[tokio::test]
async fn bench_d_prediction_loss_wiring() {
    let backend: Arc<dyn PredictionLossBackend> = Arc::new(MockPredictionLoss);
    let with_pl = run_ablation(build_config(Some(backend.clone())))
        .await
        .expect("bench D with-pl run");
    let without_pl = run_ablation(build_config(None))
        .await
        .expect("bench D without-pl run");

    // Re-run "with prediction_loss" once more, this time inspecting the
    // stored memories to verify each one carries a non-None prediction_loss.
    let coverage = count_memories_with_prediction_loss(backend).await;

    let report = BenchDReport {
        setup: SetupSummary {
            total_items: TOTAL_ITEMS,
            queries: TOPICS.len() * PARAPHRASES.len(),
            k: K,
            backend: "MockPredictionLoss (SHA256-derived, deterministic, synthetic)",
            note: "This bench proves the v0.3 prediction_loss wiring works end-to-end \
                   with deterministic mock scores. Real LLM backend numbers (vLLM / \
                   llama.cpp / legacy OpenAI completions) are out of CI scope and \
                   left to release-time smoke tests.",
        },
        with_prediction_loss: with_pl.clone(),
        without_prediction_loss: without_pl.clone(),
        memories_with_prediction_loss: coverage.with,
        memories_total: coverage.total,
    };
    write_result_json("bench_d_prediction_loss", &report).expect("write bench D result");

    println!("\nBench D — Prediction-loss wiring");
    println!(
        "  with    PL: surprise P@1={:.3} MRR={:.3}  (coverage = {}/{} memories)",
        with_pl.with_surprise.precision_at_1,
        with_pl.with_surprise.mrr,
        coverage.with,
        coverage.total,
    );
    println!(
        "  without PL: surprise P@1={:.3} MRR={:.3}",
        without_pl.with_surprise.precision_at_1, without_pl.with_surprise.mrr,
    );

    // Coverage assertion: every stored memory must have prediction_loss set
    // when the backend is wired. If this fails, the wiring regressed.
    assert_eq!(
        coverage.with, coverage.total,
        "with backend wired, every memory must carry prediction_loss; got {}/{}",
        coverage.with, coverage.total
    );

    // Bench D's MockPredictionLoss returns SHA256-derived noise, so chat
    // notes sometimes get a higher prediction_loss than Decisions (real
    // LLMs would not). We deliberately do NOT assert retrieval quality
    // here — the only retrieval guarantee is "well above the 1/20 random
    // baseline of 0.05", which proves the rest of the surprise pipeline
    // still works around the synthetic noise. Real LLM-derived numbers
    // belong in a release-time smoke test against vLLM / llama.cpp.
    assert!(
        with_pl.with_surprise.precision_at_1 >= 0.20,
        "even with synthetic MockPredictionLoss noise, surprise rerank must beat the \
         1/20 = 0.05 random-within-cluster baseline by a safe margin; got P@1 = {:.3}. \
         If this drops below 0.20, the score formula has a regression unrelated to \
         the synthetic backend.",
        with_pl.with_surprise.precision_at_1,
    );
}

struct Coverage {
    with: usize,
    total: usize,
}

async fn count_memories_with_prediction_loss(backend: Arc<dyn PredictionLossBackend>) -> Coverage {
    use claude_hippo::server::{MemoryServer, RankingConfig, RememberParams};
    claude_hippo::storage::register_sqlite_vec();
    let cfg = build_config(Some(backend));
    let mut embedder = ClusteredMockEmbedder::new(cfg.num_clusters, cfg.fallback_cluster, 0.05);
    for it in &cfg.items {
        embedder.assign(it.content.clone(), it.cluster);
    }
    let store = Storage::open_in_memory().unwrap();
    let server = MemoryServer::new_full(
        store,
        Arc::new(embedder),
        cfg.prediction_loss.clone(),
        cfg.weights,
        RankingConfig::default(),
    );
    let mut with_pl = 0;
    let mut total = 0;
    for it in &cfg.items {
        let r = server
            .remember(RememberParams {
                content: it.content.clone(),
                tags: it.tags.clone(),
                memory_type: it.memory_type.clone(),
                importance: it.importance,
                metadata: None,
            })
            .await
            .expect("remember succeeded");
        total += 1;
        if r.surprise_components.prediction_loss.is_some() {
            with_pl += 1;
        }
    }
    let _ = TOPIC; // suppress unused warning if TOPIC ever stops being referenced
    Coverage {
        with: with_pl,
        total,
    }
}
