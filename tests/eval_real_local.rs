//! v0.5 Phase D — Real-backend variant of Bench A/B/C.
//!
//! Replays the v0.2-v0.3 fixtures against the **real** FastEmbedder
//! (`all-MiniLM-L6-v2`, ONNX) instead of the deterministic mock. The
//! point is to honestly characterize how surprise rerank lift holds up
//! when "semantic similarity" stops being an orthogonal mock cluster
//! and starts being actual transformer cosines.
//!
//! All four tests are gated `#[ignore]` because:
//!   - first run downloads ~80 MB of ONNX weights (`fastembed` cache)
//!   - inference is slow vs the mock (~s instead of µs)
//!   - real-cosine ordering is not bit-deterministic across model
//!     reuploads, so equality assertions would be brittle
//!
//! Run them on demand with:
//!
//! ```bash
//! cargo test -- --ignored --nocapture bench_a_real_local
//! cargo test -- --ignored --nocapture bench_b_real_local
//! cargo test -- --ignored --nocapture bench_c_real_local
//! ```
//!
//! Each writes `target/eval_results/bench_<x>_real_local.json` so
//! `docs/SURPRISE_SELECTION.md` can quote the actual numbers and pair
//! them with the mock-fixture headlines.

#[path = "eval/mod.rs"]
mod eval;

use claude_hippo::embeddings::{Embedder, EmbeddingModelKind, FastEmbedder};
use claude_hippo::surprise::SurpriseWeights;
use eval::*;
use serde::Serialize;
use std::sync::Arc;

const TOPICS: &[&str] = &["auth", "db", "billing", "ui", "infra"];
const CHATS_PER_CLUSTER: usize = 19;
const TOTAL_ITEMS: usize = 100;
const K: usize = 5;

const PARAPHRASES: &[&str] = &[
    "what was our decision about {topic}?",
    "remind me of the {topic} choice we made",
    "the {topic} call we settled on",
    "decision summary for {topic}",
    "what did we decide on {topic}",
];

#[derive(Serialize)]
struct RealReport {
    backend: &'static str,
    fixture: &'static str,
    note: &'static str,
    ablation: AblationResult,
    /// Lift = with_surprise.precision_at_1 - baseline.precision_at_1.
    /// Sign matters more than magnitude with real embeddings — mock
    /// fixtures lock the lift at +0.95 by construction; real backends
    /// produce a smaller, noisier number that still must stay ≥ 0 to
    /// claim the rerank earns its keep.
    p1_lift: f64,
    mrr_lift: f64,
}

fn real_embedder() -> Arc<dyn Embedder> {
    let cache_dir = std::env::var("HIPPO_MODEL_CACHE")
        .ok()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(claude_hippo::embeddings::default_cache_dir);
    let e = FastEmbedder::new_with_model(cache_dir, EmbeddingModelKind::MiniLmL6V2)
        .expect("FastEmbedder construction");
    // Warm up so the load cost isn't charged to the first recall.
    let _ = e.embed_one("warmup");
    Arc::new(e)
}

// ---------- Bench A real-local ----------------------------------------

fn build_a(oversample_factor: usize, real: bool) -> EvalConfig {
    let mut items = Vec::with_capacity(TOTAL_ITEMS);
    let mut queries = Vec::with_capacity(TOPICS.len() * PARAPHRASES.len());
    for (cluster_idx, topic) in TOPICS.iter().enumerate() {
        let decision_id = format!("decision-{topic}");
        items.push(EvalItem {
            id: decision_id.clone(),
            content: format!(
                "We chose to use approach X for the {topic} module after reviewing options Y and Z. \
                 The decision was driven by latency targets and on-call ergonomics. \
                 Implementation lives in service-{topic} and is owned by the platform team. \
                 Tradeoffs: option Y was rejected because it did not satisfy the audit requirements. \
                 Revisit by 2026-Q4 if traffic crosses 50k QPS or compliance scope expands."
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
                relevant_ids: vec![decision_id.clone()],
            });
        }
    }
    EvalConfig {
        items,
        queries,
        k: K,
        weights: SurpriseWeights::default(),
        num_clusters: TOPICS.len() + 1,
        fallback_cluster: TOPICS.len(),
        noise_scale: 0.05,
        oversample_factor,
        prediction_loss: None,
        embedder_override: if real { Some(real_embedder()) } else { None },
    }
}

#[tokio::test]
#[ignore = "real fastembed: downloads ~80MB on first run, slow inference"]
async fn bench_a_real_local() {
    use claude_hippo::server::DEFAULT_OVERSAMPLE_FACTOR;
    let ablation = run_ablation(build_a(DEFAULT_OVERSAMPLE_FACTOR, true))
        .await
        .expect("bench A real-local run");
    let p1_lift = ablation.with_surprise.precision_at_1 - ablation.baseline.precision_at_1;
    let mrr_lift = ablation.with_surprise.mrr - ablation.baseline.mrr;
    let report = RealReport {
        backend: "FastEmbedder(MiniLmL6V2, ONNX)",
        fixture: "Bench A long-session noise (100 items, 25 queries)",
        note: "Real-cosine ordering is much noisier than orthogonal mock clusters. \
               Decision content shares a lot of vocabulary with chat noise of the same \
               topic (e.g. 'auth'), so the baseline pure-cosine rank is closer to a coin \
               flip within cluster, and surprise rerank's measurable lift is smaller than \
               the +0.95 the mock fixture reports — but it must remain non-negative.",
        ablation: ablation.clone(),
        p1_lift,
        mrr_lift,
    };
    write_result_json("bench_a_real_local", &report).expect("write bench A real-local");
    println!("\nBench A — real-local (FastEmbedder MiniLM)");
    println!(
        "  baseline P@1={:.3} MRR={:.3}  surprise P@1={:.3} MRR={:.3}  lift Δ={:+.3}",
        ablation.baseline.precision_at_1,
        ablation.baseline.mrr,
        ablation.with_surprise.precision_at_1,
        ablation.with_surprise.mrr,
        p1_lift,
    );
    // Honest soft-assertion: surprise rerank must not regress vs pure
    // cosine. A small positive or zero lift is plausible; a negative
    // lift is a regression worth investigating.
    assert!(
        p1_lift >= -1e-3,
        "real-local Bench A: surprise P@1 ({:.3}) regressed vs baseline ({:.3})",
        ablation.with_surprise.precision_at_1,
        ablation.baseline.precision_at_1,
    );
}

// ---------- Bench B real-local: cross-session decay -------------------

fn build_b(age_days: f32, oversample_factor: usize, real: bool) -> EvalConfig {
    // Fixture lifted from `eval_b_cross_session.rs` — keep the cluster
    // count + topic shape identical so the numbers can be diffed.
    let mut items = Vec::new();
    items.push(EvalItem {
        id: "old-decision".into(),
        content: "Old decision: we picked PostgreSQL for our primary OLTP after reviewing MySQL \
             and CockroachDB, mainly because of full-text search support and the team's \
             prior PostgreSQL operational experience. Revisit by 2026-Q4."
            .into(),
        tags: decision_tags("db"),
        memory_type: Some("Decision".into()),
        importance: Some(1.0),
        cluster: 0,
        age_days,
    });
    for i in 0..40 {
        items.push(EvalItem {
            id: format!("recent-chat-{i:02}"),
            content: format!("db chat {i:02}: short note about backups"),
            tags: chat_tags("db"),
            memory_type: Some("Observation".into()),
            importance: None,
            cluster: 0,
            age_days: 0.0,
        });
    }
    let queries: Vec<EvalQuery> = [
        "what database did we pick?",
        "what was the OLTP decision?",
        "remind me of the storage choice",
        "summarize the DB decision",
    ]
    .iter()
    .map(|q| EvalQuery {
        query: (*q).to_string(),
        cluster: 0,
        relevant_ids: vec!["old-decision".into()],
    })
    .collect();
    EvalConfig {
        items,
        queries,
        k: K,
        weights: SurpriseWeights::default(),
        num_clusters: 2,
        fallback_cluster: 1,
        noise_scale: 0.05,
        oversample_factor,
        prediction_loss: None,
        embedder_override: if real { Some(real_embedder()) } else { None },
    }
}

#[tokio::test]
#[ignore = "real fastembed: downloads ~80MB on first run, slow inference"]
async fn bench_b_real_local() {
    use claude_hippo::server::DEFAULT_OVERSAMPLE_FACTOR;
    // Mid-life: 90 days. Old enough to feel decay, not so old it falls
    // off the floor entirely.
    let ablation = run_ablation(build_b(90.0, DEFAULT_OVERSAMPLE_FACTOR, true))
        .await
        .expect("bench B real-local run");
    let p1_lift = ablation.with_surprise.precision_at_1 - ablation.baseline.precision_at_1;
    let mrr_lift = ablation.with_surprise.mrr - ablation.baseline.mrr;
    let report = RealReport {
        backend: "FastEmbedder(MiniLmL6V2, ONNX)",
        fixture: "Bench B cross-session 90-day Decision recall",
        note: "Real-cosine over a 90-day-old Decision drowning in 40 fresh chat notes. \
               half_life=30d / decay_floor=0.5 keeps the Decision competitive in theory; \
               this test measures whether real-embedding noise breaks that in practice.",
        ablation: ablation.clone(),
        p1_lift,
        mrr_lift,
    };
    write_result_json("bench_b_real_local", &report).expect("write bench B real-local");
    println!("\nBench B — real-local (FastEmbedder MiniLM, 90d age)");
    println!(
        "  baseline P@1={:.3} MRR={:.3}  surprise P@1={:.3} MRR={:.3}  lift Δ={:+.3}",
        ablation.baseline.precision_at_1,
        ablation.baseline.mrr,
        ablation.with_surprise.precision_at_1,
        ablation.with_surprise.mrr,
        p1_lift,
    );
    assert!(
        p1_lift >= -1e-3,
        "real-local Bench B: surprise P@1 regressed vs baseline by {:+.3}",
        p1_lift,
    );
}

// ---------- Bench C real-local: decision trace ------------------------

fn build_c(weights: SurpriseWeights, oversample_factor: usize, real: bool) -> EvalConfig {
    // Same shape as eval_c_decision_trace.rs: 4 Decisions + 16 chats.
    let decisions = [
        (
            "decision-auth",
            "Decision: we picked JWT with 24h expiry over session cookies",
        ),
        (
            "decision-db",
            "Decision: we picked PostgreSQL for primary OLTP",
        ),
        (
            "decision-cache",
            "Decision: we picked Redis for hot cache, 5-min TTL",
        ),
        (
            "decision-queue",
            "Decision: we picked NATS JetStream for async work",
        ),
    ];
    let mut items = Vec::new();
    for (id, content) in decisions {
        items.push(EvalItem {
            id: id.into(),
            content: content.into(),
            tags: decision_tags("infra"),
            memory_type: Some("Decision".into()),
            importance: Some(1.0),
            cluster: 0,
            age_days: 0.0,
        });
    }
    for i in 0..16 {
        items.push(EvalItem {
            id: format!("chat-{i:02}"),
            content: format!("infra chat {i:02}: routine note"),
            tags: chat_tags("infra"),
            memory_type: Some("Observation".into()),
            importance: None,
            cluster: 0,
            age_days: 0.0,
        });
    }
    let decision_ids: Vec<String> = decisions.iter().map(|(id, _)| id.to_string()).collect();
    let queries: Vec<EvalQuery> = [
        "what did we decide on?",
        "list our infra decisions",
        "summary of the choices we made",
        "what were the main calls?",
    ]
    .iter()
    .map(|q| EvalQuery {
        query: (*q).to_string(),
        cluster: 0,
        relevant_ids: decision_ids.clone(),
    })
    .collect();
    EvalConfig {
        items,
        queries,
        k: K,
        weights,
        num_clusters: 2,
        fallback_cluster: 1,
        noise_scale: 0.05,
        oversample_factor,
        prediction_loss: None,
        embedder_override: if real { Some(real_embedder()) } else { None },
    }
}

#[tokio::test]
#[ignore = "real fastembed: downloads ~80MB on first run, slow inference"]
async fn bench_c_real_local() {
    use claude_hippo::server::DEFAULT_OVERSAMPLE_FACTOR;
    let ablation = run_ablation(build_c(
        SurpriseWeights::default(),
        DEFAULT_OVERSAMPLE_FACTOR,
        true,
    ))
    .await
    .expect("bench C real-local run");
    let recall_lift = ablation.with_surprise.recall_at_k - ablation.baseline.recall_at_k;
    let report = RealReport {
        backend: "FastEmbedder(MiniLmL6V2, ONNX)",
        fixture: "Bench C decision trace (4 decisions + 16 chats, recall@5)",
        note: "Real-cosine over generic 'what did we decide on?' query. Recall@5 \
               is the headline (recall@k captures \"how many of the 4 Decisions show \
               up in top-5 over the 4 paraphrased queries\") — this is where surprise \
               rerank's effect is most legible.",
        ablation: ablation.clone(),
        p1_lift: ablation.with_surprise.precision_at_1 - ablation.baseline.precision_at_1,
        mrr_lift: ablation.with_surprise.mrr - ablation.baseline.mrr,
    };
    write_result_json("bench_c_real_local", &report).expect("write bench C real-local");
    println!("\nBench C — real-local (FastEmbedder MiniLM)");
    println!(
        "  baseline recall@5={:.3}  surprise recall@5={:.3}  lift Δ={:+.3}",
        ablation.baseline.recall_at_k, ablation.with_surprise.recall_at_k, recall_lift,
    );
    assert!(
        recall_lift >= -1e-3,
        "real-local Bench C: surprise recall@5 regressed vs baseline by {:+.3}",
        recall_lift,
    );
}
