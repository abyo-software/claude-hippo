//! Bench A — Long-session noise (v0.2 evaluation axis).
//!
//! Setup: 100 memories spread across 5 topic clusters (auth / db / billing /
//! ui / infra). Each cluster contains 1 high-importance Decision and 19
//! low-importance Observations ("chat noise"). For each topic, we ask
//! "what was our decision about <topic>?" and measure how well the actual
//! Decision surfaces.
//!
//! Hypothesis: surprise rerank pushes the Decision to rank 1 even though
//! the surrounding chat noise has comparable cosine similarity. Without
//! surprise, rank is essentially uniform within the cluster.
//!
//! Numbers consumed by `docs/SURPRISE_SELECTION.md`.

#[path = "eval/mod.rs"]
mod eval;

use claude_hippo::surprise::SurpriseWeights;
use eval::*;
use serde::Serialize;

const TOPICS: &[&str] = &["auth", "db", "billing", "ui", "infra"];
const CHATS_PER_CLUSTER: usize = 19;
const TOTAL_ITEMS: usize = 100;
const K: usize = 5;

/// Multiple phrasings per topic. Each paraphrase produces a different SHA256
/// noise pattern, so the within-cluster rank ordering varies — this gives
/// the baseline statistic real samples to average over instead of leaving
/// it at the mercy of one query's deterministic luck.
const PARAPHRASES: &[&str] = &[
    "what was our decision about {topic}?",
    "remind me of the {topic} choice we made",
    "the {topic} call we settled on",
    "decision summary for {topic}",
    "what did we decide on {topic}",
];

#[derive(Serialize)]
struct BenchAReport {
    setup: SetupSummary,
    /// Default oversample (`oversample_factor=3`, production behavior).
    default_oversample: AblationResult,
    /// Full oversample (`oversample_factor=num_items / k = 20`). Shows the
    /// ceiling of what surprise rerank can achieve when given enough
    /// candidates.
    full_oversample: AblationResult,
}

#[derive(Serialize)]
struct SetupSummary {
    total_items: usize,
    decisions: usize,
    chats: usize,
    clusters: usize,
    queries: usize,
    k: usize,
    note: &'static str,
}

fn build_config(oversample_factor: usize) -> EvalConfig {
    let mut items = Vec::with_capacity(TOTAL_ITEMS);
    let mut queries = Vec::with_capacity(TOPICS.len() * PARAPHRASES.len());

    for (cluster_idx, topic) in TOPICS.iter().enumerate() {
        // 1 decision per topic — long, multi-tag, importance=1.0
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
        // 19 chats per topic — short, single tag, no importance.
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
        // 5 paraphrased queries per topic; each has a different SHA256 noise
        // pattern, which decouples the baseline metric from any single
        // query's deterministic position bias.
        for tmpl in PARAPHRASES {
            queries.push(EvalQuery {
                query: tmpl.replace("{topic}", topic),
                cluster: cluster_idx,
                relevant_ids: vec![decision_id.clone()],
            });
        }
    }

    assert_eq!(items.len(), TOTAL_ITEMS);
    assert_eq!(queries.len(), TOPICS.len() * PARAPHRASES.len());

    // 5 topic clusters + 1 fallback (unused — every item & query is assigned)
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
        embedder_override: None,
    }
}

#[tokio::test]
async fn bench_a_long_session_noise() {
    use claude_hippo::server::DEFAULT_OVERSAMPLE_FACTOR;
    // v0.3 production default (was 3 in v0.2 — the bump is exactly to reach
    // P@1=1.0 on this bench without per-call tuning).
    let default_oversample = run_ablation(build_config(DEFAULT_OVERSAMPLE_FACTOR))
        .await
        .expect("bench A default-oversample run");

    // full-coverage: oversample_factor * k >= total items in cluster, so the
    // entire cluster is fetched before rerank.
    let full_oversample = run_ablation(build_config(20))
        .await
        .expect("bench A full-oversample run");

    let report = BenchAReport {
        setup: SetupSummary {
            total_items: TOTAL_ITEMS,
            decisions: TOPICS.len(),
            chats: TOPICS.len() * CHATS_PER_CLUSTER,
            clusters: TOPICS.len(),
            queries: TOPICS.len() * PARAPHRASES.len(),
            k: K,
            note: "1 Decision (importance=1.0) + 19 Observation chats per topic, 5 topics. \
                   5 paraphrases per topic = 25 queries total. \
                   Each query asks for the cluster's Decision; relevant set = {that one Decision}.",
        },
        default_oversample: default_oversample.clone(),
        full_oversample: full_oversample.clone(),
    };
    write_result_json("bench_a_long_session", &report).expect("write bench A result");

    // The honest claim: with v0.3's default oversample=6, KNN over-fetches
    // 30 items for k=5; that exceeds the within-cluster pool of 20 even
    // accounting for some inter-cluster bleed, so the Decision is always
    // in the rerank pool and surprise lifts precision@1 from baseline ~5%
    // (1/20 random within cluster) to 100%. Full oversample=20 fetches the
    // whole corpus and is the upper bound. v0.2 default=3 capped at 72%
    // P@1 because the Decision did not always reach the rerank pool —
    // documented in CHANGELOG v0.3.0 as an honest limitation now closed.
    println!("\nBench A — Long-session noise (v0.3)");
    println!(
        "  default oversample ({}): baseline P@1={:.3} MRR={:.3}  surprise P@1={:.3} MRR={:.3}",
        claude_hippo::server::DEFAULT_OVERSAMPLE_FACTOR,
        default_oversample.baseline.precision_at_1,
        default_oversample.baseline.mrr,
        default_oversample.with_surprise.precision_at_1,
        default_oversample.with_surprise.mrr,
    );
    println!(
        "  full   oversample (20): baseline P@1={:.3} MRR={:.3}  surprise P@1={:.3} MRR={:.3}",
        full_oversample.baseline.precision_at_1,
        full_oversample.baseline.mrr,
        full_oversample.with_surprise.precision_at_1,
        full_oversample.with_surprise.mrr,
    );

    // Headline assertion: with full oversample, surprise must perfectly
    // rank Decisions at #1. Anything less means scoring/rerank has a bug.
    assert_eq!(
        full_oversample.with_surprise.precision_at_1, 1.0,
        "with full oversample, surprise rerank must place Decision at rank 1 every time"
    );
    assert_eq!(
        full_oversample.with_surprise.mrr, 1.0,
        "MRR must be 1.0 when every Decision is rank 1"
    );

    // v0.3: production default (oversample=6) MUST also reach perfect P@1
    // on Bench A. This is the explicit acceptance criterion for the
    // v0.3 default bump.
    assert_eq!(
        default_oversample.with_surprise.precision_at_1,
        1.0,
        "v0.3: with default oversample={}, Bench A must achieve P@1=1.0 (was 0.72 in v0.2 \
         when default was 3). If this fails, the default bump regressed.",
        claude_hippo::server::DEFAULT_OVERSAMPLE_FACTOR,
    );
    assert_eq!(
        default_oversample.with_surprise.mrr, 1.0,
        "v0.3: with default oversample, Bench A MRR must = 1.0"
    );
}
