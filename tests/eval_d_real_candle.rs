//! v0.5 Phase D — Real-backend variant of Bench D (candle-local).
//!
//! Replays the v0.3 Bench D fixture with a real `CandleLocalPredictionLoss`
//! (Qwen2.5-0.5B, CPU) instead of the deterministic `MockPredictionLoss`.
//! The point is to measure whether the same wiring that passed Bench D's
//! synthetic-noise smoke also produces a *useful* surprise gradient when
//! the prediction-loss component is real LLM NLL.
//!
//! Gated `#[cfg(feature = "candle")]` + `#[ignore]` because:
//!   - downloads ~1 GB of Qwen2.5-0.5B safetensors on first run
//!   - per-position CPU forward is O(N) per scoring call (~3-4 s for a
//!     ~50-token Decision string)
//!   - 100 items × ~1.5 s ≈ ~2.5 minutes for one ablation; each
//!     `run_ablation` runs twice (baseline + with_surprise), and we
//!     compare against the v0.3 Mock variant, so a full run is ~10 min.
//!     Way out of scope for default `cargo test`.
//!
//! Run with:
//!
//! ```bash
//! cargo test --features candle -- --ignored --nocapture bench_d_real_candle
//! ```
//!
//! Writes `target/eval_results/bench_d_real_candle.json` paired with the
//! Mock variant for honest comparison in `docs/SURPRISE_SELECTION.md`.

#![cfg(feature = "candle")]

#[path = "eval/mod.rs"]
mod eval;

use claude_hippo::prediction_loss::{
    CandleLocalConfig, CandleLocalPredictionLoss, PredictionLossBackend,
};
use claude_hippo::surprise::SurpriseWeights;
use eval::*;
use serde::Serialize;
use std::sync::Arc;

const TOTAL_ITEMS: usize = 100;
const CHATS_PER_CLUSTER: usize = 19;
const TOPICS: &[&str] = &["auth", "db", "billing", "ui", "infra"];
const K: usize = 5;

const PARAPHRASES: &[&str] = &[
    "what was our decision about {topic}?",
    "remind me of the {topic} choice we made",
];

#[derive(Serialize)]
struct CandleReport {
    backend: &'static str,
    model: String,
    fixture: &'static str,
    note: &'static str,
    ablation: AblationResult,
    p1_lift: f64,
    mrr_lift: f64,
}

fn build_config(backend: Arc<dyn PredictionLossBackend>) -> EvalConfig {
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
        prediction_loss: Some(backend),
        embedder_override: None,
    }
}

#[tokio::test]
#[ignore = "candle real LLM: downloads ~1GB Qwen2.5-0.5B, CPU inference \
            takes ~10min for full ablation"]
async fn bench_d_real_candle() {
    let cfg = CandleLocalConfig {
        use_gpu: false, // deterministic CPU path; sidesteps cuDNN dep
        ..Default::default()
    };
    let model_id = cfg.model_id.clone();
    let backend: Arc<dyn PredictionLossBackend> =
        Arc::new(CandleLocalPredictionLoss::new(cfg).expect("CandleLocalPredictionLoss init"));
    let ablation = run_ablation(build_config(backend))
        .await
        .expect("bench D real-candle run");
    let p1_lift = ablation.with_surprise.precision_at_1 - ablation.baseline.precision_at_1;
    let mrr_lift = ablation.with_surprise.mrr - ablation.baseline.mrr;
    let report = CandleReport {
        backend: "CandleLocalPredictionLoss(CPU)",
        model: model_id,
        fixture: "Bench D prediction-loss real LLM (100 items, 10 queries)",
        note: "Real Qwen2.5-0.5B NLL replaces MockPredictionLoss SHA256 noise. \
               Decisions (long, structured, content-rich) score higher NLL than \
               short chat notes — same gradient the v0.4 D-spike captured (cliché \
               1.20 NLL vs specific decision 4.96 NLL). The retrieval headline \
               here is whether that gradient survives surprise's 0.7·sim + 0.3· \
               surprise·decay blend in the actual rerank.",
        ablation: ablation.clone(),
        p1_lift,
        mrr_lift,
    };
    write_result_json("bench_d_real_candle", &report).expect("write bench D real-candle");
    println!("\nBench D — real-candle (CandleLocalPredictionLoss CPU)");
    println!(
        "  baseline P@1={:.3} MRR={:.3}  surprise P@1={:.3} MRR={:.3}  lift Δ={:+.3}",
        ablation.baseline.precision_at_1,
        ablation.baseline.mrr,
        ablation.with_surprise.precision_at_1,
        ablation.with_surprise.mrr,
        p1_lift,
    );
    // Same honest soft-assertion as the real-local benches: lift must
    // not be negative. Real-LLM NLL is not guaranteed to perfectly
    // separate Decision from chat noise on every fixture, but rerank
    // must not actively hurt.
    assert!(
        p1_lift >= -1e-3,
        "real-candle Bench D: surprise P@1 ({:.3}) regressed vs baseline ({:.3})",
        ablation.with_surprise.precision_at_1,
        ablation.baseline.precision_at_1,
    );
}
