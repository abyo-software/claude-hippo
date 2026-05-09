//! Bench C — Decision trace (v0.2 evaluation axis).
//!
//! Setup: a single topic cluster with 20 items mixed across memory types:
//!   4 Decision (importance=1.0, multi-tag, long content) +
//!   4 Observation + 4 Pattern + 4 Discovery + 4 Learning
//!     (each importance=0.3, single tag, short content)
//!
//! For paraphrased "decision trace" queries on this topic, we measure how
//! many of the 4 Decisions surface in the top-K results.
//!
//! Hypothesis: surprise rerank concentrates Decisions at the top because
//! the explicit + engagement components reward high-importance long-form
//! memories. Pure cosine doesn't see memory_type or importance, so it
//! distributes top-K roughly uniformly across the 20 cluster items.
//!
//! This is the "why this design" use case: a developer asks "why did we
//! build it this way" → they want the 4 actual Decisions, not the 16
//! Observations/Learnings that share the same topic.

#[path = "eval/mod.rs"]
mod eval;

use claude_hippo::surprise::SurpriseWeights;
use eval::*;
use serde::Serialize;

const TOPIC: &str = "auth";
const NON_DECISION_TYPES: &[&str] = &["Observation", "Pattern", "Discovery", "Learning"];
const PER_TYPE: usize = 4;
const TOTAL_ITEMS: usize = (1 + NON_DECISION_TYPES.len()) * PER_TYPE; // 5 * 4 = 20
const K: usize = 5;

const PARAPHRASES: &[&str] = &[
    "show me the auth decisions",
    "auth decision trace",
    "why did we build auth this way",
    "auth design rationale",
    "auth-related architectural calls",
    "decision history for auth",
    "what calls did we make on auth",
    "the auth decisions we shipped",
];

#[derive(Serialize)]
struct BenchCReport {
    setup: SetupSummary,
    /// Run with default `SurpriseWeights` (production).
    default_weights: WeightsRun,
    /// Run with explicit-heavy weights to verify that importance dominates
    /// rerank when configured to do so. Documents the v0.2 `--surprise-weights`
    /// CLI flag's actual effect.
    explicit_heavy_weights: WeightsRun,
}

#[derive(Serialize)]
struct WeightsRun {
    weights: Weights,
    default_oversample: AblationResult,
    full_oversample: AblationResult,
}

#[derive(Serialize)]
struct Weights {
    w_outlier: f32,
    w_engagement: f32,
    w_explicit: f32,
    w_prediction: f32,
}

impl From<SurpriseWeights> for Weights {
    fn from(w: SurpriseWeights) -> Self {
        Self {
            w_outlier: w.w_outlier,
            w_engagement: w.w_engagement,
            w_explicit: w.w_explicit,
            w_prediction: w.w_prediction,
        }
    }
}

#[derive(Serialize)]
struct SetupSummary {
    total_items: usize,
    decisions: usize,
    other_types: &'static [&'static str],
    per_type: usize,
    queries: usize,
    k: usize,
    note: &'static str,
}

fn build_config(weights: SurpriseWeights, oversample_factor: usize) -> EvalConfig {
    let mut items = Vec::with_capacity(TOTAL_ITEMS);
    let mut decision_ids = Vec::with_capacity(PER_TYPE);

    for i in 0..PER_TYPE {
        let id = format!("decision-{TOPIC}-{i:02}");
        decision_ids.push(id.clone());
        items.push(EvalItem {
            id,
            content: format!(
                "Decision {i:02} on {TOPIC}: We chose approach A_{i} after weighing performance, \
                 audit posture, and migration cost. Rejected alternatives: B_{i} (latency), C_{i} \
                 (compliance). Owned by platform-{TOPIC}; revisit by 2026-Q4."
            ),
            tags: decision_tags(TOPIC),
            memory_type: Some("Decision".into()),
            importance: Some(1.0),
            cluster: 0,
            age_days: 0.0,
        });
    }
    for mt in NON_DECISION_TYPES {
        for i in 0..PER_TYPE {
            items.push(EvalItem {
                id: format!("{}-{TOPIC}-{i:02}", mt.to_lowercase()),
                content: format!(
                    "{TOPIC} {} note {i:02}: short observation",
                    mt.to_lowercase()
                ),
                tags: chat_tags(TOPIC),
                memory_type: Some((*mt).to_string()),
                importance: Some(0.3),
                cluster: 0,
                age_days: 0.0,
            });
        }
    }
    assert_eq!(items.len(), TOTAL_ITEMS);

    let queries: Vec<EvalQuery> = PARAPHRASES
        .iter()
        .map(|q| EvalQuery {
            query: q.to_string(),
            cluster: 0,
            relevant_ids: decision_ids.clone(),
        })
        .collect();

    EvalConfig {
        items,
        queries,
        k: K,
        weights,
        num_clusters: 2, // 1 topic + 1 unused fallback
        fallback_cluster: 1,
        noise_scale: 0.05,
        oversample_factor,
        prediction_loss: None,
    }
}

async fn run_for_weights(weights: SurpriseWeights) -> WeightsRun {
    let default_oversample = run_ablation(build_config(
        weights,
        claude_hippo::server::DEFAULT_OVERSAMPLE_FACTOR,
    ))
    .await
    .expect("bench C default-oversample run");
    let full_oversample = run_ablation(build_config(weights, TOTAL_ITEMS))
        .await
        .expect("bench C full-oversample run");
    WeightsRun {
        weights: weights.into(),
        default_oversample,
        full_oversample,
    }
}

#[tokio::test]
async fn bench_c_decision_trace() {
    let default_weights = run_for_weights(SurpriseWeights::default()).await;
    // Explicit-heavy: lift `w_explicit` from 0.1 to 0.5 by trading down
    // outlier and engagement. Must still sum to 1.0.
    let explicit_heavy_weights = run_for_weights(SurpriseWeights {
        w_outlier: 0.2,
        w_engagement: 0.1,
        w_explicit: 0.5,
        w_prediction: 0.2,
    })
    .await;

    println!("\nBench C — Decision trace");
    println!(
        "  default weights, full oversample : surprise P@K={:.3} R@K={:.3}  baseline P@K={:.3} R@K={:.3}",
        default_weights.full_oversample.with_surprise.precision_at_k,
        default_weights.full_oversample.with_surprise.recall_at_k,
        default_weights.full_oversample.baseline.precision_at_k,
        default_weights.full_oversample.baseline.recall_at_k,
    );
    println!(
        "  explicit-heavy weights, full     : surprise P@K={:.3} R@K={:.3}",
        explicit_heavy_weights
            .full_oversample
            .with_surprise
            .precision_at_k,
        explicit_heavy_weights
            .full_oversample
            .with_surprise
            .recall_at_k,
    );

    let report = BenchCReport {
        setup: SetupSummary {
            total_items: TOTAL_ITEMS,
            decisions: PER_TYPE,
            other_types: NON_DECISION_TYPES,
            per_type: PER_TYPE,
            queries: PARAPHRASES.len(),
            k: K,
            note: "1 cluster, 4 Decisions (importance=1.0, multi-tag) + 16 non-Decision items \
                   (4 each Observation/Pattern/Discovery/Learning, importance=0.3, single tag). \
                   Per query, relevant set = 4 Decisions; ideal P@K = min(4,5)/5 = 0.8.",
        },
        default_weights,
        explicit_heavy_weights,
    };
    write_result_json("bench_c_decision_trace", &report).expect("write bench C result");

    // Headline assertions

    // 1. Default weights, full oversample: surprise rerank must surface
    //    enough Decisions that recall@K is materially above baseline.
    let lift_recall = report
        .default_weights
        .full_oversample
        .with_surprise
        .recall_at_k
        - report.default_weights.full_oversample.baseline.recall_at_k;
    assert!(
        lift_recall >= 0.4,
        "default weights, full oversample must lift Recall@K by ≥0.4 over baseline; \
         got lift={lift_recall:.3} (surprise={:.3} baseline={:.3})",
        report
            .default_weights
            .full_oversample
            .with_surprise
            .recall_at_k,
        report.default_weights.full_oversample.baseline.recall_at_k,
    );

    // 2. Explicit-heavy weights must give an even stronger lift, proving
    //    that the `--surprise-weights` knob actually changes behavior.
    let lift_eh = report
        .explicit_heavy_weights
        .full_oversample
        .with_surprise
        .recall_at_k;
    let lift_default = report
        .default_weights
        .full_oversample
        .with_surprise
        .recall_at_k;
    assert!(
        lift_eh >= lift_default,
        "explicit-heavy weights must produce ≥ recall as default weights; \
         got default={lift_default:.3} explicit_heavy={lift_eh:.3}"
    );

    // 3. Baseline (pure cosine) on this fixture should NOT be at ceiling.
    //    If baseline is already 1.0, the bench fixture is too easy and the
    //    surprise lift would not be measurable.
    assert!(
        report.default_weights.full_oversample.baseline.recall_at_k < 0.9,
        "baseline recall must leave headroom for surprise to demonstrate lift; \
         got {:.3}",
        report.default_weights.full_oversample.baseline.recall_at_k,
    );
}
