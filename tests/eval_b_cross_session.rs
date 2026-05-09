//! Bench B — Cross-session retrieval (v0.2 axis, recalibrated for v0.3).
//!
//! Setup: 1 high-importance Decision from N days ago, surrounded by 49
//! fresh low-importance chats in the same topic cluster. We sweep N over
//! {0, 30, 90, 365} (today, one half-life, three half-lives, twelve
//! half-lives) and measure how often the old Decision still surfaces under
//! paraphrased queries.
//!
//! v0.3 hypothesis (post `decay_floor` fix):
//! - At every age, the Decision's `surprise · max(decay, decay_floor)` keeps
//!   it above fresh chats' `surprise · 1.0` because the Decision's raw
//!   surprise (importance=1.0 + long+tagged content) dominates the chats'
//!   engagement-only surprise even when Decision's decay term is reduced
//!   to `decay_floor=0.5`. v0.2's failure mode (365d negative lift) is
//!   eliminated.
//! - With full oversample, P@1 should hold at 1.0 across all ages.
//! - Default 6× oversample widens the pool so Bench B's harder one-cluster
//!   workload still benefits, but full coverage is still required for
//!   guaranteed retrieval when all items live in one tight cluster.

#[path = "eval/mod.rs"]
mod eval;

use claude_hippo::surprise::SurpriseWeights;
use eval::*;
use serde::Serialize;

const TOPIC: &str = "auth";
const FRESH_CHATS: usize = 49;
const TOTAL_ITEMS: usize = 50;
const K: usize = 5;
const AGES_DAYS: &[f32] = &[0.0, 30.0, 90.0, 365.0];

const PARAPHRASES: &[&str] = &[
    "what was our auth choice?",
    "remind me of the auth decision we settled on",
    "the auth call we made earlier",
    "auth decision history",
    "earlier auth verdict",
    "what did we decide on auth",
    "previous auth ruling",
    "the auth pick we made",
];

#[derive(Serialize)]
struct AgedAblation {
    decision_age_days: f32,
    /// Sub-result with default 3× oversample.
    default_oversample: AblationResult,
    /// Sub-result with full coverage (oversample = total_items).
    full_oversample: AblationResult,
}

#[derive(Serialize)]
struct BenchBReport {
    setup: SetupSummary,
    runs: Vec<AgedAblation>,
}

#[derive(Serialize)]
struct SetupSummary {
    total_items: usize,
    decision_count: usize,
    fresh_chats: usize,
    queries: usize,
    k: usize,
    half_life_days_in_server: f32,
    note: &'static str,
}

fn build_config(age_days: f32, oversample_factor: usize) -> EvalConfig {
    let mut items = Vec::with_capacity(TOTAL_ITEMS);

    items.push(EvalItem {
        id: "old-decision".into(),
        content: format!(
            "We chose to use approach X for the {TOPIC} module after reviewing options Y and Z. \
             The decision was driven by latency targets and on-call ergonomics. \
             Implementation lives in service-{TOPIC} and is owned by the platform team. \
             Tradeoffs: option Y was rejected because it did not satisfy the audit requirements. \
             Revisit by 2026-Q4 if traffic crosses 50k QPS or compliance scope expands."
        ),
        tags: decision_tags(TOPIC),
        memory_type: Some("Decision".into()),
        importance: Some(1.0),
        cluster: 0,
        age_days,
    });
    for i in 0..FRESH_CHATS {
        items.push(EvalItem {
            id: format!("fresh-chat-{i:02}"),
            content: format!("{TOPIC} chat {i:02}: small note"),
            tags: chat_tags(TOPIC),
            memory_type: Some("Observation".into()),
            importance: None,
            cluster: 0,
            age_days: 0.0,
        });
    }
    assert_eq!(items.len(), TOTAL_ITEMS);

    let queries = PARAPHRASES
        .iter()
        .map(|q| EvalQuery {
            query: q.to_string(),
            cluster: 0,
            relevant_ids: vec!["old-decision".into()],
        })
        .collect();

    EvalConfig {
        items,
        queries,
        k: K,
        weights: SurpriseWeights::default(),
        num_clusters: 2, // 1 topic + 1 unused fallback
        fallback_cluster: 1,
        noise_scale: 0.05,
        oversample_factor,
        prediction_loss: None,
    }
}

#[tokio::test]
async fn bench_b_cross_session_retrieval() {
    use claude_hippo::server::DEFAULT_OVERSAMPLE_FACTOR;
    let mut runs = Vec::with_capacity(AGES_DAYS.len());
    for &age in AGES_DAYS {
        let default_oversample = run_ablation(build_config(age, DEFAULT_OVERSAMPLE_FACTOR))
            .await
            .expect("bench B default-oversample run");
        let full_oversample = run_ablation(build_config(age, TOTAL_ITEMS))
            .await
            .expect("bench B full-oversample run");
        println!(
            "  age={:>4.0}d  default surprise P@1={:.3} MRR={:.3}  full surprise P@1={:.3} MRR={:.3}  \
             baseline (default) P@1={:.3} MRR={:.3}",
            age,
            default_oversample.with_surprise.precision_at_1,
            default_oversample.with_surprise.mrr,
            full_oversample.with_surprise.precision_at_1,
            full_oversample.with_surprise.mrr,
            default_oversample.baseline.precision_at_1,
            default_oversample.baseline.mrr,
        );
        runs.push(AgedAblation {
            decision_age_days: age,
            default_oversample,
            full_oversample,
        });
    }

    let report = BenchBReport {
        setup: SetupSummary {
            total_items: TOTAL_ITEMS,
            decision_count: 1,
            fresh_chats: FRESH_CHATS,
            queries: PARAPHRASES.len(),
            k: K,
            half_life_days_in_server: 30.0,
            note: "1 old Decision (importance=1.0) + 49 fresh chats. Decision is backdated by \
                   the indicated age_days. Baseline = pure cosine (no decay applies). \
                   surprise+decay = v0.3 production default: half_life=30d, decay_floor=0.5, \
                   default_oversample=6. The decay_floor preserves a baseline surprise \
                   contribution so high-importance items are not demoted by fresh \
                   low-surprise items past ~12 half-lives (the v0.2 failure mode).",
        },
        runs,
    };
    write_result_json("bench_b_cross_session", &report).expect("write bench B result");

    // Headline assertions — these are the "spec" of v0.3's decay calibration.

    let r0 = report
        .runs
        .iter()
        .find(|r| r.decision_age_days == 0.0)
        .unwrap();
    assert_eq!(
        r0.full_oversample.with_surprise.precision_at_1, 1.0,
        "fresh decision must be perfectly retrieved at age=0"
    );

    let r30 = report
        .runs
        .iter()
        .find(|r| r.decision_age_days == 30.0)
        .unwrap();
    assert!(
        r30.full_oversample.with_surprise.precision_at_1 >= 0.95,
        "30-day-old decision (1 half-life) must still rank #1 in ≥95% of queries; \
         got P@1={:.3}",
        r30.full_oversample.with_surprise.precision_at_1
    );

    let r90 = report
        .runs
        .iter()
        .find(|r| r.decision_age_days == 90.0)
        .unwrap();
    let r90_lift = r90.full_oversample.with_surprise.mrr - r90.full_oversample.baseline.mrr;
    let r0_lift = r0.full_oversample.with_surprise.mrr - r0.full_oversample.baseline.mrr;
    // v0.3: decay_floor keeps the lift from collapsing — r90_lift should be
    // close to r0_lift (≤ +1e-6 rounding). This is the inverse of v0.2's
    // "monotonic shrinkage" expectation.
    assert!(
        (r0_lift - r90_lift).abs() < 1e-3 || r90_lift <= r0_lift,
        "with decay_floor in effect, r90_lift must not exceed r0_lift by more than rounding; \
         got r0_lift={r0_lift:.3} r90_lift={r90_lift:.3}"
    );

    let r365 = report
        .runs
        .iter()
        .find(|r| r.decision_age_days == 365.0)
        .unwrap();
    // v0.3 fix: at 365 days (~12 half-lives) the raw `decay()` term is ~2e-4,
    // but `decay_floor=0.5` clamps it to 0.5, so the Decision still gets
    // 0.3 * surprise * 0.5 of contribution. Combined with importance=1.0
    // dominating the chats' engagement-only surprise, the Decision wins.
    // This is the explicit reversal of v0.2's "negative lift" honest
    // limitation (CHANGELOG v0.3.0).
    let r365_lift = r365.full_oversample.with_surprise.mrr - r365.full_oversample.baseline.mrr;
    assert!(
        r365_lift > 0.5,
        "v0.3: after 365 days the surprise rerank MUST add positive lift (>0.5) over \
         baseline thanks to decay_floor; got lift={r365_lift:.3}. If this fails, \
         decay_floor regression — see CHANGELOG v0.3.0."
    );
    assert_eq!(
        r365.full_oversample.with_surprise.precision_at_1, 1.0,
        "v0.3: 365-day-old decision must still rank #1 with full oversample"
    );
}
