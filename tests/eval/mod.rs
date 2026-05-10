//! Shared evaluation harness for the v0.2 surprise-selection benches.
//!
//! Why this exists: pure cosine similarity is what every other memory MCP
//! does; claude-hippo's pitch is that **surprise rerank** materially changes
//! retrieval ordering on workloads where naive vector search loses. The
//! harness gives those claims a number.
//!
//! Design choices that matter for re-readers:
//!
//! 1. We use a **ClusteredMockEmbedder**, not the real ONNX model. This makes
//!    the bench deterministic, fast, and re-runnable in CI without a model
//!    download. The trade-off is that "semantic similarity" becomes an
//!    explicit fixture: items in the same cluster are near, items in
//!    different clusters are far. Real LLMs produce more nuanced
//!    embeddings, but the *ranking effect of surprise rerank* doesn't
//!    depend on the embedding distribution — it depends on score arithmetic,
//!    which the harness exercises faithfully.
//!
//! 2. The harness backdates `created_at` directly via SQL. There's no
//!    public API for time travel because production memories should never
//!    be backdated, only audited.
//!
//! 3. Every bench writes a JSON artefact under `target/eval_results/`.
//!    `docs/SURPRISE_SELECTION.md` quotes those numbers; updating the
//!    docs without re-running this harness is a process bug.
#![allow(dead_code)]

use claude_hippo::embeddings::Embedder;
use claude_hippo::prediction_loss::PredictionLossBackend;
use claude_hippo::server::{
    MemoryServer, RankingConfig, RecallOptions, RecallParams, RememberParams, RememberResult,
};
use claude_hippo::storage::Storage;
use claude_hippo::surprise::SurpriseWeights;
use claude_hippo::{HippoError, EMBEDDING_DIM};
use serde::Serialize;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

/// Designer-controlled mock embedder. Each cluster has an orthogonal basis
/// center; per-text noise is small and deterministic from SHA256(text).
///
/// Texts not registered via `assign` land in `fallback_cluster`, which is
/// also designer-controlled: pick a cluster that's *not* used by any item
/// to make summary queries unrelated to all corpus items.
pub struct ClusteredMockEmbedder {
    centers: Vec<Vec<f32>>,
    assignments: HashMap<String, usize>,
    fallback_cluster: usize,
    noise_scale: f32,
}

impl ClusteredMockEmbedder {
    pub fn new(num_clusters: usize, fallback_cluster: usize, noise_scale: f32) -> Self {
        assert!(num_clusters >= 1);
        assert!(fallback_cluster < num_clusters);
        let centers = (0..num_clusters).map(orthogonal_basis).collect();
        Self {
            centers,
            assignments: HashMap::new(),
            fallback_cluster,
            noise_scale,
        }
    }

    pub fn assign(&mut self, text: impl Into<String>, cluster: usize) {
        let s = text.into();
        let n = self.centers.len();
        assert!(
            cluster < n,
            "cluster {cluster} >= num_clusters {n} for text {s:?}"
        );
        self.assignments.insert(s, cluster);
    }
}

fn orthogonal_basis(idx: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; EMBEDDING_DIM];
    v[idx % EMBEDDING_DIM] = 1.0;
    v
}

fn deterministic_noise(text: &str, scale: f32) -> Vec<f32> {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(text.as_bytes());
    let seed = h.finalize();
    let mut v = vec![0.0_f32; EMBEDDING_DIM];
    for (i, b) in (0..EMBEDDING_DIM).zip(seed.iter().cycle()) {
        v[i] = ((*b as f32 / 127.5) - 1.0) * scale;
    }
    v
}

impl Embedder for ClusteredMockEmbedder {
    fn embed_one(&self, text: &str) -> claude_hippo::Result<Vec<f32>> {
        let cluster = self
            .assignments
            .get(text)
            .copied()
            .unwrap_or(self.fallback_cluster);
        if cluster >= self.centers.len() {
            return Err(HippoError::Embedding(format!(
                "cluster {cluster} out of range for {} clusters",
                self.centers.len()
            )));
        }
        let mut v = self.centers[cluster].clone();
        let noise = deterministic_noise(text, self.noise_scale);
        for (a, b) in v.iter_mut().zip(noise.iter()) {
            *a += *b;
        }
        // L2 normalize
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        for x in v.iter_mut() {
            *x /= norm;
        }
        Ok(v)
    }

    fn embed_batch(&self, texts: &[&str]) -> claude_hippo::Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed_one(t)).collect()
    }
}

#[derive(Clone, Debug)]
pub struct EvalItem {
    pub id: String,
    pub content: String,
    pub tags: Vec<String>,
    pub memory_type: Option<String>,
    pub importance: Option<f32>,
    pub cluster: usize,
    /// Backdate this item by `age_days` days (0 = today). Bench B uses this
    /// to test forgetting-curve interaction with surprise.
    pub age_days: f32,
}

#[derive(Clone, Debug)]
pub struct EvalQuery {
    pub query: String,
    pub cluster: usize,
    /// Item IDs (matching `EvalItem.id`) that should be retrieved for this
    /// query. The metric runner compares against this set.
    pub relevant_ids: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Metrics {
    pub k: usize,
    pub queries: usize,
    /// Mean reciprocal rank: average over queries of `1 / rank_of_first_relevant`.
    /// 0.0 if no relevant retrieved.
    pub mrr: f64,
    /// Mean over queries of `1 if first-rank is relevant else 0`.
    pub precision_at_1: f64,
    /// Mean over queries of `relevant_in_top_k / k`.
    pub precision_at_k: f64,
    /// Mean over queries of `relevant_in_top_k / total_relevant_for_query`.
    pub recall_at_k: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct AblationResult {
    /// no_surprise_boost = true (pure cosine similarity).
    pub baseline: Metrics,
    /// no_surprise_boost = false (production default: 0.7 sim + 0.3 surprise·decay).
    pub with_surprise: Metrics,
}

#[derive(Clone)]
pub struct EvalConfig {
    pub items: Vec<EvalItem>,
    pub queries: Vec<EvalQuery>,
    /// Top-K for `recall_at_k` / `precision_at_k`.
    pub k: usize,
    /// Surprise score weights (default: SurpriseWeights::default()).
    pub weights: SurpriseWeights,
    /// Number of orthogonal clusters in the embedding space. Must be >= max
    /// cluster id used by any item or query, plus the fallback.
    pub num_clusters: usize,
    /// Texts not explicitly assigned land here. Pick a cluster reserved for
    /// "summary"/general queries that should not match any specific topic.
    pub fallback_cluster: usize,
    /// Per-dimension noise (uniform `[-scale, scale]`) added to each text's
    /// cluster center before L2 normalization. 0.05 yields intra-cluster
    /// `cos_sim ≈ 0.76` (with σ ≈ 0.04 spread, enough to differentiate items
    /// for KNN ordering) and inter-cluster `cos_sim ≈ 0`. Lowering the
    /// scale tightens within-cluster ordering toward perfect ties; raising
    /// it makes between-cluster overlap more likely.
    pub noise_scale: f32,
    /// Oversample factor for KNN before surprise rerank. Larger values give
    /// rerank a wider candidate pool. Default 3 matches production. Set
    /// `num_items / k` to fully disable the over-fetch ceiling.
    pub oversample_factor: usize,
    /// Optional prediction-loss backend. When `None`, the run uses the v0.2
    /// fallback (`prediction_loss = None`, `w_prediction` redistributed).
    /// When `Some`, every `remember()` populates `prediction_loss` from the
    /// backend and the surprise score uses the full 4-component formula.
    /// Bench D exercises this; A/B/C leave it `None` for v0.2 parity.
    #[allow(clippy::type_complexity)]
    pub prediction_loss: Option<Arc<dyn PredictionLossBackend>>,
}

pub async fn run_ablation(cfg: EvalConfig) -> anyhow::Result<AblationResult> {
    let baseline = run_one(&cfg, /* no_surprise_boost */ true).await?;
    let with_surprise = run_one(&cfg, /* no_surprise_boost */ false).await?;
    Ok(AblationResult {
        baseline,
        with_surprise,
    })
}

async fn run_one(cfg: &EvalConfig, no_surprise_boost: bool) -> anyhow::Result<Metrics> {
    claude_hippo::storage::register_sqlite_vec();

    // Build embedder with all known texts pre-registered to clusters.
    let mut embedder =
        ClusteredMockEmbedder::new(cfg.num_clusters, cfg.fallback_cluster, cfg.noise_scale);
    for it in &cfg.items {
        embedder.assign(it.content.clone(), it.cluster);
    }
    for q in &cfg.queries {
        embedder.assign(q.query.clone(), q.cluster);
    }
    let embedder: Arc<dyn Embedder> = Arc::new(embedder);

    let store = Storage::open_in_memory()?;
    let server = MemoryServer::new_full(
        store,
        embedder,
        cfg.prediction_loss.clone(),
        cfg.weights,
        RankingConfig::default(),
    );

    // Insert items in deterministic order. content_hash uniqueness assumed
    // (fixtures must avoid duplicate content strings).
    let mut id_to_hash: HashMap<String, String> = HashMap::new();
    for it in &cfg.items {
        let r: RememberResult = server
            .remember(RememberParams {
                content: it.content.clone(),
                tags: it.tags.clone(),
                memory_type: it.memory_type.clone(),
                importance: it.importance,
                metadata: None,
            })
            .await
            .map_err(|e| anyhow::anyhow!("remember failed for {}: {:?}", it.id, e))?;
        id_to_hash.insert(it.id.clone(), r.content_hash);
        if it.age_days > 0.0 {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs_f64();
            let backdated = now - (it.age_days as f64) * 86400.0;
            let storage = server.storage_arc();
            let s = storage.lock().await;
            s.debug_set_created_at(r.id, backdated)?;
        }
    }

    // Now evaluate.
    let opts = RecallOptions {
        oversample_factor: cfg.oversample_factor,
    };
    let mut mrr_sum = 0.0_f64;
    let mut p1_sum = 0.0_f64;
    let mut pk_sum = 0.0_f64;
    let mut rk_sum = 0.0_f64;
    for q in &cfg.queries {
        let hits = server
            .recall_with_options(
                RecallParams {
                    query: q.query.clone(),
                    limit: cfg.k,
                    no_surprise_boost,
                    oversample_factor: None,
                    mode: None,
                    seed_id: None,
                },
                opts,
            )
            .await
            .map_err(|e| anyhow::anyhow!("recall failed for {:?}: {:?}", q.query, e))?;
        let relevant_hashes: std::collections::HashSet<String> = q
            .relevant_ids
            .iter()
            .filter_map(|id| id_to_hash.get(id).cloned())
            .collect();
        let total_relevant = relevant_hashes.len().max(1) as f64;

        // first relevant rank (1-indexed)
        let mut first_relevant_rank: Option<usize> = None;
        let mut relevant_in_topk = 0usize;
        for (i, h) in hits.iter().enumerate() {
            let rank = i + 1;
            if relevant_hashes.contains(&h.memory.content_hash) {
                if first_relevant_rank.is_none() {
                    first_relevant_rank = Some(rank);
                }
                if rank <= cfg.k {
                    relevant_in_topk += 1;
                }
            }
        }
        if let Some(r) = first_relevant_rank {
            mrr_sum += 1.0 / r as f64;
            if r == 1 {
                p1_sum += 1.0;
            }
        }
        pk_sum += relevant_in_topk as f64 / cfg.k as f64;
        rk_sum += relevant_in_topk as f64 / total_relevant;
    }
    let n = cfg.queries.len() as f64;
    Ok(Metrics {
        k: cfg.k,
        queries: cfg.queries.len(),
        mrr: mrr_sum / n,
        precision_at_1: p1_sum / n,
        precision_at_k: pk_sum / n,
        recall_at_k: rk_sum / n,
    })
}

/// Persist a result to `target/eval_results/<bench>.json` for downstream
/// docs consumption. Writes the canonical artefact path that
/// `docs/SURPRISE_SELECTION.md` references.
pub fn write_result_json<T: serde::Serialize>(bench_name: &str, payload: &T) -> anyhow::Result<()> {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("target");
    p.push("eval_results");
    std::fs::create_dir_all(&p)?;
    p.push(format!("{bench_name}.json"));
    let f = std::fs::File::create(&p)?;
    serde_json::to_writer_pretty(f, payload)?;
    eprintln!("[eval] wrote {}", p.display());
    Ok(())
}

/// Small helper: stable, terse default tag set for a "chat" item.
pub fn chat_tags(topic: &str) -> Vec<String> {
    vec![topic.to_string()]
}

/// Small helper: tag set for a "decision" item.
pub fn decision_tags(topic: &str) -> Vec<String> {
    vec![
        topic.to_string(),
        "decision".to_string(),
        "important".to_string(),
    ]
}
