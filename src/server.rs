//! MCP server (rmcp) — claude-hippo の 5 ツール + SHODH 互換エイリアス。
//!
//! Tools:
//! - `hippo_remember` (alias: `store_memory`): 記憶保存 + surprise score 算出
//! - `hippo_recall`   (alias: `retrieve_memory`): semantic search + surprise-weighted ranking
//! - `hippo_list_recent` (alias: `list_memories`): 直近 N 件
//! - `hippo_forget`   (alias: `delete_memory`): soft-delete by content_hash
//! - `hippo_session_summary`: 直近セッションの compact summary
//! - `ping`: health probe (vec_version, memory_count を返す)

use crate::embeddings::Embedder;
use crate::memory_tool::{self, MemoryToolParams};
use crate::prediction_loss::PredictionLossBackend;
use crate::storage::{self, MemoryRow, Storage};
use crate::surprise::{self, SurpriseComponents, SurpriseWeights};
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, Content, Implementation, ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
    transport::io::stdio,
    ErrorData, ServerHandler, ServiceExt,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

pub const DEFAULT_HALF_LIFE_DAYS: f32 = 30.0;
pub const DEFAULT_DECAY_FLOOR: f32 = 0.5;
pub const DEFAULT_OVERSAMPLE_FACTOR: usize = 6;
/// v0.5 Phase B: per-co-recall edge increment. Small enough that a single
/// session of ~10 recalls won't saturate edges to 1.0; large enough that
/// repeated co-activation across sessions converges visibly. Capped at
/// 1.0 inside `Storage::reinforce_co_recalled`.
pub const DEFAULT_CO_RECALL_ALPHA: f32 = 0.1;
/// v0.5 Phase B: associative recall returns up to this many neighbors of
/// the seed. Used only when `RecallParams.mode == "associative"` or
/// `"hybrid"`.
pub const DEFAULT_ASSOCIATIVE_LIMIT: usize = 20;
const DEFAULT_RETRIEVE_K: usize = 10;
const DEFAULT_LIST_N: i64 = 20;

/// Server-wide ranking config. Plumbed into every recall.
///
/// `decay_floor` was added in v0.3 to fix the "old high-surprise Decision
/// gets demoted by a fresh low-surprise chat after ~12 half-lives" failure
/// mode that v0.2 Bench B surfaced. `default_oversample_factor` was bumped
/// from 3 to 6 in v0.3 so production Bench A reaches precision@1 = 1.0
/// without callers needing to tune anything.
#[derive(Debug, Clone, Copy)]
pub struct RankingConfig {
    pub half_life_days: f32,
    pub decay_floor: f32,
    pub default_oversample_factor: usize,
    /// v0.5 Phase B: enable Hebbian co-recall reinforcement. When `recall`
    /// returns ≥2 alive results, all unordered pairs in the result get
    /// their `memory_associations` edge weight bumped by `co_recall_alpha`.
    /// Default: true. Disable with `--no-hebbian-reinforce` for read-only
    /// benchmarking or when DB write amplification matters.
    pub reinforce_co_recall: bool,
    /// v0.5 Phase B: per-co-recall edge increment. See
    /// `DEFAULT_CO_RECALL_ALPHA`.
    pub co_recall_alpha: f32,
}

impl Default for RankingConfig {
    fn default() -> Self {
        Self {
            half_life_days: DEFAULT_HALF_LIFE_DAYS,
            decay_floor: DEFAULT_DECAY_FLOOR,
            default_oversample_factor: DEFAULT_OVERSAMPLE_FACTOR,
            reinforce_co_recall: true,
            co_recall_alpha: DEFAULT_CO_RECALL_ALPHA,
        }
    }
}

/// Per-call retrieval override. The MCP `RecallParams.oversample_factor`
/// field also flows through here. Callers that need full-corpus coverage
/// (e.g. eval harness, "summary" queries) bypass the server default by
/// passing `recall_with_options` directly.
#[derive(Debug, Clone, Copy)]
pub struct RecallOptions {
    /// Multiplier applied to `RecallParams.limit` to determine how many
    /// candidates KNN returns before surprise rerank trims to `limit`.
    /// Set higher when the corpus is large and you want more items
    /// considered for rerank; set to 1 to disable over-fetch entirely.
    pub oversample_factor: usize,
}

impl Default for RecallOptions {
    fn default() -> Self {
        Self {
            oversample_factor: DEFAULT_OVERSAMPLE_FACTOR,
        }
    }
}

pub struct MemoryServer {
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
    storage: Arc<Mutex<Storage>>,
    embedder: Arc<dyn Embedder>,
    /// Optional backend for filling `SurpriseComponents.prediction_loss`.
    /// `None` falls back to the v0.2 redistribution behavior.
    prediction_loss: Option<Arc<dyn PredictionLossBackend>>,
    /// When true, the `memory` tool (Anthropic Memory Tool compat) is
    /// active. Default false — opt in via `--anthropic-memory-tool`.
    enable_memory_tool: bool,
    weights: SurpriseWeights,
    ranking: RankingConfig,
    started_at: std::time::Instant,
}

// ---------- ping ----------

#[derive(Debug, Default, Deserialize, schemars::JsonSchema)]
pub struct PingParams {}

#[derive(Debug, Serialize)]
pub struct PingResult {
    pub status: &'static str,
    pub backend: &'static str,
    pub vec_version: String,
    pub alive: i64,
    pub total: i64,
    pub uptime_seconds: u64,
    pub claude_hippo_version: &'static str,
}

// ---------- hippo_remember / store_memory ----------

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RememberParams {
    /// The memory content to store.
    pub content: String,
    /// Tags as array of strings (e.g. ["auth", "decision"]).
    #[serde(default)]
    pub tags: Vec<String>,
    /// SHODH MemoryType (Decision / Learning / Discovery / Pattern / etc).
    /// Defaults to "Observation".
    #[serde(default)]
    pub memory_type: Option<String>,
    /// User-marked importance, 0.0..=1.0. Increases the surprise score.
    #[serde(default)]
    pub importance: Option<f32>,
    /// Free-form metadata stored as JSON. Reserved namespace `_hippo`.
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct RememberResult {
    pub success: bool,
    pub id: i64,
    pub content_hash: String,
    pub duplicate: bool,
    pub surprise_score: f32,
    pub surprise_components: SurpriseComponents,
}

// ---------- hippo_recall / retrieve_memory ----------

fn default_k() -> usize {
    DEFAULT_RETRIEVE_K
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RecallParams {
    pub query: String,
    /// Top-N results to return.
    #[serde(default = "default_k")]
    pub limit: usize,
    /// Disable surprise-weighted ranking (pure cosine similarity only).
    #[serde(default)]
    pub no_surprise_boost: bool,
    /// Per-call override for the KNN over-fetch multiplier. Larger values
    /// give surprise rerank a wider candidate pool at the cost of more SQL
    /// work. Default = server-wide setting (6 in v0.3, was 3 in v0.2). Set
    /// to `limit / 1` to disable over-fetch entirely. Caps at 1 minimum.
    #[serde(default)]
    pub oversample_factor: Option<usize>,
    /// v0.5 Phase B: recall mode. `"semantic"` (default) is the v0.4
    /// behavior. `"associative"` follows Hebbian edges from a seed
    /// (semantic top-1 if `seed_id` is absent). `"hybrid"` merges semantic
    /// hits with associative neighbors, deduped, ranked by combined score.
    #[serde(default)]
    pub mode: Option<String>,
    /// v0.5 Phase B: explicit seed for `mode == "associative"` or
    /// `"hybrid"`. When unset, the seed is the top-1 semantic match for
    /// `query`. Useful for clients that already have an id from a prior
    /// recall and want to expand the local subgraph without re-running
    /// semantic search.
    #[serde(default)]
    pub seed_id: Option<i64>,
}

#[derive(Debug, Serialize)]
pub struct RecalledMemory {
    pub memory: MemoryRow,
    /// 0.0..=1.0, higher is better. Combines cosine sim + surprise * decay.
    pub score: f32,
    pub cosine_similarity: f32,
    pub surprise_score: Option<f32>,
}

/// v0.5 Phase B: recall dispatch mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RecallMode {
    /// Pure semantic search + surprise rerank (v0.4 behavior). Default.
    #[default]
    Semantic,
    /// Hebbian neighbors of a seed memory only.
    Associative,
    /// Union of semantic hits + associative neighbors, deduped, ranked by
    /// blended score.
    Hybrid,
}

/// Parse `RecallParams.mode` (`None` defaults to Semantic). Unknown
/// strings fall back to Semantic with a `warn` trace so clients are not
/// silently broken; callers that need strict parsing should validate
/// upstream.
pub fn parse_recall_mode(s: Option<&str>) -> RecallMode {
    match s.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        None | Some("") | Some("semantic") => RecallMode::Semantic,
        Some("associative") | Some("assoc") | Some("hebbian") => RecallMode::Associative,
        Some("hybrid") | Some("mixed") => RecallMode::Hybrid,
        Some(other) => {
            tracing::warn!(
                mode = other,
                "unknown recall mode, falling back to semantic"
            );
            RecallMode::Semantic
        }
    }
}

// ---------- hippo_list_recent / list_memories ----------

fn default_list_n() -> i64 {
    DEFAULT_LIST_N
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ListRecentParams {
    #[serde(default = "default_list_n")]
    pub n: i64,
}

#[derive(Debug, Serialize)]
pub struct ListRecentResult {
    pub memories: Vec<MemoryRow>,
    pub count: usize,
}

// ---------- hippo_forget / delete_memory ----------

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ForgetParams {
    /// content_hash to delete (preferred selector).
    pub content_hash: Option<String>,
    /// id to delete (alternative).
    pub id: Option<i64>,
    /// Dry-run: report match without deleting.
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Debug, Serialize)]
pub struct ForgetResult {
    pub success: bool,
    pub deleted: usize,
    pub dry_run: bool,
}

// ---------- hippo_session_summary ----------

#[derive(Debug, Default, Deserialize, schemars::JsonSchema)]
pub struct SessionSummaryParams {
    /// Lookback window in hours.
    #[serde(default)]
    pub hours: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct SessionSummary {
    pub window_hours: u32,
    pub total_memories: usize,
    pub by_type: Vec<(String, usize)>,
    pub top_tags: Vec<(String, usize)>,
    pub highlights: Vec<MemoryRow>,
    pub mean_surprise: f32,
}

// ---------- impl ----------

#[tool_router]
impl MemoryServer {
    pub fn new(storage: Storage, embedder: Arc<dyn Embedder>) -> Self {
        Self::new_with_config(
            storage,
            embedder,
            SurpriseWeights::default(),
            RankingConfig::default(),
        )
    }

    pub fn new_with_weights(
        storage: Storage,
        embedder: Arc<dyn Embedder>,
        weights: SurpriseWeights,
    ) -> Self {
        Self::new_with_config(storage, embedder, weights, RankingConfig::default())
    }

    pub fn new_with_config(
        storage: Storage,
        embedder: Arc<dyn Embedder>,
        weights: SurpriseWeights,
        ranking: RankingConfig,
    ) -> Self {
        Self::new_full(storage, embedder, None, weights, ranking)
    }

    /// Full constructor including the optional prediction-loss backend.
    /// Used by [`run_stdio_full`] and the CLI when
    /// `--prediction-loss-backend` is set to a non-`none` value.
    pub fn new_full(
        storage: Storage,
        embedder: Arc<dyn Embedder>,
        prediction_loss: Option<Arc<dyn PredictionLossBackend>>,
        weights: SurpriseWeights,
        ranking: RankingConfig,
    ) -> Self {
        Self::new_full_with_memory_tool(storage, embedder, prediction_loss, weights, ranking, false)
    }

    /// Most-explicit constructor. Adds the `enable_memory_tool` flag for
    /// the v0.3 Anthropic Memory Tool compatibility layer. Default off
    /// (CLI surface: `--anthropic-memory-tool`).
    pub fn new_full_with_memory_tool(
        storage: Storage,
        embedder: Arc<dyn Embedder>,
        prediction_loss: Option<Arc<dyn PredictionLossBackend>>,
        weights: SurpriseWeights,
        ranking: RankingConfig,
        enable_memory_tool: bool,
    ) -> Self {
        Self::from_shared_storage(
            Arc::new(Mutex::new(storage)),
            embedder,
            prediction_loss,
            weights,
            ranking,
            enable_memory_tool,
        )
    }

    /// v0.4: build a `MemoryServer` from an already-shared `Arc<Mutex<Storage>>`.
    /// Lets two server instances (e.g. one for stdio MCP and one for SHODH
    /// REST) share the same SQLite connection and `sqlite-vec` virtual table
    /// in the same process, removing the v0.3 caveat that `--shodh-rest`
    /// disabled stdio MCP.
    pub fn from_shared_storage(
        storage: Arc<Mutex<Storage>>,
        embedder: Arc<dyn Embedder>,
        prediction_loss: Option<Arc<dyn PredictionLossBackend>>,
        weights: SurpriseWeights,
        ranking: RankingConfig,
        enable_memory_tool: bool,
    ) -> Self {
        Self {
            tool_router: Self::tool_router(),
            storage,
            embedder,
            prediction_loss,
            enable_memory_tool,
            weights,
            ranking,
            started_at: std::time::Instant::now(),
        }
    }

    pub fn weights(&self) -> SurpriseWeights {
        self.weights
    }

    pub fn ranking_config(&self) -> RankingConfig {
        self.ranking
    }

    pub fn has_prediction_loss_backend(&self) -> bool {
        self.prediction_loss.is_some()
    }

    pub fn uptime_seconds(&self) -> u64 {
        self.started_at.elapsed().as_secs()
    }

    /// **Tests / advanced use only.** Returns the underlying storage Arc for
    /// direct DB manipulation (e.g. backdating timestamps in evaluation
    /// scenarios). Production callers should go through MCP tools.
    pub fn storage_arc(&self) -> Arc<Mutex<Storage>> {
        self.storage.clone()
    }

    #[tool(
        name = "ping",
        description = "Health probe. Returns sqlite-vec version, memory count, and uptime."
    )]
    async fn ping(
        &self,
        Parameters(_): Parameters<PingParams>,
    ) -> std::result::Result<CallToolResult, ErrorData> {
        let store = self.storage.lock().await;
        let vec_version = store.vec_version().map_err(internal_err)?;
        let alive = store.count_alive().map_err(internal_err)?;
        let total = store.count_total().map_err(internal_err)?;
        json_result(&PingResult {
            status: "ok",
            backend: "sqlite_vec_hippo",
            vec_version,
            alive,
            total,
            uptime_seconds: self.started_at.elapsed().as_secs(),
            claude_hippo_version: crate::VERSION,
        })
    }

    #[tool(
        name = "hippo_remember",
        description = "Store a memory with semantic embedding and compute its surprise score. \
                       Surprise is high for outlier content, long/tagged content, and \
                       user-marked importance. Dedup by SHA256 content hash. \
                       SHODH-compatible alias: store_memory."
    )]
    async fn hippo_remember(
        &self,
        Parameters(p): Parameters<RememberParams>,
    ) -> std::result::Result<CallToolResult, ErrorData> {
        self.do_remember(p).await
    }

    #[tool(
        name = "store_memory",
        description = "SHODH-compatible alias for hippo_remember."
    )]
    async fn store_memory_alias(
        &self,
        Parameters(p): Parameters<RememberParams>,
    ) -> std::result::Result<CallToolResult, ErrorData> {
        self.do_remember(p).await
    }

    #[tool(
        name = "hippo_recall",
        description = "Semantic search over memories. Default ranking blends cosine \
                       similarity with surprise score and time-decay. Set no_surprise_boost=true \
                       for pure vector similarity. SHODH-compatible alias: retrieve_memory."
    )]
    async fn hippo_recall(
        &self,
        Parameters(p): Parameters<RecallParams>,
    ) -> std::result::Result<CallToolResult, ErrorData> {
        self.do_recall(p).await
    }

    #[tool(
        name = "retrieve_memory",
        description = "SHODH-compatible alias for hippo_recall."
    )]
    async fn retrieve_memory_alias(
        &self,
        Parameters(p): Parameters<RecallParams>,
    ) -> std::result::Result<CallToolResult, ErrorData> {
        self.do_recall(p).await
    }

    #[tool(
        name = "hippo_list_recent",
        description = "List the most recent N memories ordered by created_at DESC. \
                       SHODH-compatible alias: list_memories."
    )]
    async fn hippo_list_recent(
        &self,
        Parameters(p): Parameters<ListRecentParams>,
    ) -> std::result::Result<CallToolResult, ErrorData> {
        self.do_list_recent(p).await
    }

    #[tool(
        name = "list_memories",
        description = "SHODH-compatible alias for hippo_list_recent."
    )]
    async fn list_memories_alias(
        &self,
        Parameters(p): Parameters<ListRecentParams>,
    ) -> std::result::Result<CallToolResult, ErrorData> {
        self.do_list_recent(p).await
    }

    #[tool(
        name = "hippo_forget",
        description = "Soft-delete a memory by content_hash or id. The DB row is kept with \
                       deleted_at set, so retrieval ignores it but audit history is preserved. \
                       SHODH-compatible alias: delete_memory."
    )]
    async fn hippo_forget(
        &self,
        Parameters(p): Parameters<ForgetParams>,
    ) -> std::result::Result<CallToolResult, ErrorData> {
        self.do_forget(p).await
    }

    #[tool(
        name = "delete_memory",
        description = "SHODH-compatible alias for hippo_forget."
    )]
    async fn delete_memory_alias(
        &self,
        Parameters(p): Parameters<ForgetParams>,
    ) -> std::result::Result<CallToolResult, ErrorData> {
        self.do_forget(p).await
    }

    #[tool(
        name = "hippo_session_summary",
        description = "Summarize recent activity: counts by memory_type, top tags, highlights \
                       (highest-surprise memories), and mean surprise. Default lookback 24h."
    )]
    async fn hippo_session_summary(
        &self,
        Parameters(p): Parameters<SessionSummaryParams>,
    ) -> std::result::Result<CallToolResult, ErrorData> {
        self.do_session_summary(p).await
    }

    #[tool(
        name = "memory",
        description = "Anthropic Memory Tool compatibility surface (v0.3, opt-in via \
                       --anthropic-memory-tool). Filesystem-shaped operations (view / create / \
                       str_replace / insert / delete / rename) under /memories. Returns plain \
                       text matching Anthropic's documented response format. When the flag is \
                       off, returns an instructional error."
    )]
    async fn memory(
        &self,
        Parameters(p): Parameters<MemoryToolParams>,
    ) -> std::result::Result<CallToolResult, ErrorData> {
        if !self.enable_memory_tool {
            return Err(invalid_input(
                "memory tool not enabled. Restart `hippo serve --anthropic-memory-tool` to \
                 expose the Anthropic Memory Tool compatibility surface.",
            ));
        }
        let mut store = self.storage.lock().await;
        let reply = memory_tool::dispatch(&mut store, p);
        Ok(CallToolResult::success(vec![Content::text(reply.content)]))
    }
}

impl MemoryServer {
    /// Typed remember. Returns the `RememberResult` directly instead of the
    /// JSON-encoded MCP `CallToolResult`. Used by tests / eval harness; the
    /// MCP `do_remember` is a thin wrapper.
    pub async fn remember(
        &self,
        p: RememberParams,
    ) -> std::result::Result<RememberResult, ErrorData> {
        if p.content.trim().is_empty() {
            return Err(invalid_input("content is empty"));
        }
        // Compute embedding (lazy load on first call).
        let embedding = self.embedder.embed_one(&p.content).map_err(internal_err)?;

        // Compute surprise components.
        let history_emb = {
            let store = self.storage.lock().await;
            history_embeddings(&store, 50).map_err(internal_err)?
        };
        let outlier = surprise::embedding_outlier(&embedding, &history_emb);
        let engagement = surprise::engagement(&p.content, p.tags.len());
        let explicit = surprise::explicit(p.importance);
        // v0.3: if a prediction-loss backend is wired (--prediction-loss-backend
        // openai-compat), score the content's predictability and feed it into
        // the surprise score. Otherwise leave it None and the score formula
        // re-distributes w_prediction onto outlier + engagement (v0.2 fallback).
        let prediction_loss = if let Some(pl) = &self.prediction_loss {
            match pl.predict_loss(&p.content) {
                Ok(v) => Some(v.clamp(0.0, 1.0)),
                Err(e) => {
                    tracing::warn!(error = %e, "prediction_loss backend failed; falling back to None");
                    None
                }
            }
        } else {
            None
        };
        let comps = SurpriseComponents {
            embedding_outlier: outlier,
            engagement,
            explicit,
            prediction_loss,
        };
        let score = surprise::score(&comps, &self.weights);

        // Build row + attach surprise to metadata.
        let mut metadata = p.metadata.unwrap_or_else(|| serde_json::json!({}));
        storage::attach_surprise(&mut metadata, score, &comps);

        let row = storage::new_memory_row(
            p.content,
            p.tags,
            Some(p.memory_type.unwrap_or_else(|| "Observation".to_string())),
            metadata,
        );

        let mut store = self.storage.lock().await;
        let (id, dup) = store.insert(&row, Some(&embedding)).map_err(internal_err)?;

        Ok(RememberResult {
            success: true,
            id,
            content_hash: row.content_hash,
            duplicate: dup,
            surprise_score: score,
            surprise_components: comps,
        })
    }

    /// Typed recall using server-wide ranking config and per-call
    /// `RecallParams.oversample_factor` (if set).
    pub async fn recall(
        &self,
        p: RecallParams,
    ) -> std::result::Result<Vec<RecalledMemory>, ErrorData> {
        let factor = p
            .oversample_factor
            .unwrap_or(self.ranking.default_oversample_factor);
        self.recall_with_options(
            p,
            RecallOptions {
                oversample_factor: factor,
            },
        )
        .await
    }

    /// Typed recall with custom oversample factor. Used by the eval harness
    /// to ensure full corpus coverage before surprise rerank, and by the
    /// `recall` wrapper to apply the per-call `RecallParams.oversample_factor`.
    ///
    /// v0.5 Phase B: dispatches on `RecallParams.mode`:
    /// - `None` / `"semantic"`: v0.4 behavior.
    /// - `"associative"`: Hebbian neighbors of seed (top-1 semantic match
    ///   unless `seed_id` is set). `query` is still required for
    ///   seed-by-query; pass any non-empty placeholder when using
    ///   `seed_id` directly.
    /// - `"hybrid"`: union of semantic results and associative neighbors.
    ///
    /// When `RankingConfig.reinforce_co_recall` is true and the result
    /// set has ≥2 alive memories, all unordered pairs are reinforced.
    pub async fn recall_with_options(
        &self,
        p: RecallParams,
        opts: RecallOptions,
    ) -> std::result::Result<Vec<RecalledMemory>, ErrorData> {
        if p.query.trim().is_empty() {
            return Err(invalid_input("query is empty"));
        }
        let mode = parse_recall_mode(p.mode.as_deref());
        let k = p.limit.max(1);
        let factor = opts.oversample_factor.max(1);
        let now = unix_now();

        // 1. Compute semantic hits (always — even associative needs them
        //    to pick a seed when `seed_id` is unset).
        let query_emb = self.embedder.embed_one(&p.query).map_err(internal_err)?;
        let mut store = self.storage.lock().await;
        let fetch_k = if p.no_surprise_boost { k } else { k * factor };
        let semantic_hits = store.knn(&query_emb, fetch_k).map_err(internal_err)?;

        let mut results: Vec<RecalledMemory> = Vec::new();
        let mut seen_ids: std::collections::HashSet<i64> = std::collections::HashSet::new();

        let want_semantic = matches!(mode, RecallMode::Semantic | RecallMode::Hybrid);
        let want_associative = matches!(mode, RecallMode::Associative | RecallMode::Hybrid);

        if want_semantic {
            for (id, dist) in &semantic_hits {
                if !seen_ids.insert(*id) {
                    continue;
                }
                let mem = match store.get_by_id(*id).map_err(internal_err)? {
                    Some(m) => m,
                    None => continue,
                };
                let cos_sim = (1.0 - (*dist / 2.0)).clamp(0.0, 1.0);
                let surprise_score = storage::read_surprise(&mem.metadata);
                let age_days = ((now - mem.created_at).max(0.0) / 86400.0) as f32;
                let score = if p.no_surprise_boost || surprise_score.is_none() {
                    cos_sim
                } else {
                    surprise::ranking(
                        cos_sim,
                        surprise_score.unwrap_or(0.0),
                        age_days,
                        self.ranking.half_life_days,
                        self.ranking.decay_floor,
                    )
                };
                results.push(RecalledMemory {
                    memory: mem,
                    score,
                    cosine_similarity: cos_sim,
                    surprise_score,
                });
            }
        }

        // 2. Associative expansion. Seed = `seed_id` if provided, else the
        //    top-1 semantic id.
        if want_associative {
            let seed = p
                .seed_id
                .or_else(|| semantic_hits.first().map(|(id, _)| *id));
            if let Some(seed_id) = seed {
                // For pure associative mode, include the seed itself
                // first (it's literally what the caller asked for).
                if matches!(mode, RecallMode::Associative) && seen_ids.insert(seed_id) {
                    if let Some(mem) = store.get_by_id(seed_id).map_err(internal_err)? {
                        let cos_sim = semantic_hits
                            .iter()
                            .find(|(id, _)| *id == seed_id)
                            .map(|(_, d)| (1.0 - d / 2.0).clamp(0.0, 1.0))
                            .unwrap_or(1.0);
                        let surprise_score = storage::read_surprise(&mem.metadata);
                        results.push(RecalledMemory {
                            memory: mem,
                            score: 1.0,
                            cosine_similarity: cos_sim,
                            surprise_score,
                        });
                    }
                }
                let neighbors = store
                    .neighbors_by_id(seed_id, DEFAULT_ASSOCIATIVE_LIMIT)
                    .map_err(internal_err)?;
                for (nbr_id, weight, _last) in neighbors {
                    if !seen_ids.insert(nbr_id) {
                        continue;
                    }
                    let mem = match store.get_by_id(nbr_id).map_err(internal_err)? {
                        Some(m) => m,
                        None => continue,
                    };
                    let surprise_score = storage::read_surprise(&mem.metadata);
                    // Associative score = edge weight, clamped. For
                    // hybrid, this competes with semantic scores in [0,1]
                    // — edges saturate at 1.0 so the scales are aligned.
                    let score = weight.clamp(0.0, 1.0);
                    results.push(RecalledMemory {
                        memory: mem,
                        score,
                        cosine_similarity: 0.0,
                        surprise_score,
                    });
                }
            }
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);

        // 3. Hebbian reinforcement on the final set.
        if self.ranking.reinforce_co_recall && results.len() >= 2 {
            let ids: Vec<i64> = results.iter().filter_map(|r| r.memory.id).collect();
            // Best-effort — a reinforcement failure shouldn't break recall.
            if let Err(e) = store.reinforce_co_recalled(&ids, self.ranking.co_recall_alpha) {
                tracing::warn!(error = %e, "co-recall reinforcement failed");
            }
        }

        Ok(results)
    }

    pub async fn do_remember(
        &self,
        p: RememberParams,
    ) -> std::result::Result<CallToolResult, ErrorData> {
        let r = self.remember(p).await?;
        json_result(&r)
    }

    pub async fn do_recall(
        &self,
        p: RecallParams,
    ) -> std::result::Result<CallToolResult, ErrorData> {
        let r = self.recall(p).await?;
        json_result(&r)
    }

    pub async fn do_list_recent(
        &self,
        p: ListRecentParams,
    ) -> std::result::Result<CallToolResult, ErrorData> {
        let store = self.storage.lock().await;
        let memories = store.list_recent(p.n.max(1)).map_err(internal_err)?;
        let count = memories.len();
        json_result(&ListRecentResult { memories, count })
    }

    pub async fn do_forget(
        &self,
        p: ForgetParams,
    ) -> std::result::Result<CallToolResult, ErrorData> {
        if p.content_hash.is_none() && p.id.is_none() {
            return Err(invalid_input("hippo_forget requires content_hash or id"));
        }
        if p.dry_run {
            // 存在確認のみ
            let store = self.storage.lock().await;
            let exists = if let Some(ref h) = p.content_hash {
                store.get_by_hash(h).map_err(internal_err)?.is_some()
            } else if let Some(id) = p.id {
                store.get_by_id(id).map_err(internal_err)?.is_some()
            } else {
                false
            };
            return json_result(&ForgetResult {
                success: true,
                deleted: if exists { 1 } else { 0 },
                dry_run: true,
            });
        }
        let mut store = self.storage.lock().await;
        let n = if let Some(h) = p.content_hash {
            store.soft_delete_by_hash(&h).map_err(internal_err)?
        } else if let Some(id) = p.id {
            store.soft_delete_by_id(id).map_err(internal_err)?
        } else {
            0
        };
        json_result(&ForgetResult {
            success: true,
            deleted: n,
            dry_run: false,
        })
    }

    pub async fn do_session_summary(
        &self,
        p: SessionSummaryParams,
    ) -> std::result::Result<CallToolResult, ErrorData> {
        let hours = p.hours.unwrap_or(24).max(1);
        let cutoff = unix_now() - (hours as f64) * 3600.0;
        let store = self.storage.lock().await;

        // 直近 200 件まで取って window で filter (cheap で実用十分)。
        let recent = store.list_recent(500).map_err(internal_err)?;
        let in_window: Vec<MemoryRow> = recent
            .into_iter()
            .filter(|m| m.created_at >= cutoff)
            .collect();

        let total_memories = in_window.len();

        // by_type
        let mut by_type_map: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for m in &in_window {
            let t = m.memory_type.clone().unwrap_or_else(|| "(none)".into());
            *by_type_map.entry(t).or_insert(0) += 1;
        }
        let mut by_type: Vec<(String, usize)> = by_type_map.into_iter().collect();
        by_type.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

        // top_tags
        let mut tag_map: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for m in &in_window {
            for t in &m.tags {
                *tag_map.entry(t.clone()).or_insert(0) += 1;
            }
        }
        let mut top_tags: Vec<(String, usize)> = tag_map.into_iter().collect();
        top_tags.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        top_tags.truncate(10);

        // highlights = surprise score top 5
        let mut scored: Vec<(f32, MemoryRow)> = in_window
            .into_iter()
            .map(|m| (storage::read_surprise(&m.metadata).unwrap_or(0.0), m))
            .collect();
        let mean_surprise = if scored.is_empty() {
            0.0
        } else {
            scored.iter().map(|(s, _)| *s).sum::<f32>() / scored.len() as f32
        };
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let highlights: Vec<MemoryRow> = scored.into_iter().take(5).map(|(_, m)| m).collect();

        json_result(&SessionSummary {
            window_hours: hours,
            total_memories,
            by_type,
            top_tags,
            highlights,
            mean_surprise,
        })
    }
}

#[tool_handler]
impl ServerHandler for MemoryServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.server_info = Implementation::new(env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.instructions = Some(
            "Claude Code に海馬を足す surprise-aware memory MCP. \
             5 hippo_* tools + SHODH-compatible aliases (store_memory, retrieve_memory, \
             list_memories, delete_memory) + ping. Storage is schema-compatible with \
             mcp-memory-service (SHODH spec). Surprise scoring is on by default in recall."
                .into(),
        );
        info
    }
}

// ---------- helpers ----------

fn json_result<T: Serialize>(payload: &T) -> std::result::Result<CallToolResult, ErrorData> {
    let json = serde_json::to_string(payload).map_err(|e| internal_err(e.to_string()))?;
    Ok(CallToolResult::success(vec![Content::text(json)]))
}

fn internal_err(e: impl std::fmt::Display) -> ErrorData {
    ErrorData::internal_error(e.to_string(), None)
}

fn invalid_input(msg: &str) -> ErrorData {
    ErrorData::invalid_params(msg.to_string(), None)
}

fn unix_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// 既存の最新 N 件 memory の embedding を取り出す。surprise の outlier 判定に使う。
fn history_embeddings(store: &Storage, n: i64) -> crate::Result<Vec<Vec<f32>>> {
    use rusqlite::types::Value;
    use zerocopy::FromBytes;
    let mut stmt = store.conn().prepare(
        "SELECT memory_embeddings.content_embedding
         FROM memories JOIN memory_embeddings ON memories.id = memory_embeddings.rowid
         WHERE memories.deleted_at IS NULL
         ORDER BY memories.created_at DESC
         LIMIT ?1",
    )?;
    let rows = stmt.query_map(rusqlite::params![n], |r| r.get::<_, Value>(0))?;
    let mut out = Vec::new();
    for row in rows {
        let v = row?;
        if let Value::Blob(bytes) = v {
            // FLOAT[384] = 384 * 4 bytes
            if bytes.len() != crate::EMBEDDING_DIM * 4 {
                continue;
            }
            let floats: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::read_from(c).unwrap_or(0.0))
                .collect();
            out.push(floats);
        }
    }
    Ok(out)
}

/// MCP server を stdio で起動する (run loop を await)。
pub async fn run_stdio(storage: Storage, embedder: Arc<dyn Embedder>) -> anyhow::Result<()> {
    run_stdio_with_config(
        storage,
        embedder,
        SurpriseWeights::default(),
        RankingConfig::default(),
    )
    .await
}

pub async fn run_stdio_with_weights(
    storage: Storage,
    embedder: Arc<dyn Embedder>,
    weights: SurpriseWeights,
) -> anyhow::Result<()> {
    run_stdio_with_config(storage, embedder, weights, RankingConfig::default()).await
}

pub async fn run_stdio_with_config(
    storage: Storage,
    embedder: Arc<dyn Embedder>,
    weights: SurpriseWeights,
    ranking: RankingConfig,
) -> anyhow::Result<()> {
    run_stdio_full(storage, embedder, None, weights, ranking).await
}

pub async fn run_stdio_full(
    storage: Storage,
    embedder: Arc<dyn Embedder>,
    prediction_loss: Option<Arc<dyn PredictionLossBackend>>,
    weights: SurpriseWeights,
    ranking: RankingConfig,
) -> anyhow::Result<()> {
    run_stdio_full_with_memory_tool(storage, embedder, prediction_loss, weights, ranking, false)
        .await
}

pub async fn run_stdio_full_with_memory_tool(
    storage: Storage,
    embedder: Arc<dyn Embedder>,
    prediction_loss: Option<Arc<dyn PredictionLossBackend>>,
    weights: SurpriseWeights,
    ranking: RankingConfig,
    enable_memory_tool: bool,
) -> anyhow::Result<()> {
    let server = MemoryServer::new_full_with_memory_tool(
        storage,
        embedder,
        prediction_loss,
        weights,
        ranking,
        enable_memory_tool,
    );
    let service = server
        .serve(stdio())
        .await
        .map_err(|e| anyhow::anyhow!("rmcp serve init failed: {e}"))?;
    service.waiting().await.ok();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::MockEmbedder;

    fn make_server() -> MemoryServer {
        crate::storage::register_sqlite_vec();
        let store = Storage::open_in_memory().unwrap();
        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new());
        MemoryServer::new(store, embedder)
    }

    #[tokio::test]
    async fn ping_works() {
        let s = make_server();
        let r = s.ping(Parameters(PingParams::default())).await.unwrap();
        assert!(!r.is_error.unwrap_or(false));
    }

    #[tokio::test]
    async fn remember_and_recall() {
        let s = make_server();
        // store 3 memories
        for (i, content) in ["alpha note", "bravo decision", "charlie discovery"]
            .iter()
            .enumerate()
        {
            let p = RememberParams {
                content: content.to_string(),
                tags: vec!["bench".into(), format!("i{i}")],
                memory_type: Some("Observation".into()),
                importance: Some(0.5),
                metadata: None,
            };
            let _ = s.do_remember(p).await.unwrap();
        }
        // recall query similar to "alpha"
        let r = s
            .do_recall(RecallParams {
                query: "alpha".into(),
                limit: 3,
                no_surprise_boost: false,
                oversample_factor: None,
                mode: None,
                seed_id: None,
            })
            .await
            .unwrap();
        assert!(!r.is_error.unwrap_or(false));
    }

    #[tokio::test]
    async fn forget_dry_run_does_not_delete() {
        let s = make_server();
        let p = RememberParams {
            content: "to forget".into(),
            tags: vec![],
            memory_type: None,
            importance: None,
            metadata: None,
        };
        s.do_remember(p).await.unwrap();
        // dry_run forget by hash
        let hash = crate::storage::content_hash("to forget");
        let r = s
            .do_forget(ForgetParams {
                content_hash: Some(hash.clone()),
                id: None,
                dry_run: true,
            })
            .await
            .unwrap();
        assert!(!r.is_error.unwrap_or(false));
        // 確実に削除されていない
        let store = s.storage.lock().await;
        assert_eq!(store.count_alive().unwrap(), 1);
    }

    #[tokio::test]
    async fn forget_actually_deletes() {
        let s = make_server();
        s.do_remember(RememberParams {
            content: "delete me".into(),
            tags: vec![],
            memory_type: None,
            importance: None,
            metadata: None,
        })
        .await
        .unwrap();
        let hash = crate::storage::content_hash("delete me");
        s.do_forget(ForgetParams {
            content_hash: Some(hash),
            id: None,
            dry_run: false,
        })
        .await
        .unwrap();
        let store = s.storage.lock().await;
        assert_eq!(store.count_alive().unwrap(), 0);
        assert_eq!(store.count_total().unwrap(), 1);
    }

    #[tokio::test]
    async fn session_summary_groups_by_type_and_tags() {
        let s = make_server();
        for (content, mt) in &[
            ("note 1", "Observation"),
            ("note 2", "Observation"),
            ("dec 1", "Decision"),
        ] {
            s.do_remember(RememberParams {
                content: content.to_string(),
                tags: vec!["proj-x".into()],
                memory_type: Some(mt.to_string()),
                importance: Some(0.5),
                metadata: None,
            })
            .await
            .unwrap();
        }
        let r = s
            .do_session_summary(SessionSummaryParams { hours: Some(24) })
            .await
            .unwrap();
        assert!(!r.is_error.unwrap_or(false));
    }

    // ---- v0.5 Phase B: Hebbian recall tests ------------------------------

    #[test]
    fn parse_recall_mode_handles_aliases() {
        assert_eq!(parse_recall_mode(None), RecallMode::Semantic);
        assert_eq!(parse_recall_mode(Some("")), RecallMode::Semantic);
        assert_eq!(parse_recall_mode(Some("semantic")), RecallMode::Semantic);
        assert_eq!(
            parse_recall_mode(Some("Associative")),
            RecallMode::Associative
        );
        assert_eq!(parse_recall_mode(Some("hebbian")), RecallMode::Associative);
        assert_eq!(parse_recall_mode(Some("hybrid")), RecallMode::Hybrid);
        assert_eq!(parse_recall_mode(Some("MIXED")), RecallMode::Hybrid);
        // Unknown falls back to Semantic.
        assert_eq!(parse_recall_mode(Some("nonsense")), RecallMode::Semantic);
    }

    /// Co-recall reinforcement creates one edge per pair in the result set.
    #[tokio::test]
    async fn semantic_recall_reinforces_co_recalled_pairs() {
        let s = make_server();
        for content in ["alpha note", "alpha story", "alpha tale"] {
            s.do_remember(RememberParams {
                content: content.into(),
                tags: vec![],
                memory_type: None,
                importance: Some(0.5),
                metadata: None,
            })
            .await
            .unwrap();
        }
        // Pure-cosine recall returns the 3 alpha-similar items.
        let _ = s
            .recall(RecallParams {
                query: "alpha".into(),
                limit: 3,
                no_surprise_boost: true,
                oversample_factor: None,
                mode: None,
                seed_id: None,
            })
            .await
            .unwrap();
        let store = s.storage.lock().await;
        // 3 results → C(3,2) = 3 edges
        assert_eq!(store.count_associations().unwrap(), 3);
    }

    /// `reinforce_co_recall = false` short-circuits the write side.
    #[tokio::test]
    async fn disabled_reinforcement_writes_no_edges() {
        crate::storage::register_sqlite_vec();
        let store = Storage::open_in_memory().unwrap();
        let embedder: Arc<dyn Embedder> = Arc::new(MockEmbedder::new());
        let s = MemoryServer::new_with_config(
            store,
            embedder,
            SurpriseWeights::default(),
            RankingConfig {
                reinforce_co_recall: false,
                ..Default::default()
            },
        );
        for content in ["alpha note", "alpha story"] {
            s.do_remember(RememberParams {
                content: content.into(),
                tags: vec![],
                memory_type: None,
                importance: Some(0.5),
                metadata: None,
            })
            .await
            .unwrap();
        }
        let _ = s
            .recall(RecallParams {
                query: "alpha".into(),
                limit: 5,
                no_surprise_boost: true,
                oversample_factor: None,
                mode: None,
                seed_id: None,
            })
            .await
            .unwrap();
        let store = s.storage.lock().await;
        assert_eq!(store.count_associations().unwrap(), 0);
    }

    /// Associative mode returns the seed and its neighbors, ordered by
    /// edge weight.
    #[tokio::test]
    async fn associative_mode_returns_neighbors() {
        let s = make_server();
        let mut ids = Vec::new();
        for content in [
            "seed alpha",
            "neighbor1 alpha",
            "neighbor2 alpha",
            "outsider beta",
        ] {
            let r = s
                .remember(RememberParams {
                    content: content.into(),
                    tags: vec![],
                    memory_type: None,
                    importance: Some(0.5),
                    metadata: None,
                })
                .await
                .unwrap();
            ids.push(r.id);
        }
        // Manually reinforce edges between the first 3 (alpha-cluster).
        // Skip ids[3] so it should NOT show up in associative recall from
        // ids[0].
        {
            let mut store = s.storage.lock().await;
            store.reinforce_co_recalled(&ids[..3], 0.5).unwrap();
        }
        let r = s
            .recall(RecallParams {
                query: "seed alpha".into(),
                limit: 10,
                no_surprise_boost: true,
                oversample_factor: None,
                mode: Some("associative".into()),
                seed_id: Some(ids[0]),
            })
            .await
            .unwrap();
        let returned_ids: Vec<i64> = r.iter().filter_map(|m| m.memory.id).collect();
        // Seed + 2 neighbors, NOT the outsider.
        assert!(returned_ids.contains(&ids[0]));
        assert!(returned_ids.contains(&ids[1]));
        assert!(returned_ids.contains(&ids[2]));
        assert!(!returned_ids.contains(&ids[3]));
    }

    /// Hybrid merges semantic + associative without duplicating any id.
    #[tokio::test]
    async fn hybrid_mode_dedupes_results() {
        let s = make_server();
        let mut ids = Vec::new();
        for content in ["alpha one", "alpha two", "beta one"] {
            let r = s
                .remember(RememberParams {
                    content: content.into(),
                    tags: vec![],
                    memory_type: None,
                    importance: Some(0.5),
                    metadata: None,
                })
                .await
                .unwrap();
            ids.push(r.id);
        }
        // Pre-seed an edge between alpha-one and alpha-two.
        {
            let mut store = s.storage.lock().await;
            store.reinforce_co_recalled(&[ids[0], ids[1]], 0.8).unwrap();
        }
        let r = s
            .recall(RecallParams {
                query: "alpha".into(),
                limit: 10,
                no_surprise_boost: true,
                oversample_factor: None,
                mode: Some("hybrid".into()),
                seed_id: None,
            })
            .await
            .unwrap();
        let returned_ids: Vec<i64> = r.iter().filter_map(|m| m.memory.id).collect();
        // No id appears more than once.
        let mut seen = std::collections::HashSet::new();
        for id in &returned_ids {
            assert!(seen.insert(*id), "id {id} appeared twice in hybrid result");
        }
    }
}
