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
}

impl Default for RankingConfig {
    fn default() -> Self {
        Self {
            half_life_days: DEFAULT_HALF_LIFE_DAYS,
            decay_floor: DEFAULT_DECAY_FLOOR,
            default_oversample_factor: DEFAULT_OVERSAMPLE_FACTOR,
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
}

#[derive(Debug, Serialize)]
pub struct RecalledMemory {
    pub memory: MemoryRow,
    /// 0.0..=1.0, higher is better. Combines cosine sim + surprise * decay.
    pub score: f32,
    pub cosine_similarity: f32,
    pub surprise_score: Option<f32>,
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
        Self {
            tool_router: Self::tool_router(),
            storage: Arc::new(Mutex::new(storage)),
            embedder,
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
        let comps = SurpriseComponents {
            embedding_outlier: outlier,
            engagement,
            explicit,
            prediction_loss: None, // abyo-llm-probe 統合 (v0.3) で埋まる
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
    pub async fn recall_with_options(
        &self,
        p: RecallParams,
        opts: RecallOptions,
    ) -> std::result::Result<Vec<RecalledMemory>, ErrorData> {
        if p.query.trim().is_empty() {
            return Err(invalid_input("query is empty"));
        }
        let k = p.limit.max(1);
        let factor = opts.oversample_factor.max(1);
        let query_emb = self.embedder.embed_one(&p.query).map_err(internal_err)?;

        let store = self.storage.lock().await;
        // KNN over-fetch when surprise boost (rerank では hit が落ちないように)
        let fetch_k = if p.no_surprise_boost { k } else { k * factor };
        let hits = store.knn(&query_emb, fetch_k).map_err(internal_err)?;

        let mut results: Vec<RecalledMemory> = Vec::with_capacity(hits.len());
        let now = unix_now();
        for (id, dist) in hits {
            let mem = match store.get_by_id(id).map_err(internal_err)? {
                Some(m) => m,
                None => continue,
            };
            // distance ∈ [0,2] for cosine; sim = 1 - distance/2 ∈ [0,1]
            let cos_sim = (1.0 - (dist / 2.0)).clamp(0.0, 1.0);
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
        // 並べ替え (rerank)
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);
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
    let server = MemoryServer::new_with_config(storage, embedder, weights, ranking);
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
}
