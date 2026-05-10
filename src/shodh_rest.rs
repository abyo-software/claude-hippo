//! SHODH OpenAPI v1.0.0 compatibility — REST server (v0.3, opt-in).
//!
//! Spins up an axum HTTP server alongside the MCP stdio one when
//! `hippo serve --shodh-rest --shodh-rest-bind 127.0.0.1:8765` is set.
//! Logs go to stderr so the MCP stdio protocol on stdout/stdin is
//! unaffected — both can coexist in the same process.
//!
//! # Endpoint coverage in v0.3
//!
//! | Method | Path                    | Backed by                    |
//! |--------|-------------------------|------------------------------|
//! | GET    | `/api/health`           | `MemoryServer::ping`         |
//! | POST   | `/api/remember`         | `MemoryServer::remember`     |
//! | POST   | `/api/recall`           | `MemoryServer::recall`       |
//! | GET    | `/api/memories`         | `MemoryServer::list_recent`  |
//! | DELETE | `/api/forget/:id`       | `MemoryServer::forget(id)`   |
//! | GET    | `/api/stats`            | `MemoryServer::session_summary` |
//!
//! 7 SHODH endpoints remain unimplemented (consolidate, recall/by-tags,
//! context auto-ingest, PATCH metadata, forget/by-tags, list tags,
//! per-id GET). They return 501 Not Implemented with an explanatory body
//! pointing at the canonical MCP tools so users have a clear path forward
//! rather than a silent partial. v0.4 will close the gap.
//!
//! # Why a separate server (vs reusing rmcp's HTTP transport)
//!
//! rmcp's HTTP transport speaks the MCP wire protocol over HTTP, not REST.
//! SHODH clients expect REST verbs + JSON bodies matching its OpenAPI
//! spec, so we ship a thin axum facade that translates REST → typed
//! `MemoryServer` calls. This keeps the MCP and REST surfaces independent
//! and lets users adopt either without forcing the other.

use crate::server::{
    ForgetParams, MemoryServer, RecallParams, RememberParams, SessionSummaryParams,
};
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;

/// Build the REST router. Exposed for testing.
pub fn router(server: Arc<MemoryServer>) -> Router {
    Router::new()
        .route("/api/health", get(handle_health))
        .route("/api/remember", post(handle_remember))
        .route("/api/recall", post(handle_recall))
        .route("/api/recall/by-tags", post(handle_recall_by_tags))
        .route("/api/context", post(handle_context))
        .route("/api/memories", get(handle_list))
        .route(
            "/api/memories/{id}",
            get(handle_get_by_id).patch(handle_patch_by_id),
        )
        .route("/api/forget/{id}", delete(handle_forget))
        .route("/api/forget/by-tags", post(handle_forget_by_tags))
        .route("/api/tags", get(handle_tags))
        .route("/api/stats", get(handle_stats))
        .route("/api/consolidate", post(handle_consolidate))
        .with_state(server)
}

/// Bind and serve the REST API. Returns when the listener errors or the
/// caller drops the future. Coexists with MCP stdio in the same tokio
/// runtime via `tokio::join!` in the CLI.
pub async fn serve(server: Arc<MemoryServer>, bind: SocketAddr) -> anyhow::Result<()> {
    let app = router(server);
    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .map_err(|e| anyhow::anyhow!("SHODH REST: bind {bind} failed: {e}"))?;
    tracing::info!(?bind, "SHODH REST server listening");
    axum::serve(listener, app)
        .await
        .map_err(|e| anyhow::anyhow!("SHODH REST: serve loop failed: {e}"))?;
    Ok(())
}

// ---------- handlers ----------

#[derive(Serialize)]
struct HealthReply {
    status: &'static str,
    backend: &'static str,
    vec_version: String,
    alive: i64,
    total: i64,
    uptime_seconds: u64,
    claude_hippo_version: &'static str,
}

async fn handle_health(State(server): State<Arc<MemoryServer>>) -> impl IntoResponse {
    let storage = server.storage_arc();
    let store = storage.lock().await;
    let vec_version = store.vec_version().unwrap_or_else(|_| "unknown".into());
    let alive = store.count_alive().unwrap_or(0);
    let total = store.count_total().unwrap_or(0);
    Json(HealthReply {
        status: "ok",
        backend: "sqlite_vec_hippo",
        vec_version,
        alive,
        total,
        uptime_seconds: server.uptime_seconds(),
        claude_hippo_version: crate::VERSION,
    })
}

async fn handle_remember(
    State(server): State<Arc<MemoryServer>>,
    Json(p): Json<RememberParams>,
) -> impl IntoResponse {
    match server.remember(p).await {
        Ok(r) => (StatusCode::OK, Json(serde_json::to_value(&r).unwrap())).into_response(),
        Err(e) => mcp_error_to_http(e),
    }
}

async fn handle_recall(
    State(server): State<Arc<MemoryServer>>,
    Json(p): Json<RecallParams>,
) -> impl IntoResponse {
    match server.recall(p).await {
        Ok(rs) => (StatusCode::OK, Json(serde_json::to_value(&rs).unwrap())).into_response(),
        Err(e) => mcp_error_to_http(e),
    }
}

#[derive(Deserialize)]
struct ListQuery {
    #[serde(default)]
    n: Option<i64>,
}

async fn handle_list(
    State(server): State<Arc<MemoryServer>>,
    Query(q): Query<ListQuery>,
) -> impl IntoResponse {
    let storage = server.storage_arc();
    let store = storage.lock().await;
    match store.list_recent(q.n.unwrap_or(20).max(1)) {
        Ok(rows) => (
            StatusCode::OK,
            Json(serde_json::json!({"memories": rows, "count": rows.len()})),
        )
            .into_response(),
        Err(e) => storage_error_to_http(e),
    }
}

async fn handle_forget(
    State(server): State<Arc<MemoryServer>>,
    Path(id): Path<i64>,
) -> impl IntoResponse {
    let p = ForgetParams {
        content_hash: None,
        id: Some(id),
        dry_run: false,
    };
    let storage = server.storage_arc();
    let mut store = storage.lock().await;
    match store.soft_delete_by_id(p.id.unwrap()) {
        Ok(n) => (
            StatusCode::OK,
            Json(serde_json::json!({"success": true, "deleted": n, "id": id})),
        )
            .into_response(),
        Err(e) => storage_error_to_http(e),
    }
}

async fn handle_stats(
    State(server): State<Arc<MemoryServer>>,
    Query(q): Query<SessionSummaryQuery>,
) -> impl IntoResponse {
    let p = SessionSummaryParams { hours: q.hours };
    match server.do_session_summary(p).await {
        Ok(call) => {
            let txt = call_result_to_text(call).unwrap_or_else(|| "{}".into());
            let v: serde_json::Value = serde_json::from_str(&txt).unwrap_or(serde_json::json!({}));
            (StatusCode::OK, Json(v)).into_response()
        }
        Err(e) => mcp_error_to_http(e),
    }
}

#[derive(Deserialize, Default)]
struct SessionSummaryQuery {
    #[serde(default)]
    hours: Option<u32>,
}

// ---------- v0.4 Phase B-2: tag / by-id / context endpoints ----------

#[derive(Deserialize)]
struct TagSearch {
    tags: Vec<String>,
    /// `"any"` (default) or `"all"`. Maps to `Storage::TagMatch`.
    #[serde(default)]
    r#match: Option<String>,
    #[serde(default)]
    memory_type: Option<String>,
    #[serde(default)]
    limit: Option<i64>,
    /// `forget/by-tags` only.
    #[serde(default)]
    dry_run: bool,
}

fn parse_tag_match(s: Option<&str>) -> crate::storage::TagMatch {
    crate::storage::TagMatch::parse(s.unwrap_or("any"))
}

async fn handle_recall_by_tags(
    State(server): State<Arc<MemoryServer>>,
    Json(p): Json<TagSearch>,
) -> impl IntoResponse {
    let storage = server.storage_arc();
    let store = storage.lock().await;
    let mode = parse_tag_match(p.r#match.as_deref());
    let limit = p.limit.unwrap_or(20).max(0);
    match store.search_by_tag(&p.tags, mode, p.memory_type.as_deref(), limit) {
        Ok(rows) => (
            StatusCode::OK,
            Json(serde_json::json!({"memories": rows, "count": rows.len()})),
        )
            .into_response(),
        Err(e) => storage_error_to_http(e),
    }
}

async fn handle_forget_by_tags(
    State(server): State<Arc<MemoryServer>>,
    Json(p): Json<TagSearch>,
) -> impl IntoResponse {
    let storage = server.storage_arc();
    let mut store = storage.lock().await;
    let mode = parse_tag_match(p.r#match.as_deref());
    let limit = p.limit.unwrap_or(1000).max(0);
    let rows = match store.search_by_tag(&p.tags, mode, p.memory_type.as_deref(), limit) {
        Ok(r) => r,
        Err(e) => return storage_error_to_http(e),
    };
    let mut deleted = Vec::with_capacity(rows.len());
    for r in &rows {
        if let Some(id) = r.id {
            deleted.push(id);
        }
    }
    if !p.dry_run {
        for id in &deleted {
            let _ = store.soft_delete_by_id(*id);
        }
    }
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "matched": deleted.len(),
            "deleted_ids": deleted,
            "dry_run": p.dry_run,
        })),
    )
        .into_response()
}

async fn handle_get_by_id(
    State(server): State<Arc<MemoryServer>>,
    Path(id): Path<i64>,
) -> impl IntoResponse {
    let storage = server.storage_arc();
    let store = storage.lock().await;
    match store.get_by_id(id) {
        Ok(Some(row)) => {
            (StatusCode::OK, Json(serde_json::to_value(&row).unwrap())).into_response()
        }
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "not_found", "id": id})),
        )
            .into_response(),
        Err(e) => storage_error_to_http(e),
    }
}

#[derive(Deserialize, Default)]
struct PatchRequest {
    /// Replaces the whole `metadata` JSON. Server reserves the `_hippo`
    /// namespace; if the caller's metadata omits it, the existing
    /// `_hippo` block is preserved (so surprise score round-trips).
    #[serde(default)]
    metadata: Option<serde_json::Value>,
    /// Replaces the tag list. Empty array clears tags.
    #[serde(default)]
    tags: Option<Vec<String>>,
    /// Replaces memory_type. Pass JSON `null` to leave unchanged (Option
    /// semantics) — JSON omission also leaves unchanged.
    #[serde(default)]
    memory_type: Option<String>,
}

async fn handle_patch_by_id(
    State(server): State<Arc<MemoryServer>>,
    Path(id): Path<i64>,
    Json(p): Json<PatchRequest>,
) -> impl IntoResponse {
    let storage = server.storage_arc();
    let mut store = storage.lock().await;
    let existing = match store.get_by_id(id) {
        Ok(Some(r)) => r,
        Ok(None) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": "not_found", "id": id})),
            )
                .into_response();
        }
        Err(e) => return storage_error_to_http(e),
    };
    // Preserve _hippo namespace when caller replaces metadata.
    let mut new_metadata = p.metadata.unwrap_or_else(|| existing.metadata.clone());
    if !new_metadata.is_object() {
        new_metadata = serde_json::Value::Object(Default::default());
    }
    if let Some(map) = new_metadata.as_object_mut() {
        if !map.contains_key("_hippo") {
            if let Some(hippo) = existing.metadata.get("_hippo") {
                map.insert("_hippo".into(), hippo.clone());
            }
        }
    }
    let n = match store.update_metadata_by_id(
        id,
        &new_metadata,
        p.tags.as_deref(),
        p.memory_type.as_deref().map(Some),
    ) {
        Ok(n) => n,
        Err(e) => return storage_error_to_http(e),
    };
    (
        StatusCode::OK,
        Json(serde_json::json!({"updated": n, "id": id})),
    )
        .into_response()
}

async fn handle_tags(State(server): State<Arc<MemoryServer>>) -> impl IntoResponse {
    let storage = server.storage_arc();
    let store = storage.lock().await;
    match store.list_tags() {
        Ok(tags) => {
            let v: Vec<serde_json::Value> = tags
                .into_iter()
                .map(|(t, c)| serde_json::json!({"tag": t, "count": c}))
                .collect();
            (StatusCode::OK, Json(serde_json::json!({"tags": v}))).into_response()
        }
        Err(e) => storage_error_to_http(e),
    }
}

#[derive(Deserialize)]
struct ContextRequest {
    /// The current chat / query text. Used both as a recall query and
    /// (if `auto_ingest` is true) as a new memory to store.
    query: String,
    /// Top-N relevant memories to return. Default 5.
    #[serde(default)]
    limit: Option<usize>,
    /// Store `query` as a new `Observation`-typed memory before recall,
    /// so the next call sees it. Default false.
    #[serde(default)]
    auto_ingest: bool,
}

async fn handle_context(
    State(server): State<Arc<MemoryServer>>,
    Json(p): Json<ContextRequest>,
) -> impl IntoResponse {
    if p.auto_ingest {
        let _ = server
            .remember(crate::server::RememberParams {
                content: p.query.clone(),
                tags: vec!["context-auto-ingest".into()],
                memory_type: Some("Context".into()),
                importance: None,
                metadata: None,
            })
            .await;
        // Ignore errors here so a failed ingest doesn't poison the recall —
        // SHODH's spec calls this best-effort proactive surfacing.
    }
    match server
        .recall(crate::server::RecallParams {
            query: p.query,
            limit: p.limit.unwrap_or(5),
            no_surprise_boost: false,
            oversample_factor: None,
            mode: None,
            seed_id: None,
        })
        .await
    {
        Ok(rs) => (
            StatusCode::OK,
            Json(serde_json::json!({"context": rs, "auto_ingest": p.auto_ingest})),
        )
            .into_response(),
        Err(e) => mcp_error_to_http(e),
    }
}

// ---------- consolidate (v0.4 Phase B-1, extended in v0.5 Phase B) ----------
//
// SHODH `/api/consolidate` is documented to do four things:
//
// 1. **exponential decay scoring** — recompute time-decayed surprise per memory
//    (v0.4)
// 2. **quality-based archival of low-value memories** — soft-delete items
//    whose decayed surprise has fallen below a threshold and are old enough
//    not to be "fresh chat noise the user might still touch" (v0.4)
// 3. **association discovery between memories** — Hebbian edges (v0.5: edges
//    are *grown* on the read path via co-recall reinforcement; consolidate
//    decays + prunes them to keep the graph sparse)
// 4. **semantic clustering and compression** — still deferred (Phase C)

#[derive(Deserialize, Default)]
struct ConsolidateRequest {
    /// Soft-delete items whose `surprise * decay(age, half_life)` falls
    /// below this. Default 0.05.
    #[serde(default)]
    archive_threshold: Option<f32>,
    /// Don't touch memories younger than this. Default 30 days — protects
    /// fresh material from being wiped before it's had a chance to be
    /// referenced. (`0.0` disables this guard, useful for tests.)
    #[serde(default)]
    grace_period_days: Option<f32>,
    /// Cap on how many memories to scan per call. Default unlimited
    /// (passes `i64::MAX` to `list_recent`).
    #[serde(default)]
    limit: Option<i64>,
    /// When true, don't actually soft-delete — just report what would have
    /// been archived. Edge prune still runs in dry-run, so caller can see
    /// the steady-state edge count.
    #[serde(default)]
    dry_run: bool,
    /// v0.5: drop edges whose decayed weight falls below this. Default
    /// 0.01 (matches SHODH spec's "weak association" cutoff). Set to 0
    /// to keep all edges (the dangling-edge sweep against soft-deleted
    /// memories still runs).
    #[serde(default)]
    edge_prune_threshold: Option<f32>,
}

#[derive(Serialize)]
struct ConsolidateReply {
    /// Memories selected for archival.
    archived: Vec<ConsolidateArchived>,
    /// Total alive count BEFORE this call. Useful for clients that want
    /// to track corpus shrinkage over time.
    total_alive_before: i64,
    /// Total alive count AFTER (= before - archived.len() if not dry_run).
    total_alive_after: i64,
    /// True iff caller passed `dry_run = true`.
    dry_run: bool,
    /// v0.5: number of `memory_associations` rows removed (decay below
    /// threshold OR pointing to a soft-deleted memory).
    pruned_edges: i64,
    /// v0.5: total alive edges after this call's prune.
    associations_total: i64,
    /// Honest disclosure: SHODH spec lists association discovery and
    /// semantic clustering under "consolidate". v0.5 wires association
    /// discovery (Hebbian edges grown on the read path, decayed/pruned
    /// here). Semantic clustering is still deferred.
    deferred: Vec<&'static str>,
}

#[derive(Serialize)]
struct ConsolidateArchived {
    id: i64,
    content_hash: String,
    decayed_surprise: f32,
    age_days: f32,
}

async fn handle_consolidate(
    State(server): State<Arc<MemoryServer>>,
    Json(req): Json<ConsolidateRequest>,
) -> impl IntoResponse {
    let archive_threshold = req.archive_threshold.unwrap_or(0.05).clamp(0.0, 1.0);
    let grace_days = req.grace_period_days.unwrap_or(30.0).max(0.0);
    let limit = req.limit.unwrap_or(i64::MAX).max(0);
    let half_life = server.ranking_config().half_life_days;
    let decay_floor = server.ranking_config().decay_floor;

    let storage = server.storage_arc();
    let mut store = storage.lock().await;
    let total_before = match store.count_alive() {
        Ok(n) => n,
        Err(e) => return storage_error_to_http(e),
    };
    let rows = match store.list_recent(limit) {
        Ok(r) => r,
        Err(e) => return storage_error_to_http(e),
    };

    let now = unix_now();
    let mut archived: Vec<ConsolidateArchived> = Vec::new();
    for mem in rows {
        let age_days = ((now - mem.created_at).max(0.0) / 86400.0) as f32;
        if age_days < grace_days {
            continue;
        }
        let surprise = crate::storage::read_surprise(&mem.metadata).unwrap_or(0.0);
        let decay = crate::surprise::decay(age_days, half_life).max(decay_floor);
        let decayed = (surprise * decay).clamp(0.0, 1.0);
        if decayed >= archive_threshold {
            continue;
        }
        let id = match mem.id {
            Some(i) => i,
            None => continue,
        };
        archived.push(ConsolidateArchived {
            id,
            content_hash: mem.content_hash,
            decayed_surprise: decayed,
            age_days,
        });
    }

    if !req.dry_run {
        for a in &archived {
            let _ = store.soft_delete_by_id(a.id);
        }
    }

    let total_after = if req.dry_run {
        total_before
    } else {
        store.count_alive().unwrap_or(total_before)
    };

    // v0.5 Phase B: edge decay + prune. Reuses memory half-life for edge
    // half-life — the conceptual unit is the same ("how long does
    // co-activation matter"), and gives admins a single knob.
    let edge_threshold = req.edge_prune_threshold.unwrap_or(0.01).clamp(0.0, 1.0);
    let pruned_edges = store
        .prune_associations(half_life, edge_threshold, now)
        .unwrap_or(0);
    let associations_total = store.count_associations().unwrap_or(0);

    (
        StatusCode::OK,
        Json(ConsolidateReply {
            archived,
            total_alive_before: total_before,
            total_alive_after: total_after,
            dry_run: req.dry_run,
            pruned_edges,
            associations_total,
            deferred: vec!["semantic_clustering"],
        }),
    )
        .into_response()
}

fn unix_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

// ---------- helpers ----------

fn mcp_error_to_http(e: rmcp::ErrorData) -> axum::response::Response {
    // ErrorData carries a code; map a couple of well-known ones.
    let status = if e.message.contains("invalid")
        || e.message.contains("empty")
        || e.message.contains("required")
    {
        StatusCode::BAD_REQUEST
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    };
    (
        status,
        Json(serde_json::json!({"error": e.message.as_ref()})),
    )
        .into_response()
}

fn storage_error_to_http(e: crate::HippoError) -> axum::response::Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({"error": e.to_string()})),
    )
        .into_response()
}

fn call_result_to_text(r: rmcp::model::CallToolResult) -> Option<String> {
    r.content.into_iter().find_map(|c| {
        // Content is a tagged enum; extract text variant if present.
        match c.raw {
            rmcp::model::RawContent::Text(t) => Some(t.text),
            _ => None,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::MockEmbedder;
    use crate::server::{MemoryServer, RankingConfig};
    use crate::storage::{register_sqlite_vec, Storage};
    use crate::surprise::SurpriseWeights;
    use axum::body::Body;
    use axum::http::{Method, Request};
    use std::sync::Arc;
    use tower::util::ServiceExt;

    fn test_server() -> Arc<MemoryServer> {
        register_sqlite_vec();
        let store = Storage::open_in_memory().unwrap();
        let embedder = Arc::new(MockEmbedder::new());
        Arc::new(MemoryServer::new_full(
            store,
            embedder,
            None,
            SurpriseWeights::default(),
            RankingConfig::default(),
        ))
    }

    #[tokio::test]
    async fn health_returns_ok() {
        let app = router(test_server());
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn remember_then_list_round_trip() {
        let app = router(test_server());
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/remember")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"content":"REST smoke","tags":["smoke"]}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/memories?n=5")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 64)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["count"], 1);
    }

    #[tokio::test]
    async fn all_13_endpoints_route_to_a_handler() {
        // v0.4 Phase B-2 closed the last 6 SHODH endpoints. None should
        // return 405 / 404 for the canonical method+path combo. We probe
        // each with a minimal valid body and just check the routing
        // dispatched (status != 405 Method Not Allowed and != 404 Not
        // Found-due-to-missing-route — distinct from 404 returned for a
        // missing per-id resource, which is fine).
        let app = router(test_server());
        let probes = vec![
            ("GET", "/api/health", None),
            ("POST", "/api/remember", Some(r#"{"content":"x"}"#)),
            ("POST", "/api/recall", Some(r#"{"query":"x"}"#)),
            ("POST", "/api/recall/by-tags", Some(r#"{"tags":["x"]}"#)),
            ("POST", "/api/context", Some(r#"{"query":"x"}"#)),
            ("GET", "/api/memories", None),
            ("GET", "/api/memories/9999", None),
            ("PATCH", "/api/memories/9999", Some(r#"{}"#)),
            ("DELETE", "/api/forget/9999", None),
            ("POST", "/api/forget/by-tags", Some(r#"{"tags":["x"]}"#)),
            ("GET", "/api/tags", None),
            ("GET", "/api/stats", None),
            ("POST", "/api/consolidate", Some(r#"{}"#)),
        ];
        for (method, path, body) in probes {
            let mut req = Request::builder().method(method).uri(path);
            if body.is_some() {
                req = req.header("content-type", "application/json");
            }
            let req = req
                .body(body.map(Body::from).unwrap_or_else(Body::empty))
                .unwrap();
            let resp = app.clone().oneshot(req).await.unwrap();
            assert_ne!(
                resp.status(),
                StatusCode::METHOD_NOT_ALLOWED,
                "{method} {path}: routing dispatched 405"
            );
            assert_ne!(
                resp.status(),
                StatusCode::NOT_IMPLEMENTED,
                "{method} {path}: still 501"
            );
        }
    }

    #[tokio::test]
    async fn invalid_remember_returns_400() {
        let app = router(test_server());
        // Missing content
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/remember")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"content":""}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// Helper: create N memories with no `importance` so their surprise is
    /// engagement-only (~ 0.04 for short content, no tag) — easy material
    /// for `/api/consolidate` to flag once they age past the grace period.
    async fn create_low_surprise(server: &Arc<MemoryServer>, n: usize) {
        for i in 0..n {
            server
                .remember(crate::server::RememberParams {
                    content: format!("noise {i}"),
                    tags: vec![],
                    memory_type: None,
                    importance: None,
                    metadata: None,
                })
                .await
                .unwrap();
        }
    }

    /// Helper: backdate every alive memory by `days` so the consolidate
    /// grace_period gate doesn't keep them safe.
    async fn backdate_all(server: &Arc<MemoryServer>, days: f32) {
        let storage = server.storage_arc();
        let store = storage.lock().await;
        let rows = store.list_recent(10_000).unwrap();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        let backdated = now - (days as f64) * 86400.0;
        for row in rows {
            if let Some(id) = row.id {
                store.debug_set_created_at(id, backdated).unwrap();
            }
        }
    }

    #[tokio::test]
    async fn consolidate_grace_period_protects_fresh() {
        let s = test_server();
        create_low_surprise(&s, 5).await;
        // No backdating → all 5 are within grace period → archived = []
        let app = router(s);
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/consolidate")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"archive_threshold":0.5}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 64)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["archived"].as_array().unwrap().len(), 0);
        assert_eq!(v["total_alive_before"], 5);
        assert_eq!(v["total_alive_after"], 5);
    }

    #[tokio::test]
    async fn consolidate_dry_run_does_not_delete() {
        let s = test_server();
        create_low_surprise(&s, 3).await;
        backdate_all(&s, 365.0).await;
        let app = router(s.clone());
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/consolidate")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"archive_threshold":0.5,"grace_period_days":30,"dry_run":true}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 64)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["dry_run"], true);
        assert!(!v["archived"].as_array().unwrap().is_empty());
        assert_eq!(v["total_alive_before"], 3);
        assert_eq!(v["total_alive_after"], 3); // unchanged in dry_run
    }

    #[tokio::test]
    async fn recall_by_tags_returns_matching() {
        let s = test_server();
        s.remember(crate::server::RememberParams {
            content: "auth note".into(),
            tags: vec!["auth".into()],
            memory_type: None,
            importance: None,
            metadata: None,
        })
        .await
        .unwrap();
        s.remember(crate::server::RememberParams {
            content: "db note".into(),
            tags: vec!["db".into()],
            memory_type: None,
            importance: None,
            metadata: None,
        })
        .await
        .unwrap();
        let app = router(s);
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/recall/by-tags")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"tags":["auth"],"match":"any"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 64)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["count"], 1);
        assert_eq!(v["memories"][0]["content"], "auth note");
    }

    #[tokio::test]
    async fn forget_by_tags_dry_run_does_not_delete() {
        let s = test_server();
        for c in ["a", "b", "c"] {
            s.remember(crate::server::RememberParams {
                content: c.to_string(),
                tags: vec!["bulk".into()],
                memory_type: None,
                importance: None,
                metadata: None,
            })
            .await
            .unwrap();
        }
        let app = router(s.clone());
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/forget/by-tags")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"tags":["bulk"],"dry_run":true}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 64)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["matched"], 3);
        assert_eq!(v["dry_run"], true);
        assert_eq!(s.storage_arc().lock().await.count_alive().unwrap(), 3);
    }

    #[tokio::test]
    async fn forget_by_tags_actually_deletes() {
        let s = test_server();
        for c in ["x", "y"] {
            s.remember(crate::server::RememberParams {
                content: c.to_string(),
                tags: vec!["wipe".into()],
                memory_type: None,
                importance: None,
                metadata: None,
            })
            .await
            .unwrap();
        }
        let app = router(s.clone());
        let _ = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/forget/by-tags")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"tags":["wipe"]}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(s.storage_arc().lock().await.count_alive().unwrap(), 0);
    }

    #[tokio::test]
    async fn get_by_id_returns_404_when_missing() {
        let app = router(test_server());
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/memories/9999")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn get_by_id_returns_existing() {
        let s = test_server();
        let r = s
            .remember(crate::server::RememberParams {
                content: "by-id smoke".into(),
                tags: vec![],
                memory_type: None,
                importance: None,
                metadata: None,
            })
            .await
            .unwrap();
        let app = router(s);
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri(format!("/api/memories/{}", r.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 64)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["content"], "by-id smoke");
    }

    #[tokio::test]
    async fn patch_preserves_hippo_namespace() {
        let s = test_server();
        let r = s
            .remember(crate::server::RememberParams {
                content: "patch smoke".into(),
                tags: vec!["original".into()],
                memory_type: None,
                importance: None,
                metadata: None,
            })
            .await
            .unwrap();
        let app = router(s.clone());
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::PATCH)
                    .uri(format!("/api/memories/{}", r.id))
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"metadata":{"user_field":42},"tags":["renamed"]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let store = s.storage_arc();
        let store = store.lock().await;
        let row = store.get_by_id(r.id).unwrap().unwrap();
        assert_eq!(row.tags, vec!["renamed"]);
        assert_eq!(row.metadata["user_field"], 42);
        // _hippo namespace must be preserved (surprise score stays)
        assert!(row.metadata.get("_hippo").is_some());
    }

    #[tokio::test]
    async fn list_tags_aggregates_with_counts() {
        let s = test_server();
        for (c, tags) in [
            ("a", vec!["x", "y"]),
            ("b", vec!["x"]),
            ("c", vec!["y"]),
            ("d", vec!["z"]),
        ] {
            s.remember(crate::server::RememberParams {
                content: c.into(),
                tags: tags.into_iter().map(String::from).collect(),
                memory_type: None,
                importance: None,
                metadata: None,
            })
            .await
            .unwrap();
        }
        let app = router(s);
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/tags")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 64)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let tags = v["tags"].as_array().unwrap();
        let map: std::collections::HashMap<String, i64> = tags
            .iter()
            .map(|t| {
                (
                    t["tag"].as_str().unwrap().to_string(),
                    t["count"].as_i64().unwrap(),
                )
            })
            .collect();
        assert_eq!(map.get("x").copied(), Some(2));
        assert_eq!(map.get("y").copied(), Some(2));
        assert_eq!(map.get("z").copied(), Some(1));
    }

    #[tokio::test]
    async fn context_auto_ingest_stores_query() {
        let s = test_server();
        let app = router(s.clone());
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/context")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"query":"context smoke","auto_ingest":true}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        // Auto-ingest should have stored the query as a Context-typed memory.
        let store = s.storage_arc();
        let store = store.lock().await;
        assert_eq!(store.count_alive().unwrap(), 1);
    }

    #[tokio::test]
    async fn consolidate_archives_aged_low_surprise() {
        let s = test_server();
        create_low_surprise(&s, 4).await;
        backdate_all(&s, 365.0).await;
        let app = router(s.clone());
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/consolidate")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"archive_threshold":0.5,"grace_period_days":30}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 64)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["dry_run"], false);
        assert_eq!(v["archived"].as_array().unwrap().len(), 4);
        assert_eq!(v["total_alive_before"], 4);
        assert_eq!(v["total_alive_after"], 0);
        // v0.5: only semantic_clustering remains deferred — association
        // discovery is wired (edges grown on read, decayed/pruned here).
        let deferred: Vec<&str> = v["deferred"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap())
            .collect();
        assert!(!deferred.contains(&"association_discovery"));
        assert!(deferred.contains(&"semantic_clustering"));
        // Edge prune fields must surface even when no edges exist.
        assert_eq!(v["pruned_edges"], 0);
        assert_eq!(v["associations_total"], 0);
    }
}
