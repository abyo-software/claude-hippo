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
        .route("/api/memories", get(handle_list))
        .route("/api/forget/{id}", delete(handle_forget))
        .route("/api/stats", get(handle_stats))
        // Stubs for the unimplemented SHODH endpoints — return 501 with a
        // clear pointer rather than 404 / silent omission.
        .route("/api/context", post(handle_not_implemented))
        .route("/api/recall/by-tags", post(handle_not_implemented))
        .route("/api/forget/by-tags", post(handle_not_implemented))
        .route("/api/memories/{id}", get(handle_not_implemented))
        .route(
            "/api/memories/{id}",
            axum::routing::patch(handle_not_implemented),
        )
        .route("/api/consolidate", post(handle_not_implemented))
        .route("/api/tags", get(handle_not_implemented))
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

async fn handle_not_implemented(uri: axum::http::Uri) -> impl IntoResponse {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": "not_implemented",
            "path": uri.path(),
            "message": "v0.3 SHODH REST exposes 6 endpoints (health/remember/recall/memories/\
                        forget/stats). The remaining 7 (consolidate, by-tags variants, \
                        context auto-ingest, per-id GET/PATCH, list tags) are tracked for v0.4. \
                        For now, use the MCP stdio interface or the documented hippo_* tools.",
        })),
    )
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
    async fn unimplemented_endpoint_returns_501() {
        let app = router(test_server());
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/consolidate")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 64)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"], "not_implemented");
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
}
