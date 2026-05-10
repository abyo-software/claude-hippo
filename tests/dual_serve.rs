//! Integration test: stdio MCP and SHODH REST in the same process,
//! sharing one `Arc<Mutex<Storage>>`.
//!
//! This is the v0.4 "in-process dual-serve" path that v0.3 explicitly
//! deferred (the v0.3 caveat was: enable `--shodh-rest` and stdio MCP got
//! disabled). The fix: build two `MemoryServer` instances via
//! `from_shared_storage` and run them concurrently. This test exercises
//! only the REST half end-to-end against an axum router built from a
//! shared MemoryServer, plus a direct call against a second instance to
//! prove they observe the same data — the stdio MCP transport itself is
//! covered by `tests/mcp_stdio.rs`.

use claude_hippo::embeddings::MockEmbedder;
use claude_hippo::server::{MemoryServer, RankingConfig, RememberParams};
use claude_hippo::shodh_rest::router;
use claude_hippo::storage::{register_sqlite_vec, Storage};
use claude_hippo::surprise::SurpriseWeights;
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rest_and_direct_share_storage() {
    register_sqlite_vec();
    let store = Storage::open_in_memory().unwrap();
    let shared = Arc::new(Mutex::new(store));
    let embedder = Arc::new(MockEmbedder::new());

    // Two MemoryServer instances pointing at the same shared storage.
    let rest_instance = Arc::new(MemoryServer::from_shared_storage(
        shared.clone(),
        embedder.clone(),
        None,
        SurpriseWeights::default(),
        RankingConfig::default(),
        false,
    ));
    let direct_instance = MemoryServer::from_shared_storage(
        shared.clone(),
        embedder,
        None,
        SurpriseWeights::default(),
        RankingConfig::default(),
        false,
    );

    // Write through the REST surface.
    let app = router(rest_instance);
    let resp = tower::ServiceExt::oneshot(
        app,
        axum::http::Request::builder()
            .method(axum::http::Method::POST)
            .uri("/api/remember")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                r#"{"content":"shared-store smoke","tags":["dual"]}"#,
            ))
            .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(resp.status(), axum::http::StatusCode::OK);

    // Read through the direct (would-be MCP) surface — must see what REST wrote.
    let recall = direct_instance
        .recall(claude_hippo::server::RecallParams {
            query: "shared-store smoke".into(),
            limit: 5,
            no_surprise_boost: false,
            oversample_factor: None,
            mode: None,
            seed_id: None,
        })
        .await
        .unwrap();
    assert_eq!(recall.len(), 1);
    assert_eq!(recall[0].memory.content, "shared-store smoke");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn writes_through_either_instance_are_visible_to_both() {
    register_sqlite_vec();
    let store = Storage::open_in_memory().unwrap();
    let shared = Arc::new(Mutex::new(store));
    let embedder = Arc::new(MockEmbedder::new());

    let a = MemoryServer::from_shared_storage(
        shared.clone(),
        embedder.clone(),
        None,
        SurpriseWeights::default(),
        RankingConfig::default(),
        false,
    );
    let b = MemoryServer::from_shared_storage(
        shared,
        embedder,
        None,
        SurpriseWeights::default(),
        RankingConfig::default(),
        false,
    );

    // Write to A
    let _ = a
        .remember(RememberParams {
            content: "from a".into(),
            tags: vec![],
            memory_type: None,
            importance: None,
            metadata: None,
        })
        .await
        .unwrap();
    // Write to B
    let _ = b
        .remember(RememberParams {
            content: "from b".into(),
            tags: vec![],
            memory_type: None,
            importance: None,
            metadata: None,
        })
        .await
        .unwrap();

    // Each side should see both items.
    for inst in [&a, &b] {
        let hits = inst
            .recall(claude_hippo::server::RecallParams {
                query: "from".into(),
                limit: 5,
                no_surprise_boost: true, // pure cosine, deterministic
                oversample_factor: None,
                mode: None,
                seed_id: None,
            })
            .await
            .unwrap();
        let bodies: Vec<String> = hits.into_iter().map(|h| h.memory.content).collect();
        assert!(bodies.contains(&"from a".to_string()), "{:?}", bodies);
        assert!(bodies.contains(&"from b".to_string()), "{:?}", bodies);
    }
}
