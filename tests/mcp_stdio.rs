//! Integration test: spawn the `hippo serve` binary as a child process,
//! speak MCP JSON-RPC over stdio, and exercise every tool end-to-end.
//!
//! Uses MockEmbedder via `HIPPO_TEST_MOCK_EMBEDDER=1` is NOT supported (the
//! binary always uses FastEmbedder). To avoid downloading the real model in
//! CI, this test is gated behind `RUST_TEST_HEAVY=1`. Local runs work as long
//! as the model is pre-cached (which is the case after the first `hippo embed`).

use serde_json::json;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::time::Duration;

fn binary_path() -> PathBuf {
    // CARGO_BIN_EXE_<name> is set by cargo when running integration tests.
    // Fall back to target/{debug,release}/hippo for non-cargo invocations.
    if let Some(p) = option_env!("CARGO_BIN_EXE_hippo") {
        return PathBuf::from(p);
    }
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("target");
    p.push("release");
    p.push("hippo");
    if !p.exists() {
        p.pop();
        p.pop();
        p.push("debug");
        p.push("hippo");
    }
    p
}

struct McpClient {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
}

impl McpClient {
    fn spawn(db_path: &std::path::Path, model_cache: &std::path::Path) -> Self {
        let mut cmd = Command::new(binary_path());
        cmd.arg("serve")
            .arg("--db")
            .arg(db_path)
            .arg("--model-cache")
            .arg(model_cache)
            .env("RUST_LOG", "warn")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        let mut child = cmd.spawn().expect("spawn hippo serve");
        let stdin = child.stdin.take().expect("stdin");
        let stdout = BufReader::new(child.stdout.take().expect("stdout"));
        let mut me = Self {
            child,
            stdin,
            stdout,
        };
        me.handshake();
        me
    }

    fn send(&mut self, msg: &serde_json::Value) {
        let line = serde_json::to_string(msg).expect("serialize");
        self.stdin.write_all(line.as_bytes()).expect("write stdin");
        self.stdin.write_all(b"\n").expect("newline");
        self.stdin.flush().expect("flush");
    }

    fn read_until(&mut self, want_id: i64) -> serde_json::Value {
        loop {
            let mut buf = String::new();
            let n = self.stdout.read_line(&mut buf).expect("read stdout");
            assert!(n > 0, "server closed stdout (EOF) waiting for id={want_id}");
            let v: serde_json::Value = serde_json::from_str(buf.trim()).expect("parse jsonrpc");
            if v.get("id").and_then(|x| x.as_i64()) == Some(want_id) {
                return v;
            }
            // Other messages (notifications, log frames) are ignored.
        }
    }

    fn handshake(&mut self) {
        self.send(&json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "hippo-it", "version": "0"}
            }
        }));
        let _ = self.read_until(1);
        self.send(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"}));
    }

    fn call(&mut self, id: i64, name: &str, args: serde_json::Value) -> serde_json::Value {
        self.send(&json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "tools/call",
            "params": {"name": name, "arguments": args}
        }));
        self.read_until(id)
    }

    fn shutdown(mut self) {
        drop(self.stdin);
        let _ = self.child.wait_timeout_or_kill(Duration::from_secs(5));
    }
}

trait WaitTimeout {
    fn wait_timeout_or_kill(&mut self, dur: Duration) -> Option<std::process::ExitStatus>;
}

impl WaitTimeout for Child {
    fn wait_timeout_or_kill(&mut self, dur: Duration) -> Option<std::process::ExitStatus> {
        let start = std::time::Instant::now();
        while start.elapsed() < dur {
            match self.try_wait() {
                Ok(Some(s)) => return Some(s),
                Ok(None) => std::thread::sleep(Duration::from_millis(50)),
                Err(_) => break,
            }
        }
        let _ = self.kill();
        self.wait().ok()
    }
}

fn extract_text_payload(reply: &serde_json::Value) -> serde_json::Value {
    let result = reply
        .get("result")
        .unwrap_or_else(|| panic!("no result in reply: {reply}"));
    let content = result
        .get("content")
        .and_then(|c| c.as_array())
        .expect("result.content array");
    let first = content.first().expect("at least one content");
    let text = first
        .get("text")
        .and_then(|t| t.as_str())
        .expect("first.content.text");
    serde_json::from_str(text).expect("parse tool payload as json")
}

fn requires_model() -> Option<PathBuf> {
    // Use the same cache that mcp-memory-service-rs and a manual `hippo embed`
    // populate. If absent and RUST_TEST_HEAVY != 1, skip.
    let home = dirs::home_dir().expect("home");
    let mms_cache = home
        .join(".cache")
        .join("mcp_memory")
        .join("onnx_models")
        .join("all-MiniLM-L6-v2")
        .join("onnx")
        .join("model.onnx");
    let hippo_cache = home.join(".cache").join("claude-hippo").join("models");
    // fastembed maintains its own subdirs under model_cache. We pass the
    // dir to hippo and let fastembed download into it (cached after first run).
    if !mms_cache.exists() && std::env::var("RUST_TEST_HEAVY").unwrap_or_default() != "1" {
        eprintln!(
            "skip: ONNX model not cached at {} and RUST_TEST_HEAVY != 1",
            mms_cache.display()
        );
        return None;
    }
    Some(hippo_cache)
}

#[test]
fn end_to_end_remember_recall_forget() {
    let Some(model_cache) = requires_model() else {
        return;
    };
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("e2e.db");
    let mut cli = McpClient::spawn(&db, &model_cache);

    // ping
    let r = cli.call(2, "ping", json!({}));
    let payload = extract_text_payload(&r);
    assert_eq!(payload["status"], "ok");
    assert_eq!(payload["alive"], 0);

    // remember 3 entries
    for (i, content) in [
        "use JWT 24h expiry for auth",
        "Postgres index on tenant_id is critical",
        "switched billing from Stripe to Paddle",
    ]
    .iter()
    .enumerate()
    {
        let r = cli.call(
            10 + i as i64,
            "hippo_remember",
            json!({
                "content": content,
                "tags": ["bench", format!("i{i}")],
                "memory_type": "Decision",
                "importance": 0.5
            }),
        );
        let p = extract_text_payload(&r);
        assert_eq!(p["success"], true);
        assert_eq!(p["duplicate"], false);
        assert!(p["surprise_score"].as_f64().unwrap() > 0.0);
    }

    // recall by query
    let r = cli.call(20, "hippo_recall", json!({"query": "auth", "limit": 3}));
    let p = extract_text_payload(&r);
    let arr = p.as_array().expect("recall returns array");
    assert!(!arr.is_empty(), "recall must return at least 1 result");

    // SHODH alias
    let r2 = cli.call(21, "retrieve_memory", json!({"query": "auth", "limit": 3}));
    let p2 = extract_text_payload(&r2);
    assert!(!p2.as_array().expect("alias returns array").is_empty());

    // list_recent
    let r = cli.call(30, "hippo_list_recent", json!({"n": 10}));
    let p = extract_text_payload(&r);
    assert_eq!(p["count"], 3);

    // session summary
    let r = cli.call(40, "hippo_session_summary", json!({"hours": 1}));
    let p = extract_text_payload(&r);
    assert_eq!(p["total_memories"], 3);
    assert!(p["mean_surprise"].as_f64().unwrap() > 0.0);

    // forget by hash (dry run first)
    let hash = claude_hippo::storage::content_hash("use JWT 24h expiry for auth");
    let r = cli.call(
        50,
        "hippo_forget",
        json!({"content_hash": hash.clone(), "dry_run": true}),
    );
    let p = extract_text_payload(&r);
    assert_eq!(p["dry_run"], true);
    assert_eq!(p["deleted"], 1);

    // actual forget
    let r = cli.call(
        51,
        "hippo_forget",
        json!({"content_hash": hash, "dry_run": false}),
    );
    let p = extract_text_payload(&r);
    assert_eq!(p["dry_run"], false);
    assert_eq!(p["deleted"], 1);

    // verify count after forget
    let r = cli.call(52, "ping", json!({}));
    let p = extract_text_payload(&r);
    assert_eq!(p["alive"], 2);
    assert_eq!(p["total"], 3);

    cli.shutdown();
}

#[test]
fn empty_query_rejected() {
    let Some(model_cache) = requires_model() else {
        return;
    };
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("err.db");
    let mut cli = McpClient::spawn(&db, &model_cache);

    let r = cli.call(2, "hippo_recall", json!({"query": ""}));
    assert!(
        r.get("result").is_some_and(|res| {
            // tool errors are returned as result.is_error or as JSON-RPC error
            res.get("isError")
                .and_then(|x| x.as_bool())
                .unwrap_or(false)
        }) || r.get("error").is_some()
    );

    cli.shutdown();
}

#[test]
fn dedup_returns_duplicate_flag() {
    let Some(model_cache) = requires_model() else {
        return;
    };
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("dedup.db");
    let mut cli = McpClient::spawn(&db, &model_cache);

    let args = json!({
        "content": "deduplicate me",
        "tags": ["dedup"],
        "memory_type": "Observation",
    });
    let r1 = cli.call(2, "hippo_remember", args.clone());
    let p1 = extract_text_payload(&r1);
    assert_eq!(p1["duplicate"], false);

    let r2 = cli.call(3, "hippo_remember", args);
    let p2 = extract_text_payload(&r2);
    assert_eq!(p2["duplicate"], true);
    assert_eq!(p2["id"], p1["id"]);

    cli.shutdown();
}
