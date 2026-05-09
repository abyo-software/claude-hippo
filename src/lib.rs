//! claude-hippo — Claude Code に海馬を足す MCP サーバ。
//!
//! 全部覚える代わりに、特異性が高い瞬間だけを長期記憶化する surprise-aware
//! memory store。schema は mcp-memory-service-rs と互換 (SHODH spec 準拠)、
//! ライセンスは Apache-2.0 / MIT dual。
//!
//! # 構成
//! - [`storage`][]: SQLite + sqlite-vec の記憶ストア (mcp-memory-service-rs 互換 schema)
//! - [`embeddings`][]: fastembed (all-MiniLM-L6-v2, 384 dim, L2 normalized)
//! - [`surprise`][]: surprise score 算出 (embedding outlier + engagement + explicit)
//! - [`server`][]: rmcp で MCP stdio server (5 hippo_* tools + SHODH alias)
//! - [`cli`][]: clap CLI (serve / verify / embed / bench)

pub mod cli;
pub mod embeddings;
pub mod error;
pub mod memory_tool;
pub mod prediction_loss;
pub mod server;
pub mod shodh_rest;
pub mod storage;
pub mod surprise;

pub use error::{HippoError, Result};

/// このバイナリのバージョン。Cargo.toml の `version` を反映。
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// 埋め込み次元 (BGE-small-en-v1.5 / all-MiniLM-L6-v2 共通の 384)。
/// これは mcp-memory-service-rs と DB swap 互換性を保つために固定。
pub const EMBEDDING_DIM: usize = 384;
