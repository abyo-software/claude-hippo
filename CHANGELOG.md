# Changelog

すべての変更点はこのファイルに記載します。
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) に従います。

## [Unreleased]

### Planned (v0.2)
- 独自評価ベンチ（Long-session noise / Cross-session retrieval / Decision trace）
- `SurpriseWeights` の CLI / config 露出
- ONNX 量子化モデル（BGESmallENV15Q）切替フラグ
- External embedding API mode (`--embedding-backend external`、RSS <50 MB 経路)

### Planned (v0.3)
- `abyo-llm-probe` 統合で `prediction_loss` を埋める
- `abyo-filters` 内蔵で memory 存在判定を空間効率化
- 多 MCP client 動作確認（Cursor / Continue / Aider）
- Anthropic Memory Tool 互換レイヤ
- SHODH OpenAPI REST 互換 endpoint (`--shodh-rest`)

## [0.1.0] - 2026-05-10

### Added
- MCP stdio server (`hippo serve`) with rmcp 1.6
- 5 native tools: `hippo_remember`, `hippo_recall`, `hippo_list_recent`, `hippo_forget`, `hippo_session_summary`
- 4 SHODH-compatible aliases: `store_memory`, `retrieve_memory`, `list_memories`, `delete_memory`
- `ping` health probe (vec_version, alive count, total count, uptime, version)
- SQLite + sqlite-vec storage with **verbatim** schema-compatibility against `mcp-memory-service-rs` (11 columns + memory_embeddings vec0 FLOAT[384] cosine)
- `fastembed` embedding via `all-MiniLM-L6-v2` (384 dim, L2 normalized) — same vector space as `mcp-memory-service-rs`, enabling DB swap with retrieval semantics intact
- Lazy embedding model load (cold-start `~5 ms`, model load deferred to first embed call)
- Surprise scoring: `embedding_outlier` + `engagement` + `explicit` + `prediction_loss?` (last is None until v0.3)
- Surprise-weighted retrieval ranking with 30-day exponential decay (`no_surprise_boost=true` to disable)
- soft-delete only (audit trail preserved); KNN over-sample only when tombstones exist
- CLI subcommands: `serve`, `verify`, `embed`, `bench`
- `HIPPO_DB_PATH` and `HIPPO_MODEL_CACHE` env support
- Mock embedder for tests
- 29 unit tests + 3 integration tests (`tests/mcp_stdio.rs`)
- DB swap conformance test (`scripts/conformance_swap.py`) — both directions pass
- Head-to-head bench (`scripts/bench_competitor.py`) — claude-hippo wins on every metric vs `mcp-memory-service-rs`:
  - cold-start 4.5 ms vs 117.3 ms (26× faster)
  - store p50 3.1 ms vs 5.9 ms (1.9× faster)
  - retrieve p50 2.7 ms vs 6.7 ms (2.5× faster)
  - RSS 150.5 MB vs 186.3 MB (19% lighter)
- Apache-2.0 / MIT dual license (commercial-friendly alternative to `mcp-memory-service-rs` PolyForm Noncommercial)
- README, SURPRISE_SELECTION, SHODH_COMPAT, COMPETITOR_BENCH, CLAUDE_CODE_SETUP, ARCHITECTURE docs
