# Changelog

すべての変更点はこのファイルに記載します。
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) に従います。

## [Unreleased]

### Planned (v0.3)
- `abyo-llm-probe` 統合で `prediction_loss` を埋める
- `abyo-filters` 内蔵で memory 存在判定を空間効率化
- 多 MCP client 動作確認（Cursor / Continue / Aider）
- Anthropic Memory Tool 互換レイヤ
- SHODH OpenAPI REST 互換 endpoint (`--shodh-rest`)
- External embedding API backend (`--embedding-backend external`) — 設計済 (`docs/EXTERNAL_EMBEDDING.md`)、実装は v0.3
- Forgetting curve floor / `--half-life-days` CLI（Bench B が surfacした 365 日越え demotion 問題への対処）
- `oversample_factor` を MCP `RecallParams` に expose（現状は eval harness のみ）

## [0.2.0] - 2026-05-10

### Added
- **独自評価ベンチ Bench A/B/C** (`tests/eval_*.rs`) — surprise rerank の数値証拠：
  - **Bench A** (Long-session noise, 100 items / 25 queries): precision@1 を **8% → 72%** (既定 oversample) / **100%** (完全 oversample) にリフト
  - **Bench B** (Cross-session, 50 items / 8 queries × 4 ages): half-life=30d 設計の挙動を実測。30 日まで perfect、90 日で baseline 同等、365 日で **負の lift** という limit を honestly 公開
  - **Bench C** (Decision trace, 20 items / 8 queries): 4 Decision を 16 non-Decision の中から見つける recall を **0.44 → 0.84** (既定 weights) / **1.00** (explicit-heavy weights `--surprise-weights "0.2,0.1,0.5,0.2"`) に
  - 全 bench は `target/eval_results/bench_*.json` に書き出し、`docs/SURPRISE_SELECTION.md` から参照
- `--surprise-weights "w_o,w_e,w_x,w_p"` フラグ（`serve` / `bench`）。`SurpriseWeights::parse_csv` が sum=1 (±1e-3) を検証
- `HIPPO_SURPRISE_WEIGHTS` env でも同設定可
- `--embedding-model {minilm-l6-v2,bge-small-en-v15-q}` フラグ（`serve` / `embed` / `bench`）。`EmbeddingModelKind` enum で 384 dim 縛り保持、`HIPPO_EMBEDDING_MODEL` env も可
- `MemoryServer::new_with_weights(...)`、`MemoryServer::recall(...) -> Vec<RecalledMemory>` (typed)、`MemoryServer::recall_with_options(...)` (oversample_factor 調整可)、`MemoryServer::storage_arc()` (eval harness 用)、`Storage::debug_set_created_at(...)` (eval harness 用 backdating)
- `RecallOptions { oversample_factor }` 構造体（既定 3、production 互換）。fetch_k = limit × factor
- `cargo audit` を CI 必須 job として追加 (`audit.toml` で `paste` unmaintained を ignore — 理由付きコメント込み、tokenizers の上流が変わったら再評価)
- `docs/EXTERNAL_EMBEDDING.md` — External embedding API mode の v0.3 設計書（CLI フラグ、API 形状、retry/timeout、RSS 目標、セキュリティ、テストプラン）

### Changed
- `MemoryServer::do_remember` / `do_recall` を typed wrapper に refactor（内部 `remember(...) -> RememberResult` / `recall(...) -> Vec<RecalledMemory>` を eval/test から直接呼べる）
- `FastEmbedder` を `EmbeddingModelKind` 受け取りに拡張（既定は MiniLM、互換性維持）

### Documentation
- `docs/SURPRISE_SELECTION.md` の §「評価軸」を実数値に書き換え。各ベンチの再現コマンド、honest limitations（365 日 demotion、既定 oversample 取りこぼし）を本文に記載

### Tests
- 40 unit + 3 integration + 3 eval = **46 tests** (v0.1 の 32 から +14)、全 release green、clippy clean、cargo fmt clean

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
