# Changelog

すべての変更点はこのファイルに記載します。
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) に従います。

## [Unreleased]

## [0.4.0] - 2026-05-10

### Added (Phase A + B + C + D-spike + E)

**Phase A — In-process dual-serve (stdio MCP + SHODH REST 同居)**:
- `MemoryServer::from_shared_storage(...)` 新コンストラクタ — `Arc<Mutex<Storage>>` + `Arc<dyn Embedder>` + `Option<Arc<dyn PredictionLossBackend>>` を 2 instance で共有
- CLI `run_serve_with_optional_rest` を refactor: `--shodh-rest` 設定時に rest_instance + mcp_instance を build、`tokio::spawn` 2 本を `tokio::select!` で監視。どちらかの exit/error で process 終了
- v0.3 caveat 「1 process = 1 transport」を解消
- 2 integration tests (`tests/dual_serve.rs`): REST 経由で書いた memory が直接 instance から見える / 双方向 write 可視性

**Phase B-1 — SHODH `/api/consolidate` endpoint**:
- exponential decay scoring + quality-based archival of low-value memories
- 入力: `archive_threshold` (default 0.05) / `grace_period_days` (default 30) / `limit` / `dry_run`
- decay は `server.ranking_config().{half_life_days, decay_floor}` を流用
- 出力: archived list + total_alive before/after + dry_run flag + `deferred=["association_discovery", "semantic_clustering"]` (SHODH spec の残り 2 機能の honest disclosure)
- association discovery (Hebbian edges) + semantic clustering は schema 変更要のため v0.5 候補
- 3 axum unit tests

**Phase B-2 — SHODH 残り 6 endpoint**:
- `POST /api/recall/by-tags` — tag AND-OR search
- `POST /api/forget/by-tags` — bulk soft-delete with dry_run
- `GET /api/memories/{id}` — fetch by id, 404 if missing
- `PATCH /api/memories/{id}` — metadata 編集、`_hippo` namespace 自動保存 (surprise score 維持)、content/hash 不変
- `GET /api/tags` — alive tags の (tag, count) 集計
- `POST /api/context` — query → optional auto_ingest → recall 関連メモリ
- `Storage::update_metadata_by_id(...)` + `Storage::list_tags()` 新 API
- 8 axum unit tests + `all_13_endpoints_route_to_a_handler` regression net
- **SHODH OpenAPI v1.0.0 全 13 endpoint 実装完了**

**Phase C — 実 Ollama smoke (embedding only)**:
- Ollama 1.x install + `all-minilm` (384 dim) pull
- `hippo bench --embedding-backend external --external-embedding-url http://localhost:11434/v1/embeddings --external-embedding-model all-minilm --external-embedding-api-key-env NONE`:
  - cold-start: 802 ms (Ollama warm + first round-trip)
  - store p50: 11.4 ms / retrieve p50: 14.9 ms (network round-trip 主)
  - **peak RSS: 26.4 MB** (local fastembed 150 MB の 17%)
- prediction-loss: Ollama の `/v1/completions` は `echo + max_tokens=0 + logprobs` を honour せず、honest disclosure (vLLM / llama.cpp / candle-rs native v0.5 が必要) を `docs/SURPRISE_SELECTION.md` に追記
- Bench A/B/C は ClusteredMockEmbedder で決定的を維持 (real backend に swap すると baseline 比較壊れる、honest disclosure 込み)

**Phase D-spike — candle-rs native prediction-loss (PoC 成功)**:
- `examples/candle_spike.rs` — Qwen2.5-0.5B BF16, CPU 推論 (CUDA は cuDNN system install 必要のため v0.5)
- 4 sample で実 NLL 取得:
  - `the quick brown fox jumps over the lazy dog`: NLL=1.20 → surprise=0.20
  - `todo: fix the bug`: NLL=4.34 → surprise=0.72
  - `After auditing 47k OpenTelemetry spans we picked OTLP over Jaeger because of native TLS 1.3 support`: NLL=4.96 → **surprise=0.83**
  - `the proton-to-electron mass ratio decreased by 12% under quantum gravity at noon`: NLL=4.12 → surprise=0.69
- 期待通りの勾配 (predictable cliché 低 / specific decision 高) → v0.5 で `--features candle` flag 経由で production 化判断
- candle-rs (candle-core / candle-nn / candle-transformers / tokenizers) は dev-dep のみ追加、main crate を lean 維持

**Phase E — Traction**:
- `docs/SHOW_HN_DRAFT.md` 新設 — HN / Lobsters / r/rust / r/ClaudeAI 向け投稿 draft (英 + alt 短縮版)
- `README.md` に v0.4 highlights 4 行追加 + Downloads badge
- v0.3 release momentum 活用ねらい

### Stats

- 87 → **98 tests** (82 unit + 9 wiremock embedding + 9 wiremock prediction_loss + 3 integration + 4 eval + 2 dual-serve + 11 axum REST)
- clippy clean、fmt clean、cargo audit 0 new advisories
- 新規依存 (dev only): candle-core 0.9 / candle-nn 0.9 / candle-transformers 0.9 / tokenizers 0.22 / hf-hub 0.5 (dev-dep)
- crates.io: `cargo install claude-hippo`
- GitHub: https://github.com/abyo-software/claude-hippo/releases/tag/v0.4.0

### Honest disclosures (v0.5 で対処)

1. SHODH `consolidate` の association discovery (Hebbian edges) + semantic clustering は schema 要変更で deferred
2. prediction-loss real LLM bench は local D-spike のみ、Bench A/B/C 全自動再走は未走 (real backend が決定的でないため設計が要再考)
3. candle-rs CUDA path は cuDNN system install 必要、v0.5 で `--features candle-cuda` として opt-in
4. candle-rs spike は examples/ 経由のみ、main lib への production 統合は v0.5
5. GUI client (Cursor / Continue) 自動検証は引き続き手動 smoke per release

### Planned (v0.5)

- `--features candle` で candle-rs native prediction-loss を main lib に統合 (`CandleLocalPredictionLoss`)
- `--features candle-cuda` で GPU acceleration (cuDNN install required)
- SHODH consolidate に association discovery + semantic clustering を追加 (Hebbian `memory_associations` 別テーブル新設)
- abyo-llm-probe Stage 2 (Vast.ai 4090) 完走 → 大モデル (Phi-3.5-mini / Llama 3.1 8B) の verdict 取得
- Bench A-D に real backend variant を追加 (再現性 vs 実機の trade-off を整理)

## [0.3.0] - 2026-05-10

### Added (Phase A + B + C + D)

**Phase A — surprise rerank の honest limitations 解消**:
- **`decay_floor` ranking parameter**（CLI `--decay-floor`、env `HIPPO_DECAY_FLOOR`、既定 0.5）— Bench B が v0.2 で surface した「365 日越え (12 half-lives) で `surprise·decay → 0` する一方で fresh chat の `surprise·1` が old high-importance Decision を demote する」失敗モードを構造的に解消。`max(decay(age, half_life), decay_floor)` で old item の surprise 寄与に下限を設定。Bench B 365d で **negative lift → +0.875 lift** にflip
- **`--half-life-days` CLI**（env `HIPPO_HALF_LIFE_DAYS`、既定 30.0）— v0.2 でハードコードだった `DEFAULT_HALF_LIFE_DAYS = 30.0` をユーザ tunable 化。0 で decay 完全 disable
- **`--oversample-factor` CLI**（env `HIPPO_OVERSAMPLE_FACTOR`、既定 6）— v0.2 既定 3 から bump。Bench A 既定で **precision@1 0.72 → 1.000** に。`MCP RecallParams.oversample_factor: Option<usize>` で per-call override も提供
- `RankingConfig` 構造体 + `MemoryServer::new_with_config(...)` / `server::run_stdio_with_config(...)` — 上記 3 ノブを 1 か所にまとめた server-wide config
- `RecallParams.oversample_factor: Option<usize>` を MCP schema に expose（v0.2 では `RecallOptions` 内のみで eval-only）

**Phase C — Prediction-loss backend**:
- `src/prediction_loss/` 新規 module — `PredictionLossBackend` trait + `PredictionLossBackendKind { None, OpenAiCompat }` enum + `MockPredictionLoss` (テスト用 SHA256 deterministic) + `ExternalPredictionLossBackend` (OpenAI legacy `/v1/completions` 互換、`echo + max_tokens=0 + logprobs`)
- `MemoryServer::new_full(...)` + `server::run_stdio_full(...)` で optional な backend を受け、`remember()` フローが `surprise_components.prediction_loss = Some(value)` を埋める。backend が `None` の時は v0.2 fallback (`w_prediction` 再分配)
- CLI: `--prediction-loss-backend {none, openai-compat}` + `--prediction-loss-{url,model,api-key-env,timeout-ms,max-retries,scale}` (env 全対応)
- mean NLL → surprise マッピング: `clamp(mean_nll / loss_scale, 0, 1)` 既定 scale = 6.0 nats/token
- 9 wiremock integration tests (`tests/prediction_loss.rs`): mean NLL scaling / clamp upper / clamp lower / 空 content short-circuit / logprobs 欠落の actionable error / 401 fail-fast / 429 retry / MemoryServer end-to-end (with/without backend)
- Bench D 追加 (`tests/eval_d_prediction_loss.rs`): MockPredictionLoss で wiring smoke (coverage 100/100)、`target/eval_results/bench_d_prediction_loss.json` 出力。実 LLM 数値は release-time smoke 送り (Ollama/vLLM が必要)
- 対応 backend: vLLM `/v1/completions`、llama.cpp `/completion`、Ollama (shim)、legacy OpenAI Completions。OpenAI Chat Completions は prompt logprobs を返さないため非対応

**Phase B — External embedding backend**:
- **`--embedding-backend {local,external}`** + **`--external-embedding-{url,model,api-key-env,timeout-ms,batch-size,max-retries}`** + 全対応 env (`HIPPO_EXTERNAL_EMBEDDING_*`)
- `src/embeddings/` を module 化、新規 `external.rs` で OpenAI 互換 `/v1/embeddings` HTTP backend (reqwest + rustls-tls、L2 正規化強制、384 dim 検証 fail-loud、indexed re-order、429/5xx exponential backoff、batch chunking)
- `EmbeddingBackendKind` enum + `EmbeddingFlags` clap 共有 struct で serve/embed/bench 全 subcommand に backend selector を追加
- async-from-sync ブリッジ: `tokio::task::block_in_place` + `Handle::block_on` で MCP の sync Embedder trait を維持しつつ async reqwest を呼ぶ
- 9 wiremock integration tests (`tests/external_embedding.rs`): happy path / dim mismatch / 401 fail-fast / 429 retry-then-succeed / 429 max-retries-exhausted / 5xx retry / batch chunk order preservation / concurrent embeds / un-normalized input → output normalized
- `examples/external_embedding_smoke.rs` — OpenAI / Ollama / TEI / custom 4 種類の手動 smoke
- `examples/bench_external_rss.rs` — wiremock in-process で **peak RSS = 25.7 MB** 実測（target <30 MB **MET**）。store p50 0.57 ms / retrieve p50 0.75 ms（ローカルmock基準）

**Phase D — Multi-client + compat layers**:
- `docs/CLIENT_COMPAT.md` 新設 — Claude Code / Cursor / Continue / Aider 4 client の MCP 設定スニペット + per-release verification checklist
- `--anthropic-memory-tool` flag (env `HIPPO_ANTHROPIC_MEMORY_TOOL`) — Anthropic Memory Tool (`memory_20250818`) 互換 surface を `memory` MCP tool として追加。`/memories` filesystem ファサード、6 command (view / create / str_replace / insert / delete / rename) 全実装、path traversal 防御、metadata.`_hippo.memory_tool.path` で round-trip。新規 module `src/memory_tool.rs` + 7 unit tests
- `--shodh-rest` flag (env `HIPPO_SHODH_REST`) + `--shodh-rest-bind` (default `127.0.0.1:8765`) — axum 0.8 ベースの SHODH OpenAPI v1.0.0 REST server。6 endpoint (health / remember / recall / memories / forget/:id / stats) 実装、残り 7 endpoint は 501 Not Implemented + actionable error。新規 module `src/shodh_rest.rs` + 4 axum unit tests
- v0.3 design: 1 process = 1 transport (`--shodh-rest` 設定時は stdio MCP 無効、両用は 2 process で SQLite WAL 共有)。in-process dual-serve は v0.4 候補

### Stats
- 85 tests (71 unit incl. memory_tool + shodh_rest + 4 axum, 9 wiremock embedding, 9 wiremock prediction_loss, 3 integration, 4 eval)
- clippy clean、fmt clean、cargo audit 0 new advisories
- 新規依存: axum 0.8 (REST server)、tower 0.5、wiremock 0.6 (dev-only)
- crates.io: `cargo install claude-hippo` で v0.3.0 公開
- GitHub: https://github.com/abyo-software/claude-hippo/releases/tag/v0.3.0

### Planned (v0.4)
- candle-rs native local prediction-loss backend (no external HTTP service required, GPU 持ち向け)。abyo-llm-probe Stage 2 (Vast.ai 4090) 完走後に判断
- SHODH 残り 7 endpoint (consolidate / by-tags variants / context auto-ingest / per-id GET/PATCH / list tags)
- in-process dual-serve (stdio MCP + SHODH REST 同時起動) — rmcp owned-self serve refactor 後

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
