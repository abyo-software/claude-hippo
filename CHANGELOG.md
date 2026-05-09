# Changelog

すべての変更点はこのファイルに記載します。
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) に従います。

## [Unreleased]

### Added (v0.3 in progress — Phase A + B complete)

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

### Planned (v0.3 残)
- 多 MCP client 動作確認（Cursor / Continue / Aider）
- Anthropic Memory Tool 互換レイヤ
- SHODH OpenAPI REST 互換 endpoint (`--shodh-rest`)

### Planned (v0.4)
- candle-rs native local prediction-loss backend (no external HTTP service required, GPU 持ち向け)。abyo-llm-probe Stage 2 (Vast.ai 4090) 完走後に判断

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
