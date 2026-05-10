# claude-hippo 🦛

> Claude Code に **海馬（hippocampus）** を足す MCP server。
> 全部覚える代わりに、**特異性が高い瞬間だけ** を長期記憶化する surprise-aware memory store。

[![crates.io](https://img.shields.io/crates/v/claude-hippo.svg)](https://crates.io/crates/claude-hippo)
[![Downloads](https://img.shields.io/crates/d/claude-hippo.svg)](https://crates.io/crates/claude-hippo)
[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue)](#license)

```bash
cargo install claude-hippo
hippo serve  # MCP stdio server, ready for Claude Code
```

**v0.5 highlights** (released 2026-05-10):
- `--features candle` で **candle-rs native prediction-loss** が main lib に統合 (v0.4 D-spike を本番化、CPU `candle` / GPU `candle-cuda` 両 flag、Qwen2.5-0.5B 既定)
- **Hebbian associations**: `memory_associations` 別テーブル + 自動 co-recall reinforcement + `recall mode=associative/hybrid` + consolidate edge prune (SHODH `associative recall` 仕様準拠)
- **Semantic clustering**: spherical k-means + `memory_clusters` テーブル + `GET /api/clusters` + consolidate `cluster: true` flag (SHODH consolidate の deferred 全項目 wire 完了)
- Real-backend bench variants (`tests/eval_*_real_*.rs`): FastEmbedder ONNX + CandleLocalPredictionLoss CPU で v0.2-v0.4 fixture を再走 (`#[ignore]` opt-in、release smoke 用)

---

## なぜ「もう一つの memory MCP」が必要か

[mcp-memory-service](https://github.com/doobidoo/mcp-memory-service) (Python)、その Rust port [mcp-memory-service-rs](https://github.com/doobidoo/mcp-memory-service-rs)、その他複数の memory MCP が既にある。claude-hippo は次の 4 軸で差別化する：

| 軸 | claude-hippo | 既存 |
|---|---|---|
| **Surprise-based selection** | embedding outlier + engagement + explicit signal を合成して `surprise_score` を毎回計算、recall でこれを使った時間減衰込み ranking。「あの時の重要な決定」が長期セッションで薄まらない | ❌ ない |
| **Pure Rust 軽量** | 起動 4.5 ms (lazy embedding load)、warm RSS 150 MB、store p50 3.1 ms、retrieve p50 2.7 ms | mcp-memory-service-rs: cold 117 ms / RSS 186 MB / store 5.9 ms / retrieve 6.7 ms |
| **SHODH spec 互換** | mcp-memory-service-rs と完全同一 SQLite schema → **同じ DB ファイルを両者で読み書き可能**。乗り換え無痛 | ✅ mcp-memory-service-rs 系のみ |
| **Apache-2.0 / MIT dual ライセンス** | 商用利用フリー | mcp-memory-service-rs は PolyForm Noncommercial（商用は別契約必要） |

詳細は [PLAN.md](PLAN.md) §3。

---

## ベンチ（Linux x86_64、`scripts/bench_competitor.py --n 100`）

```
| Metric            | mcp-memory-service-rs          | claude-hippo                   |
|-------------------|--------------------------------|--------------------------------|
| cold-start (ms)   |                          117.3 |                            4.5 |
| store p50 (ms)    |                            5.9 |                            3.1 |
| store p95 (ms)    |                            8.1 |                            4.5 |
| retrieve p50 (ms) |                            6.7 |                            2.7 |
| retrieve p95 (ms) |                            8.5 |                            3.5 |
| RSS (MB)          |                          186.3 |                          150.5 |
```

cold-start の 26× 差は claude-hippo が embedding model を **lazy load**（最初の `store_memory` で初めて load する）するため。RSS 19% 減は ranking layer の Arc/Mutex 削減と、KNN over-sample を tombstone がある時だけに限定する最適化（mcp-memory-service-rs と同じ）の純化。

> 自分で再計測：`python3 scripts/bench_competitor.py --n 100`

---

## SHODH DB swap conformance

**証明：** claude-hippo と mcp-memory-service-rs は同じ SQLite ファイルを安全に共有できる。

`python3 scripts/conformance_swap.py` を走らせると：

```
Phase 1: claude-hippo writes 5 memories
Phase 2: mcp-memory-service-rs reads same DB        → 5/5 visible ✓
Phase 3: mcp-memory-service-rs writes 5 more
Phase 4: claude-hippo reads all 10                  → 10/10 visible ✓
Phase 5: claude-hippo semantic recall over mms-written content (cos=0.83) ✓
PASS: SHODH DB swap conformance verified ✓
```

**何を意味するか**：
- 既存の `mcp-memory-service` / `mcp-memory-service-rs` ユーザは binary を入れ替えるだけで claude-hippo に乗り換え可（過去の記憶を全部引き継ぐ）。
- 逆に PolyForm が必要な人は claude-hippo で書いた DB を `mcp-memory-service-rs` で読める。

vector space も `all-MiniLM-L6-v2` を双方使うので一致確認済（手動検証で `[-0.0217, 0.0828, 0.0426, -0.0150, -0.0737]` が両者で一致）。

---

## Claude Code から使う

```bash
cargo install claude-hippo
```

Claude Code の `~/.claude/mcp_servers.json`（または `claude mcp add` 経由）に：

```jsonc
{
  "mcpServers": {
    "hippo": {
      "command": "hippo",
      "args": ["serve"]
    }
  }
}
```

`hippo serve` はデフォルトで `~/.local/share/claude-hippo/memory.db` に保存。`HIPPO_DB_PATH` env で変更可。詳細は [docs/CLAUDE_CODE_SETUP.md](docs/CLAUDE_CODE_SETUP.md)。

---

## MCP tools

### claude-hippo native (5)

| Tool | 用途 |
|---|---|
| `hippo_remember(content, tags?, memory_type?, importance?, metadata?)` | 記憶保存 + surprise score 算出 |
| `hippo_recall(query, limit=10, no_surprise_boost=false)` | semantic search + surprise-weighted ranking |
| `hippo_list_recent(n=20)` | 直近 N 件 |
| `hippo_forget(content_hash?, id?, dry_run=false)` | soft-delete |
| `hippo_session_summary(hours=24)` | 直近活動の by_type / top_tags / highlights / mean surprise |

### SHODH alias (4) — 既存クライアント設定無変更で乗り換え可

`store_memory` / `retrieve_memory` / `list_memories` / `delete_memory` は対応する hippo_* に転送する。`mcp-memory-service-rs` と同 wire format。

### `ping` — health probe（auth 不要）

---

## Surprise scoring とは

```rust
surprise = 0.4 * embedding_outlier   // 既存記憶ベクトル群からの cosine 距離
         + 0.2 * engagement          // content 長 + tag 数の飽和関数
         + 0.1 * explicit            // ユーザの importance flag
         + 0.3 * prediction_loss     // (将来) abyo-llm-probe で LLM の驚き
```

`prediction_loss` が None（v0.1）の時は重みを `embedding_outlier` と `engagement` に按分。

recall ranking はこれと cosine sim と時間減衰の合成：

```rust
score = 0.7 * cos_sim
      + 0.3 * surprise * exp(-age_days * ln(2) / 30 days)
```

つまり **古くても surprise が高ければ浮かぶ、新しくても陳腐ならランクが下がる**。`no_surprise_boost=true` で純粋な vector similarity に戻せる。

理論本体は [docs/SURPRISE_SELECTION.md](docs/SURPRISE_SELECTION.md)。

---

## CLI

```bash
hippo serve [--db PATH] [--model-cache DIR] \
            [--surprise-weights "0.4,0.2,0.1,0.3"] \
            [--embedding-model {minilm-l6-v2|bge-small-en-v15-q}]   # MCP stdio (default)
hippo verify [--db PATH]                       # schema apply + sqlite-vec 確認
hippo embed "text" [--embedding-model X]       # 埋め込み単発、smoke test
hippo bench --n 100 [--surprise-weights X] [--embedding-model X]    # 自前 self-bench
```

env：
- `HIPPO_DB_PATH` — SQLite path（default `~/.local/share/claude-hippo/memory.db`）
- `HIPPO_MODEL_CACHE` — embedding model dir（default `~/.cache/claude-hippo/models/`）
- `HIPPO_SURPRISE_WEIGHTS` — `"w_outlier,w_engagement,w_explicit,w_prediction"`、合計 1.0 (±1e-3)。default `"0.4,0.2,0.1,0.3"`
- `HIPPO_EMBEDDING_MODEL` — `minilm-l6-v2` (default、SHODH DB swap 互換) or `bge-small-en-v15-q` (量子化、384 dim 維持)
- `RUST_LOG` — `tracing-subscriber` フィルタ

---

## ストレージレイアウト

```sql
-- mcp-memory-service-rs と verbatim 一致
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash TEXT UNIQUE NOT NULL,  -- SHA-256(content)
    content TEXT NOT NULL,
    tags TEXT,                          -- "a,b,c" comma-joined, 空白保持
    memory_type TEXT,                   -- "Decision" / "Discovery" / etc (SHODH MemoryType)
    metadata TEXT,                      -- JSON; claude-hippo は _hippo namespace を予約
    created_at REAL,
    updated_at REAL,
    created_at_iso TEXT,
    updated_at_iso TEXT,
    deleted_at REAL DEFAULT NULL        -- soft-delete tombstone
);
CREATE VIRTUAL TABLE memory_embeddings USING vec0(content_embedding FLOAT[384] distance_metric=cosine);
```

`metadata` 中の `_hippo.surprise.{score, components, version}` が claude-hippo の差別化情報。mcp-memory-service-rs はこれを未知 key として無害に無視する。詳細：[docs/SHODH_COMPAT.md](docs/SHODH_COMPAT.md)。

---

## アーキテクチャ

```text
┌────────────┐     stdio JSON-RPC     ┌─────────────────────┐
│ Claude Code│  ◀───────────────────▶ │ hippo serve         │
│ (MCP host) │                        │  ┌───────────────┐  │
└────────────┘                        │  │ rmcp router   │  │  ← MCP SDK
                                      │  └───────┬───────┘  │
                                      │  ┌───────▼───────┐  │
                                      │  │ MemoryServer  │  │  ← src/server.rs
                                      │  │ (5 hippo_*    │  │
                                      │  │  tools +      │  │
                                      │  │  SHODH alias) │  │
                                      │  └───────┬───────┘  │
                                      │   ┌──────┴──────┐    │
                                      │   ▼             ▼    │
                                      │ surprise    embeddings│
                                      │ scoring     (fastembed│
                                      │ + decay     /MiniLM   │
                                      │             ONNX)     │
                                      │   │             │    │
                                      │   ▼             ▼    │
                                      │  ┌────────────────┐  │
                                      │  │ Storage        │  │  ← src/storage.rs
                                      │  │ rusqlite +     │  │
                                      │  │ sqlite-vec     │  │
                                      │  │ (SHODH schema) │  │
                                      │  └────────────────┘  │
                                      │      │ │             │
                                      │      ▼ ▼             │
                                      │  memory.db (.wal)    │
                                      └──────────────────────┘
```

詳細は [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

---

## ステータス

**v0.2** (2026-05-10 release): production-ready as a drop-in for `mcp-memory-service-rs` with surprise scoring + **独自評価ベンチ Bench A/B/C 実数値あり**。

| Bench | 効果 | 詳細 |
|---|---|---|
| A: Long-session noise | precision@1 を **8% → 72%** (既定) / **100%** (full oversample) | 100 items, 25 paraphrased queries |
| B: Cross-session retrieval (forgetting curve) | 30 日決定は perfect、365 日で **負の lift** (honestly 公開) | 50 items × 4 ages |
| C: Decision trace | recall@5 を **0.44 → 0.97** (既定 weights) / **1.00** (`--surprise-weights "0.2,0.1,0.5,0.2"`) | 4 Decisions in 20 mixed |

40 unit + 3 integration + 3 eval = **46 tests** all pass. DB swap conformance も pass。詳細は [docs/SURPRISE_SELECTION.md](docs/SURPRISE_SELECTION.md)。

将来：
- v0.3: abyo-llm-probe 統合（`prediction_loss` を埋める）+ abyo-filters 内蔵 + External embedding API backend (`docs/EXTERNAL_EMBEDDING.md` 設計済) + decay floor / `--half-life-days` CLI（Bench B 365 日 demotion 対処）+ Anthropic Memory Tool 互換レイヤ + 多 MCP client（Cursor / Continue / Aider）対応
- v1.x: SaaS マネタイズ層（Cloudflare 同期 etc）

開発計画は [PLAN.md](PLAN.md)、変更履歴は [CHANGELOG.md](CHANGELOG.md)。

---

## ベンチを再現する

前提：`~/git/mcp-memory-service-rs` に competitor を clone + `cargo build --release`。
HF mirror から ONNX model を `~/.cache/mcp_memory/onnx_models/all-MiniLM-L6-v2/onnx/` に配置。

```bash
# self-bench
cargo build --release
target/release/hippo bench --n 100

# vs competitor (head-to-head, MCP stdio)
python3 scripts/bench_competitor.py --n 100

# SHODH DB swap conformance
python3 scripts/conformance_swap.py
```

---

## 開発

```bash
cargo test --lib                # unit (40)
cargo test --release            # + integration (3) + eval (3) — eval は MockEmbedder で ONNX 不要
cargo test --release --test eval_a_long_session    # Bench A 単発
cargo test --release --test eval_b_cross_session   # Bench B 単発
cargo test --release --test eval_c_decision_trace  # Bench C 単発
ls target/eval_results/          # bench_*.json が生成される
cargo clippy --all-targets -- -D warnings
cargo fmt --check
cargo audit                      # cargo install --locked cargo-audit が必要
```

---

## License

Dual-licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option. Commercial use is unrestricted under either license.

---

## クレジット

- 互換 schema は [doobidoo/mcp-memory-service-rs](https://github.com/doobidoo/mcp-memory-service-rs) と [doobidoo/mcp-memory-service](https://github.com/doobidoo/mcp-memory-service) から拝借（Apache-2.0、PolyForm Noncommercial）
- SHODH spec は [varun29ankuS/shodh-memory](https://github.com/varun29ankuS/shodh-memory)
- MCP は [Anthropic Model Context Protocol](https://modelcontextprotocol.io)
- 埋め込みは [Anush008/fastembed-rs](https://github.com/Anush008/fastembed-rs) + [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

— [abyo software, LLC](https://github.com/abyo-software)
