# claude-hippo 🦛

> A **hippocampus for Claude Code** — an MCP server that doesn't try to remember everything, but learns which moments are *worth* remembering.

[![crates.io](https://img.shields.io/crates/v/claude-hippo.svg)](https://crates.io/crates/claude-hippo)
[![Downloads](https://img.shields.io/crates/d/claude-hippo.svg)](https://crates.io/crates/claude-hippo)
[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue)](#license)
[![CI](https://github.com/abyo-software/claude-hippo/actions/workflows/ci.yml/badge.svg)](https://github.com/abyo-software/claude-hippo/actions/workflows/ci.yml)

🇬🇧 English: **this page** | 🇯🇵 日本語: [README.ja.md](README.ja.md)

```bash
cargo install claude-hippo
hippo serve  # MCP stdio server, ready for Claude Code
```

---

## Why claude-hippo

Most memory MCP servers store every message and rely on cosine similarity at recall time. That works until your sessions get long — at which point the *important decision* you made last month is buried under 500 routine chat messages with similar wording.

claude-hippo borrows the **surprise-driven consolidation** mechanism from neuroscience: only memories that were *surprising* at encoding time get a long-term ranking boost, and that boost decays gracefully so old-but-pivotal items stay competitive against fresh-but-mundane ones.

| Differentiator | claude-hippo | Typical memory MCP |
|---|---|---|
| **Surprise-based selection** | `surprise = embedding_outlier·0.4 + engagement·0.2 + explicit·0.1 + prediction_loss·0.3`, blended into recall ranking with exponential decay (`half_life=30d`, `decay_floor=0.5`) | Pure cosine similarity |
| **Pure-Rust, lightweight** | cold-start 4.5 ms (lazy embedding load), warm RSS 150 MB, store p50 3.1 ms, retrieve p50 2.7 ms | mcp-memory-service-rs: cold 117 ms / RSS 186 MB |
| **Drop-in DB compat** | Identical SQLite schema to [`mcp-memory-service-rs`](https://github.com/doobidoo/mcp-memory-service-rs) — the same `.db` file works in either binary | Locked into one implementation |
| **Apache-2.0 / MIT** | Commercial use unrestricted | Often PolyForm Noncommercial or AGPL |

Full positioning: [PLAN.md §3](PLAN.md).

---

## v0.5 highlights (released 2026-05-10)

- **`--features candle`**: native pure-Rust LLM prediction-loss via [candle-rs](https://github.com/huggingface/candle). CPU (`candle`) and GPU (`candle-cuda`) flags. Default install stays lean (~25 MB binary, no candle compile).
- **Hebbian associations**: a separate `memory_associations` table grows co-recall edges automatically; `recall mode=associative|hybrid` follows them; `consolidate` decays/prunes them. Implements the SHODH spec's `associative recall`.
- **Semantic clustering**: spherical k-means over alive embeddings, persisted in `memory_clusters`; `GET /api/clusters` exposes the structure; `consolidate { cluster: true }` rewrites cluster assignments. Implements the SHODH spec's `semantic clustering and compression`.
- **Real-backend bench variants**: `tests/eval_real_local.rs` (FastEmbedder ONNX) and `tests/eval_d_real_candle.rs` (Qwen2.5-0.5B CPU). Gated `#[ignore]` so CI stays fast; release smoke runs them.
- **SHODH OpenAPI v1.0.0**: all 14 endpoints (13 spec + `GET /api/clusters`) wired, `consolidate.deferred[]` is now empty.

See [CHANGELOG.md](CHANGELOG.md) for the full history.

---

## Install & wire it into Claude Code

```bash
# Default install (lean, MCP stdio + SHODH REST)
cargo install claude-hippo

# With candle-rs native prediction-loss (CPU)
cargo install claude-hippo --features candle

# With CUDA acceleration (requires cuDNN system install)
cargo install claude-hippo --features candle-cuda
```

In `~/.claude/mcp_servers.json` (or via `claude mcp add`):

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

By default, the SQLite database lives at `~/.local/share/claude-hippo/memory.db` (override with `HIPPO_DB_PATH`). Full client setup details: [docs/CLAUDE_CODE_SETUP.md](docs/CLAUDE_CODE_SETUP.md). Cursor / Continue / Aider snippets: [docs/CLIENT_COMPAT.md](docs/CLIENT_COMPAT.md).

---

## MCP tools

### claude-hippo native (5)

| Tool | Purpose |
|---|---|
| `hippo_remember(content, tags?, memory_type?, importance?, metadata?)` | Store memory + compute surprise score |
| `hippo_recall(query, limit=10, mode=?, seed_id=?, no_surprise_boost=false)` | Semantic / associative / hybrid recall with surprise-weighted ranking |
| `hippo_list_recent(n=20)` | Most recent N (alive only) |
| `hippo_forget(content_hash?, id?, dry_run=false)` | Soft-delete |
| `hippo_session_summary(hours=24)` | Activity rollup: by_type / top_tags / highlights / mean surprise |

### SHODH-spec aliases (4)

`store_memory` / `retrieve_memory` / `list_memories` / `delete_memory` forward to the corresponding `hippo_*`. Existing SHODH clients can swap binaries without config changes.

### REST surface (`--shodh-rest`)

All 14 SHODH OpenAPI v1.0.0 endpoints + `GET /api/clusters`. Coexists with MCP stdio in the same process.

---

## Surprise scoring, plain English

Every memory gets a score at write time:

```
surprise = 0.4 · embedding_outlier   // cosine distance from existing memories
         + 0.2 · engagement          // saturating function of length + tag count
         + 0.1 · explicit            // user-marked importance
         + 0.3 · prediction_loss     // LLM NLL (with `--features candle`) or external HTTP backend
```

Recall then ranks by:

```
score = 0.7 · cosine_similarity
      + 0.3 · surprise · max(exp(-age_days · ln(2) / half_life), decay_floor)
```

Net effect: **old-but-surprising stays visible, new-but-mundane sinks**. `no_surprise_boost=true` falls back to pure cosine. Theory and evaluation: [docs/SURPRISE_SELECTION.md](docs/SURPRISE_SELECTION.md).

---

## CLI

```bash
hippo serve [--db PATH] [--shodh-rest] [--shodh-rest-bind 127.0.0.1:8765] \
            [--prediction-loss-backend {none|openai-compat|candle-local}] \
            [--candle-model-id Qwen/Qwen2.5-0.5B] [--candle-cpu] \
            [--no-hebbian-reinforce] [--co-recall-alpha 0.1] \
            [--surprise-weights "0.4,0.2,0.1,0.3"] \
            [--half-life-days 30] [--decay-floor 0.5] [--oversample-factor 6]
hippo verify [--db PATH]
hippo embed "text" [--embedding-model {minilm-l6-v2|bge-small-en-v15-q}]
hippo bench --n 100 [...same flags as serve...]
```

Environment variables mirror every `--flag` (`HIPPO_DB_PATH`, `HIPPO_SHODH_REST`, `HIPPO_CANDLE_CPU`, `HIPPO_CO_RECALL_ALPHA`, etc.). `RUST_LOG=hippo=debug` for verbose tracing.

---

## Storage layout

```sql
-- Verbatim-compatible with mcp-memory-service-rs
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash TEXT UNIQUE NOT NULL,   -- SHA-256
    content TEXT NOT NULL,
    tags TEXT,                           -- comma-joined
    memory_type TEXT,                    -- "Decision" / "Discovery" / etc
    metadata TEXT,                       -- JSON; claude-hippo reserves _hippo namespace
    created_at REAL, updated_at REAL,
    created_at_iso TEXT, updated_at_iso TEXT,
    deleted_at REAL DEFAULT NULL         -- soft-delete tombstone
);
CREATE VIRTUAL TABLE memory_embeddings USING vec0(content_embedding FLOAT[384] distance_metric=cosine);

-- v0.5 additions (ignored by mcp-memory-service-rs, drop-in swap preserved)
CREATE TABLE memory_associations (from_id, to_id, weight, last_reinforced, PRIMARY KEY (from_id, to_id));
CREATE TABLE memory_clusters     (id INTEGER PRIMARY KEY AUTOINCREMENT, centroid_blob BLOB, size, last_recomputed);
```

The `metadata._hippo.{surprise, cluster_id}` namespace is what claude-hippo writes; mcp-memory-service-rs treats it as an unknown key and ignores it harmlessly. Full DB-swap rationale: [docs/SHODH_COMPAT.md](docs/SHODH_COMPAT.md).

---

## Benchmarks

Head-to-head on Linux x86_64 (`scripts/bench_competitor.py --n 100`):

| Metric | mcp-memory-service-rs | claude-hippo |
|---|---:|---:|
| cold-start (ms) | 117.3 | **4.5** |
| store p50 (ms) | 5.9 | **3.1** |
| store p95 (ms) | 8.1 | **4.5** |
| retrieve p50 (ms) | 6.7 | **2.7** |
| retrieve p95 (ms) | 8.5 | **3.5** |
| RSS (MB) | 186.3 | **150.5** |

Cold-start improvement comes from **lazy embedding load**; the first `store_memory` pays the model load, subsequent operations are warm.

### Surprise-rerank evaluation (mock fixture, deterministic)

| Bench | Headline | Detail |
|---|---|---|
| A — Long-session noise | precision@1: **0.08 → 1.00** (default oversample=6) | 100 items / 25 paraphrased queries |
| B — Cross-session decay | 365-day decision: **negative lift → +0.875 lift** with `decay_floor=0.5` | 50 items × 4 ages |
| C — Decision trace | recall@5: **0.44 → 1.00** with `--surprise-weights "0.2,0.1,0.5,0.2"` | 4 Decisions among 20 mixed |
| D — Prediction-loss wiring | 100/100 coverage with mock backend; real candle-local gated `#[ignore]` | Qwen2.5-0.5B CPU |

Real-backend variants (Bench A/B/C with FastEmbedder ONNX, Bench D with candle CPU) are gated `#[ignore]` so the default `cargo test` stays fast — run them at release time. Full disclosure of mock-vs-real tradeoffs: [docs/SURPRISE_SELECTION.md](docs/SURPRISE_SELECTION.md).

### SHODH DB-swap conformance

```
$ python3 scripts/conformance_swap.py
Phase 1: claude-hippo writes 5 memories
Phase 2: mcp-memory-service-rs reads same DB        → 5/5 visible ✓
Phase 3: mcp-memory-service-rs writes 5 more
Phase 4: claude-hippo reads all 10                  → 10/10 visible ✓
Phase 5: claude-hippo semantic recall over mms-written content (cos=0.83) ✓
PASS: SHODH DB swap conformance verified ✓
```

---

## Architecture

```text
┌────────────┐    stdio JSON-RPC    ┌──────────────────────────────┐
│ Claude Code│ ◀──────────────────▶ │ hippo serve                  │
│ (MCP host) │                      │  ┌────────────────────────┐  │
└────────────┘                      │  │ rmcp router            │  │  ← MCP SDK
                                    │  └───────────┬────────────┘  │
                                    │              ▼               │
                                    │  ┌────────────────────────┐  │
                                    │  │ MemoryServer           │  │  ← src/server.rs
                                    │  │ (5 native + 4 SHODH    │  │
                                    │  │  aliases + ping)       │  │
                                    │  └──┬──────┬──────────┬───┘  │
                                    │     ▼      ▼          ▼      │
                                    │ surprise  embeddings  prediction-loss
                                    │ scoring   (fastembed  (external HTTP
                                    │ + decay   /MiniLM)    or candle-rs)
                                    │     │      │          │      │
                                    │     ▼      ▼          ▼      │
                                    │  ┌────────────────────────┐  │
                                    │  │ Storage (rusqlite +    │  │  ← src/storage.rs
                                    │  │ sqlite-vec)            │  │
                                    │  │  - memories            │  │
                                    │  │  - memory_embeddings   │  │
                                    │  │  - memory_associations │  │ ← v0.5
                                    │  │  - memory_clusters     │  │ ← v0.5
                                    │  └────────────────────────┘  │
                                    │            │                 │
                                    │            ▼                 │
                                    │      memory.db (.wal)        │
                                    │                              │
                                    │  Optional: SHODH REST :8765  │
                                    │  (14 endpoints, --shodh-rest)│
                                    └──────────────────────────────┘
```

Full design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Status & roadmap

**v0.5.0** is production-ready as a drop-in for `mcp-memory-service-rs`, with all SHODH OpenAPI v1.0.0 spec items wired (`consolidate.deferred[]` is empty). 129 tests pass + 5 ignored real-backend gates; clippy + fmt clean.

**Planned for v0.6**:
- candle Phi-3 / Llama-3 family support + all-position-logits patch (faster scoring on long content)
- 2-hop associative recall (subgraph traversal)
- Cluster-aware archival (auto-archive redundant items in dense clusters)
- candle-cuda real-GPU latency measurements
- abyo-llm-probe Stage 2 verdict (large-model NLL gradients on Vast.ai 4090)

Plan: [PLAN.md](PLAN.md). Honest disclosures (what's still deferred and why): [CHANGELOG.md](CHANGELOG.md) §Honest disclosures.

---

## Contributing

Issues and pull requests welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow, coding conventions, and how to run the bench harness. Security reports: [SECURITY.md](SECURITY.md). Community standards: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

```bash
cargo test                                       # 129 pass + 3 ignored (default)
cargo test --features candle                     # 132 pass + 5 ignored
cargo clippy --features candle --all-targets -- -D warnings
cargo fmt --check
cargo audit
```

---

## License

Dual-licensed under either of:

- **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- **MIT License** ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option. Commercial use is unrestricted under either license. Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project shall be dual-licensed as above, without any additional terms or conditions.

---

## Credits

- The compatible SQLite schema is borrowed from [doobidoo/mcp-memory-service-rs](https://github.com/doobidoo/mcp-memory-service-rs) and [doobidoo/mcp-memory-service](https://github.com/doobidoo/mcp-memory-service) (Apache-2.0 / PolyForm Noncommercial respectively)
- The SHODH OpenAPI spec is from [varun29ankuS/shodh-memory](https://github.com/varun29ankuS/shodh-memory)
- MCP is the [Anthropic Model Context Protocol](https://modelcontextprotocol.io)
- Embeddings via [Anush008/fastembed-rs](https://github.com/Anush008/fastembed-rs) + [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- Native LLM scoring via [huggingface/candle](https://github.com/huggingface/candle) and [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)

— [abyo software, LLC](https://github.com/abyo-software)
