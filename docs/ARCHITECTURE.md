# Architecture

## モジュール

```
src/
├── lib.rs           — pub re-exports + crate-level docs
├── main.rs          — entry, hand off to cli::run()
├── cli.rs           — clap derive (serve / verify / embed / bench)
├── error.rs         — thiserror enum HippoError + Result alias
├── storage.rs       — SQLite + sqlite-vec, MemoryRow, schema, KNN, soft-delete
├── embeddings.rs    — Embedder trait, FastEmbedder (lazy load), MockEmbedder
├── surprise.rs      — SurpriseComponents, score(), embedding_outlier(), engagement(), decay(), ranking()
└── server.rs        — rmcp tool router, 5 hippo_* tools + 4 SHODH alias + ping
```

## データフロー

### `hippo_remember` flow

```
  user input (content, tags, importance?)
        │
        ▼
  ┌─────────────┐
  │ embedder    │ embed_one(content)  → Vec<f32> (384, L2 normalized)
  │ (lazy load) │
  └──────┬──────┘
         │
         ▼
  ┌─────────────────────┐
  │ history_embeddings  │ direct from DB: 直近 50 件の embedding
  │ (storage helper)    │
  └──────┬──────────────┘
         │
         ▼
  ┌─────────────────────┐
  │ surprise::score()   │ outlier(0.4) + engagement(0.2) + explicit(0.1) + pred(0.3*)
  │                     │ * pred は v0.1 では None、redistribute される
  └──────┬──────────────┘
         │
         ▼
  ┌─────────────────────┐
  │ attach_surprise()   │ metadata._hippo.surprise = {score, components, version}
  └──────┬──────────────┘
         │
         ▼
  ┌─────────────────────┐
  │ Storage::insert()   │ INSERT memories + INSERT memory_embeddings (vec0)
  │  (transactional)    │ dedup by content_hash UNIQUE
  └──────┬──────────────┘
         │
         ▼
  RememberResult { success, id, content_hash, duplicate, surprise_score, surprise_components }
```

### `hippo_recall` flow

```
  user query (string, limit=10)
        │
        ▼
  ┌─────────────┐
  │ embedder    │ embed_one(query) → Vec<f32>
  └──────┬──────┘
         │
         ▼
  ┌──────────────────────┐
  │ Storage::knn()       │ vec0 KNN, fetch_k = 3*k (oversample for surprise rerank)
  │  - tombstone-aware   │ tombstone なければ k だけ
  │    oversample        │
  └──────┬───────────────┘
         │
         ▼  for each (id, distance)
  ┌──────────────────────┐
  │ Storage::get_by_id() │ 行取得 + metadata parse
  └──────┬───────────────┘
         │
         ▼
  ┌──────────────────────┐
  │ surprise::ranking()  │ 0.7*cos_sim + 0.3*surprise_score*decay(age, 30d)
  │                      │ no_surprise_boost=true なら cos_sim のみ
  └──────┬───────────────┘
         │
         ▼  rerank by score, truncate to k
  Vec<RecalledMemory> { memory, score, cosine_similarity, surprise_score? }
```

## 同時実行モデル

- **`Storage`** は `Arc<tokio::sync::Mutex<Storage>>` で共有。SQLite はそもそも writer 1 本前提なので Mutex で十分。
- **`Embedder`** は `Arc<dyn Embedder>`、内部で `parking_lot::Mutex<Option<TextEmbedding>>`（fastembed が `&mut self` を要求するため）。
- `tokio::sync::Mutex` を `await` 越えで持ち続けるのは avoidable な所では `parking_lot` に切り替えてある。

## Lazy embedding load の意義

`hippo serve` 起動時：
- DB open + schema apply: `~5 ms`
- `FastEmbedder::new(cache_dir)`: ディレクトリ作成のみ、`~0 ms`
- rmcp serve handshake: `~1-2 ms`
- **合計 cold-start `~5 ms`** （vs mcp-memory-service-rs 117 ms）

最初の `hippo_remember` または `hippo_recall` で：
- ONNX session 構築 + tokenizer load: `~100-120 ms`（warm cache 前提）
- model.onnx mmap: 数 MB/ms

これにより MCP host (Claude Code) が server を起動した直後はまだ embedding を読まないので `Tools/list` 等が爆速で返る。

## RSS の内訳

| 段階 | RSS |
|---|---|
| `hippo verify` (DB のみ) | 23 MB |
| `hippo serve` startup（embed 未呼び出し）| ~25 MB |
| First embed call 後 | 165 MB |
| 100 store + 100 retrieve 後 | 150-186 MB |

ONNX Runtime 系（ort_sys + 周辺）が ~150 MB を占有。<50 MB を取るには external API 経由の `--embedding-backend external` モード（v0.2 予定）か、completely lazy unload + LRU process eviction が必要。

## SHODH 互換性レイヤ

claude-hippo は SHODH spec の **schema fields** に互換するが、**REST API** は提供しない（MCP 専用）。

- 11 列 SQLite schema は mcp-memory-service-rs と verbatim 一致
- SHODH の rich field（emotional valence, episode_id, source_type 等）は `metadata` JSON 列に詰める
- claude-hippo の差別化情報（surprise score）は `metadata._hippo` namespace に格納、他実装は無害に無視

詳細：[SHODH_COMPAT.md](SHODH_COMPAT.md)

## エラー処理

- crate 内の Result は `crate::Result<T> = std::result::Result<T, HippoError>`
- HippoError は thiserror で 9 variant：Io / Sqlite / Json / Schema / Embedding / Config / NotFound / Invalid / Integrity
- MCP server boundary では `internal_err()` / `invalid_input()` で `rmcp::ErrorData` に変換
- main では `anyhow::Result` で `?` 連鎖

## テスト戦略

| Layer | Where | What |
|---|---|---|
| Unit | `src/*.rs` `#[cfg(test)]` | storage 8 + surprise 8 + embeddings 5 + server 4 |
| Integration | `tests/mcp_stdio.rs` | binary spawn + JSON-RPC handshake + 全 tool E2E |
| Conformance | `scripts/conformance_swap.py` | claude-hippo ↔ mcp-memory-service-rs DB swap |
| Bench | `scripts/bench_competitor.py` | 両 binary 同条件でレイテンシ + RSS |

すべて pass を保つ。CI は `.github/workflows/ci.yml`。

## 拡張ポイント（v0.2+）

- `Embedder` trait: External API 実装、量子化 ONNX 実装、candle 実装が plug-in 可能
- `SurpriseWeights`: API で外から渡せるように
- `MemoryServer`: `SurpriseWeights` を受け取り override できるように
- Server alias 名は v0.3 で `--shodh-only` / `--hippo-only` で絞れるように
- v0.3 で SHODH OpenAPI REST 互換 layer を `--shodh-rest` で expose
