# SHODH 互換ガイド

> Status: v0.1 spike notes（2026-05-09）
> Source: [varun29ankuS/shodh-memory openapi.yaml](https://github.com/varun29ankuS/shodh-memory/blob/main/specs/openapi.yaml) v1.0.0
> Reference impl: [doobidoo/mcp-memory-service-rs/src/storage.rs](https://github.com/doobidoo/mcp-memory-service-rs/blob/main/src/storage.rs)

## TL;DR

「SHODH 互換」と一口に言っても、**仕様書（OpenAPI v1.0.0）** と **実装の最大公約数（mcp-memory-service-rs の SQLite schema）** には乖離がある。claude-hippo は **schema 互換層 → wire 互換層** の二段で実装する：

- **v0.1**：mcp-memory-service-rs と完全同一の 11-column SQLite schema を採用 → 同 DB ファイルを両者で読み書き可能（drop-in swap）
- **v0.3**：SHODH OpenAPI v1.0.0 の rich field（emotional / episodic / source）を `metadata` JSON 列に格納する形で吸収 → REST 互換層 `hippo_shodh_serve` を追加

## 1. SHODH OpenAPI v1.0.0 — 仕様面

### Memory schema（13 セクション + impl-specific）

```yaml
Memory:
  required: [id, content, content_hash, created_at]

  # Identification
  id:                 uuid
  content:            string
  content_hash:       sha256

  # Classification
  type:               MemoryType  # 13 enum (下記)
  tags:               [string]

  # Source & Trust
  source_type:        SourceType  # 7 enum
  credibility:        float (0..1, default 1.0)

  # Emotional Metadata
  emotion:            string (joy/frustration/surprise/relief/curiosity/...)
  emotional_valence:  float (-1..1)
  emotional_arousal:  float (0..1)

  # Episodic Memory
  episode_id:         string
  sequence_number:    integer

  # Timestamps
  created_at:         date-time
  updated_at:         date-time
  last_accessed_at:   date-time

  # Quality & Access
  quality_score:      float (0..1)
  access_count:       integer

  # Implementation-specific
  embedding:          [float]
  metadata:           object (free-form)
```

### MemoryType（13 enum）
`Observation` / `Decision` / `Learning` / `Error` / `Discovery` / `Pattern` / `Context` / `Task` / `CodeEdit` / `FileAccess` / `Search` / `Command` / `Conversation`

### SourceType（7 enum）
`user` / `system` / `api` / `file` / `web` / `ai_generated` / `inferred`

### Endpoint surface（13 paths）
| メソッド | path | 用途 |
|---|---|---|
| POST | /api/remember | 新規記憶保存 |
| POST | /api/recall | semantic search（mode: semantic/associative/hybrid） |
| POST | /api/recall/by-tags | タグ AND-OR 検索 |
| POST | /api/context | proactive context surfacing（auto-ingest 込み） |
| DELETE | /api/forget/{id} | 単発削除（仕様上は physical delete） |
| POST | /api/forget/by-tags | 一括削除 |
| GET | /api/memories | ページング一覧 |
| GET | /api/memories/{id} | 単発取得 |
| PATCH | /api/memories/{id} | metadata 更新（content は変えない） |
| POST | /api/consolidate | exponential decay + association discovery |
| GET | /api/stats | 集計統計 |
| GET | /api/health | ヘルスチェック（auth 不要） |
| GET | /api/tags | 全タグ列挙 |

### Recall mode の `associative` は Hebbian
> Hebbian memory edges を follow する。記憶間の連想関係グラフを別途持つ前提。claude-hippo v0.1 では未対応、v0.3 でオプション。

### Consolidation
- exponential decay scoring
- association discovery between memories
- semantic clustering and compression
- quality-based archival of low-value memories

これは claude-hippo の **forgetting curve / surprise-weighted ranking** と概念一致。SHODH の `consolidate` エンドポイントを我々の decay 機構の wire interface として再利用できる。

## 2. mcp-memory-service-rs — 実装面

### 実 SQLite schema（[storage.rs:86-98](https://github.com/doobidoo/mcp-memory-service-rs/blob/main/src/storage.rs)）

```sql
CREATE TABLE IF NOT EXISTS memories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash    TEXT UNIQUE NOT NULL,
    content         TEXT NOT NULL,
    tags            TEXT,
    memory_type     TEXT,
    metadata        TEXT,            -- JSON 文字列
    created_at      REAL,
    updated_at      REAL,
    created_at_iso  TEXT,
    updated_at_iso  TEXT,
    deleted_at      REAL DEFAULT NULL  -- soft-delete tombstone
);

CREATE INDEX idx_content_hash ON memories(content_hash);
CREATE INDEX idx_created_at  ON memories(created_at);
CREATE INDEX idx_memory_type ON memories(memory_type);
CREATE INDEX idx_deleted_at  ON memories(deleted_at);

CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE VIRTUAL TABLE memory_embeddings USING vec0(
    content_embedding FLOAT[384] distance_metric=cosine
);
```

注目点：

1. `id` は **INTEGER AUTOINCREMENT**（SHODH spec の UUID とは違う）。`content_hash` UNIQUE で実質 idempotency 担保
2. **embedding 次元 384**（all-MiniLM-L6-v2 固定）
3. **soft-delete 標準採用**（`deleted_at REAL`）。SHODH spec の DELETE エンドポイントは「permanently deletes」と書いてあるが、実装は tombstone を残す
4. SHODH の rich field（emotional / episodic / source / quality / access_count / last_accessed_at）は **存在しない** → `metadata` JSON 列に詰める設計と推察
5. コメント「Schema MUST match the upstream Python `SqliteVecMemoryStorage` exactly so both backends can share a DB file」 → **drop-in swap が明示的設計目標**

### 7 つの MCP tools（mcp-memory-service-rs）
- `store_memory`
- `retrieve_memory`
- `search_by_tag`
- `delete_memory`
- `list_memories`
- `check_database_health`
- `get_cache_stats`

claude-hippo の v0.1 5-tools と概ね対応するが、命名が異なる。**互換層 `--shodh-tool-names` フラグを v0.3 で提供** して MCP client から見たツール名を揃えると、設定ファイル変更不要で乗り換えられる。

## 3. claude-hippo の互換戦略

### v0.1（Sprint S1）：schema swap-compatible

mcp-memory-service-rs の 11-column schema を **完全コピー**で採用する。

我々の差別化機能（surprise score）は **`metadata` JSON 列に格納**：

```sql
-- 例
INSERT INTO memories (content_hash, content, tags, memory_type, metadata, ...)
VALUES (
  'sha256...',
  'Use JWT tokens with 24h expiry',
  '["auth","security"]',
  'Decision',
  '{"surprise_score":0.87,"surprise_components":{"prediction_loss":null,"embedding_outlier":0.42,"engagement":0.30,"explicit":0.15},"hippo_version":"0.1.0"}',
  ...
);
```

利点：
- mcp-memory-service-rs / mcp-memory-service の DB を `~/.local/share/mcp-memory/sqlite.db` に置いて hippo serve すれば、過去の記憶を全部そのまま読める
- 逆も真：claude-hippo で書いた DB を mcp-memory-service が読める（surprise_score は metadata の中の不明 key として無害に無視される）
- 「乗り換え無痛」が技術的事実として成立

欠点：
- `metadata->>'$.surprise_score' DESC` でしか並べられない → JSON ops のオーバーヘッド
- 将来的に hot path なら `surprise_score REAL` 列を ALTER TABLE で追加可（後方互換、他実装は無害に無視）

### v0.2（Sprint S2）：surprise score を「実質列」化

JSON ops 性能が問題になったら：

```sql
ALTER TABLE memories ADD COLUMN surprise_score REAL DEFAULT NULL;
CREATE INDEX idx_surprise_score ON memories(surprise_score);
```

mcp-memory-service-rs の `CREATE TABLE IF NOT EXISTS` は no-op なので追加列は破壊しない。彼らの SELECT は明示列指定なので「未知列」を読まない。**双方向互換維持。**

### v0.3（Sprint S3）：SHODH OpenAPI v1.0.0 wire 準拠

オプショナル機能として REST サーバを追加：

```bash
hippo serve --shodh-rest --port 8000   # SHODH OpenAPI 互換 REST
hippo serve                            # MCP stdio（デフォルト）
```

REST レイヤは MCP tools のラッパー：
- `POST /api/remember` → `hippo_remember`
- `POST /api/recall` → `hippo_recall`（mode 引数で `semantic`/`associative`/`hybrid` を分岐、associative は v0.3 の新機能）
- `POST /api/consolidate` → forgetting curve 実行

rich field は metadata JSON に格納したまま、API レイヤで de/serialize。

### 「SHODH 公式準拠」を主張する条件

[SHODH discussions](https://github.com/varun29ankuS/shodh-memory/discussions) で公認 implementation として認知されること。doobidoo の mcp-memory-service / mcp-memory-service-rs / shodh-cloudflare は OpenAPI に明記されている。claude-hippo は v0.3 公開時に PR を出して implementation table への追加を提案する。

## 4. 互換性試験計画

Sprint S1 で実装すべき conformance test：

```
# tests/compat/swap.rs
1. mcp-memory-service-rs で 100 件 store
2. その DB ファイルを claude-hippo serve に渡して list_memories
   → 100 件全件取得できる、tags/memory_type 正しく decode
3. claude-hippo で追加 100 件 store（うち 50 件は surprise_score 高、50 件は低）
4. 同 DB ファイルを mcp-memory-service-rs に戻して全件 retrieve
   → 200 件取得できる、claude-hippo の追加分も問題なく読める
5. metadata JSON の surprise_score key が破壊されていないこと
```

Sprint S2 ALTER TABLE 後：

```
# tests/compat/alter_swap.rs
1. claude-hippo（surprise_score 列あり）で 100 件 store
2. mcp-memory-service-rs（列を知らない）で list_memories
   → 100 件全件取得できる、未知列は SELECT に含まれず無害
3. mcp-memory-service-rs で 100 件追加 store
4. claude-hippo で list_memories
   → 200 件、新規 100 件は surprise_score = NULL
```

これが green になれば「**唯一、商用フリーで使え、過去の memory も全部引き継げる、surprise-aware な Rust 製 SHODH 互換 MCP server**」を主張できる。HN ブログのタイトル候補：
- 「Drop-in Rust replacement for mcp-memory-service, with a surprise twist」
- 「Why we rebuilt the SHODH-compatible memory server in Rust under Apache/MIT」

## 5. 未解決事項（Sprint S1 spike で潰す）

- [ ] sqlite-vec の vec0 仮想表が他実装で互換読み書きできるか実機確認（embedding 次元 384 固定、cosine 距離）
- [ ] `metadata` JSON の schema を decide：`hippo` プレフィックスで namespace 切るか、root 直下に置くか
- [ ] SHODH spec の `consolidate` エンドポイント実装が doobidoo 側にあるか調査（仕様にはあるが mcp-memory-service-rs には未実装の可能性）
- [ ] SHODH MemoryType 13 種に対する claude-hippo 内部表現マップ（v0.1 では `memory_type` 列にそのまま文字列で格納するだけで OK）
- [ ] 「associative recall」（Hebbian edges）を v0.3 でどう実装するか（別テーブル `memory_associations(from_id, to_id, weight)` を持つか）
