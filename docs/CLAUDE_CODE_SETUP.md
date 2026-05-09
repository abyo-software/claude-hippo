# Claude Code Setup

## 1. Install

```bash
cargo install claude-hippo
which hippo   # → ~/.cargo/bin/hippo
hippo --version
```

または source から：

```bash
git clone https://github.com/abyo-software/claude-hippo
cd claude-hippo
cargo build --release
sudo install -m 0755 target/release/hippo /usr/local/bin/hippo
```

## 2. Smoke test

```bash
hippo verify    # schema apply + sqlite-vec 確認、DB 作成のみ
hippo embed "smoke test"   # 初回は ~120 ms で MiniLM ONNX を HF から download (~85 MB)
```

cache dir: `~/.cache/claude-hippo/models/`（`HIPPO_MODEL_CACHE` で変更可）

## 3. Claude Code に登録

### 方法 A: `claude mcp add`

```bash
claude mcp add hippo hippo serve
```

### 方法 B: 設定ファイル直接

`~/.claude.json` または `~/.claude/mcp_servers.json` に：

```jsonc
{
  "mcpServers": {
    "hippo": {
      "command": "hippo",
      "args": ["serve"],
      "env": {
        "HIPPO_DB_PATH": "/home/you/.local/share/claude-hippo/memory.db"
      }
    }
  }
}
```

### 方法 C: Project-local 設定

特定のプロジェクト用に分けたい場合：

```bash
# プロジェクト直下に
echo '{
  "mcpServers": {
    "hippo": {
      "command": "hippo",
      "args": ["serve", "--db", "./.hippo/memory.db"]
    }
  }
}' > .claude/mcp_servers.json
```

## 4. 使い方（Claude Code セッション内）

Claude Code から以下が呼べるようになる：

- **保存**: 「これを覚えておいて」と言うと `hippo_remember` が走る
- **想起**: 「以前話した X について」で `hippo_recall` が走る
- **要約**: 「今日のセッションで何を決めた？」で `hippo_session_summary`
- **削除**: 「あの記憶忘れて」で `hippo_forget`

明示的にツール名を指定したい場合：

```
@hippo hippo_remember content="JWT 24h で確定" tags=["auth","decision"] importance=0.8
@hippo hippo_recall query="認証の決定" limit=5
```

## 5. 既存の mcp-memory-service / mcp-memory-service-rs から乗り換え

**DB ファイルはそのまま使える**（schema 互換）：

```bash
# 既存 DB の場所
OLD_DB=~/.mcp_memory_service.db   # mcp-memory-service デフォルト

# claude-hippo に渡す
HIPPO_DB_PATH=$OLD_DB hippo serve
```

または MCP 設定の `command` を入れ替えるだけ：

```diff
{
  "mcpServers": {
    "memory": {
-      "command": "mcp-memory-service-rs",
+      "command": "hippo",
       "args": ["serve"],
       "env": {
-        "MCP_MEMORY_DB_PATH": "/home/you/.mcp_memory_service.db"
+        "HIPPO_DB_PATH": "/home/you/.mcp_memory_service.db"
       }
    }
  }
}
```

過去の記憶は全部見える。`store_memory` / `retrieve_memory` の SHODH alias も動くので、Claude Code 側の prompt 工夫も変える必要なし。

逆も真：claude-hippo で書いた DB を mcp-memory-service-rs で読める（surprise score は metadata 中の未知 key として無害に無視される）。

## 6. トラブルシュート

### `hippo serve` が起動しない
```bash
hippo verify   # → schema が壊れていないか確認
```

### embedding model download がタイムアウト
S3 経由の自動 download が遅い場合、HF mirror から手動配置：

```bash
mkdir -p ~/.cache/claude-hippo/models/Qdrant--all-MiniLM-L6-v2-onnx
# fastembed が要求する path 構造に従う。
# 詳細は fastembed 5.13 の README を参照。
```

または `HIPPO_MODEL_CACHE=/path/to/manual/cache` で別ディレクトリを指定。

### Claude Code が tool を見つけてくれない
1. `claude mcp list` で `hippo` が登録されているか確認
2. `claude mcp test hippo` で stdio handshake が通るか確認
3. それでもダメなら `RUST_LOG=debug hippo serve` を別 terminal で起動して MCP message を観察

### DB が壊れた
sqlite-vec の virtual table は再構築できる：

```bash
sqlite3 $HIPPO_DB_PATH 'DROP TABLE memory_embeddings;'
hippo verify   # CREATE VIRTUAL TABLE が再走される
# embeddings は失われるので、必要なら再 embedding が要る (今後の re-embed CLI を予定)
```

## 7. パフォーマンスチューニング

### RSS を <50 MB にしたい
fastembed + ONNX は ~150 MB 必要。<50 MB を狙うなら external embedding API 経由（v0.2 予定の `--embedding-backend external` モード）を待つ。

現状の妥協案: idle 時に process kill + on-demand restart（systemd socket activation 等）。

### Embedding latency を下げたい
- BGE-small-en-v1.5 量子化版（`fastembed::EmbeddingModel::BGESmallENV15Q`）への切替を検討（v0.2 で flag）
- ホット query は `metadata._hippo` に embedding cache を追加（未実装）

## 8. 関連 docs

- [README.md](../README.md) — 概要 + ベンチ
- [PLAN.md](../PLAN.md) — 開発計画
- [docs/SURPRISE_SELECTION.md](SURPRISE_SELECTION.md) — surprise scoring の理論
- [docs/SHODH_COMPAT.md](SHODH_COMPAT.md) — DB schema 互換戦略
- [docs/COMPETITOR_BENCH.md](COMPETITOR_BENCH.md) — vs mcp-memory-service-rs
- [docs/ARCHITECTURE.md](ARCHITECTURE.md) — モジュール分割と data flow
