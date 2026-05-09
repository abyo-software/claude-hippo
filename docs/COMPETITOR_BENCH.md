# Competitor Benchmark — mcp-memory-service-rs 再現計測

> Status: Sprint S1 直前 spike notes（2026-05-09 → 2026-05-10、Linux x86_64 実測）
> 対象: [doobidoo/mcp-memory-service-rs](https://github.com/doobidoo/mcp-memory-service-rs) commit HEAD（2026-04-20 push、`mcp-memory-service` v0.1.1）
> 目的: 公式 README の「cold-start 68 ms / RSS 241 MB」を Linux 環境で再現可否確認、claude-hippo の <50 MB / <100 ms 目標との距離を測る

## 環境

| 項目 | 値 |
|------|------|
| OS | Linux 7.0.0-15-generic |
| CPU | x86_64 |
| rustc / cargo | 1.95.0 |
| ベンチ条件 | sequential stdio MCP, fresh DB per run, ONNX warm |

> 公式数値は **Apple Silicon Mac**。本計測は **Linux x86_64** なので CPU-bound 部分は別系統。アーキ差を踏まえた解釈が必要。

## ビルド

```bash
git clone --depth 1 https://github.com/doobidoo/mcp-memory-service-rs.git
cd mcp-memory-service-rs
cargo build --release  # 235 crates、初回 ~4 分
```

成果：
- バイナリ: `target/release/mcp-memory-service-rs`
- サイズ: **31.4 MB**（stripped, ELF 64-bit）

### `cargo bloat --release --crates` 上位

| % .text | Size | Crate |
|---|---|---|
| **48.1%** | **11.7 MiB** | **ort_sys**（ONNX Runtime）|
| 26.2% | 6.4 MiB | [Unknown]（C/asm 記号、ort 由来と推定） |
| 5.8% | 1.4 MiB | std |
| 2.1% | 522 KiB | tokenizers |
| 1.7% | 417 KiB | rmcp |
| 1.5% | 366 KiB | regex_automata |
| 1.4% | 358 KiB | rustls |
| 1.2% | 289 KiB | ring（rustls 暗号） |
| 0.9% | 220 KiB | clap_builder |
| 0.9% | 214 KiB | mcp_memory_service_rs（app 本体） |
| 0.5% | 113 KiB | libsqlite3_sys |

→ **ONNX Runtime 系（ort_sys + [Unknown]）だけで binary の 74% を占有**。これを切れば **claude-hippo binary は ~18 MB に縮められる**（目標 <30MB は余裕）。

## 計測 1: `verify` smoke test（ONNX 未ロード）

```bash
$ /usr/bin/time -v ./target/release/mcp-memory-service-rs verify --db /tmp/mms.db
mcp-memory-service-rs verify ✓
  db path       : /tmp/mms.db
  db size       : 4096 bytes
  open + schema : 0 ms
  vec_version   : v0.1.9
  memories      : 0 (undeleted)
```

| 指標 | 実測 |
|------|------|
| Wall time | < 10 ms |
| **Peak RSS** | **23 MB** |
| sqlite-vec version | v0.1.9 |

→ **ONNX を読まなければ 23 MB で運用できる**。claude-hippo の <50 MB 目標は十分達成可能（lazy load 戦略の根拠）。

## 計測 2: `embed` cold pipeline（ONNX 込み、warm cache）

```bash
$ /usr/bin/time -v ./target/release/mcp-memory-service-rs embed "hello hippo"
mcp-memory-service-rs embed ✓
  text        : "hello hippo"
  load model  : 96 ms
  embed time  : 6 ms
  dim         : 384
  L2 norm     : 1.000000
  first 5     : [-0.0217, 0.0828, 0.0426, -0.0150, -0.0737]
```

| 指標 | 実測（Linux x86） | 公式 README（Apple Silicon） |
|------|------|------|
| Load model | **96 ms** | — |
| Embed (single) | **6 ms** | — |
| Wall total | **110 ms** | — |
| **Peak RSS** | **173 MB** (177332 KB) | 241 MB（peak after 200 ops） |

→ Linux の ONNX Runtime は macOS より **lean**（173 vs 241 MB）。

## 計測 3: 公式相当 bench script — Rust-only 版

公式 `scripts/bench.py` は Python upstream を要求するため、Python 側を除外した版（`/tmp/bench_rust_only.py`）を作成して実行。

```bash
$ python3 /tmp/bench_rust_only.py
cold-start      : 99.3 ms
store    x100  : p50=5.4ms p95=7.1ms min=5.0ms max=9.6ms
retrieve x100  : p50=5.5ms p95=6.1ms min=5.2ms max=7.8ms
peak RSS        : 186.0 MB
```

### 公式 vs 本計測

| 指標 | 公式（Apple Silicon, README） | 本計測（Linux x86） | 差分 |
|------|------|------|------|
| cold-start | 68 ms | **99 ms** | +46% 遅い |
| store p50 | 8.5 ms | **5.4 ms** | -36% 速い |
| store p95 | 8.7 ms | **7.1 ms** | -18% 速い |
| retrieve p50 | 8.8 ms | **5.5 ms** | -38% 速い |
| retrieve p95 | 9.2 ms | **6.1 ms** | -34% 速い |
| RSS (live) | 241 MB | **186 MB** | -23% 軽い |

**所見**：
- cold-start は Apple Silicon が速い（NEON 最適化された ONNX Runtime + ファーストパーティ kernel）
- 一旦 warm すると Linux x86 のほうが store/retrieve とも安定して速い（CPU クロック高い）
- RSS は Linux のほうが軽い（macOS の ONNX Runtime fat binary 込みのオーバーヘッドかも）

つまり **公式数値は claude-hippo にとっての一面的な競合 baseline であり、Linux ユーザにとっては本計測値が現実**。

## 重要な所見と claude-hippo への含意

### 所見 1: 「<50 MB RSS」は ONNX を切らない限り達成不能
- verify only（ONNX なし） = 23 MB
- embed (warm) = 173 MB
- serve + 100 ops = 186 MB

ONNX Runtime + MiniLM モデルが ~150 MB を占める。**candle 純 Rust 経路への切替が <50 MB 達成の唯一の現実路線**。Sprint S1 spike で candle 経路の最小プロトを作って RSS 計測する。

### 所見 2: 「<100 ms cold-start」は ONNX 込みで届く
- 本計測の 99 ms = mcp-memory-service-rs の Linux x86 実力値
- claude-hippo は同レンジ達成可能（rmcp + sqlite-vec + fastembed のスタックは mcp-memory-service-rs と同等）
- これを **超える**には lazy ONNX load が必要（initialize 応答だけ返して、最初の embed リクエストで model load）

### 所見 3: store/retrieve の 5-7 ms は claude-hippo の現実的な目標下限
- mcp-memory-service-rs の 5.4 ms 以下を狙うのは難しい（同じ stack なら同レンジ）
- 差別化は **絶対速度よりも RSS と機能** で取る
- ただし retrieve に surprise-weighted ranking を加えると 1-2 ms 程度のオーバーヘッドは想定（許容範囲）

### 所見 4: ONNX model のダウンロードソースは脆い
- 公式 URL `https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz`
- 本セッションでは S3 からの download が ~30 KB/s に張り付いて完了不能
- 代替: HuggingFace `https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx` + `tokenizer.json`
- mcp-memory-service-rs は **extracted file の存在だけチェック**（archive SHA は archive がある場合のみ） → HF から取って手で `~/.cache/mcp_memory/onnx_models/all-MiniLM-L6-v2/onnx/` に置けば動く

**claude-hippo 戦略**:
- HF をプライマリミラー、S3 をフォールバックに
- `--no-download` フラグでオフラインモード
- `CLAUDE_HIPPO_MODEL_CACHE` env で事前配置パスを指定可能に
- `model.onnx` + `tokenizer.json` を直接取って組み立て、archive 経由しない（SHA 整合性は file 単位で SHA256 をハードコード）

### 所見 5: SHODH 互換 schema は単純（[SHODH_COMPAT.md](SHODH_COMPAT.md) 参照）
- 11 列の SQLite schema を verbatim コピーするだけで Day-1 swap-compatible
- SHODH OpenAPI v1.0.0 の rich field（emotional / episodic / source / quality）は `metadata` JSON 列に詰める
- mcp-memory-service-rs の `SPEC.md` §7「Behavior Preservation Checklist」が事実上 free design doc → claude-hippo 互換テストに転用

### 所見 6: MCP tool 命名差は互換層で吸収
- mcp-memory-service-rs: `store_memory`, `retrieve_memory`, `search_by_tag`, `delete_memory`, `list_memories`, `check_database_health`, `get_cache_stats`
- claude-hippo (PLAN §4): `hippo_*` プレフィックス付き 5 ツール
- v0.3 で `--shodh-tool-names` フラグ提供 → 既存 client 設定無変更で乗り換え可

## 次のアクション（Sprint S1 着手前）

- [x] 公式 ベンチ Linux 再現完了（cold 99ms / store 5.4ms / retrieve 5.5ms / RSS 186MB）
- [x] cargo bloat 計測完了（ort_sys 11.7 MiB が 48%）
- [x] ONNX cache 配置の代替経路特定（HuggingFace 直 download）
- [ ] Apple Silicon 機がある時に同 bench 実行 → 68/241 公式数値の再現性確認
- [ ] claude-hippo 候補スタック（rmcp + fastembed + sqlite-vec + tokio）で空 binary を作って `cargo bloat`
- [ ] candle 純 Rust 経路でのプロトタイプ binary サイズ計測（ort 切った場合の理論最小値）

## ベンチ再現手順（再実行時の最短経路）

```bash
# 1. clone + build
git clone --depth 1 https://github.com/doobidoo/mcp-memory-service-rs.git ~/git/mcp-memory-service-rs
cd ~/git/mcp-memory-service-rs && cargo build --release

# 2. ONNX model を HF から事前配置（S3 が遅い時）
mkdir -p ~/.cache/mcp_memory/onnx_models/all-MiniLM-L6-v2/onnx
cd ~/.cache/mcp_memory/onnx_models/all-MiniLM-L6-v2/onnx
curl -L -o model.onnx https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx
curl -L -o tokenizer.json https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json

# 3. smoke test
~/git/mcp-memory-service-rs/target/release/mcp-memory-service-rs verify --db /tmp/mms.db
~/git/mcp-memory-service-rs/target/release/mcp-memory-service-rs embed "hello"

# 4. full bench（Rust-only 版）
python3 /home/y1/git/hippo/scripts/bench_competitor.py  # 同一スクリプトを将来このリポに取り込む
```

## 参考

- [doobidoo/mcp-memory-service-rs](https://github.com/doobidoo/mcp-memory-service-rs)
- [SPEC.md（wire-level contract）](https://github.com/doobidoo/mcp-memory-service-rs/blob/main/SPEC.md)
- [scripts/bench.py（公式ベンチ）](https://github.com/doobidoo/mcp-memory-service-rs/blob/main/scripts/bench.py)
- [SHODH OpenAPI](https://github.com/varun29ankuS/shodh-memory/blob/main/specs/openapi.yaml)
- [本リポ docs/SHODH_COMPAT.md](./SHODH_COMPAT.md)
