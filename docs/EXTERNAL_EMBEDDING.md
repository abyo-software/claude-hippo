# External Embedding Backend — 設計ドキュメント

> Status: **Design only** (v0.3 で実装予定)
> Last updated: 2026-05-10
> 目的: ローカル ONNX (fastembed) を外部 HTTP API に差し替え、RSS を <50 MB に
> 落としつつ retrieval semantics を維持する経路を提供する。

---

## なぜ作るか

`fastembed` (ONNX Runtime ロード) は cold-start 後 ~150 MB の RSS を要する。これは
mcp-memory-service-rs (~186 MB) より軽いが、依然として「Pure Rust 軽量」として
打ち出すには重い。`cargo bloat` 計測で **binary の 74% が ort_sys + 周辺**
であることが判明している (PLAN.md §3 軸2)。

外部 embedding API モードを足すと：

- **RSS <50 MB** が現実的に達成できる (sqlite-vec + tokio + reqwest のみ)
- **モデル DL 不要** で cold-start が ~30 ms 以下
- **任意モデル** が選べる (OpenAI text-embedding-3-small、HF Inference API、
  Ollama、ローカル LM Studio、自前 vLLM 等)
- **`UserDefinedEmbeddingModel` 不要** — 推論はクラウド or 別プロセスに任せる

トレードオフ:

- ネットワーク必須 (オフライン動作不可)
- レイテンシは API のラウンドトリップに支配される (~50-300 ms)
- API 利用料が発生する場合がある (OpenAI: $0.02 / 1M tokens)
- API キー管理が必要

このため **mcp-memory-service-rs と DB swap 互換性を維持するには 384 dim 縛り
を厳守する必要がある**。embeddings.rs の `EMBEDDING_DIM = 384` 定数は
v0.3 でも変えない。

---

## CLI / 設定

```text
hippo serve \
  --embedding-backend external \
  --external-embedding-url https://api.openai.com/v1/embeddings \
  --external-embedding-model text-embedding-3-small \
  --external-embedding-dim 384 \
  --external-embedding-api-key-env OPENAI_API_KEY
```

| フラグ | デフォルト | 用途 |
|---|---|---|
| `--embedding-backend {local,external}` | `local` | バックエンド切替。`local` は v0.1 互換 |
| `--external-embedding-url` | (必須 if external) | OpenAI 互換 `/v1/embeddings` エンドポイント |
| `--external-embedding-model` | (必須 if external) | API 側のモデル名 (e.g. `text-embedding-3-small`, `bge-m3`) |
| `--external-embedding-dim` | `384` | 期待される dim。**384 以外は reject** (DB schema 整合) |
| `--external-embedding-api-key-env` | `OPENAI_API_KEY` | API key を読む env 変数名。直接 key を渡す経路は意図的に提供しない (shell history / process listing 流出防止) |
| `--external-embedding-timeout-ms` | `5000` | 個別リクエスト timeout |
| `--external-embedding-batch-size` | `64` | 1 リクエストあたりの最大 input 数 |
| `--external-embedding-max-retries` | `3` | 5xx / 429 / network error 時のリトライ回数 (exponential backoff) |
| `--external-embedding-extra-headers` | (none) | `Header: Value` 形式の追加ヘッダ (Azure リソースキー等) |

env 経由でも全て設定可能 (`HIPPO_EXTERNAL_EMBEDDING_URL` 等)。

---

## API 形状

OpenAI Embeddings API 互換 ([reference](https://platform.openai.com/docs/api-reference/embeddings/create))。これは Anthropic 純正サポートはないが、Azure / HF / Ollama / Together / Voyage / OpenRouter 等が広く準拠している事実上の標準。

リクエスト:
```http
POST {url}
Authorization: Bearer {api_key}
Content-Type: application/json
{extra_headers}

{
  "model": "text-embedding-3-small",
  "input": ["text 1", "text 2", "..."],
  "encoding_format": "float",
  "dimensions": 384
}
```

`dimensions` パラメータは OpenAI v3 系のみサポート。それ以外のプロバイダで
無視されたら、L2 sub-projection を claude-hippo 側で行わない方針 (生 dim を
そのまま使う、`--external-embedding-dim` で受け取り側 dim を確定させる)。

レスポンス:
```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.012, -0.034, ...]},
    {"object": "embedding", "index": 1, "embedding": [...]}
  ],
  "model": "text-embedding-3-small",
  "usage": {"prompt_tokens": 17, "total_tokens": 17}
}
```

claude-hippo 側で:
1. dim が `expected_dim` と一致するか検証 (差があれば fail loud)
2. **L2 正規化を強制** (`embeddings.rs` の MiniLM 経路と同じ仕様、KNN cosine で
   distance / 2 を sim にできるようにする)
3. tokio 上で `reqwest` を使う、blocking call は禁止 (server.rs は async)

---

## 実装スケッチ

`src/embeddings/external.rs` (新規):

```rust
use crate::{Embedder, HippoError, Result, EMBEDDING_DIM};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use std::time::Duration;

pub struct ExternalEmbedder {
    client: reqwest::blocking::Client, // or async + tokio::task::block_in_place
    url: String,
    model_name: String,
    dim: usize,
    batch_size: usize,
    max_retries: u32,
    headers: HeaderMap,
}

impl ExternalEmbedder {
    pub fn new(cfg: ExternalEmbeddingConfig) -> Result<Self> {
        if cfg.dim != EMBEDDING_DIM {
            return Err(HippoError::Config(format!(
                "external embedding dim {} != schema-required {} (DB swap compat)",
                cfg.dim, EMBEDDING_DIM
            )));
        }
        // ... build headers, client with timeout
    }
}

impl Embedder for ExternalEmbedder {
    fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_batch(&[text])?.into_iter().next()
            .ok_or_else(|| HippoError::Embedding("empty external response".into()))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.batch_size) {
            let body = json!({
                "model": self.model_name,
                "input": chunk,
                "encoding_format": "float",
                "dimensions": self.dim,
            });
            let resp = self.with_retry(|| self.client.post(&self.url)
                .headers(self.headers.clone())
                .json(&body)
                .send())?;
            let parsed: EmbeddingsResponse = resp.json()?;
            for e in parsed.data {
                if e.embedding.len() != self.dim {
                    return Err(HippoError::Embedding(format!(
                        "dim mismatch: expected {}, got {}", self.dim, e.embedding.len()
                    )));
                }
                let mut v = e.embedding;
                let n: f32 = v.iter().map(|x| x*x).sum::<f32>().sqrt().max(1e-8);
                for x in v.iter_mut() { *x /= n; }
                out.push(v);
            }
        }
        Ok(out)
    }
}
```

retry 戦略:
- 429 (rate limit) → `Retry-After` ヘッダ尊重、なければ exponential backoff (200ms, 800ms, 3200ms, max 3 retries)
- 5xx → 同上
- network error (DNS, connect, read timeout) → 同上
- 4xx (400, 401, 403) → 即時 fail (retry 無意味)

**並行性**: stdio MCP server は基本的にシングルクライアント (Claude Code が 1
コネクション)。バッチサイズに収まる量を 1 リクエストで投げる。並列リクエストは
v0.3 では実装しない (複雑さ vs MCP の現実的負荷で見合わない)。

---

## RSS 目標

| 構成 | RSS 目標 | 備考 |
|---|---|---|
| `--embedding-backend external` (HTTP only) | **<30 MB** | sqlite + sqlite-vec + tokio + reqwest + rustls |
| `--embedding-backend external` + `--external-embedding-url localhost` (Ollama 同居) | **<35 MB** | ローカル Ollama / vLLM が別プロセスでメモリを別建てで持つ |
| `--embedding-backend local` (v0.1 既定) | ~150 MB | fastembed + ONNX runtime + model |

実測は v0.3 で `--bench-rss` を追加し、`/proc/self/status` の `VmHWM` を
記録する (CLI bench に既に実装済の経路を使い回し)。

---

## エラーモデル

外部 API は不安定なので、現存する `HippoError::Embedding(String)` で十分
網羅できるが、内訳ログは細かく: tracing で `target=hippo::external_embedding`
として、status code, model, retry count, total elapsed を構造化出力。

ユーザに見せる error は対症療法的に分類:
- 401 → "API key invalid or env var missing: $VAR_NAME"
- 429 → "rate limited; reduce batch size or upgrade plan"
- network → "could not reach $URL; check network and DNS"
- dim mismatch → "model returned dim {X}, expected {Y} for DB schema compatibility"

---

## セキュリティ

- API key は **env 経由のみ**。`--external-embedding-api-key VALUE` のような
  直渡し flag は提供しない (ps / shell history 流出)。
- リクエスト body は **content をそのまま送信** する (これは external API の
  仕様上避けられない)。**README と SURPRISE_SELECTION の冒頭でユーザに警告**:
  「external mode で OpenAI 等を使うと記憶内容が API プロバイダに送信される」。
- TLS は `rustls` 強制 (`reqwest` の `rustls-tls` feature)。openssl は使わない。

---

## テストプラン

1. **mock HTTP server** (`wiremock` crate) で `/v1/embeddings` を立てる
2. dim mismatch を返した場合の reject 確認
3. 401/429/500 の retry behavior
4. batch size 跨ぎ (`batch_size=2` で 5 input → 3 リクエスト)
5. Concurrent embed (server.rs の `do_remember` 同時並行) で deadlock しないこと
6. **L2 正規化検証**: dummy 非正規化 vector を返す mock で、出力が正規化済か

CI では mock のみ。実 OpenAI API への smoke は `examples/external_embedding_smoke.rs`
として `cargo run --example` で手動実行。CI で API key を持たない方針。

---

## v0.3 実装の Phase 分解

1. **Phase 1**: `ExternalEmbedder` 実装 + `--embedding-backend external` 追加
   (mock 駆動 unit test 込み)
2. **Phase 2**: retry / timeout / batch chunk / Authorization header
3. **Phase 3**: `examples/external_embedding_smoke.rs` で OpenAI / Ollama / HF
   実 smoke 動作確認
4. **Phase 4**: README に「external mode で何がトレードオフか」を追記、
   セキュリティ警告を README 冒頭の Quickstart に必ず混ぜる

---

## なぜ v0.2 で実装しないか

v0.2 は **surprise selection の評価軸を実数値で示す** ことが目的。external
embedding は backend 切替であり、surprise の効き目とは独立。両方を v0.2 に
詰めると、ベンチ結果が「実 ONNX 384 dim」と「外部 API 384 dim」のどちらの
モデルで取られたか曖昧になる。混同を避けるため v0.3 に分離する。

---

## 関連

- 実装予定先: `src/embeddings/external.rs` (現状の `src/embeddings.rs` を
  module 化して `local.rs` / `external.rs` / `mod.rs` に分割)
- 設計の根: PLAN.md §3 軸2 「<50 MB 達成戦略」(c) External embedding API
- DB swap: docs/SHODH_COMPAT.md (384 dim 縛りの根拠)
- 競合数値: docs/COMPETITOR_BENCH.md (mcp-memory-service-rs の 186 MB を超える軽量化)
