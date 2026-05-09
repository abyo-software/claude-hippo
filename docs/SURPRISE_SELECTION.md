# Surprise-based Selection — claude-hippo の差別化軸の核

> Status: v0.1 documented (production)、v0.2 で評価ベンチ追加予定
> Last updated: 2026-05-10

## なぜ「全部覚える」ではダメか

既存の memory MCP（mcp-memory-service / mcp-memory-service-rs / Mem0 / MemPalace 等）は、ユーザの発話を **すべて埋め込んでベクトル検索** する。これは短期セッションでは機能するが、**1 万セッション後に重要な決定がノイズに埋もれる**。

人間の海馬は違う。Tulving (1985) や McGaugh (2000) が示すように、**特異性が高い瞬間だけ**を長期記憶化し、ありふれた瞬間は薄く保存して時間と共に忘れる。これにより 30 年経っても「あの時の重要な判断」を引き出せる。

claude-hippo はこの選択性を **モデル化可能な metric** に落とし込んで実装する。

---

## Surprise score の定義

各記憶 `m` に対して、保存時に 4 成分から合成スコアを計算する：

```
surprise(m) = w_o · embedding_outlier(m)
            + w_e · engagement(m)
            + w_x · explicit(m)
            + w_p · prediction_loss(m)
```

デフォルト重み（`SurpriseWeights::default()`）：

| 重み | 値 | 役割 |
|---|---|---|
| `w_o` (embedding_outlier) | 0.4 | 既存記憶ベクトル群からの平均 cosine 距離。novelty を測る |
| `w_e` (engagement) | 0.2 | 文長と tag 数の飽和関数。「真剣に書かれた記憶」を hint |
| `w_x` (explicit) | 0.1 | ユーザが `importance` flag で明示マークした重み |
| `w_p` (prediction_loss) | 0.3 | abyo-llm-probe で計測する LLM の予測誤差（v0.3 で実装） |

`prediction_loss` が None（v0.1）の時は `w_p` を `w_o:w_e = 2:1` で按分する：

```rust
fn score_no_pred() {
    let extra = w_p;  // 0.3
    let w_o' = w_o + extra * 2/3;  // 0.4 + 0.2 = 0.6
    let w_e' = w_e + extra * 1/3;  // 0.2 + 0.1 = 0.3
    surprise = w_o' * outlier + w_e' * engagement + w_x * explicit
}
```

これで成分が全部 1.0 でも合計が 1.0 を超えない。

### 各成分の詳細

#### `embedding_outlier`

既存の記憶埋め込み `H = {h_1, ..., h_n}` と新規 query `q` (両方 L2 正規化済) について：

```
cos_sim(q, h_i) ∈ [-1, 1]
distance(q, h_i) = (1 - cos_sim) / 2 ∈ [0, 1]
embedding_outlier = mean_i distance(q, h_i)
```

`H = ∅` の時は `1.0`（最初の記憶は常に novel）。

直近 50 件を `H` として使う（src/server.rs `history_embeddings(store, 50)`）。

**意味**: 既存の記憶と意味的に近いものは「ありふれた」、遠いものは「新しい話題」と判定。

#### `engagement`

```rust
len_score = tanh(content.len() / 1000)        // 1000 chars で ~0.76
tag_score = tanh(tags.len() / 5)               // 5 tags で ~0.76
engagement = 0.7 * len_score + 0.3 * tag_score
```

**意味**: 長文 + 多 tag = ユーザが時間をかけて整理した記憶 = 重要度が高い可能性。

ヒューリスティックなので外しもあるが、コストはゼロ。

#### `explicit`

```rust
explicit = importance.unwrap_or(0.0).clamp(0.0, 1.0)
```

`hippo_remember` の `importance` 引数で 0.0..=1.0 を指定。ユーザの明示判断。

#### `prediction_loss` （v0.3）

LLM (Claude Code 自身か別の小型ローカル LLM) に context + content を見せて、content の `next-token prediction loss` を計算。loss が高い = LLM にとって予測しづらい = 情報量が高い。

abyo-llm-probe のローカル LLM 内部状態 hook で実装予定。v0.1 では `None`。

---

## Forgetting curve

Ebbinghaus 風の指数減衰：

```rust
decay(age_days, half_life_days) = exp(-age_days * ln(2) / half_life_days)
```

- `age_days = 0` → `1.0`
- `age_days = half_life` → `0.5`
- `age_days = 2 * half_life` → `0.25`

デフォルト `half_life_days = 30`。

**設計判断**: decay は **物理削除に紐づけない**。記憶は常に保持され、`recall` のランキングで decay が効くだけ。法務 / audit 用途のユーザには、`no_surprise_boost=true` で純粋な vector similarity に戻せる。

---

## Recall ranking

```rust
score = 0.7 * cos_sim(query, memory)
      + 0.3 * surprise(memory) * decay(age_days, 30)
```

**意味**:
- 70% は意味的近さ（既存の semantic search の標準）
- 30% は「**古くても surprise が高ければ浮かび、新しくても陳腐ならランクが下がる**」効果

KNN 第一段で `3k` 件 oversample してから surprise rerank で `k` 件に絞る（src/server.rs `do_recall` の `fetch_k = k * 3`）。

`no_surprise_boost=true` を渡すと oversample せず、純 cosine sim のみで動く。

---

## 評価軸（v0.2 で実装）

LongMemEval 等の汎用 vector 検索ベンチでは負ける可能性が高い（MemPalace は ChromaDB に最適化されている）。代わりに **特異性ベース選別が活きるシナリオ** を独自定義する：

### Bench A: Long-session noise

100 ターンの仮想会話に **5 件の重要決定** + **95 件の雑談** を混ぜる。100 ターン後に「重要決定だけ」を recall できる精度（precision@5、recall@5）を測る。

仮説：claude-hippo の surprise rerank は雑談を suppress するため、precision で勝つ。

### Bench B: Cross-session retrieval

1 ヶ月前の決定を、関連だが言い換え query から想起できるか。decay が活きる。

### Bench C: Decision trace

「なぜこう設計したか」を辿る reasoning chain。memory_type=Decision の高 surprise を優先表示する効果を測る。

これらは v0.2 で実装し、独自評価セットを公開する。

---

## チューニング

- `SurpriseWeights` を `MemoryServer::new` で受け取れるように API 拡張する余地あり（v0.2）
- `half_life_days` も同様
- `0.7 * sim + 0.3 * surprise*decay` のブレンド係数も tunable に

現状（v0.1）はハードコード（src/server.rs 上部の `DEFAULT_HALF_LIFE_DAYS = 30.0` 等）。

---

## 限界と honest disclosures

1. **Engagement ヒューリスティックは外す**: 短文だが超重要な決定（例: `"go ahead"`）を埋もれさせる可能性。`importance` で補える。
2. **Embedding outlier は cluster の中心が近いと低く出る**: 同種の決定が大量にある時、新しい決定が「すでにある」と判定される。Bench A で観測予定。
3. **Forgetting curve の `half_life=30 days` は経験値**: 個人ユーザによって最適値は変わる。v0.2 で per-user tuning を予定。
4. **`prediction_loss` 未実装**: v0.1 は w_p を再分配で吸収しているが、本来の差別化は abyo-llm-probe 統合後に発揮される。
5. **mcp-memory-service-rs / Python upstream は surprise score を読まない**: DB swap 時、彼らの retrieve は素の cosine sim になる。これは仕様（互換のため）。

---

## 関連

- 実装: [src/surprise.rs](../src/surprise.rs)
- 統合: [src/server.rs](../src/server.rs) `do_remember` / `do_recall`
- 互換: [docs/SHODH_COMPAT.md](SHODH_COMPAT.md) — surprise は `metadata._hippo.surprise.{score, components}` に格納
- ベンチ: [docs/COMPETITOR_BENCH.md](COMPETITOR_BENCH.md) — store/retrieve レイテンシは surprise 込みで mcp-memory-service-rs より速い
