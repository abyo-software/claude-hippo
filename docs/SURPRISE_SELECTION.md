# Surprise-based Selection — claude-hippo の差別化軸の核

> Status: v0.2 evaluation harness shipped — Bench A/B/C 実数値あり (production)
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

## 評価軸（v0.3 実装済 — 実測値）

LongMemEval 等の汎用 vector 検索ベンチでは負ける可能性が高い（MemPalace は ChromaDB に最適化されている）。代わりに **特異性ベース選別が活きるシナリオ** を独自定義した。すべて `tests/eval_*.rs` で実装、`cargo test --test eval_a_long_session --release` 等で再現可能。生 JSON は `target/eval_results/bench_*.json`。

> **v0.3 で score arithmetic を改良**: `decay_floor=0.5` 追加 + `default_oversample_factor=6` (was 3) で Bench A 既定 P@1 を **0.72 → 1.000** に、Bench B 365d 既定 lift を **negative → +0.875** に flip。下表は v0.3 数値（`tests/eval_*.rs` から実機）。v0.2 数値は `git log` で過去 commit 参照。

> **embedding は ClusteredMockEmbedder**（決定的、CI で再現可能）。実 ONNX
> embedding に切り替えた時の数値は別で計測する（v0.3 の `--embedding-backend
> external` 経由）が、surprise rerank の寄与は score 算術に依存しており、
> embedding 分布には依存しないため、定性的な結論は変わらない。

### Bench A: Long-session noise

100 ターン会話 (5 トピックに各 1 Decision + 19 chat = 100 items)。各トピック
5 種類の paraphrase = 25 query。Decision 1 件を ground truth に。

| metric (k=5)         | baseline (純 cosine) | v0.2 surprise (oversample=3) | v0.3 surprise (oversample=6 既定) | surprise full (oversample=20) |
|----------------------|----------------------|------------------------------|-----------------------------------|-------------------------------|
| precision@1          | 0.080                | 0.720                        | **1.000**                         | **1.000**                     |
| MRR                  | 0.145                | 0.720                        | **1.000**                         | **1.000**                     |
| recall@5             | 0.280                | 0.720                        | **1.000**                         | **1.000**                     |

**読み方**:
- 純 cosine では Decision がクラスタ内 20 件中ランダム位置に埋もれる（rank 1 は 8%、top-5 は 28%）。
- v0.2 既定 (oversample=3): surprise rerank が precision@1 を 8% → 72% にリフト。fetch_k=15 で cluster サイズ 20 を超えないため 72% で頭打ち（v0.2 の honest limit）。
- v0.3 既定 (oversample=6): fetch_k=30 で 20 item cluster をカバーし切り、Decision が常に rerank プールに入って **precision@1 = 1.000**。v0.2 の頭打ちを解消。
- 完全 oversample (k×20=100)：rerank が全候補を見て、毎回 Decision を rank 1 に置く（v0.3 既定と同値、上限）。

### Bench B: Cross-session retrieval (forgetting curve calibration)

1 件の Decision を `age_days` 分だけ backdating、49 件の fresh chat と同 cluster
で混ぜ、8 paraphrase で recall。`half_life_days = 30` (既定)、v0.3 で `decay_floor = 0.5` (既定) 追加。

| Decision age | baseline P@1 / MRR | v0.2 surprise full P@1 / MRR | v0.3 surprise full P@1 / MRR | v0.3 lift |
|-------------:|:-------------------|:-----------------------------|:-----------------------------|:----------|
|    0 days    | 0.125 / 0.125      | 1.000 / 1.000                | **1.000 / 1.000**            | +0.875    |
|   30 days    | 0.125 / 0.125      | 1.000 / 1.000                | **1.000 / 1.000**            | +0.875    |
|   90 days    | 0.125 / 0.125      | 0.125 / 0.125                | **1.000 / 1.000**            | +0.875    |
|  365 days    | 0.125 / 0.125      | 0.000 / 0.000 (negative lift) | **1.000 / 1.000**            | **+0.875** |

**v0.3 の構造的修正 (`decay_floor=0.5`)**: 古い Decision でも `surprise · max(decay, 0.5)` で
最低 0.5 倍の surprise 寄与が残る。importance=1.0 の Decision の raw surprise
(~0.96) は fresh chat の engagement-only surprise (~0.024) を圧倒するため、
365 日経過後も rerank で勝つ。

- v0.2: `score = 0.7·cos_sim + 0.3·surprise·decay`
  - 365 日 Decision: `0.96 × 2e-4 ≈ 0.0002` (decay でほぼ消滅) → fresh chat に負ける
- v0.3: `score = 0.7·cos_sim + 0.3·surprise·max(decay, decay_floor)`
  - 365 日 Decision: `0.96 × max(2e-4, 0.5) = 0.96 × 0.5 = 0.48` → fresh chat (0.024) を圧倒

`decay_floor=0` で v0.2 挙動を再現できる（CLI/env で調整可）。**現実的なユースケース** (1〜365 日のセッション継続) で surprise rerank が baseline より下にランクされる挙動は v0.3 で完全に消去。

### Bench C: Decision trace

1 cluster に 4 Decision + 16 non-Decision (Observation/Pattern/Discovery/Learning
各 4)、8 paraphrase で「decision を全部出して」query。relevant set = 4 Decision、ideal P@K = min(4,5)/5 = 0.8。

#### Default weights (`0.4,0.2,0.1,0.3`)

| metric (k=5)        | baseline | surprise oversample=3 | surprise oversample=20 (full) |
|---------------------|----------|-----------------------|-------------------------------|
| precision@K         | 0.350    | **0.675**             | **0.775** (97% of ideal)      |
| recall@K            | 0.438    | **0.844**             | **0.969**                     |
| precision@1         | 0.500    | **1.000**             | **1.000**                     |

#### Explicit-heavy weights (`0.2,0.1,0.5,0.2`)

`--surprise-weights "0.2,0.1,0.5,0.2"` で `w_explicit` を 0.1 → 0.5 にあげると:

| metric (k=5)        | baseline | surprise oversample=20 (full) |
|---------------------|----------|-------------------------------|
| precision@K         | 0.350    | **0.800** (= ideal)           |
| recall@K            | 0.438    | **1.000**                     |

**読み方**:
- baseline は Decisions と non-Decisions を区別できず、recall@5 は 0.438 (4 Decision のうち平均 1.75 件が top-5)。
- production 既定の surprise rerank は recall を 0.844 (3.4/4) に押し上げる。
- 完全 oversample で 0.969 (3.9/4)、explicit-heavy weights で **理論最大の 1.000**。
- これは `--surprise-weights` フラグが意味のある効果を持つ証拠 (CLI 知識が正しく rerank に反映されている)。

### 再現

```bash
cargo test --release --test eval_a_long_session    # Bench A
cargo test --release --test eval_b_cross_session   # Bench B
cargo test --release --test eval_c_decision_trace  # Bench C

# 数値は target/eval_results/bench_*.json に書き出される
ls target/eval_results/
```

### v0.2 で見えた限界 (v0.3 Phase A で全件解決)

| # | v0.2 限界                                            | v0.3 解決                                                    |
|---|------------------------------------------------------|--------------------------------------------------------------|
| 1 | 既定 oversample=3 で fetch_k=15 が決定を取りこぼす    | `default_oversample_factor` を 3→6 に bump、`RecallParams.oversample_factor: Option<usize>` を MCP に expose、CLI `--oversample-factor` + env 追加 |
| 2 | 365 日越え Decision を surprise rerank が demote する | `decay_floor=0.5` (既定) 追加、`surprise · max(decay, floor)` で構造的解決。Bench B 365d で **negative lift → +0.875** |
| 3 | half-life 30 日固定                                    | CLI `--half-life-days` + env `HIPPO_HALF_LIFE_DAYS` 追加（既定 30、0 で disable） |
| 4 | mock embedding ベース、real ONNX 数値はまだ            | Phase B (external embedding backend) + Phase C (prediction_loss 実値) 進行中 |

---

## チューニング

- [x] **v0.2**: `SurpriseWeights` を `--surprise-weights "w_o,w_e,w_x,w_p"` で CLI から注入。`HIPPO_SURPRISE_WEIGHTS` env でも可。`MemoryServer::new_with_weights` 経由
- [x] **v0.2**: `EmbeddingModelKind` を `--embedding-model {minilm-l6-v2|bge-small-en-v15-q}` で切替
- [x] **v0.3 Phase A**: `--half-life-days` + env `HIPPO_HALF_LIFE_DAYS` (既定 30、0 で disable)
- [x] **v0.3 Phase A**: `--decay-floor` + env `HIPPO_DECAY_FLOOR` (既定 0.5、0 で v0.2 挙動に戻す)。`surprise · max(decay, floor)` 形式で blending
- [x] **v0.3 Phase A**: `--oversample-factor` + env `HIPPO_OVERSAMPLE_FACTOR` (既定 6、was 3)、`RecallParams.oversample_factor: Option<usize>` を MCP schema に expose
- [x] **v0.3 Phase B**: `--embedding-backend external` (OpenAI/Ollama/vLLM/HF TEI 互換)、9 wiremock test、RSS 25.7 MB 実測
- [x] **v0.3 Phase C**: `--prediction-loss-backend openai-compat` で `prediction_loss` を実値化 (vLLM/llama.cpp/Ollama)、9 wiremock test、Bench D wiring smoke。実 LLM 数値は release smoke 送り、candle-rs native は v0.4

---

## 限界と honest disclosures

1. **Engagement ヒューリスティックは外す**: 短文だが超重要な決定（例: `"go ahead"`）を埋もれさせる可能性。`importance` で補える。Bench C は importance=1.0 を前提にしているため、explicit signal がないケースでは効果は弱まる。
2. **Embedding outlier は cluster の中心が近いと低く出る**: 同種の決定が大量にある時、新しい決定が「すでにある」と判定される。Bench A の chat-vs-decision の outlier 差は engagement 経路で稼いでいる。
3. **(解決済 — v0.3 Phase A)** Forgetting curve `half_life=30 days` ハードコード + 365 日越え demotion: v0.2 では Bench B 365d で fresh chat に追い越される（負の lift）挙動。v0.3 で `--half-life-days` (CLI/env, 既定 30) + `--decay-floor` (既定 0.5) を追加し、`surprise · max(decay, decay_floor)` で構造的解決。Bench B 365d が **negative lift → +0.875** に flip。
4. **(v0.3 で wiring 完了)** `prediction_loss`: v0.3 で `--prediction-loss-backend openai-compat` (vLLM / llama.cpp / Ollama / legacy OpenAI completions 互換) を追加し、`SurpriseComponents.prediction_loss` を実値で埋められる経路を提供。`backend = none` (既定) では v0.2 fallback (w_prediction を outlier+engagement に再分配)。**実 LLM での bench 数値は release-time smoke 送り** (Bench D は MockPredictionLoss で wiring smoke のみ)。candle-rs native backend は v0.4 候補
5. **mcp-memory-service-rs / Python upstream は surprise score を読まない**: DB swap 時、彼らの retrieve は素の cosine sim になる。これは仕様（互換のため）。彼らへ swap した瞬間に Bench A の baseline 数値に劣化する想定。
6. **(解決済 — v0.3 Phase A)** Default `oversample_factor=3` の取りこぼし: v0.2 では Bench A 既定で 72% 頭打ち。v0.3 で `default_oversample_factor` を 3→6 に bump、`RecallParams.oversample_factor: Option<usize>` を MCP schema に expose、CLI `--oversample-factor` + env `HIPPO_OVERSAMPLE_FACTOR` 追加。Bench A 既定で **precision@1 = 1.000**。

---

## 関連

- 実装: [src/surprise.rs](../src/surprise.rs)
- 統合: [src/server.rs](../src/server.rs) `do_remember` / `do_recall`
- 互換: [docs/SHODH_COMPAT.md](SHODH_COMPAT.md) — surprise は `metadata._hippo.surprise.{score, components}` に格納
- ベンチ: [docs/COMPETITOR_BENCH.md](COMPETITOR_BENCH.md) — store/retrieve レイテンシは surprise 込みで mcp-memory-service-rs より速い
