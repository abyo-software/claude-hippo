# claude-hippo 企画書

| 項目 | 値 |
|---|---|
| Status | Draft v0.2（推敲版） |
| Updated | 2026-05-09 |
| Author | Youichi / abyo software, LLC |
| Repo (予約) | `github.com/abyo-software/claude-hippo` |
| Crate (予約) | `hippo`, `claude-hippo` |
| Command | `hippo` |
| License | Apache-2.0 / MIT dual |

---

## TL;DR（60秒）

Claude Code に **海馬（hippocampus）** を足す MCP サーバ。全部覚える代わりに、**特異性が高い瞬間だけ**を長期記憶化する。Pure Rust 単一バイナリで起動 100ms 以下を目標、ローカル動作、Anthropic Memory Tool との互換レイヤを持つ。差別化は 3 軸：

1. **Surprise-based selection**（人間の海馬模倣、abyo-recall 哲学の継承）
2. **Pure Rust 高速**（Python 系 MCP memory に対する起動時間・メモリ常駐優位）
3. **abyo software 既存資産との統合**（probe / speculate / filters）

直接収益はゼロ前提、abyo software ブランド構築 + Youichi 自身のペイン解決 + Ferro 売却バリュエーション補強の三重投資。

---

## 1. 市場認識（レッドオーシャン）

Claude Code 用 memory MCP は既に競合多数。素朴に「Rust で書きました」では負ける。

### 主要競合（2026-05 時点 / 数値はリポジトリ README・記事ベース推定、要実測検証）

| 競合 | 言語 | 特徴 | 弱点（推定） |
|------|------|------|-------------|
| mcp-memory-service (doobidoo) | Python | 25+ AI app 対応、Cloudflare 同期、ナレッジグラフ可視化 | Python、汎用、特化なし |
| MemPalace | Python | ChromaDB ベース、LongMemEval ベンチで話題 | Issue #27 にアーキ批判、ベンチが vector store 由来との指摘 |
| MemCP | Python | `/compact` フック、自動グラフ、トークン削減主張 | Python、デバッグツール寄り |
| mcp-memory-keeper (mkreyman) | TypeScript | Claude Code 特化、SQLite、シンプル | 機能が限定的 |
| claude-code-memory (ViralVoodoo) | Python | Neo4j ベース、ナレッジグラフ | Neo4j 依存で重い |
| claude-memory-mcp (WhenMoon-afk) | Python | Tiered memory、Docker | 機能が素朴 |
| Anthropic Memory Tool（公式） | - | beta、公式ロードマップで進行中 | 仕様未確定、特異性選別なし |

> 競合の数値（メモリ常駐、起動時間）は記事・README ベースの推定値であり、Sprint S1 の比較ベンチで自前計測する。

### 既に解かれている部分には踏み込まない

- ベクトル検索の精度競争（Mem0/MemPalace に分がある）
- ナレッジグラフ可視化（mcp-memory-service が先行）
- 多 client 対応（mcp-memory-service が 25+ で先行）

### 勝負する領域

- **長期セッションでの想起精度**（特異性選別が効く）
- **起動時間 / メモリ常駐**（Pure Rust の物理優位）
- **LLM 内部状態を活かした surprise 推定**（abyo-llm-probe があるからできる）

---

## 2. なぜ作るか

### 一次：Youichi 自身のペイン解決
Claude Code ヘビーユーザとして context rot と cross-session continuity に毎日困る。自分用に作って商品化する。dogfooding が最強の品質保証。

### 二次：abyo software ブランドの結節点
probe / speculate / filters がバラバラだった。claude-hippo は **ユーザが直接体験できる統合プロダクト** になり、他ライブラリの認知導線になる。

### 三次：Ferro 売却バリュエーション補強
「Claude Code に海馬」は HN 受け抜群。abyo-llm-probe + abyo-speculate + claude-hippo の 3 点で「ローカル LLM 推論基盤を書ける検索エンジン作者」のストーリーが完成。

---

## 3. 差別化軸の詳細

### 軸1：Surprise-based Selection

**他社**：全発話を保存してベクトル検索

**claude-hippo**：重要な瞬間だけ強く保存、瑣末は薄く保存または忘却

```rust
// 複数 metric の合成（係数は経験的に調整）
let surprise =
    prediction_loss(event, context) * 0.4   // LLM 予測誤差 (abyo-llm-probe)
    + embedding_outlier(event, history) * 0.3  // 既存記憶からの距離
    + user_engagement_signal(event)     * 0.2  // 応答の長さ・修正頻度
    + explicit_marker(event)            * 0.1; // ユーザ明示マーク
```

解く問題：
- 1 万セッション後にノイズで薄まらない
- 「あの時の重要な決定」だけが想起される
- 人間の記憶っぽい挙動

**評価難度のリスク**：「賢く忘れる」の良さは既存ベンチでは出ない。独自評価軸（§6）と定性評価で証明する。

### 軸2：Pure Rust 高速（目標値、要実測）

| 指標 | claude-hippo 目標 | Python 系（推定） |
|------|------------------|------------------|
| 起動時間（cold） | < 100 ms | 1–3 s |
| メモリ常駐 | < 50 MB | 200–800 MB |
| recall レイテンシ | < 5 ms | 10–100 ms |

> これらは **目標値**。Sprint S1 で自前計測し、未達なら正直に書く。

### 軸3：abyo 既存資産との統合

- **abyo-llm-probe**：ローカル LLM の内部状態（hidden state, attention）を読んで surprise 推定の精度を上げる
- **abyo-speculate**（将来）：内部処理をローカル LLM で speculative に高速化
- **abyo-filters**：メモリ存在判定（probabilistic filter）で空間効率改善

これは他社に複製困難な濠。ただし v0.1 では abyo 統合 **なし** で動くこと。abyo 側が遅延しても claude-hippo は止まらない設計（§9 リスク5）。

---

## 4. 機能仕様

### v0.1 MVP（Sprint S1, 10–14 日）

競合と「最低限同等に動く」ところまで。

```jsonc
// Claude Code 設定例
{
  "mcpServers": {
    "hippo": { "command": "hippo", "args": ["serve"] }
  }
}
```

提供する MCP tools：

- `hippo_remember(content, tags, importance?)` — 明示保存
- `hippo_recall(query, limit?)` — 関連記憶取り出し
- `hippo_list_recent(n?)` — 直近一覧
- `hippo_forget(id)` — 削除
- `hippo_session_summary()` — セッション要約

実装範囲（未着手）：

- [ ] MCP プロトコル（公式 Rust SDK or rust-mcp 系を選定）
- [ ] SQLite (rusqlite) による記憶ストア
- [ ] FastEmbed or candle によるローカル埋め込み
- [ ] ベクトル検索（HNSW、後で abyo-filters 連携）
- [ ] Claude Code 統合の動作確認
- [ ] 比較ベンチ基盤（vs mcp-memory-keeper, MemCP）

### v0.2 Surprise-based Selection（Sprint S2, 5–7 日）

差別化の核。

- [ ] surprise score 算出パイプライン
- [ ] 重要度ベース保存判定
- [ ] forgetting curve（時間減衰モデル）
- [ ] 「特異な瞬間」のハイライト機能
- [ ] 独自評価セットでの実測

### v0.3 abyo 統合（Sprint S3, 5–7 日）

- [ ] abyo-llm-probe 統合（オプション）
- [ ] abyo-filters 内蔵
- [ ] 多 MCP client 対応（Cursor, Continue, Aider）
- [ ] Anthropic Memory Tool 互換レイヤ

### v1.0 仕上げ

- ドキュメント・サンプル・ベンチ結果
- crates.io 公開
- HN / Lobsters / r/ClaudeAI 投稿

---

## 5. 競合比較表（ブログ素材）

> 数値は **目標 / 推定**。Sprint S1 の自前ベンチで上書きする。

| 機能 | claude-hippo | mcp-memory-service | MemPalace | MemCP | mcp-memory-keeper |
|------|--------------|--------------------|-----------|-------|-------------------|
| 言語 | **Rust** | Python | Python | Python | TypeScript |
| 起動時間 | **<100ms (目標)** | 1–3s | 1–3s | 1–3s | <500ms |
| メモリ常駐 | **<50MB (目標)** | 200–500MB | 300–800MB | 200–400MB | <100MB |
| 特異性ベース選別 | **⭕** | ❌ | ❌ | ❌ | ❌ |
| Forgetting curve | **⭕** | ❌ | ❌ | △ | ❌ |
| ローカル動作 | ⭕ | ⭕ | ⭕ | ⭕ | ⭕ |
| Cloudflare 同期 | △ (将来) | ⭕ | ❌ | ❌ | ❌ |
| ナレッジグラフ | △ (v0.3+) | ⭕ | ⭕ | ⭕ | ❌ |
| 多 MCP client | ⭕ | ⭕⭕ (25+) | ⭕ | ⭕ | △ |
| LLM 内部状態活用 | **⭕** | ❌ | ❌ | ❌ | ❌ |

---

## 6. ベンチマーク戦略

「数値で殴る」が HN 受けの鉄則。**最初から測定基盤を作る**。

### 測定対象
- 起動時間（cold / warm）
- メモリ常駐
- レイテンシ（remember / recall）
- 1 万件記憶後のレイテンシ劣化
- 想起精度（独自評価セット）

### 比較対象
mcp-memory-service / MemPalace / MemCP / mcp-memory-keeper

### 独自評価軸（差別化の核）

LongMemEval 等の既存ベンチでは負ける前提。「**特異性ベース選別が活きるシナリオ**」を独自定義する：

1. **Long-session noise**：100 ターン会話に重要決定 5 + 雑談 95 を混ぜ、後から重要決定を想起する精度
2. **Cross-session retrieval**：1 ヶ月前の決定を関連質問から想起できるか
3. **Decision trace**：「なぜこう設計したか」を辿れるか

### 「正直に負ける」ベンチも公開

汎用ベクトル検索精度では MemPalace に負ける可能性が高い。**負ける軸も honestly 公開**する。これが信頼の核。

---

## 7. ネーミング

### claude-hippo / hippo
**hippocampus**（海馬）の親しみやすい愛称。比喩が一発で伝わる、HN タイトルが書ける、覚えやすい、二番煎じが作りにくい。

### マスコット
カバの「Hippo」キャラクター。イラスト、ブログアイコン、エラーメッセージのキャラ化等。

---

## 8. マネタイズ

直接収益はゼロ前提、ブランディング投資。

### 短期（〜半年）
- crates.io 公開、GitHub Star 獲得
- ブログ「Pure Rust で MCP memory server を書いた話」「Claude Code に海馬を追加した話」
- HN / r/ClaudeAI / Lobsters 投稿
- Anthropic MCP コミュニティでの認知獲得

### 中期（半年〜1年）
- 商用サポート契約（特定企業向けカスタマイズ）
- マネージド版 SaaS（Cloudflare ベースの同期、月額 $9–$29）
- エンタープライズ版（チーム間記憶共有、年契約）

### Ferro 売却への効果
abyo-llm-probe + abyo-speculate + claude-hippo の 3 点で「ローカル LLM 推論基盤の専門家」ポジションが完成。Anthropic 自身や AI スタートアップの二次買収候補の関心も期待。

---

## 9. リスク（確率 × 影響度）

| # | リスク | 確率 | 影響 | 対策 |
|---|--------|------|------|------|
| R1 | Anthropic 公式 Memory Tool が claude-hippo の差別化を消す | 中 | 大 | 公式互換レイヤを最初から提供。公式が手を出さない領域（特異性選別、Pure Rust 高速）に集中 |
| R2 | Python 系競合が Cloudflare 同期等で先行進化 | 高 | 中 | SHODH 風の互換仕様を v0.3 で対応（仕様の正式名称は要確認）。速度差は埋められない優位 |
| R3 | 「特異性選別」の良さが既存ベンチで数値化できない | 高 | 中 | 独自評価軸 + 10 人ヒューマン評価 + ブログで定性評価多用 |
| R4 | 「全部覚えてくれ」派ユーザの拒絶（特に法務・規制） | 中 | 中 | デフォルトは全保存 + 重要度スコア付与。「全保存モード」を Pro 機能化 |
| R5 | abyo-llm-probe / abyo-speculate の遅延がコア差別化を遅らせる | 中 | 大 | v0.1 は abyo 統合 **なし** で完全動作。abyo 統合は v0.2/v0.3 でオプション機能 |
| R6 | MCP プロトコル変化に追従コスト | 低 | 小 | 公式 SDK を使い、自前実装しない |
| R7 | Youichi 1 人の手数（probe / speculate / FerroSearch / hippo 並走不可） | 高 | 大 | claude-hippo は abyo-speculate Phase 1 完了後に着手。dogfooding で Claude Code 自身に書かせる |

---

## 10. オープン質問（着手前に決める）

- [ ] MCP SDK は公式 Rust SDK か `rust-mcp` 系か。最新版調査必要
- [ ] ローカル埋め込みは FastEmbed (Rust port) か candle か。バイナリサイズと依存性で判断
- [ ] §9 R2 の「SHODH 仕様」は正式には何の仕様か。mcp-memory-service の互換 spec と思われるが要確認
- [ ] abyo-recall 哲学のドキュメント（surprise based selection の理論的根拠）を別ファイルに切り出すか
- [ ] crates.io 名前空間（`hippo` / `claude-hippo` / `abyo-hippo` / `hippocampus`）の予約タイミング
- [ ] ライセンスは Apache-2.0 / MIT dual で確定か（abyo software 他リポと整合）
- [ ] Cloudflare 同期は v0.3 でやるか v1.x で先送りか
- [ ] 「全保存モード」のデフォルト ON / OFF はどちらか

---

## 11. 実装計画（絶対日付）

> 起点：今日 = 2026-05-09。abyo-speculate Phase 1 完了を Week 6 末（≈2026-06-20）と仮置き。実 Phase 完了に応じて再調整。

### Sprint S1 / v0.1 MVP（10–14 日）

| Day | タスク |
|-----|--------|
| 1–2 | リポジトリ初期化、MCP プロトコル選定、基本 store スケルトン |
| 3–4 | SQLite 統合、ローカル埋め込み（FastEmbed / candle）、Claude Code 統合 |
| 5–6 | 5 つの基本 MCP tools 実装、E2E 動作確認 |
| 7–8 | mcp-memory-keeper / MemCP との比較ベンチ（自前計測） |
| 9–10 | README / docs / 公開準備 |
| 11–14 | バグ修正、ブログ草稿、crates.io 公開、HN 投稿 |

成果物：crates.io 公開、ブログ第 1 弾、HN 投稿

### Sprint S2 / 特異性ベース選別（5–7 日）

| Day | タスク |
|-----|--------|
| 1–2 | surprise score パイプライン |
| 3–4 | 重要度判定、forgetting curve |
| 5–6 | 独自評価ベンチ（Long-session noise / Cross-session retrieval） |
| 7 | ブログ第 2 弾、HN 再投稿 |

成果物：v0.2、独自ベンチ結果、ブログ第 2 弾

### Sprint S3 / abyo 統合（5–7 日）

| Day | タスク |
|-----|--------|
| 1–2 | abyo-llm-probe 統合（surprise 高度化） |
| 3–4 | abyo-filters 内蔵 |
| 5–6 | 多 MCP client 対応（Cursor / Continue / Aider）、Anthropic Memory Tool 互換レイヤ |
| 7 | v1.0 公開、総括ブログ |

### 統合ロードマップでの位置

| Sprint | 本筋 | 並行 |
|--------|------|------|
| Sprint 1 (Week 1–2) | ExaLogLog + abyo-filters | — |
| Sprint 2 (Week 3–4) | FerroSearch 強化 | abyo-llm-probe（実行中） |
| Sprint 3 (Week 5–6) | FerroStream | abyo-speculate Phase 1 |
| Sprint 4 (Week 7–8) | AI Forecaster | abyo-speculate Phase 2–3 |
| **Sprint 5 (Week 9–10)** | — | **claude-hippo v0.1 + v0.2** |
| **Sprint 6 (Week 11–12)** | — | **claude-hippo v0.3 + v1.0** |

---

## 12. 成功基準

### 最低限（実験として成立）
- v0.1 が動く、Claude Code から呼び出せる
- crates.io 公開
- ブログ + HN 投稿

### 部分成功（プロダクトとして立ち上がり）
- GitHub Star 200+（半年）
- 採用 10 リポ以上
- HN フロントページ入り 1 回以上
- 既存 memory MCP からの乗り換え事例

### フル成功
- Claude Code memory MCP の default 選択肢の 1 つに認知される
- Anthropic 公式 docs で言及
- 商用採用 1 社以上
- SaaS で月収 $1000+（半年〜1 年）

---

## 13. 着手前 TODO

- [ ] crates.io 名前空間予約：`claude-hippo`, `hippo`, `abyo-hippo`, `hippocampus`
- [ ] GitHub リポジトリ作成：`abyo-software/claude-hippo`
- [ ] 競合の最新版を実機試用（特に MemCP, mcp-memory-service）
- [ ] MCP SDK 最新版確認（公式 Rust SDK の状況）
- [ ] FastEmbed / candle 比較（バイナリサイズ、起動時間、依存）
- [ ] §10 オープン質問の決着

---

## 14. まとめ

claude-hippo は **abyo software の結節点**であり、**Youichi が毎日使う**プロダクトであり、**Ferro 売却バリュエーション補強**のブランディング装置。

正直に：

- レッドオーシャン（memory MCP は既に複数）
- Pure Rust + 特異性選別 + abyo 統合 の 3 軸で差別化
- Anthropic 公式 Memory Tool 本格化で差別化が薄れるリスクは大
- 直接収益はゼロ前提

それでも作る理由：

1. Youichi 自身のペインを解決する（dogfooding）
2. abyo software 群の認知導線として最強
3. 「Claude Code に海馬」は HN タイトルが立つ
4. abyo-recall 哲学を統合機能として復活できる
5. probe / speculate / filters が全部活きる唯一のプロダクト

過剰な期待をせず、地道に作って公開して、ユーザの反応を見ながら育てる。

---

## Appendix A. Sprint S1 着手プロンプト（Claude Code 用）

```
@PLAN.md を読んで、claude-hippo Sprint S1 (v0.1 MVP) を開始してくれ。

タスク:
1. リポジトリ初期化（abyo-software/claude-hippo）
2. MCP SDK 統合（公式 Rust SDK か rust-mcp 系を選定し PLAN.md §10 を更新）
3. SQLite + ローカル埋め込み（FastEmbed / candle）の記憶ストア
4. 5 つの基本 MCP tools 実装:
   - hippo_remember
   - hippo_recall
   - hippo_list_recent
   - hippo_forget
   - hippo_session_summary
5. Claude Code から動作確認（手動 E2E）
6. mcp-memory-keeper, MemCP との起動時間・メモリ使用量比較ベンチ

ライセンス: Apache-2.0 / MIT dual
authors: abyo software, LLC
コマンド名: hippo / crate 名: hippo + claude-hippo（両方押さえる）
最初のコミットメッセージは「初期化」レベル可、後で物語を作る。
PLAN.md の未着手チェックボックスを進捗に応じて更新すること。
```
