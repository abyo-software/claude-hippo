# claude-hippo 企画書

| 項目 | 値 |
|---|---|
| Status | Draft v0.3（オープン質問全決着版） |
| Updated | 2026-05-09 |
| Author | Youichi / abyo software, LLC |
| Repo (予約) | `github.com/abyo-software/claude-hippo` |
| Crate (確定) | `claude-hippo`（primary）, `abyo-hippo`（防御）。`hippo`/`hippocampus` は他者保有のため断念 |
| Command | `hippo`（binary 名は crate 名と独立） |
| License | Apache-2.0 / MIT dual（mcp-memory-service-rs の PolyForm NC に対する商用フリーポジション） |
| MCP SDK | `rmcp` 1.6.x（Anthropic 公式 Tier 2、9.46M DL、daily merges） |
| 埋め込み | `fastembed` 5.13.x（30+ モデル enum、ort 自動同梱） |

---

## TL;DR（60秒）

Claude Code に **海馬（hippocampus）** を足す MCP サーバ。全部覚える代わりに、**特異性が高い瞬間だけ**を長期記憶化する。Pure Rust 単一バイナリで起動 100ms 以下・常駐 50MB 以下を目標、ローカル動作、Anthropic Memory Tool / SHODH spec の互換レイヤを持つ。差別化は 4 軸：

1. **Surprise-based selection**（人間の海馬模倣、abyo-recall 哲学の継承）
2. **Pure Rust 軽量**（Python 系に対する優位は当然として、唯一の Rust 競合 `mcp-memory-service-rs` の 241MB RSS に対しても <50MB を狙う）
3. **abyo software 既存資産との統合**（probe / speculate / filters）
4. **Apache-2.0 / MIT dual ライセンス**（`mcp-memory-service-rs` は PolyForm Noncommercial で商用クローズド → claude-hippo は商用フリー版のポジション）

直接収益はゼロ前提、abyo software ブランド構築 + Youichi 自身のペイン解決 + Ferro 売却バリュエーション補強の三重投資。

---

## 1. 市場認識（レッドオーシャン）

Claude Code 用 memory MCP は既に競合多数。素朴に「Rust で書きました」では負ける。

### 主要競合（2026-05 時点 / 数値はリポジトリ README・記事ベース推定、要実測検証）

| 競合 | 言語 | 特徴 | 弱点（推定） |
|------|------|------|-------------|
| **mcp-memory-service-rs (doobidoo)** ⚠️ | **Rust** | **直接競合**。SHODH 互換 schema、cold-start 68ms、RSS 241MB、`rmcp` 1.5 採用、M0 scaffold 段階（2026-04 リリース、★2） | **PolyForm Noncommercial** で商用閉鎖、特異性選別なし、abyo 統合なし |
| mcp-memory-service (doobidoo) | Python | 25+ AI app 対応、Cloudflare 同期、ナレッジグラフ可視化、SHODH spec オリジン | Python、汎用、特化なし |
| MemPalace | Python | ChromaDB ベース、LongMemEval ベンチで話題 | Issue #27 にアーキ批判、ベンチが vector store 由来との指摘 |
| MemCP | Python | `/compact` フック、自動グラフ、トークン削減主張 | Python、デバッグツール寄り |
| mcp-memory-keeper (mkreyman) | TypeScript | Claude Code 特化、SQLite、シンプル | 機能が限定的 |
| claude-code-memory (ViralVoodoo) | Python | Neo4j ベース、ナレッジグラフ | Neo4j 依存で重い |
| claude-memory-mcp (WhenMoon-afk) | Python | Tiered memory、Docker | 機能が素朴 |
| Anthropic Memory Tool（公式） | - | beta、公式ロードマップで進行中 | 仕様未確定、特異性選別なし |

> 競合の数値（メモリ常駐、起動時間）は README ベース。`mcp-memory-service-rs` の 68ms / 241MB は公式ベンチ実測値（Apple Silicon）。Sprint S1 で自前再計測する。

### SHODH spec とは
**Shodh Unified Memory API Specification v1.0.0**（[varun29ankuS/shodh-memory](https://github.com/varun29ankuS/shodh-memory)）。サンスクリット語「शोध（探究）」由来。emotional metadata（valence / arousal）、episodic memory（episode_id, sequence_number）、source credibility scoring を含む memory schema 仕様。doobidoo/mcp-memory-service が公式準拠。互換性確保で SQLite ファイルレベルの swap が可能になる。

### 既に解かれている部分には踏み込まない

- ベクトル検索の精度競争（Mem0/MemPalace に分がある）
- ナレッジグラフ可視化（mcp-memory-service が先行）
- 多 client 対応（mcp-memory-service が 25+ で先行）
- 「Python から Rust へ書き換えただけ」の差別化（`mcp-memory-service-rs` が先着）

### 勝負する領域

- **長期セッションでの想起精度**（特異性選別が効く、`mcp-memory-service-rs` にもない）
- **常駐メモリの一段下**（mcp-memory-service-rs 241MB → claude-hippo <50MB を狙う。ONNX を切るか lazy load で達成）
- **LLM 内部状態を活かした surprise 推定**（abyo-llm-probe があるからできる）
- **商用フリー license**（PolyForm NC な mcp-memory-service-rs の代替）

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

### 軸2：Pure Rust 軽量（実測検証済の競合数値あり）

| 指標 | claude-hippo 目標 | mcp-memory-service-rs（公式 Apple Silicon） | mcp-memory-service-rs（本リポ Linux x86 実測） | Python 系（推定） |
|------|------|------|------|------|
| 起動時間（cold） | **< 100 ms** | 68 ms | 99 ms | 1–3 s |
| **メモリ常駐** | **< 50 MB**（一段下） | 241 MB | **186 MB** | 200–800 MB |
| recall レイテンシ p50 | < 5 ms | 8.8 ms | 5.5 ms | 10–100 ms |
| store レイテンシ p50 | < 5 ms | 8.5 ms | 5.4 ms | 10–100 ms |

> 本リポ Linux 実測値の詳細は [docs/COMPETITOR_BENCH.md](docs/COMPETITOR_BENCH.md) 参照。store/retrieve レイテンシは **Linux x86 のほうが Apple Silicon より速い**ことが分かった（warm 後の CPU クロック差）。
>
> **<50 MB 達成戦略**：`cargo bloat` で **ONNX Runtime（ort_sys + 周辺）が binary の 74% を占有**することが判明。これが ~150 MB 常駐の主因。達成パスは:
> - (a) **candle 純 Rust に倒す**（ort 系を全部切る、自前 BertModel 実装が必要、binary は ~18 MB に縮む）
> - (b) `fastembed` のまま **lazy load + idle unload**（cold-start は 23 MB を維持、最初の embed で 173 MB ジャンプ）
> - (c) **External embedding API**（OpenAI/HF/Ollama 経由、binary <20 MB / RSS <30 MB だがネットワーク必須）
>
> Sprint S1 spike で (a)(b) のプロトを作って RSS 計測 → 採用判断。

### 軸3：abyo 既存資産との統合

- **abyo-llm-probe**：ローカル LLM の内部状態（hidden state, attention）を読んで surprise 推定の精度を上げる
- **abyo-speculate**（将来）：内部処理をローカル LLM で speculative に高速化
- **abyo-filters**：メモリ存在判定（probabilistic filter）で空間効率改善

これは他社に複製困難な濠。ただし v0.1 では abyo 統合 **なし** で動くこと。abyo 側が遅延しても claude-hippo は止まらない設計（§9 リスク5）。

### 軸4：Apache-2.0 / MIT dual ライセンス（戦略軸）

唯一の Rust 直接競合 `mcp-memory-service-rs` は **PolyForm Noncommercial 1.0.0**（商用は別契約、henry.krupp@gmail.com 連絡）。エンタープライズ・スタートアップ・コンサル等の商用ユーザは事実上排除されている。claude-hippo は Apache/MIT デュアル → **「商用フリーで使える唯一の Rust 製 SHODH 互換 memory MCP」** という言い切れるポジション。

これは技術差別化ではなく **法務差別化**。HN ブログでも「PolyForm vs Apache/MIT」の対比を前面に出す。

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
- `hippo_forget(id)` — 削除（soft-delete、SHODH 互換のため tombstone 残す）
- `hippo_session_summary()` — セッション要約

設計原則 — **storage と retrieval を分離**：
- **保存**は常にフル（surprise score を column として attach）。法務派ユーザの audit trail を確保
- **取り出し**は surprise-weighted ranking がデフォルト ON、threshold 設定可能
- 「忘れる」は物理削除に紐付けず、recall ランキング下げ + decay model で表現
- これで「全保存モード派 vs 特異性選別派」の対立を解消（§10 (8) の決定）

実装範囲（未着手）：

- [ ] MCP プロトコル：`rmcp` 1.6.x（features: `server`, `macros`, `transport-io`）
- [ ] SQLite：`rusqlite` 0.39 (bundled) + `sqlite-vec` 0.1（mcp-memory-service-rs と同 schema で互換性確保）
- [ ] ローカル埋め込み：`fastembed` 5.13.x（モデルは BGE-small-en-v1.5 デフォルト、`UserDefinedEmbeddingModel` で同梱パス対応）
- [ ] ベクトル検索：sqlite-vec で KNN、`memories.deleted_at IS NOT NULL LIMIT 1` 条件付き oversample（mcp-memory-service-rs の最適化を踏襲）
- [ ] Claude Code 統合の動作確認（`npx @modelcontextprotocol/inspector cargo run` で smoke test）
- [ ] 比較ベンチ基盤（vs mcp-memory-service-rs, mcp-memory-service Python, mcp-memory-keeper, MemCP）

> S1 spike 注意点（埋め込み調査 agent からの finding）：
> 1. `ort` dylib のクロスコンパイル → CI matrix を `macos-14`（arm64）と `ubuntu-latest` で分割
> 2. モデル weight の vendoring vs runtime DL → `~/.cache/claude-hippo/models/` キャッシュ + fallback DL
> 3. `tokenizers` の `onig` feature → CI に C コンパイラ必須、Alpine musl で詰む

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

| 機能 | claude-hippo | mcp-memory-service-rs | mcp-memory-service | MemPalace | MemCP | mcp-memory-keeper |
|------|--------------|----------------------|--------------------|-----------|-------|-------------------|
| 言語 | **Rust** | Rust | Python | Python | Python | TypeScript |
| 起動時間 | **<100ms (目標)** | 68ms (実測) | 3.6s (実測) | 1–3s | 1–3s | <500ms |
| メモリ常駐 | **<50MB (目標)** | 241MB (実測) | 561MB (実測) | 300–800MB | 200–400MB | <100MB |
| ライセンス | **Apache/MIT** | PolyForm NC | Apache-2.0 | — | — | MIT |
| 商用利用 | **⭕ 自由** | ❌ 別契約必要 | ⭕ | ⭕ | ⭕ | ⭕ |
| 特異性ベース選別 | **⭕** | ❌ | ❌ | ❌ | ❌ | ❌ |
| Forgetting curve / decay | **⭕** | ❌ | ❌ | ❌ | △ | ❌ |
| SHODH spec 互換 | **⭕ (v0.3)** | ⭕ | ⭕ (origin) | ❌ | ❌ | ❌ |
| ローカル動作 | ⭕ | ⭕ | ⭕ | ⭕ | ⭕ | ⭕ |
| Cloudflare 同期 | △ (v1.x) | ❌ | ⭕ | ❌ | ❌ | ❌ |
| ナレッジグラフ | △ (v0.3+) | ❌ | ⭕ | ⭕ | ⭕ | ❌ |
| 多 MCP client | ⭕ | ⭕ | ⭕⭕ (25+) | ⭕ | ⭕ | △ |
| LLM 内部状態活用 | **⭕** | ❌ | ❌ | ❌ | ❌ | ❌ |

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

### crate / binary 名（2026-05-09 確定）
- **crate primary**: `claude-hippo`（AVAILABLE 確認済、今週中に v0.0.1 placeholder で予約）
- **crate 防御**: `abyo-hippo`（AVAILABLE 確認済、同上）
- **binary**: `hippo`（crate 名と独立、衝突なし）
- **断念**:
  - `hippo` → 2021 年公開の web asset preprocessor（max_version 0.1.1, downloads 1664）が保有。squat ではないため negotiate 困難
  - `hippocampus` → 2026-01-21 に b0xtch が `Hello, world!` 中身で squat（github.com/b0xtch/hippo）。同コンセプト名のため不快だが、争わず無視

### コンセプト
**hippocampus**（海馬）の親しみやすい愛称。比喩が一発で伝わる、HN タイトルが書ける、覚えやすい。`claude-hippo` は「Claude 専用の海馬」と読めて MCP サーバの所属が明確。

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
| R1 | Anthropic 公式 Memory Tool が claude-hippo の差別化を消す | 中 | 大 | 公式互換レイヤを v0.3 で提供。公式が手を出さない領域（特異性選別、Pure Rust 軽量、abyo 統合）に集中 |
| R2 | Python 系競合が Cloudflare 同期等で先行進化 | 高 | 中 | **SHODH Unified Memory API Spec v1.0.0** に v0.3 で公式準拠（emotional metadata + episodic memory + source credibility）。SQLite ファイルレベル swap で乗り換え動線を確保。Cloudflare 同期は doobidoo の `shodh-cloudflare` に乗っかる |
| R3 | 「特異性選別」の良さが既存ベンチで数値化できない | 高 | 中 | 独自評価軸 + 10 人ヒューマン評価 + ブログで定性評価多用 |
| R4 | 「全部覚えてくれ」派ユーザの拒絶（特に法務・規制） | 中 | 中 | §4 設計で解決済：storage は常にフル、retrieval が surprise-weighted。物理削除は soft-delete のみ |
| R5 | abyo-llm-probe / abyo-speculate の遅延がコア差別化を遅らせる | 中 | 大 | v0.1 は abyo 統合 **なし** で完全動作。abyo 統合は v0.2/v0.3 でオプション機能 |
| R6 | MCP プロトコル変化に追従コスト | 低 | 小 | `rmcp` 1.6.x（Anthropic 公式 Tier 2、daily merges、v1.0 GA 済）を使い、自前実装しない |
| R7 | Youichi 1 人の手数（probe / speculate / FerroSearch / hippo 並走不可） | 高 | 大 | claude-hippo は abyo-speculate Phase 1 完了後に着手。dogfooding で Claude Code 自身に書かせる |
| **R8** | **`mcp-memory-service-rs` が先に traction を獲得し「Rust の選択肢」枠を埋める** | 中 | 大 | 4 軸差別化を全面展開：(a) 特異性選別、(b) RSS 50MB（vs 241MB）、(c) abyo 統合、(d) **Apache/MIT 商用フリー（vs PolyForm NC）**。HN ブログでは正面から名指しで対比、誠実さで認知奪取 |
| **R9** | **`hippocampus` crate squat（b0xtch、2026-01）の絡みでブランド混乱** | 低 | 小 | 名前空間は `claude-hippo` / `abyo-hippo` で確定（§7）。`hippocampus` は争わず無視 |

---

## 10. オープン質問（2026-05-09 決着）

- [x] **(1) MCP SDK** → `rmcp` 1.6.x（Anthropic 公式 Tier 2、9.46M cumulative DL、daily merges、v1.0 GA 済、SEP 取り込み継続）。`rust-mcp-sdk` は OAuth プロバイダー連携が必要になった場合の runner-up。自前実装は仕様改版追従コストで論外
- [x] **(2) ローカル埋め込み** → `fastembed` 5.13.x（30+ モデル enum、ort 自動 fetch + 同梱、5 行で動く）。RSS <50MB が達成不能と判明したら candle 純 Rust に乗り換え（Sprint S1 spike で判断）
- [x] **(3) SHODH 仕様** → **Shodh Unified Memory API Specification v1.0.0**（[varun29ankuS/shodh-memory](https://github.com/varun29ankuS/shodh-memory)、サンスクリット「探究」由来）。emotional metadata、episodic memory、source credibility scoring を含む。doobidoo/mcp-memory-service が公式準拠。**v0.3 で claude-hippo も準拠**して SQLite swap を可能に
- [x] **(4) abyo-recall 哲学のドキュメント分離** → **別ファイル化**。Sprint S2 着手時に `docs/SURPRISE_SELECTION.md` を作成、v0.2 実装と並走で肉付け（§11 タスクに追加）。差別化軸の核なので、外部に説明可能な独立ドキュメントを持つ
- [x] **(5) crates.io 名前空間予約** → **今週中（〜2026-05-16）に `claude-hippo` v0.0.1 と `abyo-hippo` v0.0.1 placeholder を publish**。`hippo`（2021 web preprocessor）と `hippocampus`（2026-01 b0xtch squat）は他者保有なので断念。binary 名 `hippo` は crate 名と独立で衝突しない
- [x] **(6) ライセンス** → **Apache-2.0 / MIT dual で確定**。戦略的意義：`mcp-memory-service-rs` の PolyForm Noncommercial に対する商用フリーポジション（§3 軸4、§9 R8）
- [x] **(7) Cloudflare 同期** → **v1.x 送り**。v0.3 で SHODH wire compat だけ取り、Cloudflare 同期は doobidoo の `shodh-cloudflare` に schema 互換で乗っかる。1 人開発キャパで自前実装するなら SaaS マネタイズ層として後回し
- [x] **(8) 全保存モードのデフォルト** → **storage と retrieval を分離**（§4 設計原則）。保存は常にフル + surprise score column、retrieval が surprise-weighted ranking デフォルト ON。物理削除なし（soft-delete のみ）。これで法務派と特異性選別派を両立

---

## 11. 実装計画（絶対日付）

> 起点：今日 = 2026-05-09。abyo-speculate Phase 1 完了を Week 6 末（≈2026-06-20）と仮置き。実 Phase 完了に応じて再調整。

### Sprint S1 / v0.1 MVP（10–14 日）

| Day | タスク |
|-----|--------|
| 1–2 | リポジトリ初期化、`rmcp` 1.6 で hello tool 動作、`cargo bloat` で binary サイズ計測 |
| 3–4 | `rusqlite` + `sqlite-vec` 統合（SHODH 互換 schema）、`fastembed` で BGE-small 埋め込み、Claude Code 統合 |
| 5–6 | 5 つの基本 MCP tools 実装、`@modelcontextprotocol/inspector` で E2E 動作確認 |
| 7–8 | 比較ベンチ（vs `mcp-memory-service-rs`, `mcp-memory-service`, `mcp-memory-keeper`, `MemCP`）、`criterion` で再現可能化 |
| 9–10 | README / docs / 公開準備、RSS <50MB 達成可否判定（未達なら candle 乗り換え検討） |
| 11–14 | バグ修正、ブログ草稿、crates.io publish、HN 投稿 |

成果物：crates.io 公開、ブログ第 1 弾、HN 投稿

### Sprint S2 / 特異性ベース選別（5–7 日）

| Day | タスク |
|-----|--------|
| 1 | `docs/SURPRISE_SELECTION.md` スタブ作成（理論・式・評価軸を別ドキュメント化） |
| 1–2 | surprise score パイプライン（embedding outlier + engagement signal + explicit marker から開始、prediction loss は abyo-llm-probe 統合 = v0.3） |
| 3–4 | 重要度判定、forgetting curve（decay model）、retrieval ranking |
| 5–6 | 独自評価ベンチ（Long-session noise / Cross-session retrieval） |
| 7 | ブログ第 2 弾、HN 再投稿、`docs/SURPRISE_SELECTION.md` 肉付け完了 |

成果物：v0.2、独自ベンチ結果、ブログ第 2 弾、SURPRISE_SELECTION.md 公開

### Sprint S3 / abyo 統合 + SHODH 互換（5–7 日）

| Day | タスク |
|-----|--------|
| 1–2 | abyo-llm-probe 統合（prediction loss を surprise score に追加） |
| 3 | abyo-filters 内蔵（メモリ存在判定の空間効率化） |
| 4 | SHODH Unified Memory API Spec v1.0.0 準拠（emotional metadata、episodic memory、source credibility column 追加） |
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

## 13. 着手前 TODO（2026-05-09 時点、Sprint S1 開始までに完了）

- [ ] **今週中（〜2026-05-16）**: crates.io に `claude-hippo` v0.0.1 placeholder を publish（squat 防止）
- [ ] **今週中（〜2026-05-16）**: crates.io に `abyo-hippo` v0.0.1 placeholder を publish（防御）
- [ ] **今週中（〜2026-05-16）**: GitHub リポジトリ作成 `abyo-software/claude-hippo`（README に「Sprint S1 着手予定」だけ書く）
- [x] Sprint S1 直前: `mcp-memory-service-rs` を実機 clone + `cargo build` + ベンチ再現（Linux x86 で cold 99ms / store 5.4ms / retrieve 5.5ms / RSS 186MB を計測、[docs/COMPETITOR_BENCH.md](docs/COMPETITOR_BENCH.md)）
- [ ] Sprint S1 直前: `mcp-memory-service` (Python) と MemCP の docker / pip インストールでベンチ環境を整備（claude-hippo Sprint S1 で並走）
- [x] Sprint S1 直前: SHODH spec OpenAPI 読解 + schema 互換性差分を [docs/SHODH_COMPAT.md](docs/SHODH_COMPAT.md) に記載
- [x] §10 オープン質問の決着（本ドキュメントで完了）
- [x] MCP SDK 最新版確認（rmcp 1.6.x で確定）
- [x] 埋め込み比較（fastembed 5.13.x で確定、candle は runner-up）

---

## 14. まとめ

claude-hippo は **abyo software の結節点**であり、**Youichi が毎日使う**プロダクトであり、**Ferro 売却バリュエーション補強**のブランディング装置。

正直に（v0.3 推敲時点で更新）：

- レッドオーシャン（memory MCP は既に複数 + Rust 直接競合 `mcp-memory-service-rs` も 4 月リリース済）
- 4 軸差別化：(a) **特異性選別**、(b) **Pure Rust 軽量（<50MB）**、(c) **abyo 統合**、(d) **Apache/MIT 商用フリー**
- Anthropic 公式 Memory Tool 本格化で (a)(b) が薄れるリスクは大
- 直接収益はゼロ前提

それでも作る理由：

1. Youichi 自身のペインを解決する（dogfooding）
2. abyo software 群の認知導線として最強
3. 「Claude Code に海馬」は HN タイトルが立つ
4. abyo-recall 哲学を `docs/SURPRISE_SELECTION.md` として独立ドキュメント化、独自評価軸で実証する
5. probe / speculate / filters が全部活きる唯一のプロダクト
6. 唯一の Rust 競合が PolyForm Noncommercial で商用閉鎖、Apache/MIT の開放枠が空いている

過剰な期待をせず、地道に作って公開して、ユーザの反応を見ながら育てる。`mcp-memory-service-rs` を素直に名指しで対比し、棲み分けを誠実に書くことで HN/Lobsters の信頼を獲りに行く。

---

## Appendix A. Sprint S1 着手プロンプト（Claude Code 用）

```
@PLAN.md を読んで、claude-hippo Sprint S1 (v0.1 MVP) を開始してくれ。

技術選定は §10 で決着済：
- MCP SDK: rmcp 1.6.x (features: server, macros, transport-io)
- DB: rusqlite 0.39 (bundled) + sqlite-vec 0.1 (SHODH 互換 schema)
- 埋め込み: fastembed 5.13.x (BGE-small-en-v1.5 デフォルト)
- async: tokio 1
- ライセンス: Apache-2.0 / MIT dual
- crate 名: claude-hippo (primary) / abyo-hippo (防御)
- binary 名: hippo

タスク:
1. リポジトリ初期化（abyo-software/claude-hippo）
2. rmcp 1.6 で hello_world tool が動くことを確認 (npx @modelcontextprotocol/inspector)
3. rusqlite + sqlite-vec で記憶ストア（mcp-memory-service-rs と同 schema を最初から）
4. fastembed で埋め込みパイプライン（モデル DL は ~/.cache/claude-hippo/models/ にキャッシュ）
5. 5 つの基本 MCP tools 実装:
   - hippo_remember(content, tags, importance?)
   - hippo_recall(query, limit?)
   - hippo_list_recent(n?)
   - hippo_forget(id)  # soft-delete のみ
   - hippo_session_summary()
   設計原則: storage は常にフル、retrieval が surprise-weighted (v0.2 で本格化、v0.1 は recency + tag match のみ)
6. Claude Code から動作確認（手動 E2E）
7. 比較ベンチ:
   - vs mcp-memory-service-rs (Rust 直接競合) ← 最重要
   - vs mcp-memory-service (Python 上流)
   - vs mcp-memory-keeper (TypeScript)
   - vs MemCP (Python)
   指標: cold-start, RSS, store p50/p95, retrieve p50/p95
8. RSS <50MB 達成可否を判定。未達なら candle 純 Rust に乗り換え検討（PLAN.md §3 軸2 の判断条件）

注意点 (S1 spike で詰まる箇所):
- ort dylib のクロスコンパイル (CI matrix を macos-14 / ubuntu-latest で分割)
- モデル weight の vendoring vs runtime DL (fallback DL を入れる)
- tokenizers の onig feature (CI に C コンパイラ必須)

最初のコミットメッセージは「初期化」レベル可、後で物語を作る。
PLAN.md の未着手チェックボックスを進捗に応じて更新すること。
ベンチ結果は docs/BENCH.md に記録、未達なら正直に書く（CLAUDE.md 「honest limitations 重視」）。
```
