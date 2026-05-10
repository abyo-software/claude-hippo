# Show HN draft — claude-hippo v0.4

> ステータス: ドラフト v1 (2026-05-10、v0.4.0 release 直後)
> 投稿先候補: HN / Lobsters / r/ClaudeAI / r/rust / r/LocalLLaMA
> 投稿前に: GitHub stars, crates.io DL を確認、自分の hippo serve smoke を最終チェック

---

## Title 候補

- **Show HN: claude-hippo — surprise-aware memory MCP for Claude Code, in Pure Rust**
- **Show HN: I added a hippocampus to Claude Code (Pure Rust MCP server)**
- **Show HN: Memory MCP that scores surprise instead of dumping everything (Rust, 25 MB RSS)**

「surprise」に引きが集まりやすいので 1番目 or 3番目を推奨。

---

## Body (英語、HN 想定)

claude-hippo is a Memory MCP server for Claude Code (and Cursor / Continue / any
MCP client) that does something other memory layers don't: it computes a
**surprise score** for every memory and uses it to re-rank retrieval. The
intuition is that on a long session, the most useful "memory" of your past
work isn't "what did I just say five turns ago" — it's "what was the
non-obvious decision I made three weeks ago that I'd forget without help."

### What's there

- **MCP stdio** for Claude Code, Cursor, Continue
- **SHODH OpenAPI v1.0.0 REST** (all 13 endpoints) for hosted-runner / cron use
- **Anthropic Memory Tool compatibility** (`memory_20250818` filesystem
  facade with view/create/str_replace/insert/delete/rename under `/memories`)
- **External embedding backend** (OpenAI / Azure / Ollama / vLLM / HF TEI),
  L2 + 384 dim enforced for DB swap with mcp-memory-service-rs
- **External prediction-loss backend** for the `prediction_loss`
  surprise component via `/v1/completions echo + max_tokens=0 + logprobs`
  (works with vLLM, llama.cpp, legacy OpenAI completions)
- Schema-compatible with mcp-memory-service-rs — same SQLite file works in
  either tool, so you can swap the binary without losing memories

### What the surprise scoring does

A small bench fixture (5 topic clusters × 1 important Decision + 19 chat
notes per topic = 100 items, 25 paraphrased queries) shows:

```
                          baseline (pure cosine)   v0.4 surprise rerank
precision@1 (find Decision at rank 1)        0.08                   1.000
recall@5                                     0.28                   1.000
```

Without surprise, the actual decision is buried in 19 chat notes that
share the same topic; cosine similarity can't tell them apart. With
surprise rerank (engagement + outlier + explicit + decay-floored time
weight), the decision floats to the top.

For "old high-importance decisions still surface 1 year later" we added a
`decay_floor=0.5` in v0.3 — without it, fresh chat noise was actively
demoting old decisions because the decay term went to zero. Bench B at
365 days (12 half-lives): MRR went from a v0.2 negative-lift result to
**+0.875 over baseline** in v0.4.

### Why Pure Rust

- `cargo install claude-hippo` is a single static binary, no Python venv
  or Node runtime
- Local fastembed (ONNX) backend: warm RSS 150 MB, store p50 3.1 ms
- External embedding backend (Ollama / OpenAI): in-process RSS 25.7 MB,
  store p50 11 ms (the network round-trip dominates)
- License is Apache-2.0 OR MIT (commercial-friendly), unlike the
  PolyForm Noncommercial license of mcp-memory-service-rs

### v0.4 also includes

- In-process dual-serve: stdio MCP + SHODH REST in the same process via a
  shared `Arc<Mutex<Storage>>`
- 98 tests including 9 wiremock embedding tests, 9 wiremock prediction-
  loss tests, 4 axum REST tests, and 4 deterministic surprise benches
- A spike showing pure-Rust local prediction-loss with candle-rs (Qwen2.5-
  0.5B on CPU) — gives the right surprise gradient (cliché ~0.20,
  specific technical decision ~0.83), v0.5 will productionize behind
  `--features candle`

### Honest limitations

- **prediction_loss real LLM bench is local-spike only**, not a full
  Bench A/B/C re-run with a real model. Ollama doesn't honor
  `echo + max_tokens=0 + logprobs`, so the wired-in path needs vLLM or
  llama.cpp (or wait for the candle-rs native backend in v0.5).
- The "no users yet" caveat: this is a fresh release. crates.io DL count
  is in the double digits at posting time.
- Bench A/B/C use a deterministic mock embedder so the score arithmetic
  is reproducible in CI. Real ONNX embeddings move the absolute numbers
  but the surprise rerank effect is independent of embedding distribution
  (it's a re-ranking layer).

### Repo / install

- crates.io: <https://crates.io/crates/claude-hippo>
- GitHub: <https://github.com/abyo-software/claude-hippo>
- `cargo install claude-hippo && hippo serve` — done

Happy to answer questions about the surprise scoring math, the decay
floor decision, the SHODH compat layer, or what didn't work along the way.

---

## 投稿時の注意

- 投稿時刻は HN なら平日 8-10 AM PT (US 西海岸出社時間)
- 「Show HN:」は title の頭に必ず付ける
- 自演ブースト禁止 (HN guideline)、純粋に技術寄せて議論誘導
- 質問 (FAQ 想定):
  - Q: Why Rust over Python? → 「single static binary, no venv, RSS 25 MB」
  - Q: Difference vs mcp-memory-service-rs? → 「surprise rerank + Apache MIT + 4 transport」
  - Q: How does this differ from RAG? → 「per-memory surprise score informs ranking, not just cosine」
  - Q: Production ready? → 「85 tests, schema-compatible with prior tool, but pre-1.0 — feedback welcome」

## A/B テスト用 alternate body (短縮版、Lobsters / r/rust 向け)

claude-hippo is a memory MCP server in Pure Rust that computes a
**surprise score** for each memory and uses it to re-rank retrieval. On a
fixture with 5 topic clusters of 1 Decision + 19 chat notes each, this
lifts precision@1 from 0.08 (cosine baseline) to 1.000.

Schema-compatible with mcp-memory-service-rs (same SQLite file works in
either), 4 transports (MCP stdio / SHODH REST / Anthropic Memory Tool /
external embedding HTTP), 25 MB in-process RSS in external mode.

Apache-2.0 / MIT, `cargo install claude-hippo`, repo at
<https://github.com/abyo-software/claude-hippo>.
