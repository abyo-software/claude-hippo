# Contributing to claude-hippo

Thank you for your interest in contributing! This project is built in the open and small, focused contributions are welcome.

🇯🇵 日本語の貢献者向けノート: 文書は英語ベースで管理していますが、issue / PR は **英語または日本語のどちらでも** 構いません。コードコメント・docs は英語推奨ですが、補足コメントとして日本語も歓迎です。

## Quick start

```bash
git clone https://github.com/abyo-software/claude-hippo
cd claude-hippo
cargo test                                   # default suite (129 pass + 3 ignored)
cargo test --features candle                 # full suite incl. candle wiring
cargo clippy --features candle --all-targets -- -D warnings
cargo fmt --check
```

If `cargo test` is green and `clippy --all-targets -- -D warnings` is clean, your local environment is ready.

## Workflow

1. **Open an issue first** for non-trivial changes (new MCP tools, schema changes, scoring formula tweaks). For small fixes (typos, docs, tightening assertions), feel free to send the PR directly.
2. **Branch** off `main`. Use `feat/<topic>` or `fix/<topic>` naming.
3. **One logical change per PR.** A bench-harness change shouldn't share a PR with an unrelated CLI flag rename.
4. **Tests are required** for behavior changes. The harness style is documented inline in `tests/eval/mod.rs` — extend the existing pattern rather than introducing a parallel one.
5. **Honest disclosures**: if your change leaves a corner unfinished (e.g. "works for Qwen2 only, Llama-3 in v0.6"), say so explicitly in the PR description and the CHANGELOG entry. We treat partial-but-disclosed as strictly better than appearing-complete-but-broken.

## Coding conventions

- **Rust edition 2021**, MSRV `1.80`. `cargo fmt` enforced; `cargo clippy -- -D warnings` enforced both with and without `--features candle`.
- **Comments**: only when the *why* is non-obvious (hidden constraint, surprising invariant, workaround for a specific upstream bug). Don't restate what well-named identifiers already say.
- **Docs strings**: required on public types and public functions in `src/`. Short paragraph + bullet points beats a multi-page docstring.
- **Tests**: small, targeted, deterministic. If a test needs network or downloads >10 MB, gate it `#[ignore]` and document the manual run command.
- **No backwards-compat shims** for code that's only ever lived in this repo. If you're removing an unused field, just remove it.
- **CHANGELOG.md is part of the change.** Add an entry under `## [Unreleased]` describing what shipped in user-visible terms.

## Areas that especially welcome contributions

- **Additional client compat snippets** (`docs/CLIENT_COMPAT.md`) — Cursor, Continue, Aider, Goose, etc.
- **Bench fixtures** that surface failure modes the current A/B/C/D set doesn't cover (e.g. multilingual content, very long context, code-heavy memories).
- **Schema migrations**: if you find a real-world DB swap edge case with `mcp-memory-service-rs`, a regression test in `scripts/conformance_swap.py` is gold.
- **Embedding backends**: BGE / E5 / multilingual-MiniLM behind the same 384-dim contract.
- **Real-GPU smoke**: candle-cuda actual-GPU latency numbers (Phase E left this as v0.6 work).

## Reporting bugs

- Include `hippo --version`, `cargo --version`, OS, and the relevant `RUST_LOG=hippo=debug` excerpt.
- For data-corruption or schema-related issues, include the output of `hippo verify --db <path>` if possible.
- For surprise-scoring questions, include the input content and expected vs actual ranking.

## Releases

Maintainers cut releases via:

1. `Cargo.toml` version bump
2. `CHANGELOG.md` entry under `## [Unreleased]` → moved to a dated section
3. `chore(release): vX.Y.Z` commit
4. `git tag -a vX.Y.Z`, `git push origin main && git push origin vX.Y.Z`
5. `cargo publish`
6. `gh release create vX.Y.Z --notes-file <notes>`

External contributors don't need to worry about steps 4–6; flag the release intent in your PR description and a maintainer will handle it.

## Security

Please **do not** open public issues for security-sensitive bugs. See [SECURITY.md](SECURITY.md) for the responsible-disclosure path.

## License

By contributing, you agree that your contributions will be dual-licensed under Apache-2.0 OR MIT, matching the project's existing license terms.
