# Security Policy

🇯🇵 日本語: 機密性の高い脆弱性報告は **GitHub Issue ではなく** メールでお願いします。詳細は下記。

## Supported versions

Only the latest minor release on `main` receives security fixes. We do not maintain back-ports.

| Version | Status |
|---|---|
| 0.5.x | ✅ Active support |
| 0.4.x and earlier | ❌ Please upgrade |

## Reporting a vulnerability

**Please do not open a public GitHub issue for security bugs.**

Email **youichi.uda@gmail.com** with:

- A description of the issue and its potential impact
- Steps to reproduce (a minimal failing example or PoC if possible)
- Affected versions
- Your preferred name / handle for credit (or "anonymous" if you prefer)

You can encrypt sensitive reports with GPG; request the maintainer's public key in the same email and a key fingerprint will be returned out-of-band before details are exchanged.

## What to expect

- **Acknowledgment within 72 hours** (often within 24 hours).
- **Triage and fix-or-mitigate plan within 7 days** for High / Critical issues.
- **Coordinated disclosure**: we'll work with you on a public-disclosure timeline. Default is 90 days from initial report or until a fix ships in a published `cargo install claude-hippo` release, whichever comes first. Earlier disclosure is fine if the fix is trivial; later is fine if active exploitation requires careful coordination.
- **Credit in the release notes** unless you prefer anonymity.

## Scope

In scope:

- Memory-corruption or panics in `claude-hippo` itself (not in upstream `rusqlite`, `sqlite-vec`, `fastembed`, `candle-*` — please report those upstream and CC us)
- SQL injection or schema-confusion against the SHODH DB-swap surface
- Authentication / authorization gaps in the SHODH REST endpoints (`--shodh-rest`)
- Path-traversal or arbitrary-file-read via `--db PATH` / `HIPPO_DB_PATH` / model cache flags
- Prompt-injection or content-injection vectors in the surprise-scoring path that lead to denial of service, integrity corruption of the memory store, or unintended Claude tool calls
- Supply-chain concerns about the published crate (`cargo install claude-hippo`)

Out of scope:

- DoS via legitimate but expensive recall queries against a corpus of the user's choosing — that's a configuration and capacity issue, not a vulnerability
- Findings that require a malicious local user with write access to the user's `~/.claude/` or `~/.local/share/`
- Findings against `mcp-memory-service-rs` itself (please file with [doobidoo/mcp-memory-service-rs](https://github.com/doobidoo/mcp-memory-service-rs))
- Theoretical timing-side-channels in cosine similarity (we don't claim any timing guarantees)

## Honest disclosures

claude-hippo embeds optional LLM weight loading via `hf-hub` when built with `--features candle`. If you're concerned about supply-chain integrity of those weights, see <https://huggingface.co/docs/hub/security-malware>; we don't independently verify weight files beyond what `safetensors` does.

When `--shodh-rest` is enabled, the REST surface listens on `127.0.0.1:8765` by default with **no built-in authentication**. This is intentional — the design assumes a localhost-only deployment or a reverse proxy in front. Exposing the bind address to a public interface without an auth proxy is **not supported** and would be considered user error rather than a vulnerability.

— [abyo software, LLC](https://github.com/abyo-software)
