# MCP Client Compatibility

claude-hippo speaks MCP over stdio (rmcp 1.6) and exposes 5 native tools
plus 4 SHODH-compatible aliases. Any MCP client that can spawn an stdio
subprocess can use it. This doc lists the per-client setup snippets and
the verification status for each.

## Tools exposed

| Tool                       | SHODH alias       | Purpose                                                |
|----------------------------|-------------------|--------------------------------------------------------|
| `hippo_remember`           | `store_memory`    | Store memory + compute surprise score                  |
| `hippo_recall`             | `retrieve_memory` | Semantic search + surprise-weighted ranking            |
| `hippo_list_recent`        | `list_memories`   | Most recent N memories                                 |
| `hippo_forget`             | `delete_memory`   | Soft-delete by content_hash or id                      |
| `hippo_session_summary`    | —                 | by_type / top_tags / highlights / mean_surprise        |
| `ping`                     | —                 | Health probe (vec_version, alive count, uptime)        |

All clients see the same tool list. Surprise rerank is on by default in
`hippo_recall`; pass `no_surprise_boost: true` for pure cosine.

---

## Claude Code (reference)

Already covered in [`docs/CLAUDE_CODE_SETUP.md`](CLAUDE_CODE_SETUP.md).
Verified end-to-end (`claude mcp add hippo hippo serve`, then natural
language `remember` / `recall` / `summary` from inside the chat).

---

## Cursor

Cursor 0.43+ supports MCP servers via `~/.cursor/mcp.json` (global) or
`.cursor/mcp.json` (project-local).

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

Restart Cursor; the `hippo_*` tools should appear in the tool list. To
verify:

1. Open the chat panel.
2. Ask `Use the hippo_remember tool to store "Cursor smoke test" with tag "smoke"`.
3. Ask `Use hippo_list_recent to fetch 1 memory`. The response should
   include `"Cursor smoke test"`.

**Verification status**: documented; per-release smoke-tested by hand
(Cursor cannot be installed in CI, so this is a release checklist item,
not an automated assertion).

---

## Continue.dev

Continue 0.9.x supports MCP via `~/.continue/config.json` (or the
project-local `.continue/config.json`):

```jsonc
{
  "models": [/* ... your existing models ... */],
  "experimental": {
    "modelContextProtocolServers": [
      {
        "transport": {
          "type": "stdio",
          "command": "hippo",
          "args": ["serve"],
          "env": {
            "HIPPO_DB_PATH": "/home/you/.local/share/claude-hippo/memory.db"
          }
        }
      }
    ]
  }
}
```

Continue's MCP support is under `experimental` and the schema may
churn; keep an eye on
<https://github.com/continuedev/continue/blob/main/docs/docs/customize/deep-dives/mcp.md>.

**Verification status**: documented; smoke-tested per release.

---

## Aider

Aider does **not** speak MCP natively. To wire claude-hippo into an Aider
workflow, treat it as an out-of-band store and have Aider shell out:

```bash
# In your aider session, you can register a custom command that calls
# the binary directly:
hippo embed "search query"          # one-off embedding for ad-hoc use
hippo verify                         # check the DB
```

For semantic recall, run `hippo serve` in a side terminal and use a
small Python script (`scripts/aider_recall.py` — TODO) that calls the
MCP tools and pastes results into Aider's prompt. There is no official
Aider MCP integration as of this writing; the recommended approach is
"use Claude Code or Cursor for MCP-driven memory, use Aider for raw
file editing".

**Verification status**: known limitation; no integration shipped.

---

## OpenAI Agent SDK / generic MCP runners

Any MCP-compliant runner works. Spawn the server:

```bash
hippo serve --db ./memory.db
```

The handshake follows the standard MCP `initialize` → `tools/list` →
`tools/call` flow. See <https://github.com/modelcontextprotocol/specification>
for protocol details.

For HTTP transport (some hosted runners require it), wait for v0.3.x's
`--shodh-rest` flag (Phase D-3) which exposes the same tool surface
over a REST endpoint.

---

## Verification checklist (per release)

Before tagging `vX.Y.Z`, confirm at least one of:

- [ ] Claude Code: `claude mcp test hippo` returns ok + a manual
  remember+recall round-trip succeeds.
- [ ] Cursor: snippet above produces a working tool list and a manual
  remember+recall round-trip succeeds.
- [ ] Continue: snippet above shows the tools in the chat panel and a
  manual round-trip succeeds.

CI does not exercise any of these — they require GUI clients. The
on-the-record proof is the rmcp stdio integration test
(`tests/mcp_stdio.rs`), which exercises the wire protocol directly.

---

## See also

- [`docs/CLAUDE_CODE_SETUP.md`](CLAUDE_CODE_SETUP.md) — Claude Code installer + walkthrough
- [`docs/SHODH_COMPAT.md`](SHODH_COMPAT.md) — DB schema + tool alias rationale
- [`tests/mcp_stdio.rs`](../tests/mcp_stdio.rs) — automated wire-protocol test
