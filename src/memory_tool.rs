//! Anthropic Memory Tool compatibility layer.
//!
//! Anthropic's Memory Tool (`type: "memory_20250818"`) is a *filesystem*-shaped
//! API: clients implement `view` / `create` / `str_replace` / `insert` /
//! `delete` / `rename` operations under a `/memories` namespace. claude-hippo
//! is a *semantic* memory store — these are complementary models.
//!
//! This module bridges them so an MCP client (e.g. Claude Code with the
//! Memory Tool exposure enabled) can store/list/edit memories using
//! Memory-Tool semantics while claude-hippo's surprise-aware retrieval and
//! schema compatibility with `mcp-memory-service-rs` continue to apply.
//!
//! # Mapping
//!
//! - **Path → memory**: every memory is a single "file" at
//!   `/memories/{name}.md`. The `name` is taken from the path; if the user
//!   doesn't pick one, a slug derived from the content hash is used.
//! - **Round-trip storage**: the path is persisted in
//!   `metadata._hippo.memory_tool.path` so subsequent `view` / `delete` /
//!   `str_replace` calls can find the memory by path.
//! - **Tag and surprise**: `create` accepts an optional tag set via the
//!   Memory Tool extension `tags: ["tag1", "tag2"]` shoved into a YAML-ish
//!   front matter; otherwise the memory inherits a single tag derived from
//!   the path (`memory-tool-{name}`). Surprise scoring runs as normal.
//!
//! # Coverage in v0.3
//!
//! All six Memory Tool commands are implemented. `str_replace` and `insert`
//! are *immutable*-friendly: they fetch the file, mutate the in-memory
//! string, and store a new memory + soft-delete the previous one. The
//! claude-hippo metadata `previous_hash` records the lineage so audit
//! history is preserved.
//!
//! # What this is NOT
//!
//! - This does not implement Anthropic's hosted-tool wire format directly
//!   (you don't wire `tools: [{type: "memory_20250818"}]` against the MCP
//!   server). Use this when your MCP client is already brokering tools
//!   to Claude and you want a Memory-Tool-shaped surface.
//! - This is a compatibility layer; the canonical claude-hippo API
//!   (`hippo_remember` / `hippo_recall` / etc.) remains the recommended
//!   surface for surprise-aware retrieval.
//!
//! # Reference
//!
//! Anthropic Memory Tool spec:
//! <https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool>
//! (read 2026-05-10 for this implementation; re-verify before relying on
//! field names in production).

use crate::storage::{self, MemoryRow, Storage};
use crate::HippoError;
use serde::{Deserialize, Serialize};

const MEMORIES_ROOT: &str = "/memories";

/// One unified Params for the `memory` MCP tool — Anthropic's spec uses
/// a discriminated union over `command`, so all per-command fields are
/// `Option<...>` and validated at dispatch time.
#[derive(Debug, Clone, Deserialize, schemars::JsonSchema)]
pub struct MemoryToolParams {
    /// One of: "view", "create", "str_replace", "insert", "delete", "rename".
    pub command: String,
    /// Operand path (most commands). Must start with `/memories`.
    #[serde(default)]
    pub path: Option<String>,
    /// `view`-only: optional `[start, end]` 1-indexed line range. Out of
    /// range silently clamps.
    #[serde(default)]
    pub view_range: Option<Vec<u32>>,
    /// `create`-only: full file text to store.
    #[serde(default)]
    pub file_text: Option<String>,
    /// `str_replace`-only: substring to find (must be unique in the file).
    #[serde(default)]
    pub old_str: Option<String>,
    /// `str_replace`-only: replacement string.
    #[serde(default)]
    pub new_str: Option<String>,
    /// `insert`-only: 0-indexed line after which to insert (0 = before line 1).
    #[serde(default)]
    pub insert_line: Option<u32>,
    /// `insert`-only: text to insert.
    #[serde(default)]
    pub insert_text: Option<String>,
    /// `rename`-only: source path.
    #[serde(default)]
    pub old_path: Option<String>,
    /// `rename`-only: destination path.
    #[serde(default)]
    pub new_path: Option<String>,
    /// claude-hippo extension: tags to attach to the stored memory. Not part
    /// of Anthropic's spec but harmless if unused.
    #[serde(default)]
    pub tags: Option<Vec<String>>,
}

/// Reply from the Memory Tool surface. Anthropic returns plain text
/// strings; we mirror that exactly so a client expecting the upstream
/// shape sees identical responses.
#[derive(Debug, Serialize)]
pub struct MemoryToolReply {
    pub content: String,
}

/// Apply a Memory Tool command against the given storage. Sync — caller
/// holds the `Storage` lock and decides whether to wrap in async.
pub fn dispatch(storage: &mut Storage, params: MemoryToolParams) -> MemoryToolReply {
    let res = match params.command.as_str() {
        "view" => cmd_view(storage, &params),
        "create" => cmd_create(storage, &params),
        "str_replace" => cmd_str_replace(storage, &params),
        "insert" => cmd_insert(storage, &params),
        "delete" => cmd_delete(storage, &params),
        "rename" => cmd_rename(storage, &params),
        other => Err(format!(
            "Unknown command {other:?}. Expected one of: view, create, str_replace, insert, \
             delete, rename."
        )),
    };
    MemoryToolReply {
        content: res.unwrap_or_else(|e| format!("Error: {e}")),
    }
}

// ---------- view ----------

fn cmd_view(storage: &Storage, p: &MemoryToolParams) -> Result<String, String> {
    let path = require_path(p)?;
    validate_under_memories(&path)?;
    if path == MEMORIES_ROOT || path == format!("{MEMORIES_ROOT}/") {
        return view_directory(storage);
    }
    view_file(storage, &path, p.view_range.as_deref())
}

fn view_directory(storage: &Storage) -> Result<String, String> {
    let rows = storage
        .list_recent(1000)
        .map_err(|e| format!("storage list: {e}"))?;
    let mut out = format!(
        "Here're the files and directories up to 2 levels deep in {MEMORIES_ROOT}, excluding \
         hidden items and node_modules:\n"
    );
    let total: usize = rows.iter().map(|m| m.content.len()).sum();
    out.push_str(&format!(
        "{}\t{MEMORIES_ROOT}\n",
        human_readable_size(total as u64)
    ));
    let mut listed: Vec<(String, usize)> = rows
        .iter()
        .filter_map(|m| memory_path(m).map(|p| (p, m.content.len())))
        .collect();
    listed.sort_by(|a, b| a.0.cmp(&b.0));
    for (p, sz) in listed {
        out.push_str(&format!("{}\t{p}\n", human_readable_size(sz as u64)));
    }
    Ok(out)
}

fn view_file(storage: &Storage, path: &str, view_range: Option<&[u32]>) -> Result<String, String> {
    let mem = find_by_path(storage, path)?
        .ok_or_else(|| format!("The path {path} does not exist. Please provide a valid path."))?;
    let lines: Vec<&str> = mem.content.split('\n').collect();
    let n = lines.len();
    if n > 999_999 {
        return Err(format!(
            "File {path} exceeds maximum line limit of 999,999 lines."
        ));
    }
    let (start, end) = match view_range {
        Some([s, e]) if *s >= 1 && *e >= *s => ((*s - 1) as usize, (*e as usize).min(n)),
        _ => (0, n),
    };
    let mut out = format!("Here's the content of {path} with line numbers:\n");
    for (i, line) in lines.iter().enumerate().take(end).skip(start) {
        out.push_str(&format!("{:>6}\t{line}\n", i + 1));
    }
    Ok(out)
}

// ---------- create ----------

fn cmd_create(storage: &mut Storage, p: &MemoryToolParams) -> Result<String, String> {
    let path = require_path(p)?;
    validate_under_memories(&path)?;
    if path == MEMORIES_ROOT || path.ends_with('/') {
        return Err(format!(
            "Cannot create directory entry {path}. claude-hippo's Memory Tool layer is flat."
        ));
    }
    let file_text = p
        .file_text
        .clone()
        .ok_or_else(|| "create requires `file_text`".to_string())?;
    if find_by_path(storage, &path)?.is_some() {
        return Err(format!("File {path} already exists"));
    }
    let tags = p
        .tags
        .clone()
        .unwrap_or_else(|| vec![format!("memory-tool-{}", path_basename(&path))]);
    insert_memory(storage, &path, file_text, tags, None)?;
    Ok(format!("File created successfully at: {path}"))
}

// ---------- str_replace ----------

fn cmd_str_replace(storage: &mut Storage, p: &MemoryToolParams) -> Result<String, String> {
    let path = require_path(p)?;
    validate_under_memories(&path)?;
    let old = p
        .old_str
        .clone()
        .ok_or_else(|| "str_replace requires `old_str`".to_string())?;
    let new = p.new_str.clone().unwrap_or_default();
    let mem = find_by_path(storage, &path)?
        .ok_or_else(|| format!("The path {path} does not exist. Please provide a valid path."))?;
    let occurrences: Vec<usize> = mem
        .content
        .match_indices(&old)
        .map(|(idx, _)| {
            // line number = count of '\n' before idx + 1
            mem.content[..idx].chars().filter(|c| *c == '\n').count() + 1
        })
        .collect();
    match occurrences.len() {
        0 => Err(format!(
            "No replacement was performed, old_str `{old}` did not appear verbatim in {path}."
        )),
        1 => {
            let new_content = mem.content.replacen(&old, &new, 1);
            replace_memory(storage, &mem, new_content)?;
            // Snippet around the change: the modified line + 2 above and below.
            let snippet = format_snippet(
                storage,
                &path,
                occurrences[0].saturating_sub(2),
                occurrences[0] + 2,
            )
            .unwrap_or_default();
            Ok(format!("The memory file has been edited.\n{snippet}"))
        }
        _ => Err(format!(
            "No replacement was performed. Multiple occurrences of old_str `{old}` in lines: \
             {occurrences:?}. Please ensure it is unique"
        )),
    }
}

// ---------- insert ----------

fn cmd_insert(storage: &mut Storage, p: &MemoryToolParams) -> Result<String, String> {
    let path = require_path(p)?;
    validate_under_memories(&path)?;
    let line = p
        .insert_line
        .ok_or_else(|| "insert requires `insert_line`".to_string())?;
    let text = p
        .insert_text
        .clone()
        .ok_or_else(|| "insert requires `insert_text`".to_string())?;
    let mem =
        find_by_path(storage, &path)?.ok_or_else(|| format!("The path {path} does not exist"))?;
    let mut lines: Vec<String> = mem.content.split('\n').map(String::from).collect();
    let n = lines.len();
    if (line as usize) > n {
        return Err(format!(
            "Invalid `insert_line` parameter: {line}. It should be within the range of lines of \
             the file: [0, {n}]"
        ));
    }
    // Memory Tool semantics: insert AFTER the given 0-indexed line.
    let split_text: Vec<String> = text.split('\n').map(String::from).collect();
    let mut new_lines = Vec::with_capacity(n + split_text.len());
    new_lines.extend(lines.drain(..line as usize));
    new_lines.extend(split_text);
    new_lines.extend(lines);
    let new_content = new_lines.join("\n");
    replace_memory(storage, &mem, new_content)?;
    Ok(format!("The file {path} has been edited."))
}

// ---------- delete ----------

fn cmd_delete(storage: &mut Storage, p: &MemoryToolParams) -> Result<String, String> {
    let path = require_path(p)?;
    validate_under_memories(&path)?;
    let mem =
        find_by_path(storage, &path)?.ok_or_else(|| format!("The path {path} does not exist"))?;
    let id = mem.id.ok_or_else(|| "memory has no id".to_string())?;
    storage
        .soft_delete_by_id(id)
        .map_err(|e| format!("delete: {e}"))?;
    Ok(format!("Successfully deleted {path}"))
}

// ---------- rename ----------

fn cmd_rename(storage: &mut Storage, p: &MemoryToolParams) -> Result<String, String> {
    let old_path = p
        .old_path
        .clone()
        .ok_or_else(|| "rename requires `old_path`".to_string())?;
    let new_path = p
        .new_path
        .clone()
        .ok_or_else(|| "rename requires `new_path`".to_string())?;
    validate_under_memories(&old_path)?;
    validate_under_memories(&new_path)?;
    let mem = find_by_path(storage, &old_path)?
        .ok_or_else(|| format!("The path {old_path} does not exist"))?;
    if find_by_path(storage, &new_path)?.is_some() {
        return Err(format!("The destination {new_path} already exists"));
    }
    // Update metadata in-place. We don't change content / hash / created_at.
    let id = mem.id.ok_or_else(|| "memory has no id".to_string())?;
    let mut new_meta = mem.metadata.clone();
    set_path_in_metadata(&mut new_meta, &new_path);
    let new_meta_str =
        serde_json::to_string(&new_meta).map_err(|e| format!("metadata serialize: {e}"))?;
    storage
        .conn()
        .execute(
            "UPDATE memories SET metadata = ?1 WHERE id = ?2",
            rusqlite::params![new_meta_str, id],
        )
        .map_err(|e| format!("rename update: {e}"))?;
    Ok(format!("Successfully renamed {old_path} to {new_path}"))
}

// ---------- helpers ----------

fn require_path(p: &MemoryToolParams) -> Result<String, String> {
    p.path
        .clone()
        .ok_or_else(|| "missing required `path` field".to_string())
}

fn validate_under_memories(path: &str) -> Result<(), String> {
    if !path.starts_with(MEMORIES_ROOT) {
        return Err(format!(
            "path {path} must start with {MEMORIES_ROOT} (Memory Tool security \
             contract — restrict all operations to {MEMORIES_ROOT})"
        ));
    }
    if path.contains("..") {
        return Err(format!("path {path} contains traversal pattern; rejected"));
    }
    Ok(())
}

fn path_basename(path: &str) -> String {
    path.rsplit('/')
        .next()
        .unwrap_or(path)
        .trim_end_matches(".md")
        .to_string()
}

fn memory_path(mem: &MemoryRow) -> Option<String> {
    mem.metadata
        .get("_hippo")
        .and_then(|h| h.get("memory_tool"))
        .and_then(|t| t.get("path"))
        .and_then(|p| p.as_str())
        .map(String::from)
}

fn set_path_in_metadata(meta: &mut serde_json::Value, path: &str) {
    if !meta.is_object() {
        *meta = serde_json::Value::Object(Default::default());
    }
    let m = meta.as_object_mut().expect("ensured object above");
    let hippo = m
        .entry("_hippo")
        .or_insert_with(|| serde_json::Value::Object(Default::default()));
    if !hippo.is_object() {
        *hippo = serde_json::Value::Object(Default::default());
    }
    let h = hippo.as_object_mut().expect("ensured object above");
    let mt = h
        .entry("memory_tool")
        .or_insert_with(|| serde_json::Value::Object(Default::default()));
    if !mt.is_object() {
        *mt = serde_json::Value::Object(Default::default());
    }
    mt.as_object_mut()
        .expect("ensured object above")
        .insert("path".into(), serde_json::Value::String(path.into()));
}

fn find_by_path(storage: &Storage, path: &str) -> Result<Option<MemoryRow>, String> {
    // Linear scan over alive memories — fine for human-scale memory counts
    // (Memory Tool workloads are O(100 to 10k) files, not O(1M)).
    let rows = storage
        .list_recent(10_000)
        .map_err(|e| format!("storage scan: {e}"))?;
    Ok(rows
        .into_iter()
        .find(|m| memory_path(m).as_deref() == Some(path)))
}

fn insert_memory(
    storage: &mut Storage,
    path: &str,
    content: String,
    tags: Vec<String>,
    previous_hash: Option<String>,
) -> Result<i64, String> {
    let mut metadata = serde_json::json!({});
    set_path_in_metadata(&mut metadata, path);
    if let Some(prev) = previous_hash {
        if let Some(obj) = metadata.as_object_mut() {
            if let Some(serde_json::Value::Object(h)) = obj.get_mut("_hippo") {
                if let Some(serde_json::Value::Object(mt)) = h.get_mut("memory_tool") {
                    mt.insert("previous_hash".into(), serde_json::Value::String(prev));
                }
            }
        }
    }
    let row = storage::new_memory_row(content, tags, Some("MemoryToolFile".to_string()), metadata);
    // No embedding inside this layer — semantic search isn't part of the
    // Memory Tool contract. Callers who want surprise-aware retrieval must
    // round-trip through hippo_remember.
    let (id, _dup) = storage
        .insert(&row, None)
        .map_err(|e: HippoError| format!("insert: {e}"))?;
    Ok(id)
}

fn replace_memory(
    storage: &mut Storage,
    old: &MemoryRow,
    new_content: String,
) -> Result<(), String> {
    let path = memory_path(old).ok_or_else(|| "memory missing path".to_string())?;
    let tags = old.tags.clone();
    let previous_hash = old.content_hash.clone();
    let old_id = old.id.ok_or_else(|| "memory has no id".to_string())?;
    insert_memory(storage, &path, new_content, tags, Some(previous_hash))?;
    storage
        .soft_delete_by_id(old_id)
        .map_err(|e| format!("retire previous: {e}"))?;
    Ok(())
}

fn human_readable_size(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    let b = bytes as f64;
    if b < KB {
        format!("{bytes}B")
    } else if b < MB {
        format!("{:.1}K", b / KB)
    } else {
        format!("{:.1}M", b / MB)
    }
}

fn format_snippet(
    storage: &Storage,
    path: &str,
    start_line: usize,
    end_line: usize,
) -> Option<String> {
    let mem = find_by_path(storage, path).ok().flatten()?;
    let lines: Vec<&str> = mem.content.split('\n').collect();
    let n = lines.len();
    let s = start_line.max(1);
    let e = end_line.min(n);
    if s > e {
        return None;
    }
    let mut out = String::new();
    for (i, line) in lines.iter().enumerate().take(e).skip(s - 1) {
        out.push_str(&format!("{:>6}\t{line}\n", i + 1));
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{register_sqlite_vec, Storage};

    fn store() -> Storage {
        register_sqlite_vec();
        Storage::open_in_memory().unwrap()
    }

    #[test]
    fn human_size_formats() {
        assert_eq!(human_readable_size(500), "500B");
        assert_eq!(human_readable_size(2048), "2.0K");
        assert_eq!(human_readable_size(2 * 1024 * 1024), "2.0M");
    }

    #[test]
    fn validate_rejects_outside_memories() {
        assert!(validate_under_memories("/etc/passwd").is_err());
        assert!(validate_under_memories("/memories/../etc/passwd").is_err());
        assert!(validate_under_memories("/memories/foo.md").is_ok());
    }

    #[test]
    fn create_then_view_round_trip() {
        let mut s = store();
        let r1 = dispatch(
            &mut s,
            MemoryToolParams {
                command: "create".into(),
                path: Some("/memories/notes.md".into()),
                file_text: Some("hello\nworld".into()),
                view_range: None,
                old_str: None,
                new_str: None,
                insert_line: None,
                insert_text: None,
                old_path: None,
                new_path: None,
                tags: None,
            },
        );
        assert!(
            r1.content.contains("created successfully"),
            "{}",
            r1.content
        );
        let r2 = dispatch(
            &mut s,
            MemoryToolParams {
                command: "view".into(),
                path: Some("/memories/notes.md".into()),
                view_range: None,
                file_text: None,
                old_str: None,
                new_str: None,
                insert_line: None,
                insert_text: None,
                old_path: None,
                new_path: None,
                tags: None,
            },
        );
        assert!(r2.content.contains("hello"), "{}", r2.content);
        assert!(r2.content.contains("world"), "{}", r2.content);
        // line numbers
        assert!(r2.content.contains("     1"), "{}", r2.content);
    }

    #[test]
    fn view_directory_lists_files() {
        let mut s = store();
        for name in ["alpha", "bravo"] {
            dispatch(
                &mut s,
                MemoryToolParams {
                    command: "create".into(),
                    path: Some(format!("/memories/{name}.md")),
                    file_text: Some(format!("body of {name}")),
                    view_range: None,
                    old_str: None,
                    new_str: None,
                    insert_line: None,
                    insert_text: None,
                    old_path: None,
                    new_path: None,
                    tags: None,
                },
            );
        }
        let r = dispatch(
            &mut s,
            MemoryToolParams {
                command: "view".into(),
                path: Some("/memories".into()),
                view_range: None,
                file_text: None,
                old_str: None,
                new_str: None,
                insert_line: None,
                insert_text: None,
                old_path: None,
                new_path: None,
                tags: None,
            },
        );
        assert!(r.content.contains("/memories/alpha.md"), "{}", r.content);
        assert!(r.content.contains("/memories/bravo.md"), "{}", r.content);
    }

    #[test]
    fn create_rejects_duplicate() {
        let mut s = store();
        let p = MemoryToolParams {
            command: "create".into(),
            path: Some("/memories/x.md".into()),
            file_text: Some("a".into()),
            view_range: None,
            old_str: None,
            new_str: None,
            insert_line: None,
            insert_text: None,
            old_path: None,
            new_path: None,
            tags: None,
        };
        let _ = dispatch(&mut s, p.clone());
        let r = dispatch(&mut s, p);
        assert!(r.content.contains("already exists"), "{}", r.content);
    }

    #[test]
    fn str_replace_unique_match() {
        let mut s = store();
        dispatch(
            &mut s,
            MemoryToolParams {
                command: "create".into(),
                path: Some("/memories/p.md".into()),
                file_text: Some("color: blue".into()),
                view_range: None,
                old_str: None,
                new_str: None,
                insert_line: None,
                insert_text: None,
                old_path: None,
                new_path: None,
                tags: None,
            },
        );
        let r = dispatch(
            &mut s,
            MemoryToolParams {
                command: "str_replace".into(),
                path: Some("/memories/p.md".into()),
                old_str: Some("blue".into()),
                new_str: Some("green".into()),
                view_range: None,
                file_text: None,
                insert_line: None,
                insert_text: None,
                old_path: None,
                new_path: None,
                tags: None,
            },
        );
        assert!(r.content.contains("edited"), "{}", r.content);
        let v = dispatch(
            &mut s,
            MemoryToolParams {
                command: "view".into(),
                path: Some("/memories/p.md".into()),
                view_range: None,
                file_text: None,
                old_str: None,
                new_str: None,
                insert_line: None,
                insert_text: None,
                old_path: None,
                new_path: None,
                tags: None,
            },
        );
        assert!(v.content.contains("green"), "{}", v.content);
        assert!(!v.content.contains("blue"), "{}", v.content);
    }

    #[test]
    fn delete_removes() {
        let mut s = store();
        dispatch(
            &mut s,
            MemoryToolParams {
                command: "create".into(),
                path: Some("/memories/d.md".into()),
                file_text: Some("doomed".into()),
                view_range: None,
                old_str: None,
                new_str: None,
                insert_line: None,
                insert_text: None,
                old_path: None,
                new_path: None,
                tags: None,
            },
        );
        let r = dispatch(
            &mut s,
            MemoryToolParams {
                command: "delete".into(),
                path: Some("/memories/d.md".into()),
                view_range: None,
                file_text: None,
                old_str: None,
                new_str: None,
                insert_line: None,
                insert_text: None,
                old_path: None,
                new_path: None,
                tags: None,
            },
        );
        assert!(r.content.contains("Successfully deleted"), "{}", r.content);
        // Subsequent view fails.
        let v = dispatch(
            &mut s,
            MemoryToolParams {
                command: "view".into(),
                path: Some("/memories/d.md".into()),
                view_range: None,
                file_text: None,
                old_str: None,
                new_str: None,
                insert_line: None,
                insert_text: None,
                old_path: None,
                new_path: None,
                tags: None,
            },
        );
        assert!(v.content.contains("does not exist"), "{}", v.content);
    }
}
