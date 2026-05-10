//! Storage layer — SQLite + sqlite-vec, mcp-memory-service-rs と schema 互換。
//!
//! - 11 列 schema を verbatim コピー → 同じ DB ファイルを両者で読み書き可能
//! - surprise_score / surprise_components は `metadata` JSON 列に格納する
//!   (専用列を増やすと competitor の `CREATE TABLE IF NOT EXISTS` で
//!   自動マイグレーションが走らないため、JSON に閉じ込める)
//! - soft-delete only (`deleted_at REAL`)、物理削除は API 層で禁止
//! - tag は comma-separated 文字列、case-sensitive、whitespace-stripped GLOB

use crate::{HippoError, Result, EMBEDDING_DIM};
use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;
use zerocopy::AsBytes;

/// mcp-memory-service-rs と verbatim 一致する schema。
/// DB ファイルレベルで両者が drop-in swap 可能。
pub const SCHEMA_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS memories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash    TEXT UNIQUE NOT NULL,
    content         TEXT NOT NULL,
    tags            TEXT,
    memory_type     TEXT,
    metadata        TEXT,
    created_at      REAL,
    updated_at      REAL,
    created_at_iso  TEXT,
    updated_at_iso  TEXT,
    deleted_at      REAL DEFAULT NULL
);

CREATE INDEX IF NOT EXISTS idx_content_hash ON memories(content_hash);
CREATE INDEX IF NOT EXISTS idx_created_at  ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_deleted_at  ON memories(deleted_at);

CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings USING vec0(
    content_embedding FLOAT[384] distance_metric=cosine
);
"#;

const PRAGMAS_SQL: &str = r#"
PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 5000;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = 10000;
PRAGMA temp_store = MEMORY;
"#;

/// sqlite-vec extension をすべての connection で auto-load する。
/// `Connection::open` を呼ぶ前に **一度だけ** 呼び出すこと。
pub fn register_sqlite_vec() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // SAFETY: sqlite_vec::sqlite3_vec_init is the canonical extension entry
        // point with the exact ABI sqlite3 expects.
        unsafe {
            #[allow(clippy::missing_transmute_annotations)]
            let f: unsafe extern "C" fn(
                *mut rusqlite::ffi::sqlite3,
                *mut *mut std::os::raw::c_char,
                *const rusqlite::ffi::sqlite3_api_routines,
            ) -> std::os::raw::c_int =
                std::mem::transmute(sqlite_vec::sqlite3_vec_init as *const ());
            rusqlite::ffi::sqlite3_auto_extension(Some(f));
        }
    });
}

/// Memory record. `metadata` は free-form JSON、surprise score は
/// `_hippo.surprise` namespace で詰める。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct MemoryRow {
    pub id: Option<i64>,
    pub content_hash: String,
    pub content: String,
    pub tags: Vec<String>,
    pub memory_type: Option<String>,
    /// JSON object, free-form. claude-hippo は `_hippo` キー namespace を予約。
    pub metadata: serde_json::Value,
    pub created_at: f64,
    pub updated_at: f64,
    pub created_at_iso: String,
    pub updated_at_iso: String,
    /// soft-delete tombstone (None = alive)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deleted_at: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TagMatch {
    Any,
    All,
}

impl TagMatch {
    pub fn parse(s: &str) -> Self {
        if s.eq_ignore_ascii_case("all") {
            Self::All
        } else {
            Self::Any
        }
    }
}

pub struct Storage {
    conn: Connection,
}

impl Storage {
    /// DB を開いて schema を適用する。`register_sqlite_vec()` が事前に
    /// 呼ばれていることが必要。
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(PRAGMAS_SQL)?;
        conn.execute_batch(SCHEMA_SQL)
            .map_err(|e| HippoError::Schema(format!("apply schema: {e}")))?;
        Ok(Self { conn })
    }

    /// in-memory DB (tests).
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(PRAGMAS_SQL)?;
        conn.execute_batch(SCHEMA_SQL)
            .map_err(|e| HippoError::Schema(format!("apply schema: {e}")))?;
        Ok(Self { conn })
    }

    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    pub fn vec_version(&self) -> Result<String> {
        Ok(self
            .conn
            .query_row("SELECT vec_version()", [], |r| r.get(0))?)
    }

    pub fn count_alive(&self) -> Result<i64> {
        Ok(self.conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL",
            [],
            |r| r.get(0),
        )?)
    }

    pub fn count_total(&self) -> Result<i64> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))?)
    }

    /// Insert a memory + its embedding. content_hash UNIQUE collision → no-op
    /// で `Ok((existing_id, true))` を返す (duplicate=true)。
    /// embedding が None の場合は memory 行のみ insert (テスト用)。
    pub fn insert(&mut self, row: &MemoryRow, embedding: Option<&[f32]>) -> Result<(i64, bool)> {
        if let Some(emb) = embedding {
            if emb.len() != EMBEDDING_DIM {
                return Err(HippoError::Embedding(format!(
                    "embedding dim mismatch: expected {EMBEDDING_DIM}, got {}",
                    emb.len()
                )));
            }
        }

        let tx = self.conn.transaction()?;

        // Dedup check.
        let existing: Option<i64> = tx
            .query_row(
                "SELECT id FROM memories WHERE content_hash = ?1",
                params![row.content_hash],
                |r| r.get(0),
            )
            .optional()?;
        if let Some(id) = existing {
            return Ok((id, true));
        }

        let tags_str = encode_tags(&row.tags);
        let metadata_str = serde_json::to_string(&row.metadata)?;

        tx.execute(
            "INSERT INTO memories (
                content_hash, content, tags, memory_type, metadata,
                created_at, updated_at, created_at_iso, updated_at_iso
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                row.content_hash,
                row.content,
                tags_str,
                row.memory_type,
                metadata_str,
                row.created_at,
                row.updated_at,
                row.created_at_iso,
                row.updated_at_iso,
            ],
        )?;
        let id = tx.last_insert_rowid();

        if let Some(emb) = embedding {
            tx.execute(
                "INSERT INTO memory_embeddings (rowid, content_embedding) VALUES (?1, ?2)",
                params![id, emb.as_bytes()],
            )?;
        }

        tx.commit()?;
        Ok((id, false))
    }

    /// content_hash で 1 件取得。
    pub fn get_by_hash(&self, hash: &str) -> Result<Option<MemoryRow>> {
        let row = self
            .conn
            .query_row(
                "SELECT id, content_hash, content, tags, memory_type, metadata,
                        created_at, updated_at, created_at_iso, updated_at_iso, deleted_at
                 FROM memories WHERE content_hash = ?1",
                params![hash],
                row_from_sql,
            )
            .optional()?;
        Ok(row)
    }

    /// id で 1 件取得。
    pub fn get_by_id(&self, id: i64) -> Result<Option<MemoryRow>> {
        let row = self
            .conn
            .query_row(
                "SELECT id, content_hash, content, tags, memory_type, metadata,
                        created_at, updated_at, created_at_iso, updated_at_iso, deleted_at
                 FROM memories WHERE id = ?1",
                params![id],
                row_from_sql,
            )
            .optional()?;
        Ok(row)
    }

    /// 直近 n 件 (alive) を created_at DESC で。
    pub fn list_recent(&self, n: i64) -> Result<Vec<MemoryRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, content_hash, content, tags, memory_type, metadata,
                    created_at, updated_at, created_at_iso, updated_at_iso, deleted_at
             FROM memories
             WHERE deleted_at IS NULL
             ORDER BY created_at DESC
             LIMIT ?1",
        )?;
        let rows = stmt
            .query_map(params![n], row_from_sql)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// tag GLOB 検索。mcp-memory-service-rs / Python upstream と完全同一の
    /// pattern: `(',' || REPLACE(tags, ' ', '') || ',') GLOB '*,<tag>,*'`
    pub fn search_by_tag(
        &self,
        tags: &[String],
        match_mode: TagMatch,
        memory_type: Option<&str>,
        limit: i64,
    ) -> Result<Vec<MemoryRow>> {
        if tags.is_empty() {
            return Ok(Vec::new());
        }

        let mut where_parts: Vec<String> = Vec::new();
        let mut params_vec: Vec<rusqlite::types::Value> = Vec::new();

        where_parts.push("deleted_at IS NULL".into());

        let op = match match_mode {
            TagMatch::Any => "OR",
            TagMatch::All => "AND",
        };

        let tag_clauses: Vec<String> = (0..tags.len())
            .map(|_| "(',' || REPLACE(IFNULL(tags, ''), ' ', '') || ',') GLOB ?".to_string())
            .collect();
        where_parts.push(format!("({})", tag_clauses.join(&format!(" {op} "))));
        for t in tags {
            params_vec.push(rusqlite::types::Value::from(format!(
                "*,{},*",
                t.replace(' ', "")
            )));
        }

        if let Some(mt) = memory_type {
            where_parts.push("memory_type = ?".into());
            params_vec.push(rusqlite::types::Value::from(mt.to_string()));
        }

        let sql = format!(
            "SELECT id, content_hash, content, tags, memory_type, metadata,
                    created_at, updated_at, created_at_iso, updated_at_iso, deleted_at
             FROM memories
             WHERE {}
             ORDER BY created_at DESC
             LIMIT {}",
            where_parts.join(" AND "),
            limit.max(0)
        );

        let mut stmt = self.conn.prepare(&sql)?;
        let params_refs: Vec<&dyn rusqlite::ToSql> = params_vec
            .iter()
            .map(|v| v as &dyn rusqlite::ToSql)
            .collect();
        let rows = stmt
            .query_map(params_refs.as_slice(), row_from_sql)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// vector KNN search. embedding は L2 normalized 必須。
    /// 戻り値の `f32` は cosine distance (sqlite-vec 規約)、
    /// 0.0 が完全一致、2.0 が完全反対。
    pub fn knn(&self, query_embedding: &[f32], k: usize) -> Result<Vec<(i64, f32)>> {
        if query_embedding.len() != EMBEDDING_DIM {
            return Err(HippoError::Embedding(format!(
                "query embedding dim mismatch: expected {EMBEDDING_DIM}, got {}",
                query_embedding.len()
            )));
        }
        // soft-delete tombstone がある時だけ over-sample (mcp-memory-service-rs 流儀)。
        let has_tombstones: bool = self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM memories WHERE deleted_at IS NOT NULL LIMIT 1)",
            [],
            |r| r.get(0),
        )?;
        let oversample = if has_tombstones { k * 3 } else { k };

        let mut stmt = self.conn.prepare(
            "SELECT rowid, distance FROM memory_embeddings
             WHERE content_embedding MATCH ?1 AND k = ?2
             ORDER BY distance",
        )?;
        let rows: Vec<(i64, f32)> = stmt
            .query_map(
                params![query_embedding.as_bytes(), oversample as i64],
                |r| Ok((r.get::<_, i64>(0)?, r.get::<_, f32>(1)?)),
            )?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        // tombstone を弾いて k 件まで切り詰める。
        let mut alive: Vec<(i64, f32)> = Vec::with_capacity(k);
        if rows.is_empty() {
            return Ok(alive);
        }
        // 一括で alive id を引いてから filter。
        let ids: Vec<i64> = rows.iter().map(|(id, _)| *id).collect();
        let placeholders = (0..ids.len()).map(|_| "?").collect::<Vec<_>>().join(",");
        let alive_sql =
            format!("SELECT id FROM memories WHERE id IN ({placeholders}) AND deleted_at IS NULL");
        let mut stmt2 = self.conn.prepare(&alive_sql)?;
        let id_params: Vec<&dyn rusqlite::ToSql> =
            ids.iter().map(|i| i as &dyn rusqlite::ToSql).collect();
        let alive_set: std::collections::HashSet<i64> = stmt2
            .query_map(id_params.as_slice(), |r| r.get::<_, i64>(0))?
            .collect::<std::result::Result<_, _>>()?;
        for (id, dist) in rows {
            if alive_set.contains(&id) {
                alive.push((id, dist));
                if alive.len() >= k {
                    break;
                }
            }
        }
        Ok(alive)
    }

    /// soft-delete by content_hash。返り値は実際に削除された件数。
    pub fn soft_delete_by_hash(&mut self, hash: &str) -> Result<usize> {
        let now = unix_now();
        let n = self.conn.execute(
            "UPDATE memories SET deleted_at = ?1
             WHERE content_hash = ?2 AND deleted_at IS NULL",
            params![now, hash],
        )?;
        Ok(n)
    }

    /// **Tests / eval harness only.** Backdate a memory's `created_at` and
    /// `updated_at` to simulate cross-session recall scenarios. Production
    /// memories should never be backdated; this helper exists so the v0.2
    /// evaluation suite can stress the forgetting-curve interaction with
    /// surprise rerank without waiting 30 days of wall-clock.
    pub fn debug_set_created_at(&self, id: i64, ts: f64) -> Result<usize> {
        Ok(self.conn.execute(
            "UPDATE memories SET created_at = ?1, updated_at = ?1 WHERE id = ?2",
            params![ts, id],
        )?)
    }

    /// soft-delete by id。
    pub fn soft_delete_by_id(&mut self, id: i64) -> Result<usize> {
        let now = unix_now();
        let n = self.conn.execute(
            "UPDATE memories SET deleted_at = ?1
             WHERE id = ?2 AND deleted_at IS NULL",
            params![now, id],
        )?;
        Ok(n)
    }

    /// v0.4: update metadata + tags + memory_type for an existing memory.
    /// Content (and content_hash) stays — SHODH `PATCH /api/memories/{id}` is
    /// explicitly metadata-only. `metadata` overwrites the full JSON field;
    /// pass the result of merging with `get_by_id(...).metadata` if you want
    /// merge semantics (we keep the API simple and let the caller decide).
    pub fn update_metadata_by_id(
        &mut self,
        id: i64,
        metadata: &serde_json::Value,
        tags: Option<&[String]>,
        memory_type: Option<Option<&str>>,
    ) -> Result<usize> {
        let now = unix_now();
        let now_iso = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
        let metadata_str = serde_json::to_string(metadata)?;
        let n = match (tags, memory_type) {
            (Some(t), Some(mt)) => self.conn.execute(
                "UPDATE memories SET metadata = ?1, tags = ?2, memory_type = ?3,
                 updated_at = ?4, updated_at_iso = ?5
                 WHERE id = ?6 AND deleted_at IS NULL",
                params![metadata_str, encode_tags(t), mt, now, now_iso, id],
            )?,
            (Some(t), None) => self.conn.execute(
                "UPDATE memories SET metadata = ?1, tags = ?2,
                 updated_at = ?3, updated_at_iso = ?4
                 WHERE id = ?5 AND deleted_at IS NULL",
                params![metadata_str, encode_tags(t), now, now_iso, id],
            )?,
            (None, Some(mt)) => self.conn.execute(
                "UPDATE memories SET metadata = ?1, memory_type = ?2,
                 updated_at = ?3, updated_at_iso = ?4
                 WHERE id = ?5 AND deleted_at IS NULL",
                params![metadata_str, mt, now, now_iso, id],
            )?,
            (None, None) => self.conn.execute(
                "UPDATE memories SET metadata = ?1, updated_at = ?2, updated_at_iso = ?3
                 WHERE id = ?4 AND deleted_at IS NULL",
                params![metadata_str, now, now_iso, id],
            )?,
        };
        Ok(n)
    }

    /// v0.4: aggregate alive tags with counts. Used by SHODH `GET /api/tags`.
    /// Returns `[(tag, count)]` sorted by count desc then tag asc.
    pub fn list_tags(&self) -> Result<Vec<(String, i64)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT tags FROM memories WHERE deleted_at IS NULL AND tags IS NOT NULL")?;
        let rows = stmt.query_map([], |r| r.get::<_, Option<String>>(0))?;
        let mut counts: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
        for row in rows {
            if let Some(s) = row? {
                for t in s.split(',') {
                    let t = t.trim();
                    if !t.is_empty() {
                        *counts.entry(t.to_string()).or_insert(0) += 1;
                    }
                }
            }
        }
        let mut v: Vec<(String, i64)> = counts.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        Ok(v)
    }
}

fn row_from_sql(r: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryRow> {
    let metadata_str: Option<String> = r.get("metadata")?;
    let metadata = match metadata_str {
        Some(s) if !s.is_empty() => serde_json::from_str(&s).unwrap_or(serde_json::Value::Null),
        _ => serde_json::Value::Object(Default::default()),
    };
    let tags_str: Option<String> = r.get("tags")?;
    Ok(MemoryRow {
        id: Some(r.get("id")?),
        content_hash: r.get("content_hash")?,
        content: r.get("content")?,
        tags: decode_tags(tags_str.as_deref()),
        memory_type: r.get("memory_type")?,
        metadata,
        created_at: r.get::<_, Option<f64>>("created_at")?.unwrap_or(0.0),
        updated_at: r.get::<_, Option<f64>>("updated_at")?.unwrap_or(0.0),
        created_at_iso: r
            .get::<_, Option<String>>("created_at_iso")?
            .unwrap_or_default(),
        updated_at_iso: r
            .get::<_, Option<String>>("updated_at_iso")?
            .unwrap_or_default(),
        deleted_at: r.get("deleted_at")?,
    })
}

/// 新しい MemoryRow を作る。timestamps は now。content_hash は SHA256(content)。
pub fn new_memory_row(
    content: String,
    tags: Vec<String>,
    memory_type: Option<String>,
    metadata: serde_json::Value,
) -> MemoryRow {
    let now = unix_now();
    let now_iso = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
    let hash = content_hash(&content);
    MemoryRow {
        id: None,
        content_hash: hash,
        content,
        tags,
        memory_type,
        metadata,
        created_at: now,
        updated_at: now,
        created_at_iso: now_iso.clone(),
        updated_at_iso: now_iso,
        deleted_at: None,
    }
}

fn unix_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// SHA-256 hex (lowercase) — mcp-memory-service-rs と同じ content_hash 規約。
pub fn content_hash(content: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(content.as_bytes());
    hex::encode(h.finalize())
}

/// tags を「カンマ区切り文字列、空白保持、case 保持」で encode する。
/// (mcp-memory-service-rs の Python 上流挙動と同一: 検索時に空白は strip
/// するが、ストレージは生のまま保持)
fn encode_tags(tags: &[String]) -> Option<String> {
    if tags.is_empty() {
        None
    } else {
        Some(tags.join(","))
    }
}

fn decode_tags(s: Option<&str>) -> Vec<String> {
    match s {
        Some(s) if !s.is_empty() => s.split(',').map(|t| t.to_string()).collect(),
        _ => Vec::new(),
    }
}

/// Surprise score を MemoryRow.metadata に attach する helper。
/// metadata が non-object なら object に置き換える。
pub fn attach_surprise(
    metadata: &mut serde_json::Value,
    score: f32,
    components: &crate::surprise::SurpriseComponents,
) {
    if !metadata.is_object() {
        *metadata = serde_json::Value::Object(Default::default());
    }
    let map = metadata.as_object_mut().expect("ensured object above");
    let hippo = map
        .entry("_hippo")
        .or_insert_with(|| serde_json::Value::Object(Default::default()));
    if !hippo.is_object() {
        *hippo = serde_json::Value::Object(Default::default());
    }
    let hm = hippo.as_object_mut().expect("ensured object above");
    hm.insert(
        "surprise".into(),
        serde_json::json!({
            "score": score,
            "components": components,
            "version": crate::VERSION,
        }),
    );
}

/// metadata から surprise score を読み出す。なければ None。
pub fn read_surprise(metadata: &serde_json::Value) -> Option<f32> {
    metadata
        .get("_hippo")?
        .get("surprise")?
        .get("score")?
        .as_f64()
        .map(|f| f as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> Storage {
        register_sqlite_vec();
        Storage::open_in_memory().expect("open in-memory")
    }

    fn dummy_emb(seed: f32) -> Vec<f32> {
        let mut v = vec![0.0_f32; EMBEDDING_DIM];
        v[0] = seed;
        // L2 normalize
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        v.iter_mut().for_each(|x| *x /= norm);
        v
    }

    #[test]
    fn open_applies_schema() {
        let s = store();
        assert_eq!(s.count_alive().unwrap(), 0);
        assert_eq!(s.count_total().unwrap(), 0);
        // vec_version available
        let v = s.vec_version().unwrap();
        assert!(!v.is_empty(), "vec_version: {v}");
    }

    #[test]
    fn insert_and_get_roundtrip() {
        let mut s = store();
        let row = new_memory_row(
            "JWT 24h expiry".into(),
            vec!["auth".into(), "security".into()],
            Some("Decision".into()),
            serde_json::json!({}),
        );
        let (id, dup) = s.insert(&row, Some(&dummy_emb(1.0))).unwrap();
        assert!(!dup);
        assert!(id > 0);
        let fetched = s.get_by_hash(&row.content_hash).unwrap().unwrap();
        assert_eq!(fetched.content, row.content);
        assert_eq!(fetched.tags, vec!["auth", "security"]);
    }

    #[test]
    fn dedup_returns_existing() {
        let mut s = store();
        let row = new_memory_row("same content".into(), vec![], None, serde_json::json!({}));
        let (id1, dup1) = s.insert(&row, Some(&dummy_emb(1.0))).unwrap();
        let (id2, dup2) = s.insert(&row, Some(&dummy_emb(2.0))).unwrap();
        assert!(!dup1);
        assert!(dup2);
        assert_eq!(id1, id2);
    }

    #[test]
    fn soft_delete_filters_alive() {
        let mut s = store();
        let r1 = new_memory_row("first".into(), vec![], None, serde_json::json!({}));
        let r2 = new_memory_row("second".into(), vec![], None, serde_json::json!({}));
        s.insert(&r1, Some(&dummy_emb(1.0))).unwrap();
        s.insert(&r2, Some(&dummy_emb(2.0))).unwrap();
        assert_eq!(s.count_alive().unwrap(), 2);
        s.soft_delete_by_hash(&r1.content_hash).unwrap();
        assert_eq!(s.count_alive().unwrap(), 1);
        assert_eq!(s.count_total().unwrap(), 2);
    }

    #[test]
    fn list_recent_orders_by_created_at_desc() {
        let mut s = store();
        let r1 = new_memory_row("oldest".into(), vec![], None, serde_json::json!({}));
        s.insert(&r1, Some(&dummy_emb(1.0))).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let r2 = new_memory_row("newer".into(), vec![], None, serde_json::json!({}));
        s.insert(&r2, Some(&dummy_emb(2.0))).unwrap();
        let rows = s.list_recent(10).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].content, "newer");
        assert_eq!(rows[1].content, "oldest");
    }

    #[test]
    fn tag_glob_any_matches_either() {
        let mut s = store();
        let r1 = new_memory_row(
            "auth note".into(),
            vec!["auth".into()],
            None,
            serde_json::json!({}),
        );
        let r2 = new_memory_row(
            "db note".into(),
            vec!["db".into()],
            None,
            serde_json::json!({}),
        );
        s.insert(&r1, Some(&dummy_emb(1.0))).unwrap();
        s.insert(&r2, Some(&dummy_emb(2.0))).unwrap();
        let hits = s
            .search_by_tag(&["auth".into(), "db".into()], TagMatch::Any, None, 10)
            .unwrap();
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn tag_glob_all_requires_both() {
        let mut s = store();
        let r1 = new_memory_row(
            "both".into(),
            vec!["auth".into(), "security".into()],
            None,
            serde_json::json!({}),
        );
        let r2 = new_memory_row(
            "one".into(),
            vec!["auth".into()],
            None,
            serde_json::json!({}),
        );
        s.insert(&r1, Some(&dummy_emb(1.0))).unwrap();
        s.insert(&r2, Some(&dummy_emb(2.0))).unwrap();
        let hits = s
            .search_by_tag(&["auth".into(), "security".into()], TagMatch::All, None, 10)
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].content, "both");
    }

    #[test]
    fn knn_returns_alive_ranked() {
        let mut s = store();
        let r1 = new_memory_row("a".into(), vec![], None, serde_json::json!({}));
        let r2 = new_memory_row("b".into(), vec![], None, serde_json::json!({}));
        s.insert(&r1, Some(&dummy_emb(1.0))).unwrap();
        s.insert(&r2, Some(&dummy_emb(-1.0))).unwrap();
        let hits = s.knn(&dummy_emb(1.0), 2).unwrap();
        assert_eq!(hits.len(), 2);
        assert!(hits[0].1 < hits[1].1, "closer match must come first");
    }

    #[test]
    fn knn_skips_soft_deleted() {
        let mut s = store();
        let r1 = new_memory_row("deleted".into(), vec![], None, serde_json::json!({}));
        let r2 = new_memory_row("alive".into(), vec![], None, serde_json::json!({}));
        s.insert(&r1, Some(&dummy_emb(1.0))).unwrap();
        s.insert(&r2, Some(&dummy_emb(0.99))).unwrap();
        s.soft_delete_by_hash(&r1.content_hash).unwrap();
        let hits = s.knn(&dummy_emb(1.0), 5).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 2);
    }

    #[test]
    fn surprise_attach_round_trip() {
        let mut meta = serde_json::json!({"user_field": 1});
        let comps = crate::surprise::SurpriseComponents {
            embedding_outlier: 0.5,
            engagement: 0.3,
            explicit: 0.2,
            prediction_loss: None,
        };
        attach_surprise(&mut meta, 0.42, &comps);
        let s = read_surprise(&meta).unwrap();
        assert!((s - 0.42).abs() < 1e-6);
        // user_field 保存されている
        assert_eq!(meta["user_field"], serde_json::json!(1));
    }
}
