//! DENDRITE SQLite persistence.
//!
//! Faithful port of Go `internal/memory/dendrite_store.go`. The schema
//! (table, index, FTS5 virtual table, triggers) and every query string are
//! kept identical so the Rust build reads and writes the same
//! `dendrite.db` as the Go original with zero migration.
//!
//! rusqlite's `Connection` is `!Sync`, so it is wrapped in a `Mutex`;
//! callers still share the store through `Arc` (single-writer semantics,
//! mirroring the Go `SetMaxOpenConns(1)`).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use rusqlite::{params, Connection, OpenFlags, OptionalExtension};

use crate::graph::{Dendrite, Node, NodeType};

/// Memory store generation error when database files are modified or swapped out underfoot.
#[derive(Debug, Clone)]
pub struct GenerationError;

impl std::fmt::Display for GenerationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Dendrite database generation mismatch (disk inode/size changed unexpectedly)")
    }
}

impl std::error::Error for GenerationError {}

#[derive(Debug)]
struct GenerationFence {
    db_path: PathBuf,
    expected_dev: u64,
    expected_ino: u64,
    write_generation: AtomicU64,
}

impl GenerationFence {
    fn new(path: &Path) -> Result<Self> {
        let (dev, ino) = Self::stat_ids(path)?;
        Ok(Self {
            db_path: path.to_path_buf(),
            expected_dev: dev,
            expected_ino: ino,
            write_generation: AtomicU64::new(1),
        })
    }

    #[cfg(unix)]
    fn stat_ids(path: &Path) -> Result<(u64, u64)> {
        use std::os::unix::fs::MetadataExt;
        if path.exists() {
            let meta = std::fs::metadata(path).context("stat db file")?;
            Ok((meta.dev(), meta.ino()))
        } else {
            Ok((0, 0))
        }
    }

    #[cfg(not(unix))]
    fn stat_ids(_path: &Path) -> Result<(u64, u64)> {
        Ok((0, 0))
    }

    fn check(&self) -> Result<u64> {
        #[cfg(unix)]
        if self.expected_ino != 0 && self.db_path.exists() {
            let (dev, ino) = Self::stat_ids(&self.db_path)?;
            if dev != self.expected_dev || ino != self.expected_ino {
                bail!(GenerationError);
            }
        }
        Ok(self.write_generation.load(Ordering::SeqCst))
    }

    fn inc(&self) -> u64 {
        self.write_generation.fetch_add(1, Ordering::SeqCst) + 1
    }
}

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS dendrite_nodes (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    content    TEXT NOT NULL DEFAULT '',
    type       TEXT NOT NULL DEFAULT 'custom',
    tags       TEXT NOT NULL DEFAULT '[]',
    links      TEXT NOT NULL DEFAULT '[]',
    backlinks  TEXT NOT NULL DEFAULT '[]',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    topic_key  TEXT DEFAULT NULL,
    revision_count INTEGER NOT NULL DEFAULT 1,
    supersedes TEXT NOT NULL DEFAULT '[]',
    conflicts_with TEXT NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_dendrite_updated ON dendrite_nodes(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_dendrite_topic_key ON dendrite_nodes(topic_key);
"#;

const FTS5_SCHEMA: &str = r#"
CREATE VIRTUAL TABLE IF NOT EXISTS dendrite_fts USING fts5(
    id         UNINDEXED,
    title,
    content,
    tags,
    tokenize = 'porter unicode61'
);
"#;

const FTS_FALLBACK_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS dendrite_fts_fallback (
    id    TEXT PRIMARY KEY,
    title TEXT,
    content TEXT,
    tags  TEXT
);
"#;

const TRIGGERS: &str = r#"
CREATE TRIGGER IF NOT EXISTS dendrite_nodes_ai
AFTER INSERT ON dendrite_nodes BEGIN
    INSERT INTO dendrite_fts(id, title, content, tags)
    VALUES (new.id, new.title, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS dendrite_nodes_au
AFTER UPDATE ON dendrite_nodes BEGIN
    DELETE FROM dendrite_fts WHERE id = old.id;
    INSERT INTO dendrite_fts(id, title, content, tags)
    VALUES (new.id, new.title, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS dendrite_nodes_ad
AFTER DELETE ON dendrite_nodes BEGIN
    DELETE FROM dendrite_fts WHERE id = old.id;
END;
"#;

/// Persists DENDRITE nodes to SQLite with atomic generation fencing & BM25 rank bounds.
pub struct DendriteStore {
    conn: Mutex<Connection>,
    has_fts: bool,
    fence: GenerationFence,
}

fn lock_conn(conn: &Mutex<Connection>) -> MutexGuard<'_, Connection> {
    conn.lock().unwrap_or_else(|e| e.into_inner())
}

impl DendriteStore {
    pub fn open<P: AsRef<Path>>(db_path: P) -> Result<DendriteStore> {
        let conn = Connection::open_with_flags(
            db_path.as_ref(),
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_CREATE
                | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .context("open graph db")?;
        conn.busy_timeout(Duration::from_millis(5000))
            .context("set busy_timeout")?;
        conn.pragma_update(None, "journal_mode", "WAL")
            .context("enable WAL")?;

        let fence = GenerationFence::new(db_path.as_ref())?;

        let mut store = DendriteStore {
            conn: Mutex::new(conn),
            has_fts: true,
            fence,
        };
        store.migrate().context("graph db migrate")?;
        Ok(store)
    }

    /// Current database generation counter (engram style fence).
    pub fn current_generation(&self) -> Result<u64> {
        self.fence.check()
    }

    fn migrate(&mut self) -> Result<()> {
        let conn = lock_conn(&self.conn);
        conn.execute_batch(SCHEMA).context("create core table")?;

        // Gracefully add topic_key and revision_count columns for existing databases
        let _ = conn.execute("ALTER TABLE dendrite_nodes ADD COLUMN topic_key TEXT DEFAULT NULL", []);
        let _ = conn.execute("ALTER TABLE dendrite_nodes ADD COLUMN revision_count INTEGER NOT NULL DEFAULT 1", []);
        let _ = conn.execute("ALTER TABLE dendrite_nodes ADD COLUMN supersedes TEXT NOT NULL DEFAULT '[]'", []);
        let _ = conn.execute("ALTER TABLE dendrite_nodes ADD COLUMN conflicts_with TEXT NOT NULL DEFAULT '[]'", []);
        let _ = conn.execute("CREATE INDEX IF NOT EXISTS idx_dendrite_topic_key ON dendrite_nodes(topic_key)", []);

        // Try FTS5; if unavailable, gracefully fall back.
        match conn.execute_batch(FTS5_SCHEMA) {
            Ok(()) => {
                self.has_fts = true;
                conn.execute_batch(TRIGGERS).context("create fts triggers")?;
            }
            Err(_) => {
                self.has_fts = false;
                let _ = conn.execute_batch(FTS_FALLBACK_SCHEMA);
            }
        }
        Ok(())
    }

    /// Upsert a node into SQLite under generational protection.
    /// If node has a `topic_key` that already exists in SQLite, it updates that record in place
    /// and increments its `revision_count`, preventing memory fragmentation.
    pub fn save(&self, node: &Node) -> Result<()> {
        self.fence.check()?;
        let tags = serde_json::to_string(&node.tags)?;
        let links = serde_json::to_string(&node.links)?;
        let backlinks = serde_json::to_string(&node.backlinks)?;
        let supersedes = serde_json::to_string(&node.supersedes)?;
        let conflicts = serde_json::to_string(&node.conflicts_with)?;

        let mut conn = lock_conn(&self.conn);
        let tx = conn.transaction()?;

        // Engram-inspired topic-key in-place update check
        let existing_id_and_rev = if let Some(ref t_key) = node.topic_key {
            tx.query_row(
                "SELECT id, revision_count FROM dendrite_nodes WHERE topic_key = ?",
                params![t_key],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, u32>(1)?)),
            ).optional()?
        } else {
            None
        };

        let target_id = if let Some((existing_id, existing_rev)) = existing_id_and_rev {
            let next_rev = existing_rev + 1;
            tx.execute(
                r#"
                UPDATE dendrite_nodes SET
                    title          = ?,
                    content        = ?,
                    type           = ?,
                    tags           = ?,
                    links          = ?,
                    backlinks      = ?,
                    updated_at     = ?,
                    revision_count = ?,
                    supersedes     = ?,
                    conflicts_with = ?
                WHERE id = ?
                "#,
                params![
                    node.title,
                    node.content,
                    node.node_type.as_str(),
                    tags,
                    links,
                    backlinks,
                    node.updated_at,
                    next_rev,
                    supersedes,
                    conflicts,
                    existing_id,
                ],
            )?;
            existing_id
        } else {
            tx.execute(
                r#"
                INSERT INTO dendrite_nodes
                    (id, title, content, type, tags, links, backlinks, created_at, updated_at, topic_key, revision_count, supersedes, conflicts_with)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title          = excluded.title,
                    content        = excluded.content,
                    type           = excluded.type,
                    tags           = excluded.tags,
                    links          = excluded.links,
                    backlinks      = excluded.backlinks,
                    updated_at     = excluded.updated_at,
                    topic_key      = excluded.topic_key,
                    revision_count = excluded.revision_count,
                    supersedes     = excluded.supersedes,
                    conflicts_with = excluded.conflicts_with
                "#,
                params![
                    node.id,
                    node.title,
                    node.content,
                    node.node_type.as_str(),
                    tags,
                    links,
                    backlinks,
                    node.created_at,
                    node.updated_at,
                    node.topic_key,
                    node.revision_count.max(1),
                    supersedes,
                    conflicts,
                ],
            )?;
            node.id.clone()
        };

        if !self.has_fts {
            tx.execute(
                r#"
                INSERT INTO dendrite_fts_fallback (id, title, content, tags)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    content = excluded.content,
                    tags = excluded.tags
                "#,
                params![target_id, node.title, node.content, tags],
            )?;
        }

        tx.commit()?;
        self.fence.inc();
        Ok(())
    }

    pub fn delete(&self, id: &str) -> Result<()> {
        self.fence.check()?;
        let mut conn = lock_conn(&self.conn);
        let tx = conn.transaction()?;
        tx.execute("DELETE FROM dendrite_nodes WHERE id = ?", [id])?;
        if !self.has_fts {
            tx.execute("DELETE FROM dendrite_fts_fallback WHERE id = ?", [id])?;
        }
        tx.commit()?;
        self.fence.inc();
        Ok(())
    }

    /// Hydrate all stored nodes directly into the graph's node map.
    pub fn load_all(&self, graph: &Dendrite) -> Result<()> {
        self.fence.check()?;
        let conn = lock_conn(&self.conn);
        let mut stmt = conn.prepare(
            r#"
            SELECT id, title, content, type, tags, links, backlinks, created_at, updated_at, topic_key, revision_count, supersedes, conflicts_with
            FROM dendrite_nodes
            ORDER BY updated_at DESC
            "#,
        )?;
        let rows = stmt.query_map([], |row| {
            let node_type: String = row.get(3)?;
            let tags: String = row.get(4)?;
            let links: String = row.get(5)?;
            let backlinks: String = row.get(6)?;
            let topic_key: Option<String> = row.get(9).unwrap_or(None);
            let revision_count: u32 = row.get(10).unwrap_or(1);
            let supersedes_json: String = row.get(11).unwrap_or_else(|_| "[]".to_string());
            let conflicts_json: String = row.get(12).unwrap_or_else(|_| "[]".to_string());
            let mut node = Node {
                id: row.get(0)?,
                title: row.get(1)?,
                content: row.get(2)?,
                node_type: NodeType::from_str(&node_type),
                tags: serde_json::from_str(&tags).unwrap_or_default(),
                links: serde_json::from_str(&links).unwrap_or_default(),
                backlinks: serde_json::from_str(&backlinks).unwrap_or_default(),
                created_at: row.get(7)?,
                updated_at: row.get(8)?,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                vx: 0.0,
                vy: 0.0,
                vz: 0.0,
                mass: 1.0,
                topic_key,
                revision_count,
                supersedes: serde_json::from_str(&supersedes_json).unwrap_or_default(),
                conflicts_with: serde_json::from_str(&conflicts_json).unwrap_or_default(),
            };
            node.mass = node.compute_mass();
            Ok(node)
        })?;

        for node in rows {
            let node = node.context("reading node row")?;
            graph.insert_hydrated(node);
        }
        // Single light relaxation pass on DB load to pin core without startup lag
        graph.simulate_forces(1);
        Ok(())
    }

    /// Escape user input into safe FTS5 match tokens.
    pub fn escape_fts5_query(query: &str) -> String {
        let clean: String = query
            .chars()
            .map(|c| if c.is_alphanumeric() || c == '_' || c == ' ' { c } else { ' ' })
            .collect();
        let words: Vec<&str> = clean.split_whitespace().collect();
        if words.is_empty() {
            return String::new();
        }
        words
            .iter()
            .map(|w| format!("\"{}\"", w))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Full-text search with calibrated BM25 ranking (engram inspired).
    /// Discards results with rank > max_bm25_rank to prevent low-signal noise
    /// from cluttering small-model context windows.
    pub fn fts_search_calibrated(&self, query: &str, limit: usize, max_bm25_rank: Option<f64>) -> Result<Vec<(String, f64)>> {
        let limit = if limit == 0 { 10 } else { limit };
        self.fence.check()?;
        let conn = lock_conn(&self.conn);

        if self.has_fts {
            let safe_query = Self::escape_fts5_query(query);
            if !safe_query.is_empty() {
                if let Ok(mut stmt) = conn.prepare(
                    r#"
                    SELECT id, rank FROM dendrite_fts
                    WHERE dendrite_fts MATCH ?
                    ORDER BY rank ASC
                    LIMIT ?
                    "#,
                ) {
                    if let Ok(rows) = stmt.query_map(params![safe_query, limit as i64], |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
                    }) {
                        let mut candidates = Vec::new();
                        for item in rows.flatten() {
                            if let Some(max_rank) = max_bm25_rank {
                                // SQLite FTS5 bm25 rank is negative or positive depending on version;
                                // smaller/more negative is better match.
                                if item.1 <= max_rank {
                                    candidates.push(item);
                                }
                            } else {
                                candidates.push(item);
                            }
                        }
                        if !candidates.is_empty() || max_bm25_rank.is_some() {
                            return Ok(candidates);
                        }
                    }
                }
            }
        }

        // Fallback: LIKE-based search across core dendrite_nodes (title, content, tags).
        let pattern = format!(
            "%{}%",
            query.replace('\\', "\\\\").replace('%', "\\%").replace('_', "\\_")
        );
        let mut stmt = conn.prepare(
            r#"
            SELECT id FROM dendrite_nodes
            WHERE title LIKE ? OR content LIKE ? OR tags LIKE ?
            ORDER BY updated_at DESC
            LIMIT ?
            "#,
        )?;
        let rows =
            stmt.query_map(params![pattern, pattern, pattern, limit as i64], |row| {
                row.get(0)
            })?;
        let mut ids = Vec::new();
        for id in rows.flatten() {
            ids.push((id, 0.0));
        }
        Ok(ids)
    }

    /// Full-text search, returning matching node IDs ordered by rank.
    pub fn fts_search(&self, query: &str, limit: usize) -> Result<Vec<String>> {
        let results = self.fts_search_calibrated(query, limit, None)?;
        Ok(results.into_iter().map(|(id, _)| id).collect())
    }

    pub fn close(self) -> Result<()> {
        let conn = self.conn.into_inner().unwrap_or_else(|e| e.into_inner());
        conn.close().map_err(|(_, e)| e.into())
    }

    /// Number of rows in the core table.
    pub fn node_count(&self) -> Result<usize> {
        let conn = lock_conn(&self.conn);
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM dendrite_nodes", [], |r| {
            r.get(0)
        })?;
        Ok(count as usize)
    }

    /// Perform SQLite database integrity check via PRAGMA quick_check.
    pub fn quick_check(&self) -> Result<String> {
        let conn = lock_conn(&self.conn);
        let status: String = conn.query_row("PRAGMA quick_check;", [], |r| r.get(0))?;
        Ok(status)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_generation_fence_and_calibrated_search() -> Result<()> {
        let store = DendriteStore::open(":memory:")?;

        let initial_gen = store.current_generation()?;
        assert!(initial_gen >= 1);

        let mut node = Node::placeholder("test-1".to_string(), 100);
        node.title = "Authentication System".to_string();
        node.content = "Handles JWT and session tokens".to_string();
        node.node_type = NodeType::Concept;
        store.save(&node)?;

        let next_gen = store.current_generation()?;
        assert_eq!(next_gen, initial_gen + 1);

        // Calibrated FTS search
        let results = store.fts_search_calibrated("Authentication", 5, None)?;
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "test-1");

        // Calibrated FTS search with strict max BM25 threshold filtering
        let filtered = store.fts_search_calibrated("Authentication", 5, Some(-999.0))?;
        // If threshold is an impossibly low score, no results should leak to the small model
        assert!(filtered.is_empty());

        Ok(())
    }

    #[test]
    fn test_store_topic_key_in_place_upsert() -> Result<()> {
        let store = DendriteStore::open(":memory:")?;

        // 1. Initial save with topic_key
        let mut node1 = Node::placeholder("uuid-aaa".to_string(), 100);
        node1.title = "Auth Model V1".to_string();
        node1.content = "Initial JWT implementation".to_string();
        node1.node_type = NodeType::Procedure;
        node1.topic_key = Some("architecture/auth-model".to_string());
        store.save(&node1)?;

        assert_eq!(store.node_count()?, 1);

        // 2. Second save with different uuid but identical topic_key
        let mut node2 = Node::placeholder("uuid-bbb".to_string(), 200);
        node2.title = "Auth Model V2".to_string();
        node2.content = "Evolved session cookie implementation".to_string();
        node2.node_type = NodeType::Procedure;
        node2.topic_key = Some("architecture/auth-model".to_string());
        store.save(&node2)?;

        // Must still be 1 row, in-place updated with incremented revision_count
        assert_eq!(store.node_count()?, 1);

        let graph = Dendrite::new();
        store.load_all(&graph)?;
        let loaded = graph.get("uuid-aaa").expect("Original canonical ID preserved");
        assert_eq!(loaded.title, "Auth Model V2");
        assert_eq!(loaded.content, "Evolved session cookie implementation");
        assert_eq!(loaded.revision_count, 2);
        assert_eq!(loaded.topic_key.as_deref(), Some("architecture/auth-model"));

        Ok(())
    }
}
