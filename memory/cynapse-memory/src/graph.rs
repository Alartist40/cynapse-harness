//! DENDRITE — the in-memory knowledge graph.
//!
//! Faithful port of Go `internal/memory/dendrite.go`. Nodes carry
//! wiki-links (`[[target]]`) and hashtags (`#tag`) that are parsed from
//! content; backlinks are auto-wired and kept in sync on every mutation.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Mutex, MutexGuard, OnceLock};

use regex::Regex;

fn link_pattern() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]").unwrap())
}

fn tag_pattern() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"#([A-Za-z0-9_-]+)").unwrap())
}

fn fenced_code_pattern() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?s)```.*?```").unwrap())
}

fn inline_code_pattern() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"`[^`\n]*`").unwrap())
}

/// Strip Markdown fenced (```) and inline (`) code spans before running the
/// wiki-link / hashtag regexes over content, so illustrative syntax like
/// `` `[[id]]` `` in documentation text isn't parsed as a real link. Not a
/// full Markdown parser — just enough to avoid the common case of code
/// spans quoting the very syntax this module looks for.
fn strip_code_spans(content: &str) -> String {
    let without_fences = fenced_code_pattern().replace_all(content, " ");
    inline_code_pattern().replace_all(&without_fences, " ").into_owned()
}

/// Classifies what kind of knowledge a node holds across the 4-tier memory model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeType {
    /// Core self / agent persona (L3)
    Identity,
    /// A real person (user, contact) (L1 / L3)
    Person,
    /// Abstract concept or topic (L2)
    Concept,
    /// A project or task (L2)
    Project,
    /// Procedural skill or workflow (L2)
    Procedure,
    /// Rule or operational lesson learned from errors or corrections (L2)
    Lesson,
    /// Something that happened / event (L1)
    Event,
    /// Atomic fact or preference (L1)
    AtomicFact,
    /// Raw chat turn log (L0)
    TurnLog,
    /// Episodic memory entry (L1)
    Memory,
    /// User-defined
    Custom,
}

impl NodeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            NodeType::Identity => "identity",
            NodeType::Person => "person",
            NodeType::Concept => "concept",
            NodeType::Project => "project",
            NodeType::Procedure => "procedure",
            NodeType::Lesson => "lesson",
            NodeType::Event => "event",
            NodeType::AtomicFact => "atomic_fact",
            NodeType::TurnLog => "turn_log",
            NodeType::Memory => "memory",
            NodeType::Custom => "custom",
        }
    }

    pub fn label(&self) -> &'static str {
        self.as_str()
    }

    pub fn tier(&self) -> u8 {
        match self {
            NodeType::TurnLog => 0,
            NodeType::AtomicFact | NodeType::Lesson | NodeType::Memory | NodeType::Event | NodeType::Person => 1,
            NodeType::Procedure | NodeType::Project | NodeType::Concept => 2,
            NodeType::Identity => 3,
            NodeType::Custom => 1,
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> NodeType {
        s.parse().unwrap_or(NodeType::Custom)
    }
}

impl std::str::FromStr for NodeType {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "identity" => NodeType::Identity,
            "person" => NodeType::Person,
            "concept" => NodeType::Concept,
            "project" => NodeType::Project,
            "procedure" => NodeType::Procedure,
            "lesson" => NodeType::Lesson,
            "event" => NodeType::Event,
            "atomic_fact" => NodeType::AtomicFact,
            "turn_log" => NodeType::TurnLog,
            "memory" => NodeType::Memory,
            _ => NodeType::Custom,
        })
    }
}

impl std::fmt::Display for NodeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Classification of memory category for multi-galaxy clusters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeCategory {
    Personal,
    Engineering,
    Preferences,
    Meta,
    Episodic,
    Transient,
}

impl NodeCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            NodeCategory::Personal => "Personal",
            NodeCategory::Engineering => "Engineering",
            NodeCategory::Preferences => "Preferences",
            NodeCategory::Meta => "Meta & Identity",
            NodeCategory::Episodic => "Episodic & Events",
            NodeCategory::Transient => "Transient Buffer",
        }
    }
}

/// A single knowledge node in the graph with spatial 3D physics coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub id: String,
    pub title: String,
    pub content: String,
    pub node_type: NodeType,
    pub tags: Vec<String>,
    /// Outgoing [[links]]
    pub links: Vec<String>,
    /// Auto-maintained incoming links
    pub backlinks: Vec<String>,
    pub created_at: i64,
    pub updated_at: i64,
    /// 3D Spatial coordinates in the galaxy topology
    pub x: f32,
    pub y: f32,
    pub z: f32,
    /// Physics velocity vectors
    pub vx: f32,
    pub vy: f32,
    pub vz: f32,
    /// Gravitational mass computed from specialization index and connectivity
    pub mass: f32,
    /// Hierarchical kebab-case topic key (Engram paradigm, e.g. "architecture/auth-model")
    pub topic_key: Option<String>,
    /// Evolution revision count (increments on topic upsert)
    pub revision_count: u32,
    /// Memory IDs or topic keys that this node obsoletes / supersedes
    pub supersedes: Vec<String>,
    /// Memory IDs or topic keys that directly contradict this node
    pub conflicts_with: Vec<String>,
}

impl Node {
    /// Classify node into a sub-galaxy category.
    pub fn category(&self) -> NodeCategory {
        if self.node_type == NodeType::Identity {
            return NodeCategory::Meta;
        }
        if self.node_type == NodeType::TurnLog {
            return NodeCategory::Transient;
        }
        for tag in &self.tags {
            let t = tag.to_lowercase();
            if t.contains("pref") || t.contains("like") || t.contains("favorite") || t.contains("food") || t.contains("book") || t.contains("color") {
                return NodeCategory::Preferences;
            }
            if t.contains("code") || t.contains("rust") || t.contains("arch") || t.contains("procedure") || t.contains("lesson") || t.contains("project") || t.contains("concept") {
                return NodeCategory::Engineering;
            }
            if t.contains("fact") || t.contains("person") || t.contains("user") {
                return NodeCategory::Personal;
            }
        }
        match self.node_type {
            NodeType::Person | NodeType::AtomicFact => NodeCategory::Personal,
            NodeType::Procedure | NodeType::Lesson | NodeType::Project | NodeType::Concept => NodeCategory::Engineering,
            NodeType::Event | NodeType::Memory => NodeCategory::Episodic,
            NodeType::Identity => NodeCategory::Meta,
            NodeType::TurnLog => NodeCategory::Transient,
            NodeType::Custom => NodeCategory::Personal,
        }
    }

    /// Specialization index spec(e) in range [0.1, 1.0].
    pub fn spec_index(&self) -> f32 {
        let tag_score = (self.tags.len() as f32 * 0.25).min(0.5);
        let link_score = ((self.links.len() + self.backlinks.len()) as f32 * 0.1).min(0.3);
        let tier_base = match self.node_type.tier() {
            3 => 0.9,
            2 => 0.7,
            1 => 0.8,
            _ => 0.2,
        };
        (tier_base + tag_score + link_score).min(1.0)
    }

    /// Compute gravitational mass based on specialization and connection degree.
    pub fn compute_mass(&self) -> f32 {
        let spec = self.spec_index();
        let degree = (self.links.len() + self.backlinks.len()) as f32;
        let tier_weight = match self.node_type.tier() {
            3 => 3.5, // Identity / Core
            2 => 2.2, // Procedure / Concept
            1 => 1.5, // Fact / Event
            _ => 0.8, // TurnLog
        };
        (spec * tier_weight * (1.0 + degree * 0.35)).max(0.5)
    }

    /// Create a minimal placeholder for a node that is referenced by a
    /// `[[link]]` but has no content yet.
    #[allow(dead_code)]
    pub fn placeholder(id: String, now: i64) -> Node {
        let mut n = Node {
            title: id.clone(),
            id,
            content: String::new(),
            node_type: NodeType::Custom,
            tags: Vec::new(),
            links: Vec::new(),
            backlinks: Vec::new(),
            created_at: now,
            updated_at: now,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            vx: 0.0,
            vy: 0.0,
            vz: 0.0,
            mass: 1.0,
            topic_key: None,
            revision_count: 1,
            supersedes: Vec::new(),
            conflicts_with: Vec::new(),
        };
        n.mass = n.compute_mass();
        n
    }
}

#[derive(Default)]
struct DendriteInner {
    nodes: HashMap<String, Node>,
    on_change: HashMap<u64, std::sync::Arc<dyn Fn() + Send + Sync>>,
    next_cb_id: u64,
}

/// The in-memory graph. Thread-safe: all mutations take an internal lock,
/// and `on_change` callbacks are invoked after the lock is released.
pub struct Dendrite {
    inner: Mutex<DendriteInner>,
}

fn lock_inner(inner: &Mutex<DendriteInner>) -> MutexGuard<'_, DendriteInner> {
    inner.lock().unwrap_or_else(|e| e.into_inner())
}

impl Default for Dendrite {
    fn default() -> Self {
        Self::new()
    }
}

impl Dendrite {
    pub fn new() -> Dendrite {
        Dendrite {
            inner: Mutex::new(DendriteInner::default()),
        }
    }

    /// Register a callback invoked on every mutation. Returns a numeric ID
    /// that can be passed to `unregister_on_change` to remove it later.
    pub fn register_on_change(&self, cb: std::sync::Arc<dyn Fn() + Send + Sync>) -> u64 {
        let mut inner = lock_inner(&self.inner);
        let id = inner.next_cb_id;
        inner.next_cb_id += 1;
        inner.on_change.insert(id, cb);
        id
    }

    /// Register a callback with a specific ID (used by DendriteContext to
    /// coordinate cleanup via `ChangeGuard`).
    pub fn register_on_change_with_id(&self, id: u64, cb: std::sync::Arc<dyn Fn() + Send + Sync>) {
        let mut inner = lock_inner(&self.inner);
        inner.on_change.insert(id, cb);
    }

    /// Remove a previously registered callback by its ID.
    pub fn unregister_on_change(&self, id: u64) {
        let mut inner = lock_inner(&self.inner);
        inner.on_change.remove(&id);
    }

    fn notify(&self) {
        let callbacks: Vec<_> = {
            let inner = lock_inner(&self.inner);
            inner.on_change.values().cloned().collect()
        };
        for cb in callbacks {
            cb();
        }
    }

    /// Insert a hydrated node directly (used by the store on load). Skips
    /// backlink recalculation because backlinks are already stored.
    pub fn insert_hydrated(&self, mut node: Node) {
        node.mass = node.compute_mass();
        if node.x == 0.0 && node.y == 0.0 && node.z == 0.0 && node.node_type != NodeType::Identity {
            let arm = Self::category_cluster_arm(&node.category());
            let hash = node.id.len() as f32;
            node.x = arm.0 + (hash % 3.0 - 1.0) * 0.4;
            node.y = arm.1 + (hash % 2.0 - 0.5) * 0.3;
            node.z = arm.2 + (hash % 4.0 - 2.0) * 0.4;
        }
        lock_inner(&self.inner).nodes.insert(node.id.clone(), node);
    }

    /// Step the 3D force-directed physics layout across all nodes in the galaxy.
    pub fn simulate_forces(&self, iterations: usize) {
        let mut inner = lock_inner(&self.inner);
        Self::simulate_forces_inner(&mut inner, iterations);
    }

    /// Pull two nodes closer in 3D coordinate space by applying a mutual gravitational impulse.
    pub fn attract(&self, node_a: &str, node_b: &str, strength: f32) {
        let mut inner = lock_inner(&self.inner);
        if let (Some(a), Some(b)) = (inner.nodes.get(node_a).cloned(), inner.nodes.get(node_b).cloned()) {
            let dx = b.x - a.x;
            let dy = b.y - a.y;
            let dz = b.z - a.z;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(0.1);
            let force = (strength * 2.0).min(10.0);

            if let Some(na) = inner.nodes.get_mut(node_a) {
                na.vx += (dx / dist) * force;
                na.vy += (dy / dist) * force;
                na.vz += (dz / dist) * force;
            }
            if let Some(nb) = inner.nodes.get_mut(node_b) {
                nb.vx -= (dx / dist) * force;
                nb.vy -= (dy / dist) * force;
                nb.vz -= (dz / dist) * force;
            }
        }
    }

    /// Identify the supermassive central node (highest gravitational mass / connectivity).
    pub fn supermassive_node(&self) -> Option<Node> {
        let inner = lock_inner(&self.inner);
        inner.nodes.values().max_by(|a, b| a.mass.partial_cmp(&b.mass).unwrap_or(std::cmp::Ordering::Equal)).cloned()
    }

    /// Create or fully replace a node, re-wiring all backlinks and relaxing galaxy layout.
    pub fn upsert(
        &self,
        id: &str,
        title: &str,
        content: &str,
        node_type: NodeType,
        tags: Option<Vec<String>>,
    ) -> Node {
        self.upsert_with_topic(id, None, title, content, node_type, tags)
    }

    /// Upsert with optional Engram-style hierarchical topic key (`family/slug`).
    /// If an existing node shares the same topic key, it is updated in-place (`revision_count += 1`).
    pub fn upsert_with_topic(
        &self,
        id: &str,
        topic_key: Option<&str>,
        title: &str,
        content: &str,
        node_type: NodeType,
        tags: Option<Vec<String>>,
    ) -> Node {
        let now = timestamp();
        let links = parse_links(content);
        let tags = tags.unwrap_or_else(|| parse_tags(content));

        let mut inner = lock_inner(&self.inner);

        // Check if an existing node already owns this topic_key
        let topic_match_id = if let Some(t_key) = topic_key {
            inner.nodes.iter().find_map(|(k, v)| {
                if v.topic_key.as_deref() == Some(t_key) {
                    Some(k.clone())
                } else {
                    None
                }
            })
        } else {
            None
        };

        let effective_id = topic_match_id.as_deref().unwrap_or(id);

        // Remove old backlinks from the previous version of this node.
        let old_links: Vec<String> = inner
            .nodes
            .get(effective_id)
            .map(|old| old.links.clone())
            .unwrap_or_default();
        for old_link in &old_links {
            if let Some(target) = inner.nodes.get_mut(old_link) {
                target.backlinks = remove_str(&target.backlinks, effective_id);
            }
        }

        let total_nodes = inner.nodes.len();
        let node = match inner.nodes.entry(effective_id.to_string()) {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                let n = e.get_mut();
                n.title = title.to_string();
                n.content = content.to_string();
                n.node_type = node_type;
                n.tags = tags;
                n.links = links.clone();
                n.updated_at = now;
                if topic_key.is_some() {
                    n.topic_key = topic_key.map(|s| s.to_string());
                }
                n.revision_count += 1;
                n.mass = n.compute_mass();
                n.clone()
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                let count = total_nodes;
                let cat = {
                    let temp = Node {
                        id: String::new(),
                        title: String::new(),
                        content: String::new(),
                        node_type,
                        tags: tags.clone(),
                        links: Vec::new(),
                        backlinks: Vec::new(),
                        created_at: 0,
                        updated_at: 0,
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                        vx: 0.0,
                        vy: 0.0,
                        vz: 0.0,
                        mass: 1.0,
                        topic_key: None,
                        revision_count: 1,
                        supersedes: Vec::new(),
                        conflicts_with: Vec::new(),
                    };
                    temp.category()
                };
                let arm = Self::category_cluster_arm(&cat);
                let jitter_angle = (count as f32 * 2.39996) % (2.0 * std::f32::consts::PI);
                let jitter_r = ((count % 5) as f32 - 2.0) * 0.35;
                let jitter_y = ((count % 3) as f32 - 1.0) * 0.25;
                let mut n = Node {
                    id: effective_id.to_string(),
                    title: title.to_string(),
                    content: content.to_string(),
                    node_type,
                    tags,
                    links: links.clone(),
                    backlinks: Vec::new(),
                    created_at: now,
                    updated_at: now,
                    x: arm.0 + jitter_r * jitter_angle.cos(),
                    y: arm.1 + jitter_y,
                    z: arm.2 + jitter_r * jitter_angle.sin(),
                    vx: 0.0,
                    vy: 0.0,
                    vz: 0.0,
                    mass: 1.0,
                    topic_key: topic_key.map(|s| s.to_string()),
                    revision_count: 1,
                    supersedes: Vec::new(),
                    conflicts_with: Vec::new(),
                };
                n.mass = n.compute_mass();
                e.insert(n.clone());
                n
            }
        };

        // Wire new backlinks — only for targets that already exist in the graph.
        for link in &links {
            if let Some(target) = inner.nodes.get_mut(link) {
                if !target.backlinks.contains(&node.id) {
                    target.backlinks.push(node.id.clone());
                    target.mass = target.compute_mass();
                }
            }
        }

        // Single light relaxation pass to anchor supermassive core at (0,0,0) and adjust positions
        Self::simulate_forces_inner(&mut inner, 1);
        drop(inner);
        self.notify();
        node
    }

    /// Delete a node and clean up all references.
    pub fn delete(&self, id: &str) -> bool {
        let mut inner = lock_inner(&self.inner);
        if !inner.nodes.contains_key(id) {
            return false;
        }

        for n in inner.nodes.values_mut() {
            n.links = remove_str(&n.links, id);
            n.backlinks = remove_str(&n.backlinks, id);
            n.mass = n.compute_mass();
        }

        inner.nodes.remove(id);
        Self::simulate_forces_inner(&mut inner, 15);
        drop(inner);
        self.notify();
        true
    }

    /// Mark that node `superseding_id` obsoletes/supersedes node `superseded_id`.
    pub fn mark_supersedes(&self, superseding_id: &str, superseded_id: &str) {
        let mut inner = lock_inner(&self.inner);
        if let Some(node) = inner.nodes.get_mut(superseding_id) {
            if !node.supersedes.iter().any(|s| s == superseded_id) {
                node.supersedes.push(superseded_id.to_string());
            }
        }
    }

    /// Mark mutual or directional semantic contradiction between two nodes.
    pub fn mark_conflict(&self, node_a: &str, node_b: &str) {
        let mut inner = lock_inner(&self.inner);
        if let Some(a) = inner.nodes.get_mut(node_a) {
            if !a.conflicts_with.iter().any(|c| c == node_b) {
                a.conflicts_with.push(node_b.to_string());
            }
        }
        if let Some(b) = inner.nodes.get_mut(node_b) {
            if !b.conflicts_with.iter().any(|c| c == node_a) {
                b.conflicts_with.push(node_a.to_string());
            }
        }
    }

    /// Returns the set of all node IDs or topic keys that have been superseded by newer knowledge.
    pub fn superseded_set(&self) -> HashSet<String> {
        let inner = lock_inner(&self.inner);
        let mut set = HashSet::new();
        for node in inner.nodes.values() {
            for s in &node.supersedes {
                set.insert(s.clone());
            }
        }
        set
    }

    /// Detects pairs of conflicting nodes among a candidate set of node IDs.
    pub fn detect_conflicts(&self, candidate_ids: &[String]) -> Vec<(String, String)> {
        let inner = lock_inner(&self.inner);
        let mut conflicts = Vec::new();
        let set: HashSet<&str> = candidate_ids.iter().map(|s| s.as_str()).collect();

        for id in candidate_ids {
            if let Some(node) = inner.nodes.get(id) {
                for conflict_target in &node.conflicts_with {
                    if set.contains(conflict_target.as_str()) && id.as_str() < conflict_target.as_str() {
                        conflicts.push((id.clone(), conflict_target.clone()));
                    }
                }
            }
        }
        conflicts
    }


/// Canonical orbital arm position for a memory category cluster in 3D space.
pub fn category_cluster_arm(cat: &NodeCategory) -> (f32, f32, f32) {
    let (angle_idx, r, y) = match cat {
        NodeCategory::Meta => (0.0f32, 9.0f32, 0.4f32),
        NodeCategory::Engineering => (1.0f32, 16.0f32, -0.4f32),
        NodeCategory::Personal => (2.0f32, 23.0f32, 0.6f32),
        NodeCategory::Preferences => (3.0f32, 30.0f32, -0.6f32),
        NodeCategory::Episodic => (4.0f32, 37.0f32, 0.3f32),
        NodeCategory::Transient => (5.0f32, 42.0f32, -0.3f32),
    };
    let angle = angle_idx * (std::f32::consts::PI / 3.0);
    (r * angle.cos(), y, r * angle.sin())
}

/// Internal force simulation engine.
/// Anchors the supermassive central node (highest mass) at (0, 0, 0),
/// while memories group into Colibri-style category clusters that orbit around the core.
fn simulate_forces_inner(inner: &mut DendriteInner, iterations: usize) {
    if inner.nodes.is_empty() {
        return;
    }

    // 1. Identify the supermassive central node (highest mass)
    let supermassive_id = inner
        .nodes
        .values()
        .max_by(|a, b| a.mass.partial_cmp(&b.mass).unwrap_or(std::cmp::Ordering::Equal))
        .map(|n| n.id.clone());

    // Lock central core at origin (0, 0, 0)
    if let Some(ref core_id) = supermassive_id {
        if let Some(core) = inner.nodes.get_mut(core_id) {
            core.x = 0.0;
            core.y = 0.0;
            core.z = 0.0;
            core.vx = 0.0;
            core.vy = 0.0;
            core.vz = 0.0;
        }
    }

    if inner.nodes.len() == 1 {
        return;
    }

    for _ in 0..iterations {
        // 2. Compute dynamic cluster centroids for each category among non-core nodes
        let mut cat_sums: HashMap<NodeCategory, (f32, f32, f32, f32)> = HashMap::new();
        for (id, node) in &inner.nodes {
            if Some(id) == supermassive_id.as_ref() {
                continue;
            }
            let cat = node.category();
            let entry = cat_sums.entry(cat).or_insert((0.0, 0.0, 0.0, 0.0));
            entry.0 += node.x;
            entry.1 += node.y;
            entry.2 += node.z;
            entry.3 += 1.0;
        }

        // Blend category averages with designated orbital arm centroids
        let mut cluster_centers: HashMap<NodeCategory, (f32, f32, f32)> = HashMap::new();
        for (cat, (sx, sy, sz, count)) in &cat_sums {
            let arm = Self::category_cluster_arm(cat);
            let avg_x = sx / count;
            let avg_y = sy / count;
            let avg_z = sz / count;
            cluster_centers.insert(
                *cat,
                (
                    avg_x * 0.4 + arm.0 * 0.6,
                    avg_y * 0.4 + arm.1 * 0.6,
                    avg_z * 0.4 + arm.2 * 0.6,
                ),
            );
        }

        // 3. Compute forces for each non-core node
        let node_ids: Vec<String> = inner.nodes.keys().cloned().collect();
        let mut forces: HashMap<String, (f32, f32, f32)> = HashMap::new();
        for id in &node_ids {
            forces.insert(id.clone(), (0.0, 0.0, 0.0));
        }

        // a) Colibri-style Cluster Cohesion & Core Orbit
        for (id, node) in &inner.nodes {
            if Some(id) == supermassive_id.as_ref() {
                continue;
            }
            let cat = node.category();
            let (target_cx, target_cy, target_cz) = cluster_centers
                .get(&cat)
                .cloned()
                .unwrap_or_else(|| Self::category_cluster_arm(&cat));

            // Spring pull towards cluster centroid (Colibri group formation)
            let cdx = target_cx - node.x;
            let cdy = target_cy - node.y;
            let cdz = target_cz - node.z;
            let c_dist = (cdx * cdx + cdy * cdy + cdz * cdz).sqrt().max(0.1);
            let cluster_f = 0.12 * c_dist.min(4.0);

            // Core gravity & orbital velocity around (0,0,0)
            let r_sq = node.x * node.x + node.z * node.z;
            let r = r_sq.sqrt().max(0.5);

            // Minimum clearance from central core (keep center clear for biggest memory)
            let core_repulsion = if r < 3.0 { (3.0 - r) * 0.4 } else { 0.0 };

            // Tangential orbital velocity (clockwise rotation around Y-axis)
            let tangent_x = -node.z / r;
            let tangent_z = node.x / r;
            let orbital_drift = 0.08;

            if let Some(f) = forces.get_mut(id) {
                f.0 += (cdx / c_dist) * cluster_f + (node.x / r) * core_repulsion + tangent_x * orbital_drift;
                f.1 += (cdy / c_dist) * cluster_f * 0.6;
                f.2 += (cdz / c_dist) * cluster_f + (node.z / r) * core_repulsion + tangent_z * orbital_drift;
            }
        }

        // b) Colibri Concept Similarity Attraction (Nodes sharing tags or categories pull together)
        let count = node_ids.len();
        for i in 0..count {
            let id1 = &node_ids[i];
            if Some(id1) == supermassive_id.as_ref() {
                continue;
            }
            let (n1_tags, n1_cat, n1_pos) = match inner.nodes.get(id1) {
                Some(n) => (&n.tags, n.category(), (n.x, n.y, n.z)),
                None => continue,
            };

            for id2 in &node_ids[(i + 1)..] {
                if Some(id2) == supermassive_id.as_ref() {
                    continue;
                }
                let (n2_tags, n2_cat, n2_pos) = match inner.nodes.get(id2) {
                    Some(n) => (&n.tags, n.category(), (n.x, n.y, n.z)),
                    None => continue,
                };

                let shared_tags = n1_tags.iter().filter(|t| n2_tags.contains(t)).count();
                if shared_tags > 0 || n1_cat == n2_cat {
                    let sdx = n2_pos.0 - n1_pos.0;
                    let sdy = n2_pos.1 - n1_pos.1;
                    let sdz = n2_pos.2 - n1_pos.2;
                    let sdist = (sdx * sdx + sdy * sdy + sdz * sdz).sqrt().max(0.1);

                    if sdist > 1.5 {
                        let pull_factor = if shared_tags > 0 { 0.05 * (shared_tags as f32) } else { 0.02 };
                        let pull_f = ((sdist - 1.5) * pull_factor).min(0.25);
                        let px = (sdx / sdist) * pull_f;
                        let py = (sdy / sdist) * pull_f * 0.5;
                        let pz = (sdz / sdist) * pull_f;

                        if let Some(f1) = forces.get_mut(id1) {
                            f1.0 += px;
                            f1.1 += py;
                            f1.2 += pz;
                        }
                        if let Some(f2) = forces.get_mut(id2) {
                            f2.0 -= px;
                            f2.1 -= py;
                            f2.2 -= pz;
                        }
                    }
                }
            }
        }

        // c) Synaptic Links & Backlinks Attraction (Hooke's Law between connected memories)
        for (id, node) in &inner.nodes {
            if Some(id) == supermassive_id.as_ref() {
                continue;
            }
            for link_id in &node.links {
                if let Some(target) = inner.nodes.get(link_id) {
                    let ldx = target.x - node.x;
                    let ldy = target.y - node.y;
                    let ldz = target.z - node.z;
                    let ldist = (ldx * ldx + ldy * ldy + ldz * ldz).sqrt().max(0.1);
                    let rest_len = if Some(link_id) == supermassive_id.as_ref() { 3.5 } else { 1.8 };
                    let spring_f = (ldist - rest_len) * 0.06;

                    if let Some(f) = forces.get_mut(id) {
                        f.0 += (ldx / ldist) * spring_f;
                        f.1 += (ldy / ldist) * spring_f * 0.5;
                        f.2 += (ldz / ldist) * spring_f;
                    }
                }
            }
        }

        // c) Pairwise Coulomb Repulsion between nearby nodes (prevents overlaps within cluster)
        let count = node_ids.len();
        for i in 0..count {
            let id1 = &node_ids[i];
            if Some(id1) == supermassive_id.as_ref() {
                continue;
            }
            let n1 = match inner.nodes.get(id1) {
                Some(n) => (n.x, n.y, n.z),
                None => continue,
            };

            for id2 in &node_ids[(i + 1)..] {
                if Some(id2) == supermassive_id.as_ref() {
                    continue;
                }
                let n2 = match inner.nodes.get(id2) {
                    Some(n) => (n.x, n.y, n.z),
                    None => continue,
                };

                let pdx = n2.0 - n1.0;
                let pdy = n2.1 - n1.1;
                let pdz = n2.2 - n1.2;
                let pdist_sq = pdx * pdx + pdy * pdy + pdz * pdz;
                if pdist_sq < 4.0 && pdist_sq > 0.0001 {
                    let pdist = pdist_sq.sqrt();
                    let rep_f = 0.35 / pdist;
                    let rx = (pdx / pdist) * rep_f;
                    let ry = (pdy / pdist) * rep_f * 0.5;
                    let rz = (pdz / pdist) * rep_f;

                    if let Some(f1) = forces.get_mut(id1) {
                        f1.0 -= rx;
                        f1.1 -= ry;
                        f1.2 -= rz;
                    }
                    if let Some(f2) = forces.get_mut(id2) {
                        f2.0 += rx;
                        f2.1 += ry;
                        f2.2 += rz;
                    }
                }
            }
        }

        // 4. Integrate Velocities and Positions
        for (id, (fx, fy, fz)) in forces {
            if Some(&id) == supermassive_id.as_ref() {
                continue;
            }
            if let Some(node) = inner.nodes.get_mut(&id) {
                node.vx = (node.vx + fx) * 0.82;
                node.vy = (node.vy + fy) * 0.82;
                node.vz = (node.vz + fz) * 0.82;

                node.x += node.vx;
                node.y += node.vy;
                node.z += node.vz;

                // Galaxy outer boundary clamp (XZ orbital radius and Y vertical bounds)
                let cur_r = (node.x * node.x + node.z * node.z).sqrt();
                if cur_r > 50.0 {
                    node.x = (node.x / cur_r) * 48.0;
                    node.z = (node.z / cur_r) * 48.0;
                    node.vx *= 0.2;
                    node.vz *= 0.2;
                }
                if node.y.abs() > 4.0 {
                    node.y = node.y.signum() * 3.8;
                    node.vy *= 0.2;
                }
            }
        }

        // 5. Ensure core remains pinned at (0, 0, 0)
        if let Some(ref core_id) = supermassive_id {
            if let Some(core) = inner.nodes.get_mut(core_id) {
                core.x = 0.0;
                core.y = 0.0;
                core.z = 0.0;
                core.vx = 0.0;
                core.vy = 0.0;
                core.vz = 0.0;
            }
        }
    }
}

    pub fn get(&self, id: &str) -> Option<Node> {
        lock_inner(&self.inner).nodes.get(id).cloned()
    }

    /// All nodes sorted by UpdatedAt descending.
    pub fn all(&self) -> Vec<Node> {
        let inner = lock_inner(&self.inner);
        let mut nodes: Vec<Node> = inner.nodes.values().cloned().collect();
        nodes.sort_by_key(|b| std::cmp::Reverse(b.updated_at));
        nodes
    }

    /// Return all nodes belonging to a specific memory tier (0..3).
    pub fn by_tier(&self, tier: u8) -> Vec<Node> {
        let inner = lock_inner(&self.inner);
        let mut nodes: Vec<Node> = inner
            .nodes
            .values()
            .filter(|n| n.node_type.tier() == tier)
            .cloned()
            .collect();
        nodes.sort_by_key(|b| std::cmp::Reverse(b.updated_at));
        nodes
    }

    /// Return all nodes and edge connections (source_id, target_id) in the graph.
    pub fn topology(&self) -> (Vec<Node>, Vec<(String, String)>) {
        let inner = lock_inner(&self.inner);
        let mut nodes: Vec<Node> = inner.nodes.values().cloned().collect();
        nodes.sort_by_key(|b| std::cmp::Reverse(b.updated_at));

        let mut edges = Vec::new();
        for node in &nodes {
            for link in &node.links {
                if inner.nodes.contains_key(link) {
                    edges.push((node.id.clone(), link.clone()));
                }
            }
        }
        (nodes, edges)
    }

    /// Fast in-memory BM25 term relevance search.
    pub fn search_bm25(&self, query: &str, limit: usize) -> Vec<(Node, f32)> {
        let inner = lock_inner(&self.inner);
        let query_tokens: Vec<String> = query
            .to_lowercase()
            .split_whitespace()
            .filter(|s| s.len() >= 2)
            .map(|s| s.to_string())
            .collect();

        if query_tokens.is_empty() {
            return Vec::new();
        }

        let total_docs = inner.nodes.len() as f32;
        if total_docs == 0.0 {
            return Vec::new();
        }

        // Single-pass doc_text pre-computation and doc_freq accumulation
        let mut docs: Vec<(&Node, String, f32)> = Vec::with_capacity(inner.nodes.len());
        let mut doc_freq: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
        let mut total_len_sum = 0.0f32;

        for node in inner.nodes.values() {
            let doc_text = format!("{} {} {}", node.title, node.content, node.tags.join(" ")).to_lowercase();
            let doc_len = doc_text.len() as f32;
            total_len_sum += doc_len;

            for token in &query_tokens {
                if doc_text.contains(token) {
                    *doc_freq.entry(token.clone()).or_insert(0.0) += 1.0;
                }
            }
            docs.push((node, doc_text, doc_len));
        }

        let avg_len = total_len_sum / total_docs.max(1.0);
        let k1 = 1.2f32;
        let b = 0.75f32;
        let mut scored: Vec<(Node, f32)> = Vec::new();

        for (node, doc_text, doc_len) in docs {
            let mut score = 0.0f32;
            for token in &query_tokens {
                let count = doc_text.matches(token).count() as f32;
                if count > 0.0 {
                    let df = doc_freq.get(token).copied().unwrap_or(0.0);
                    let idf = ((total_docs - df + 0.5) / (df + 0.5)).max(0.0001).ln();
                    let tf = (count * (k1 + 1.0)) / (count + k1 * (1.0 - b + b * (doc_len / avg_len.max(1.0))));
                    score += idf * tf;
                }
            }

            if score > 0.0 {
                scored.push((node.clone(), score));
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if limit > 0 && scored.len() > limit {
            scored.truncate(limit);
        }
        scored
    }

    /// 1-hop neighborhood (links + backlinks combined).
    pub fn neighbors(&self, id: &str) -> Vec<Node> {
        let inner = lock_inner(&self.inner);
        let node = match inner.nodes.get(id) {
            Some(n) => n,
            None => return Vec::new(),
        };

        let mut seen = HashSet::new();
        seen.insert(id.to_string());
        let mut out = Vec::new();

        for lid in &node.links {
            if !seen.contains(lid) {
                if let Some(n) = inner.nodes.get(lid) {
                    out.push(n.clone());
                    seen.insert(lid.clone());
                }
            }
        }
        for bid in &node.backlinks {
            if !seen.contains(bid) {
                if let Some(n) = inner.nodes.get(bid) {
                    out.push(n.clone());
                    seen.insert(bid.clone());
                }
            }
        }
        out
    }

    /// Nodes within 2 hops (BFS).
    pub fn neighbors_2hop(&self, id: &str) -> Vec<Node> {
        self.neighbors_nhop(id, 2)
    }

    /// Nodes within N hops (BFS).
    fn neighbors_nhop(&self, id: &str, max_depth: usize) -> Vec<Node> {
        let inner = lock_inner(&self.inner);
        if !inner.nodes.contains_key(id) {
            return Vec::new();
        }

        let mut seen = HashSet::new();
        seen.insert(id.to_string());
        let mut out = Vec::new();

        let mut queue: VecDeque<(String, usize)> = VecDeque::new();
        queue.push_back((id.to_string(), 0));
        while let Some((node_id, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }
            let n = match inner.nodes.get(&node_id) {
                Some(n) => n,
                None => continue,
            };
            let next_depth = depth + 1;
            for lid in &n.links {
                if !seen.contains(lid) {
                    seen.insert(lid.clone());
                    if let Some(target) = inner.nodes.get(lid) {
                        out.push(target.clone());
                    }
                    queue.push_back((lid.clone(), next_depth));
                }
            }
            for bid in &n.backlinks {
                if !seen.contains(bid) {
                    seen.insert(bid.clone());
                    if let Some(target) = inner.nodes.get(bid) {
                        out.push(target.clone());
                    }
                    queue.push_back((bid.clone(), next_depth));
                }
            }
        }
        out
    }

    /// Nodes within 3 hops (BFS).
    pub fn neighbors_3hop(&self, id: &str) -> Vec<Node> {
        self.neighbors_nhop(id, 3)
    }

    /// Nodes whose title, content, or tags contain the query (case-insensitive).
    pub fn search(&self, query: &str) -> Vec<Node> {
        let q = query.trim().to_lowercase();
        if q.is_empty() {
            return Vec::new();
        }
        let inner = lock_inner(&self.inner);
        inner
            .nodes
            .values()
            .filter(|n| {
                n.title.to_lowercase().contains(&q)
                    || n.content.to_lowercase().contains(&q)
                    || contains_str_fold(&n.tags, &q)
            })
            .cloned()
            .collect()
    }

    pub fn by_tag(&self, tag: &str) -> Vec<Node> {
        let inner = lock_inner(&self.inner);
        inner
            .nodes
            .values()
            .filter(|n| contains_str_fold(&n.tags, tag))
            .cloned()
            .collect()
    }

    pub fn len(&self) -> usize {
        lock_inner(&self.inner).nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Normalise a wiki-link target into a stable lowercase_underscore ID.
fn to_node_id(s: &str) -> String {
    s.trim().to_lowercase().replace(' ', "_")
}

pub(crate) fn parse_links(content: &str) -> Vec<String> {
    let content = strip_code_spans(content);
    let mut seen = HashSet::new();
    let mut links = Vec::new();
    for caps in link_pattern().captures_iter(&content) {
        if let Some(m) = caps.get(1) {
            let id = to_node_id(m.as_str());
            if seen.insert(id.clone()) {
                links.push(id);
            }
        }
    }
    links
}

pub(crate) fn parse_tags(content: &str) -> Vec<String> {
    let content = strip_code_spans(content);
    let mut seen = HashSet::new();
    let mut tags = Vec::new();
    for caps in tag_pattern().captures_iter(&content) {
        if let Some(m) = caps.get(1) {
            let t = m.as_str().to_string();
            if seen.insert(t.clone()) {
                tags.push(t);
            }
        }
    }
    tags
}

pub(crate) fn contains_str_fold(slice: &[String], item: &str) -> bool {
    slice.iter().any(|s| s.eq_ignore_ascii_case(item))
}

pub(crate) fn remove_str(slice: &[String], item: &str) -> Vec<String> {
    slice.iter().filter(|s| *s != item).cloned().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_links_ignores_inline_code_spans() {
        // Reproduces the exact TOOLS.md sentence that used to create a
        // bogus permanent placeholder node named "id".
        let content = "Wiki-links `[[id]]` and `#tags` are parsed automatically.";
        assert!(parse_links(content).is_empty());
        assert!(parse_tags(content).is_empty());
    }

    #[test]
    fn parse_links_ignores_fenced_code_blocks() {
        let content = "See below:\n```\n[[fenced_link]] #fenced_tag\n```\nReal link: [[real_target]]";
        assert_eq!(parse_links(content), vec!["real_target".to_string()]);
    }

    #[test]
    fn parse_links_still_finds_real_links_and_tags_outside_code_spans() {
        let content = "Refer to [[identity]] and [[soul]] for #personality guidance.";
        assert_eq!(
            parse_links(content),
            vec!["identity".to_string(), "soul".to_string()]
        );
        assert_eq!(parse_tags(content), vec!["personality".to_string()]);
    }

    #[test]
    fn upsert_does_not_create_placeholder_from_documentation_code_span() {
        let g = Dendrite::new();
        g.upsert(
            "tools",
            "Tools",
            "Wiki-links `[[id]]` and `#tags` are parsed automatically.",
            NodeType::Concept,
            None,
        );
        assert!(g.get("id").is_none(), "code-span example must not create a real node");
    }

    #[test]
    fn test_supermassive_core_anchored_at_origin() {
        let g = Dendrite::new();
        // Insert identity node (highest tier/mass)
        let _core = g.upsert(
            "core_identity",
            "System Identity",
            "I am Cynapse Core, central knowledge anchor.",
            NodeType::Identity,
            Some(vec!["core".into(), "identity".into()]),
        );

        // Insert peripheral concepts
        g.upsert(
            "rust_engine",
            "Rust Engine",
            "High performance kernels linked to [[core_identity]].",
            NodeType::Concept,
            Some(vec!["code".into(), "rust".into()]),
        );
        g.upsert(
            "user_pref",
            "User Preferences",
            "User favorite theme linked to [[core_identity]].",
            NodeType::Concept,
            Some(vec!["pref".into(), "theme".into()]),
        );

        let supermassive = g.supermassive_node().expect("supermassive node exists");
        assert_eq!(supermassive.id, "core_identity");

        let core_node = g.get("core_identity").expect("core node exists");
        assert_eq!((core_node.x, core_node.y, core_node.z), (0.0, 0.0, 0.0));

        let eng_node = g.get("rust_engine").expect("rust_engine exists");
        let pref_node = g.get("user_pref").expect("user_pref exists");

        // Peripheral nodes must not be at (0,0,0)
        let eng_dist = (eng_node.x * eng_node.x + eng_node.z * eng_node.z).sqrt();
        let pref_dist = (pref_node.x * pref_node.x + pref_node.z * pref_node.z).sqrt();
        assert!(eng_dist > 2.0, "Engineering cluster must orbit away from origin");
        assert!(pref_dist > 2.0, "Preferences cluster must orbit away from origin");
    }

    #[test]
    fn test_attract_z_axis_signs() {
        let g = Dendrite::new();
        let mut na = Node::placeholder("node_a".to_string(), 0);
        na.x = 1.0;
        na.y = 1.0;
        na.z = 1.0;
        let mut nb = Node::placeholder("node_b".to_string(), 0);
        nb.x = 1.0;
        nb.y = 1.0;
        nb.z = 10.0;

        g.insert_hydrated(na);
        g.insert_hydrated(nb);

        g.attract("node_a", "node_b", 1.0);

        let res_a = g.get("node_a").unwrap();
        let res_b = g.get("node_b").unwrap();

        // Node A should be pulled forward (+Z towards Node B)
        assert!(res_a.vz > 0.0, "Node A vz must be positive towards B");
        // Node B should be pulled backward (-Z towards Node A)
        assert!(res_b.vz < 0.0, "Node B vz must be negative towards A");
        // Magnitudes must be equal and opposite
        assert!((res_a.vz + res_b.vz).abs() < 1e-5, "Attraction must be equal and opposite");
    }

    #[test]
    fn test_boundary_clamping_xz_and_y() {
        let g = Dendrite::new();
        let mut runaway = Node::placeholder("runaway".to_string(), 0);
        runaway.x = 100.0;
        runaway.y = 50.0;
        runaway.z = 100.0;
        g.insert_hydrated(runaway);

        g.simulate_forces(1);

        let clamped = g.get("runaway").unwrap();
        let r_xz = (clamped.x * clamped.x + clamped.z * clamped.z).sqrt();
        assert!(r_xz <= 50.0, "Outer XZ radius must be clamped <= 50.0, got {}", r_xz);
        assert!(clamped.y.abs() <= 4.0, "Y-axis must be clamped <= 4.0, got {}", clamped.y);
    }

    #[test]
    fn test_graph_topic_key_upsert_increments_revision() {
        let g = Dendrite::new();
        let n1 = g.upsert_with_topic(
            "id-1",
            Some("decision/database-engine"),
            "Database Choice V1",
            "Decided to use SQLite with WAL",
            NodeType::Concept,
            Some(vec!["sqlite".to_string()]),
        );
        assert_eq!(n1.id, "id-1");
        assert_eq!(n1.revision_count, 1);
        assert_eq!(n1.topic_key.as_deref(), Some("decision/database-engine"));

        // Second upsert with different ID but same topic_key
        let n2 = g.upsert_with_topic(
            "id-2",
            Some("decision/database-engine"),
            "Database Choice V2",
            "Decided to use SQLite with WAL + FTS5",
            NodeType::Concept,
            Some(vec!["sqlite".to_string(), "fts5".to_string()]),
        );
        assert_eq!(n2.id, "id-1", "Canonical existing node ID preserved on topic update");
        assert_eq!(n2.revision_count, 2);
        assert_eq!(n2.title, "Database Choice V2");
        assert_eq!(g.len(), 1, "Graph count remains 1, no duplicate memory pollution");
    }
}
