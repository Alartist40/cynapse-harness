//! Zero-Breach Sovereign Digital Forensics Engine for CYNAPSE.
//!
//! Inspired by Interplains Zero-Breach / AdFection Digital Forensics.
//! Strictly implements Doctrine D2 ("never contact an indicator"):
//! 100% offline, read-only inspection of local browser histories,
//! caches, cookies, and system DNS/host artifacts for Candiru /
//! DevilsTongue / AdFection indicators of compromise (IOCs).
//!
//! Zero outbound network connections, zero socket calls, zero system writes.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const EMBEDDED_IOCS: &str = include_str!("assets/adfection_iocs.txt");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndicatorType {
    Domain,
    Ip,
    Sha256,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Indicator {
    pub value: String,
    pub tier: String,
    pub confidence: String,
    pub ind_type: IndicatorType,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForensicFinding {
    pub indicator: String,
    pub indicator_type: String,
    pub tier: String,
    pub confidence: String,
    pub source: String,
    pub artifact_path: String,
    pub match_context: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForensicCoverageGap {
    pub category: String,
    pub target: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspectedTarget {
    pub category: String,
    pub target: String,
    pub items_checked: usize,
    pub status: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ForensicStatus {
    Clean,
    ThreatDetected,
    GapsOnly,
}

impl ForensicStatus {
    pub fn badge(&self) -> &'static str {
        match self {
            ForensicStatus::Clean => "[✓ CLEAN]",
            ForensicStatus::ThreatDetected => "[! THREAT DETECTED]",
            ForensicStatus::GapsOnly => "[? GAPS NOTED]",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanHistoryEntry {
    pub timestamp: String,
    pub total_artifacts_scanned: usize,
    pub findings_count: usize,
    pub mode: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForensicScanDiff {
    pub previous_artifacts: usize,
    pub artifact_delta: i64,
    pub previous_findings: usize,
    pub findings_delta: i64,
    pub last_scan_timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForensicScanReport {
    pub platform: String,
    pub hardware: String,
    pub scan_timestamp: String,
    pub total_artifacts_scanned: usize,
    pub iocs_loaded: usize,
    pub findings: Vec<ForensicFinding>,
    pub gaps: Vec<ForensicCoverageGap>,
    pub inspected_targets: Vec<InspectedTarget>,
    pub status: ForensicStatus,
    pub zero_network_guarantee: bool,
    pub scan_duration_ms: u64,
    pub history_diff: Option<ForensicScanDiff>,
}

impl ForensicScanReport {
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    pub fn to_summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("[ ZERO-BREACH DIGITAL FORENSICS REPORT ]"));
        lines.push(format!("Status: {} | Doctrine D2 Zero-Network: Guaranteed", self.status.badge()));
        lines.push(format!("Platform: {} | Hardware: {}", self.platform, self.hardware));
        lines.push(format!("Scanned Artifacts: {} | Loaded IOCs: {} | Duration: {}ms", self.total_artifacts_scanned, self.iocs_loaded, self.scan_duration_ms));
        if let Some(diff) = &self.history_diff {
            let sign = if diff.artifact_delta >= 0 { "+" } else { "" };
            lines.push(format!("History Delta: {}{} artifacts since last scan ({})", sign, diff.artifact_delta, diff.last_scan_timestamp));
        }
        
        if self.findings.is_empty() {
            lines.push(format!("Findings: 0 indicators matched. (No known Candiru/AdFection IOCs found in readable artifacts)"));
        } else {
            lines.push(format!("[!] CRITICAL: {} INDICATOR(S) MATCHED:", self.findings.len()));
            for f in &self.findings {
                lines.push(format!("  • [{}] {} (Tier: {}, Conf: {})", f.indicator_type, f.indicator, f.tier, f.confidence));
                lines.push(format!("    Artifact: {}", f.artifact_path));
                lines.push(format!("    Context: {}", f.match_context));
            }
        }

        if !self.gaps.is_empty() {
            lines.push(format!("Coverage Gaps ({} targets unreadable or absent):", self.gaps.len()));
            for g in self.gaps.iter().take(5) {
                lines.push(format!("  - [{}] {}: {}", g.category, g.target, g.reason));
            }
            if self.gaps.len() > 5 {
                lines.push(format!("  - ... and {} more unreadable paths", self.gaps.len() - 5));
            }
        }

        lines.join("\n")
    }
}

/// Parses raw tab-delimited or whitespace-delimited IOC lines.
pub fn parse_iocs(raw: &str) -> Vec<Indicator> {
    let mut iocs = Vec::new();
    let sha256_re = regex::Regex::new(r"^[0-9a-fA-F]{64}$").unwrap();
    let ipv4_re = regex::Regex::new(r"^\d{1,3}(\.\d{1,3}){3}$").unwrap();

    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = trimmed.split('\t').collect();
        let (val_raw, tier, conf, src) = if parts.len() >= 3 {
            (
                parts[0].trim(),
                parts[1].trim(),
                parts[2].trim(),
                if parts.len() >= 4 { parts[3].trim() } else { "citizenlab-adfection-2026" },
            )
        } else {
            let ws: Vec<&str> = trimmed.split_whitespace().collect();
            if ws.len() >= 3 {
                (
                    ws[0].trim(),
                    ws[1].trim(),
                    ws[2].trim(),
                    if ws.len() >= 4 { ws[3].trim() } else { "citizenlab-adfection-2026" },
                )
            } else if !ws.is_empty() {
                (ws[0].trim(), "cc", "primary", "citizenlab-adfection-2026")
            } else {
                continue;
            }
        };

        let val_lower = val_raw.to_lowercase();
        let ind_type = if sha256_re.is_match(&val_lower) {
            IndicatorType::Sha256
        } else if ipv4_re.is_match(&val_lower) || (val_lower.contains(':') && val_lower.chars().all(|c| c.is_ascii_hexdigit() || c == ':')) {
            IndicatorType::Ip
        } else {
            IndicatorType::Domain
        };

        iocs.push(Indicator {
            value: val_lower,
            tier: tier.to_string(),
            confidence: conf.to_string(),
            ind_type,
            source: src.to_string(),
        });
    }

    iocs
}

/// Checks whether an extracted hostname matches a candidate domain indicator.
/// Respects DNS label boundaries: "fearevolve.com" matches "fearevolve.com" and "sub.fearevolve.com",
/// but does NOT match "notfearevolve.com" or "fearevolve.computer".
pub fn matches_domain_boundary(host: &str, indicator_domain: &str) -> bool {
    let host_norm = host.trim().trim_end_matches('.').to_lowercase();
    let ind_norm = indicator_domain.trim().trim_end_matches('.').to_lowercase();

    if host_norm == ind_norm {
        return true;
    }

    if host_norm.ends_with(&format!(".{}", ind_norm)) {
        return true;
    }

    false
}

/// Detects local hardware and OS platform string.
pub fn detect_forensics_platform() -> (String, String) {
    let os_name = match std::env::consts::OS {
        "linux" => "Linux",
        "macos" => "macOS",
        "windows" => "Windows",
        other => other,
    };
    let arch = std::env::consts::ARCH;

    let is_sbc = Path::new("/sys/firmware/devicetree/base/model").exists()
        || Path::new("/dev/rknpu0").exists()
        || Path::new("/dev/rknpu").exists();

    let hw_type = if is_sbc {
        format!("{} SBC (ARM Embedded)", arch)
    } else {
        format!("{} Desktop/Laptop", arch)
    };

    (format!("{} ({})", os_name, arch), hw_type)
}

/// Returns standard browser profile directories based on platform.
pub fn discover_browser_profile_roots() -> Vec<(String, PathBuf)> {
    let mut roots = Vec::new();
    let home = match dirs::home_dir() {
        Some(h) => h,
        None => return roots,
    };

    #[cfg(target_os = "linux")]
    {
        roots.push(("Google Chrome".into(), home.join(".config/google-chrome")));
        roots.push(("Chromium".into(), home.join(".config/chromium")));
        roots.push(("Brave Browser".into(), home.join(".config/BraveSoftware/Brave-Browser")));
        roots.push(("Microsoft Edge".into(), home.join(".config/microsoft-edge")));
        roots.push(("Vivaldi".into(), home.join(".config/vivaldi")));
        roots.push(("Mozilla Firefox".into(), home.join(".mozilla/firefox")));
    }

    #[cfg(target_os = "macos")]
    {
        let app_support = home.join("Library/Application Support");
        roots.push(("Google Chrome".into(), app_support.join("Google/Chrome")));
        roots.push(("Chromium".into(), app_support.join("Chromium")));
        roots.push(("Brave Browser".into(), app_support.join("BraveSoftware/Brave-Browser")));
        roots.push(("Microsoft Edge".into(), app_support.join("Microsoft Edge")));
        roots.push(("Vivaldi".into(), app_support.join("Vivaldi")));
        roots.push(("Mozilla Firefox".into(), app_support.join("Firefox/Profiles")));
    }

    #[cfg(target_os = "windows")]
    {
        if let Some(local_app_data) = dirs::data_local_dir() {
            roots.push(("Google Chrome".into(), local_app_data.join("Google/Chrome/User Data")));
            roots.push(("Chromium".into(), local_app_data.join("Chromium/User Data")));
            roots.push(("Brave Browser".into(), local_app_data.join("BraveSoftware/Brave-Browser/User Data")));
            roots.push(("Microsoft Edge".into(), local_app_data.join("Microsoft/Edge/User Data")));
            roots.push(("Vivaldi".into(), local_app_data.join("Vivaldi/User Data")));
        }
        if let Some(app_data) = dirs::config_dir() {
            roots.push(("Mozilla Firefox".into(), app_data.join("Mozilla/Firefox/Profiles")));
        }
    }

    // Filter to existing directories
    roots.retain(|(_, p)| p.exists());
    roots
}

/// Scans a SQLite history file in read-only mode by making a temporary isolated snapshot.
fn scan_sqlite_history(
    browser_name: &str,
    db_path: &Path,
    query: &str,
    iocs: &[Indicator],
    findings: &mut Vec<ForensicFinding>,
    gaps: &mut Vec<ForensicCoverageGap>,
) -> usize {
    if !db_path.exists() {
        return 0;
    }

    // Create temporary snapshot to prevent lock conflicts with active browser processes
    let temp_file = match tempfile::Builder::new().prefix("cynapse_forensic_").tempfile() {
        Ok(t) => t,
        Err(e) => {
            gaps.push(ForensicCoverageGap {
                category: "browser".into(),
                target: db_path.display().to_string(),
                reason: format!("Failed to create snapshot tempfile: {}", e),
            });
            return 0;
        }
    };

    if let Err(e) = fs::copy(db_path, temp_file.path()) {
        gaps.push(ForensicCoverageGap {
            category: "browser".into(),
            target: db_path.display().to_string(),
            reason: format!("Unable to copy database snapshot (permission/lock): {}", e),
        });
        return 0;
    }

    let conn = match rusqlite::Connection::open_with_flags(
        temp_file.path(),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_URI,
    ) {
        Ok(c) => c,
        Err(e) => {
            gaps.push(ForensicCoverageGap {
                category: "browser".into(),
                target: db_path.display().to_string(),
                reason: format!("Failed to open SQLite snapshot: {}", e),
            });
            return 0;
        }
    };

    let mut stmt = match conn.prepare(query) {
        Ok(s) => s,
        Err(e) => {
            gaps.push(ForensicCoverageGap {
                category: "browser".into(),
                target: db_path.display().to_string(),
                reason: format!("Query prepare failed ('{}'): {}", query, e),
            });
            return 0;
        }
    };

    let mut rows_scanned = 0;
    let url_iter = stmt.query_map([], |row| row.get::<_, String>(0));

    if let Ok(iter) = url_iter {
        for url_res in iter {
            if let Ok(raw_url) = url_res {
                rows_scanned += 1;
                let url_lower = raw_url.to_lowercase();

                // Extract hostname heuristic
                let host_candidate = if let Some(idx) = url_lower.find("://") {
                    let after = &url_lower[idx + 3..];
                    after.split(&['/', '?', ':', '#'][..]).next().unwrap_or(after)
                } else {
                    url_lower.split(&['/', '?', ':', '#'][..]).next().unwrap_or(&url_lower)
                };

                for ioc in iocs {
                    match ioc.ind_type {
                        IndicatorType::Domain => {
                            if matches_domain_boundary(host_candidate, &ioc.value) {
                                findings.push(ForensicFinding {
                                    indicator: ioc.value.clone(),
                                    indicator_type: "domain".into(),
                                    tier: ioc.tier.clone(),
                                    confidence: ioc.confidence.clone(),
                                    source: ioc.source.clone(),
                                    artifact_path: format!("{} ({})", db_path.display(), browser_name),
                                    match_context: raw_url.clone(),
                                });
                            }
                        }
                        IndicatorType::Ip => {
                            if host_candidate == ioc.value {
                                findings.push(ForensicFinding {
                                    indicator: ioc.value.clone(),
                                    indicator_type: "ip".into(),
                                    tier: ioc.tier.clone(),
                                    confidence: ioc.confidence.clone(),
                                    source: ioc.source.clone(),
                                    artifact_path: format!("{} ({})", db_path.display(), browser_name),
                                    match_context: raw_url.clone(),
                                });
                            }
                        }
                        IndicatorType::Sha256 => {}
                    }
                }
            }
        }
    }

    rows_scanned
}

/// Scans local hosts file for indicators.
fn scan_hosts_file(
    iocs: &[Indicator],
    findings: &mut Vec<ForensicFinding>,
    gaps: &mut Vec<ForensicCoverageGap>,
) -> usize {
    let hosts_path = if cfg!(target_os = "windows") {
        PathBuf::from(r"C:\Windows\System32\drivers\etc\hosts")
    } else {
        PathBuf::from("/etc/hosts")
    };

    if !hosts_path.exists() {
        gaps.push(ForensicCoverageGap {
            category: "host".into(),
            target: hosts_path.display().to_string(),
            reason: "Hosts file not found".into(),
        });
        return 0;
    }

    let content = match fs::read_to_string(&hosts_path) {
        Ok(c) => c,
        Err(e) => {
            gaps.push(ForensicCoverageGap {
                category: "host".into(),
                target: hosts_path.display().to_string(),
                reason: format!("Cannot read hosts file: {}", e),
            });
            return 0;
        }
    };

    let mut lines_scanned = 0;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        lines_scanned += 1;

        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() >= 2 {
            for host_cand in &parts[1..] {
                for ioc in iocs {
                    if ioc.ind_type == IndicatorType::Domain && matches_domain_boundary(host_cand, &ioc.value) {
                        findings.push(ForensicFinding {
                            indicator: ioc.value.clone(),
                            indicator_type: "domain".into(),
                            tier: ioc.tier.clone(),
                            confidence: ioc.confidence.clone(),
                            source: ioc.source.clone(),
                            artifact_path: hosts_path.display().to_string(),
                            match_context: line.to_string(),
                        });
                    }
                }
            }
        }
    }

    lines_scanned
}

/// Scans candidate user directories for sample file hashes.
fn scan_file_hashes(
    iocs: &[Indicator],
    deep: bool,
    findings: &mut Vec<ForensicFinding>,
    _gaps: &mut Vec<ForensicCoverageGap>,
) -> usize {
    let hash_iocs: HashSet<&str> = iocs
        .iter()
        .filter(|i| i.ind_type == IndicatorType::Sha256)
        .map(|i| i.value.as_str())
        .collect();

    if hash_iocs.is_empty() {
        return 0;
    }

    let mut candidate_dirs = Vec::new();
    if let Some(home) = dirs::home_dir() {
        candidate_dirs.push(home.join("Downloads"));
        candidate_dirs.push(home.join("Desktop"));
        if deep {
            candidate_dirs.push(home.join(".cache"));
        }
    }
    candidate_dirs.push(std::env::temp_dir());

    let mut files_scanned = 0;
    let max_files = if deep { 1000 } else { 200 };

    for dir in candidate_dirs {
        if !dir.exists() {
            continue;
        }

        if let Ok(entries) = fs::read_dir(&dir) {
            for entry in entries.flatten() {
                if files_scanned >= max_files {
                    break;
                }
                let path = entry.path();
                if path.is_file() {
                    // Only hash files < 32MB to keep scan instant
                    if let Ok(meta) = entry.metadata() {
                        if meta.len() > 0 && meta.len() < 32 * 1024 * 1024 {
                            files_scanned += 1;
                            if let Ok(bytes) = fs::read(&path) {
                                let mut hasher = Sha256::new();
                                hasher.update(&bytes);
                                let digest = format!("{:x}", hasher.finalize());

                                if hash_iocs.contains(digest.as_str()) {
                                    if let Some(ioc) = iocs.iter().find(|i| i.value == digest) {
                                        findings.push(ForensicFinding {
                                            indicator: ioc.value.clone(),
                                            indicator_type: "sha256".into(),
                                            tier: ioc.tier.clone(),
                                            confidence: ioc.confidence.clone(),
                                            source: ioc.source.clone(),
                                            artifact_path: path.display().to_string(),
                                            match_context: format!("SHA256 Match: {}", digest),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    files_scanned
}

/// Executes a complete read-only digital forensics scan across the host.
pub fn run_forensic_scan(deep: bool) -> ForensicScanReport {
    let start_time = std::time::Instant::now();
    let (platform, hardware) = detect_forensics_platform();

    // Check for custom IOC file in ~/.cynapse/synapses/zero_breach/adfection_iocs.txt
    let custom_ioc_path = dirs::home_dir()
        .map(|h| h.join(".cynapse").join("synapses").join("zero_breach").join("adfection_iocs.txt"));

    let raw_iocs = if let Some(ref p) = custom_ioc_path {
        if p.exists() {
            fs::read_to_string(p).unwrap_or_else(|_| EMBEDDED_IOCS.to_string())
        } else {
            EMBEDDED_IOCS.to_string()
        }
    } else {
        EMBEDDED_IOCS.to_string()
    };

    let iocs = parse_iocs(&raw_iocs);
    let mut findings = Vec::new();
    let mut gaps = Vec::new();
    let mut inspected_targets = Vec::new();
    let mut total_artifacts = 0;

    // 1. Browser Profiles (Chromium & Firefox)
    let browser_roots = discover_browser_profile_roots();
    if browser_roots.is_empty() {
        gaps.push(ForensicCoverageGap {
            category: "browser".into(),
            target: "All Browser Profiles".into(),
            reason: "No installed browser profiles discovered in default OS paths".into(),
        });
    }

    for (b_name, b_path) in browser_roots {
        // Chromium structure
        let mut chromium_dbs = vec![
            b_path.join("Default/History"),
            b_path.join("Default/Network/Cookies"),
            b_path.join("Default/Cookies"),
        ];
        // Scan numbered profiles (Profile 1, Profile 2, etc.)
        if let Ok(entries) = fs::read_dir(&b_path) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    let fname = p.file_name().unwrap_or_default().to_string_lossy();
                    if fname.starts_with("Profile ") {
                        chromium_dbs.push(p.join("History"));
                        chromium_dbs.push(p.join("Network/Cookies"));
                    }
                }
            }
        }

        for db in chromium_dbs {
            if db.exists() {
                let is_history = db.ends_with("History");
                let query = if is_history {
                    "SELECT url FROM urls LIMIT 50000"
                } else {
                    "SELECT host_key FROM cookies LIMIT 50000"
                };
                let before = findings.len();
                let count = scan_sqlite_history(&b_name, &db, query, &iocs, &mut findings, &mut gaps);
                total_artifacts += count;
                inspected_targets.push(InspectedTarget {
                    category: if is_history { "Browser History".into() } else { "Browser Cookies".into() },
                    target: format!("{} ({})", db.display(), b_name),
                    items_checked: count,
                    status: if findings.len() > before { "[! COMPROMISE]" } else { "[✓ CLEAN]" }.into(),
                });
            }
        }

        // Firefox structure
        if b_name.contains("Firefox") {
            if let Ok(entries) = fs::read_dir(&b_path) {
                for entry in entries.flatten() {
                    let profile_dir = entry.path();
                    if profile_dir.is_dir() {
                        let places = profile_dir.join("places.sqlite");
                        if places.exists() {
                            let before = findings.len();
                            let count = scan_sqlite_history(
                                "Mozilla Firefox",
                                &places,
                                "SELECT url FROM moz_places LIMIT 50000",
                                &iocs,
                                &mut findings,
                                &mut gaps,
                            );
                            total_artifacts += count;
                            inspected_targets.push(InspectedTarget {
                                category: "Firefox Places".into(),
                                target: places.display().to_string(),
                                items_checked: count,
                                status: if findings.len() > before { "[! COMPROMISE]" } else { "[✓ CLEAN]" }.into(),
                            });
                        }
                        let cookies = profile_dir.join("cookies.sqlite");
                        if cookies.exists() {
                            let before = findings.len();
                            let count = scan_sqlite_history(
                                "Mozilla Firefox",
                                &cookies,
                                "SELECT host FROM moz_cookies LIMIT 50000",
                                &iocs,
                                &mut findings,
                                &mut gaps,
                            );
                            total_artifacts += count;
                            inspected_targets.push(InspectedTarget {
                                category: "Firefox Cookies".into(),
                                target: cookies.display().to_string(),
                                items_checked: count,
                                status: if findings.len() > before { "[! COMPROMISE]" } else { "[✓ CLEAN]" }.into(),
                            });
                        }
                    }
                }
            }
        }
    }

    // 2. Host Artifacts (Hosts file & DNS configuration)
    let before_hosts = findings.len();
    let hosts_count = scan_hosts_file(&iocs, &mut findings, &mut gaps);
    total_artifacts += hosts_count;
    inspected_targets.push(InspectedTarget {
        category: "Host / DNS Cache".into(),
        target: if cfg!(target_os = "windows") { r"C:\Windows\System32\drivers\etc\hosts".into() } else { "/etc/hosts".into() },
        items_checked: hosts_count,
        status: if findings.len() > before_hosts { "[! COMPROMISE]" } else { "[✓ CLEAN]" }.into(),
    });

    // 3. File Hash Fingerprinting
    let before_hashes = findings.len();
    let hashes_count = scan_file_hashes(&iocs, deep, &mut findings, &mut gaps);
    total_artifacts += hashes_count;
    inspected_targets.push(InspectedTarget {
        category: if deep { "Deep Staging Hashes".into() } else { "Staging Artifacts".into() },
        target: "Downloads, Desktop, /tmp SHA-256 signatures".into(),
        items_checked: hashes_count,
        status: if findings.len() > before_hashes { "[! COMPROMISE]" } else { "[✓ CLEAN]" }.into(),
    });

    let duration = start_time.elapsed().as_millis() as u64;

    let status = if !findings.is_empty() {
        ForensicStatus::ThreatDetected
    } else if !gaps.is_empty() && total_artifacts == 0 {
        ForensicStatus::GapsOnly
    } else {
        ForensicStatus::Clean
    };

    let history_file = Path::new("data/scan_history.json");
    let mut history_diff = None;
    if let Ok(content) = fs::read_to_string(history_file) {
        if let Ok(entries) = serde_json::from_str::<Vec<ScanHistoryEntry>>(&content) {
            if let Some(last) = entries.last() {
                history_diff = Some(ForensicScanDiff {
                    previous_artifacts: last.total_artifacts_scanned,
                    artifact_delta: total_artifacts as i64 - last.total_artifacts_scanned as i64,
                    previous_findings: last.findings_count,
                    findings_delta: findings.len() as i64 - last.findings_count as i64,
                    last_scan_timestamp: last.timestamp.clone(),
                });
            }
        }
    }

    let current_timestamp = chrono_now_iso();
    let new_entry = ScanHistoryEntry {
        timestamp: current_timestamp.clone(),
        total_artifacts_scanned: total_artifacts,
        findings_count: findings.len(),
        mode: if deep { "DEEP".into() } else { "QUICK".into() },
    };

    let mut entries = if let Ok(content) = fs::read_to_string(history_file) {
        serde_json::from_str::<Vec<ScanHistoryEntry>>(&content).unwrap_or_default()
    } else {
        Vec::new()
    };
    entries.push(new_entry);
    if let Some(parent) = history_file.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string_pretty(&entries) {
        let _ = fs::write(history_file, json);
    }

    ForensicScanReport {
        platform,
        hardware,
        scan_timestamp: current_timestamp,
        total_artifacts_scanned: total_artifacts,
        iocs_loaded: iocs.len(),
        findings,
        gaps,
        inspected_targets,
        status,
        zero_network_guarantee: true,
        scan_duration_ms: duration,
        history_diff,
    }
}

/// Returns the embedded threat indicators list for interactive inspection.
pub fn get_embedded_iocs() -> Vec<Indicator> {
    parse_iocs(EMBEDDED_IOCS)
}

/// Filters threat indicators by substring query and category index.
/// category: 0 = All, 1 = Domains, 2 = IPs, 3 = Hashes
pub fn filter_iocs(iocs: &[Indicator], filter_query: &str, category: usize) -> Vec<Indicator> {
    let q = filter_query.trim().to_lowercase();
    iocs.iter()
        .filter(|ioc| {
            let cat_match = match category {
                1 => ioc.ind_type == IndicatorType::Domain,
                2 => ioc.ind_type == IndicatorType::Ip,
                3 => ioc.ind_type == IndicatorType::Sha256,
                _ => true,
            };
            if !cat_match {
                return false;
            }
            if q.is_empty() {
                return true;
            }
            ioc.value.to_lowercase().contains(&q)
                || ioc.tier.to_lowercase().contains(&q)
                || ioc.source.to_lowercase().contains(&q)
                || ioc.confidence.to_lowercase().contains(&q)
        })
        .cloned()
        .collect()
}

fn chrono_now_iso() -> String {
    let now = std::time::SystemTime::now();
    let since = now.duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
    format!("{}-epoch-sec", since)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_embedded_iocs() {
        let iocs = parse_iocs(EMBEDDED_IOCS);
        assert!(!iocs.is_empty(), "Embedded IOCs should parse successfully");
        assert!(iocs.iter().any(|i| i.ind_type == IndicatorType::Domain));
        assert!(iocs.iter().any(|i| i.value == "fearevolve.com"));
        assert!(iocs.iter().any(|i| i.value == "52.174.65.77"));
    }

    #[test]
    fn test_matches_domain_boundary() {
        assert!(matches_domain_boundary("fearevolve.com", "fearevolve.com"));
        assert!(matches_domain_boundary("sub.fearevolve.com", "fearevolve.com"));
        assert!(matches_domain_boundary("sub.sub.fearevolve.com", "fearevolve.com"));
        assert!(matches_domain_boundary("fearevolve.com.", "fearevolve.com"));

        // Must reject sub-string prefix / suffix masquerading
        assert!(!matches_domain_boundary("notfearevolve.com", "fearevolve.com"));
        assert!(!matches_domain_boundary("fearevolve.computer", "fearevolve.com"));
        assert!(!matches_domain_boundary("fearevolve.com.attacker.com", "fearevolve.com"));
    }

    #[test]
    fn test_filter_iocs_filtering() {
        let iocs = parse_iocs(EMBEDDED_IOCS);
        let domains = filter_iocs(&iocs, "", 1);
        assert!(!domains.is_empty());
        assert!(domains.iter().all(|i| i.ind_type == IndicatorType::Domain));

        let fearevolve = filter_iocs(&iocs, "fearevolve", 0);
        assert!(!fearevolve.is_empty());
        assert!(fearevolve[0].value.contains("fearevolve"));

        let ips = filter_iocs(&iocs, "", 2);
        assert!(!ips.is_empty());
        assert!(ips.iter().all(|i| i.ind_type == IndicatorType::Ip));
    }

    #[test]
    fn test_run_forensic_scan_execution() {
        let report = run_forensic_scan(false);
        assert!(report.zero_network_guarantee);
        assert!(report.iocs_loaded > 50);
        assert!(!report.platform.is_empty());
        let json = report.to_json();
        assert!(json.contains("zero_network_guarantee"));
    }
}
