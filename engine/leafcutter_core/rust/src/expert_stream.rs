//! Active-expert disk streaming inspired by `kimi-k3-in-c` (`k3_cache.h`).
//! Streams only the routed top-K MoE expert byte slices from disk on demand
//! with a bounded RSS cache, frequency histogram, hot-expert pinning, and
//! honest hit-rate accounting (separating prefetch reads from true cache hits).

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

pub type ExpertKey = (usize, usize);

pub struct ExpertStreamer {
    file: File,
    max_cached: usize,
    cache: HashMap<ExpertKey, Vec<u8>>,
    lru: VecDeque<ExpertKey>,
    pinned: HashSet<ExpertKey>,
    hist: HashMap<ExpertKey, u64>,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub prefetch_reads: u64,
    pub bytes_read: u64,
}

impl ExpertStreamer {
    pub fn open<P: AsRef<Path>>(path: P, max_cached: usize) -> io::Result<Self> {
        Ok(Self {
            file: File::open(path)?,
            max_cached: max_cached.max(1),
            cache: HashMap::new(),
            lru: VecDeque::new(),
            pinned: HashSet::new(),
            hist: HashMap::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
            prefetch_reads: 0,
            bytes_read: 0,
        })
    }

    fn evict_one(&mut self) -> bool {
        let n = self.lru.len();
        for _ in 0..n {
            if let Some(candidate) = self.lru.pop_front() {
                if self.pinned.contains(&candidate) {
                    self.lru.push_back(candidate);
                    continue;
                }
                self.cache.remove(&candidate);
                self.evictions += 1;
                return true;
            }
        }
        false
    }

    fn read_slice(&mut self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        self.file.seek(SeekFrom::Start(offset))?;
        let mut buf = vec![0u8; len];
        self.file.read_exact(&mut buf)?;
        self.bytes_read += len as u64;
        Ok(buf)
    }

    /// Stream only `(layer, expert_id)` from `[offset .. offset + len]` on disk.
    pub fn load_expert(
        &mut self,
        layer: usize,
        expert_id: usize,
        offset: u64,
        len: usize,
    ) -> io::Result<&[u8]> {
        let key = (layer, expert_id);
        *self.hist.entry(key).or_insert(0) += 1;

        if self.cache.contains_key(&key) {
            self.hits += 1;
            self.lru.retain(|&k| k != key);
            self.lru.push_back(key);
        } else {
            self.misses += 1;
            let buf = self.read_slice(offset, len)?;
            if self.cache.len() >= self.max_cached {
                let _ = self.evict_one();
            }
            self.cache.insert(key, buf);
            self.lru.push_back(key);
        }
        Ok(self.cache.get(&key).map(|v| v.as_slice()).unwrap_or(&[]))
    }

    /// Prefetch `(layer, expert_id)` into the packed cache without inflating effective hit rate.
    pub fn prefetch_expert(
        &mut self,
        layer: usize,
        expert_id: usize,
        offset: u64,
        len: usize,
    ) -> io::Result<bool> {
        let key = (layer, expert_id);
        if self.cache.contains_key(&key) {
            return Ok(false);
        }
        let buf = self.read_slice(offset, len)?;
        if self.cache.len() >= self.max_cached && !self.evict_one() {
            return Ok(false);
        }
        self.cache.insert(key, buf);
        self.lru.push_back(key);
        self.prefetch_reads += 1;
        Ok(true)
    }

    /// Pin a specific `(layer, expert_id)` so LRU will never evict it.
    pub fn pin(&mut self, layer: usize, expert_id: usize) {
        self.pinned.insert((layer, expert_id));
    }

    /// Pin the top-`k` most frequently requested experts currently in the histogram.
    pub fn pin_hot_experts(&mut self, top_k: usize) -> usize {
        let mut ranked: Vec<(ExpertKey, u64)> = self.hist.iter().map(|(&k, &v)| (k, v)).collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1));
        let cap = self.max_cached.saturating_sub(1);
        let mut count = 0;
        for (key, _) in ranked.into_iter().take(top_k) {
            if self.pinned.len() >= cap {
                break;
            }
            if self.pinned.insert(key) {
                count += 1;
            }
        }
        count
    }

    /// True cache hit rate excluding disk-backed prefetch reads (`(hits - prefetch_reads) / total`).
    pub fn effective_hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        let true_hits = self.hits.saturating_sub(self.prefetch_reads);
        true_hits as f64 / total as f64
    }

    pub fn cached_count(&self) -> usize {
        self.cache.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn streams_only_requested_expert_slice_and_bounds_cache() {
        let path = std::env::temp_dir().join(format!("leafcutter_expert_{}.bin", std::process::id()));
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(b"EXPERT_0EXPERT_1EXPERT_2").unwrap();
        }

        let mut streamer = ExpertStreamer::open(&path, 2).unwrap();
        assert_eq!(streamer.load_expert(0, 0, 0, 8).unwrap(), b"EXPERT_0");
        assert_eq!(streamer.load_expert(0, 2, 16, 8).unwrap(), b"EXPERT_2");
        assert_eq!(streamer.load_expert(0, 1, 8, 8).unwrap(), b"EXPERT_1");
        assert_eq!(streamer.cached_count(), 2);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn hot_expert_pinning_protects_from_eviction_and_tracks_prefetch() {
        let path = std::env::temp_dir().join(format!("leafcutter_expert_pin_{}.bin", std::process::id()));
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(b"AAAAAAAA_BBBBBBBB_CCCCCCCC").unwrap();
        }

        let mut streamer = ExpertStreamer::open(&path, 2).unwrap();
        assert_eq!(streamer.load_expert(0, 0, 0, 8).unwrap(), b"AAAAAAAA");
        assert_eq!(streamer.load_expert(0, 0, 0, 8).unwrap(), b"AAAAAAAA");
        assert_eq!(streamer.pin_hot_experts(1), 1);

        assert!(streamer.prefetch_expert(0, 1, 9, 8).unwrap());
        assert_eq!(streamer.load_expert(0, 1, 9, 8).unwrap(), b"BBBBBBBB");
        // Loading (0, 2) must evict (0, 1), keeping pinned (0, 0).
        assert_eq!(streamer.load_expert(0, 2, 18, 8).unwrap(), b"CCCCCCCC");
        assert_eq!(streamer.load_expert(0, 0, 0, 8).unwrap(), b"AAAAAAAA");
        assert_eq!(streamer.misses, 2); // (0,0) initial + (0,2)
        assert_eq!(streamer.prefetch_reads, 1);

        let _ = std::fs::remove_file(&path);
    }
}
