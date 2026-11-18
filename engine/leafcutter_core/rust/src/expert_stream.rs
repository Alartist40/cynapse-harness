//! Active-expert disk streaming inspired by `kimi-k3-in-c`.
//! Streams only the routed top-K MoE expert byte slices from disk on demand
//! with a bounded RSS cache, keeping oversized MoE models inside tight RAM limits.

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

pub struct ExpertStreamer {
    file: File,
    max_cached: usize,
    cache: HashMap<(usize, usize), Vec<u8>>,
    lru: VecDeque<(usize, usize)>,
}

impl ExpertStreamer {
    pub fn open<P: AsRef<Path>>(path: P, max_cached: usize) -> io::Result<Self> {
        Ok(Self {
            file: File::open(path)?,
            max_cached: max_cached.max(1),
            cache: HashMap::new(),
            lru: VecDeque::new(),
        })
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
        if !self.cache.contains_key(&key) {
            self.file.seek(SeekFrom::Start(offset))?;
            let mut buf = vec![0u8; len];
            self.file.read_exact(&mut buf)?;

            if self.cache.len() >= self.max_cached {
                if let Some(evict) = self.lru.pop_front() {
                    self.cache.remove(&evict);
                }
            }
            self.cache.insert(key, buf);
            self.lru.push_back(key);
        }
        Ok(self.cache.get(&key).map(|v| v.as_slice()).unwrap_or(&[]))
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
}
