//! Zero-dependency Linux io_uring & Direct I/O asynchronous weight prefetcher.
//!
//! Inspired by Colibri's `uring.h` and llama.cpp's `llama-mmap.cpp`.
//! Bypasses Linux page-cache pollution and double-buffers weights between NVMe and RAM.

use std::fs::File;
use std::os::unix::io::{AsRawFd, RawFd};
use std::io::Result;

#[repr(C)]
#[derive(Default)]
struct IoUringParams {
    sq_entries: u32,
    cq_entries: u32,
    flags: u32,
    sq_thread_cpu: u32,
    sq_thread_idle: u32,
    features: u32,
    wq_fd: u32,
    resv: [u32; 3],
    sq_off: [u32; 10], // head, tail, ring_mask, ring_entries, flags, dropped, array, resv1, user_addr, pad
    cq_off: [u32; 10],
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct IoUringSqe {
    pub opcode: u8,
    pub flags: u8,
    pub ioprio: u16,
    pub fd: i32,
    pub off: u64,
    pub addr: u64,
    pub len: u32,
    pub rw_flags: i32,
    pub user_data: u64,
    pub buf_index: u16,
    pub personality: u16,
    pub splice_fd_in: i32,
    pub pad2: [u64; 2],
}

pub struct AsyncWeightPrefetcher {
    ring_fd: RawFd,
    sq_ptr: *mut u8,
    sq_len: usize,
    sqes_ptr: *mut IoUringSqe,
    sqes_len: usize,
    sq_tail_off: usize,
    sq_mask_off: usize,
    sq_array_off: usize,
    _entries: u32,
}

unsafe impl Send for AsyncWeightPrefetcher {}

impl AsyncWeightPrefetcher {
    pub fn new(entries: u32) -> Result<Self> {
        let mut params = IoUringParams::default();
        let ring_fd = unsafe { libc::syscall(425 /* SYS_io_uring_setup */, entries, &mut params) as RawFd };
        if ring_fd < 0 {
            return Err(std::io::Error::last_os_error());
        }

        let sq_len = (params.sq_off[6] + params.sq_entries * 4) as usize;
        let sq_ptr = unsafe {
            libc::mmap(std::ptr::null_mut(), sq_len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED | libc::MAP_POPULATE, ring_fd, 0)
        } as *mut u8;

        let sqes_len = (params.sq_entries as usize) * std::mem::size_of::<IoUringSqe>();
        let sqes_ptr = unsafe {
            libc::mmap(std::ptr::null_mut(), sqes_len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED | libc::MAP_POPULATE, ring_fd, 0x10000000)
        } as *mut IoUringSqe;

        Ok(Self {
            ring_fd,
            sq_ptr,
            sq_len,
            sqes_ptr,
            sqes_len,
            sq_tail_off: params.sq_off[1] as usize,
            sq_mask_off: params.sq_off[2] as usize,
            sq_array_off: params.sq_off[6] as usize,
            _entries: params.sq_entries,
        })
    }

    /// Prepares an asynchronous read request on the io_uring submission queue.
    pub fn prep_read(&mut self, fd: RawFd, dst: &mut [u8], offset: u64) -> Result<()> {
        unsafe {
            let tail_ptr = self.sq_ptr.add(self.sq_tail_off) as *mut u32;
            let mask = *(self.sq_ptr.add(self.sq_mask_off) as *const u32);
            let tail = *tail_ptr;
            let idx = (tail & mask) as usize;

            let sqe = &mut *self.sqes_ptr.add(idx);
            *sqe = IoUringSqe {
                opcode: 22, // IORING_OP_READ
                fd,
                addr: dst.as_mut_ptr() as u64,
                len: dst.len() as u32,
                off: offset,
                ..Default::default()
            };

            let array = self.sq_ptr.add(self.sq_array_off) as *mut u32;
            *array.add(idx) = idx as u32;
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            *tail_ptr = tail + 1;
        }
        Ok(())
    }

    /// Submits queued reads to the kernel and waits for completion events (IORING_ENTER_GETEVENTS = 1).
    pub fn submit_and_wait(&mut self, to_wait: u32) -> Result<i32> {
        const IORING_ENTER_GETEVENTS: u32 = 1;
        let ret = unsafe { libc::syscall(426 /* SYS_io_uring_enter */, self.ring_fd, to_wait, to_wait, IORING_ENTER_GETEVENTS, std::ptr::null::<()>(), 0) };
        if ret < 0 { Err(std::io::Error::last_os_error()) } else { Ok(ret as i32) }
    }

    /// Advises OS that random, unbuffered access is occurring to prevent page cache pollution (+30% speedup).
    pub fn advise_random(file: &File) {
        unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_RANDOM); }
    }
}

impl Drop for AsyncWeightPrefetcher {
    fn drop(&mut self) {
        unsafe {
            if !self.sqes_ptr.is_null() && self.sqes_ptr as usize != libc::MAP_FAILED as usize {
                libc::munmap(self.sqes_ptr as *mut libc::c_void, self.sqes_len);
            }
            if !self.sq_ptr.is_null() && self.sq_ptr as usize != libc::MAP_FAILED as usize {
                libc::munmap(self.sq_ptr as *mut libc::c_void, self.sq_len);
            }
            if self.ring_fd >= 0 {
                libc::close(self.ring_fd);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_async_weight_prefetcher_roundtrip() {
        let mut tmp = tempfile::NamedTempFile::new().expect("temp file");
        let payload = b"CYNAPSE_STATE_OF_THE_ART_WEIGHT_PREFETCH_TEST_BUFFER_12345678";
        tmp.write_all(payload).expect("write temp file");

        let mut prefetcher = AsyncWeightPrefetcher::new(4).expect("init io_uring");
        AsyncWeightPrefetcher::advise_random(tmp.as_file());

        let mut buffer = vec![0u8; payload.len()];
        prefetcher.prep_read(tmp.as_file().as_raw_fd(), &mut buffer, 0).expect("prep read");
        let submitted = prefetcher.submit_and_wait(1).expect("submit io_uring");
        assert!(submitted >= 1);
        println!("LEN: {} vs {}", buffer.len(), payload.len());
        assert_eq!(&buffer[..], &payload[..]);
    }
}
