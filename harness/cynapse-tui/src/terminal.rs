//! Production-Grade RAII Terminal Guard & Custom Panic Hook.
//!
//! Inspired by jcode's `terminal.rs`. Ensures raw mode and alternate terminal
//! screen are ALWAYS restored cleanly upon exit, drop, error, or panic.

use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use anyhow::Result;
use crossterm::{
    event::{DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};

static RAW_MODE_ACTIVE: AtomicBool = AtomicBool::new(false);

/// RAII Terminal Guard ensuring clean terminal restoration on drop or panic.
pub struct TuiRuntimeGuard {
    _private: (),
}

impl TuiRuntimeGuard {
    pub fn enter() -> Result<Self> {
        set_panic_hook();
        enable_raw_mode()?;
        RAW_MODE_ACTIVE.store(true, Ordering::SeqCst);

        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableBracketedPaste, EnableMouseCapture)?;

        Ok(Self { _private: () })
    }
}

impl Drop for TuiRuntimeGuard {
    fn drop(&mut self) {
        Self::restore_terminal_sync();
    }
}

impl TuiRuntimeGuard {
    pub fn restore_terminal_sync() {
        if RAW_MODE_ACTIVE.compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
            let _ = disable_raw_mode();
            let mut stdout = io::stdout();
            let _ = execute!(stdout, LeaveAlternateScreen, DisableBracketedPaste, DisableMouseCapture);
        }
    }
}

/// Installs custom panic hook to restore terminal state before printing backtrace.
fn set_panic_hook() {
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        TuiRuntimeGuard::restore_terminal_sync();
        eprintln!("\n❌ [CYNAPSE TUI CRASH DETECTED]");
        eprintln!("   Restored terminal state cleanly.");
        original_hook(panic_info);
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guard_flag_safety() {
        assert!(!RAW_MODE_ACTIVE.load(Ordering::SeqCst));
    }
}
