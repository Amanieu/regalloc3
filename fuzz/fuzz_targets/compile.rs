//! Checks that register allocation produces correct results.

#![no_main]

use libfuzzer_sys::fuzz_target;
use regalloc3::{RegisterAllocator, debug_utils};
use regalloc3_fuzz::TestCase;

fuzz_target!(|t: TestCase| {
    // Ensure the logger is initialized.
    let _ = pretty_env_logger::try_init();

    let mut regalloc = RegisterAllocator::new();
    if let Ok(output) = regalloc.allocate_registers(&t.func, t.reginfo(), &t.options) {
        debug_utils::check_output(&output).unwrap();
    }
});
