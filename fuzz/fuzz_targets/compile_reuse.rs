//! Checks that re-using `RegisterAllocator` for multiple compilations works.

#![no_main]

use libfuzzer_sys::fuzz_target;
use regalloc3::{debug_utils, RegisterAllocator};
use regalloc3_fuzz::TestCase;

fuzz_target!(|ts: [TestCase; 4]| {
    // Ensure the logger is initialized.
    let _ = pretty_env_logger::try_init();

    let mut regalloc = RegisterAllocator::new();
    for t in ts {
        if let Ok(output) = regalloc.allocate_registers(&t.func, t.reginfo(), &Default::default()) {
            debug_utils::check_output(&output).unwrap();
        }
    }
});
