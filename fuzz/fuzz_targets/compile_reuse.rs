//! Checks that re-using `RegisterAllocator` for multiple compilations works.

#![no_main]

use libfuzzer_sys::fuzz_target;
use regalloc3::{RegAllocError, RegisterAllocator, debug_utils};
use regalloc3_fuzz::TestCase;

fuzz_target!(|ts: [TestCase; 4]| {
    // Ensure the logger is initialized.
    let _ = pretty_env_logger::try_init();

    let mut regalloc = RegisterAllocator::new();
    for t in ts {
        match regalloc.allocate_registers(&t.func, t.reginfo(), &t.options) {
            Ok(output) => debug_utils::check_output(&output).unwrap(),
            Err(RegAllocError::FunctionTooBig) => {}
            Err(error) => panic!("generated function failed allocation: {error}"),
        }
    }
});
