//! Checks that `GenericFunction::arbitrary` produces functions that pass
//! validation.

#![no_main]

use libfuzzer_sys::fuzz_target;
use regalloc3::debug_utils;
use regalloc3_fuzz::TestCase;

fuzz_target!(|t: TestCase| {
    // Ensure the logger is initialized.
    let _ = pretty_env_logger::try_init();

    debug_utils::validate_function(&t.func, t.reginfo()).unwrap();
});
