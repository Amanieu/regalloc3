//! Checks that dumping a function and then re-parsing it is lossless.

#![no_main]

use libfuzzer_sys::fuzz_target;
use regalloc3::debug_utils::{self, GenericFunction};
use regalloc3_fuzz::TestCase;

fuzz_target!(|t: TestCase| {
    // Ensure the logger is initialized.
    let _ = pretty_env_logger::try_init();

    let dumped = debug_utils::DisplayFunction(&t.func).to_string();
    let parsed = GenericFunction::parse(&dumped).unwrap();
    let dumped2 = debug_utils::DisplayFunction(&parsed).to_string();
    assert_eq!(dumped, dumped2);
});
