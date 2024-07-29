//! Checks that dumping a function and then re-parsing it is lossless.

#![no_main]

use libfuzzer_sys::fuzz_target;
use regalloc3::debug_utils::{self, GenericRegInfo};

fuzz_target!(|reginfo: GenericRegInfo| {
    // Ensure the logger is initialized.
    let _ = pretty_env_logger::try_init();

    let dumped = debug_utils::DisplayRegInfo(&reginfo).to_string();
    let parsed = GenericRegInfo::parse(&dumped).unwrap();
    let dumped2 = debug_utils::DisplayRegInfo(&parsed).to_string();
    assert_eq!(dumped, dumped2);
});
