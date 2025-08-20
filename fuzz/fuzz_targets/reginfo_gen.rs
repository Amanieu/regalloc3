//! Checks that `GenericRegInfo::arbitrary` produces register descriptions that
//! pass validation.

#![no_main]

use libfuzzer_sys::fuzz_target;
use regalloc3::debug_utils::{self, GenericRegInfo};

fuzz_target!(|reginfo: GenericRegInfo| {
    // Ensure the logger is initialized.
    let _ = pretty_env_logger::try_init();

    debug_utils::validate_reginfo(&reginfo).unwrap();
});
