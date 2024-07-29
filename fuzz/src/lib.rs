use std::fmt;
use std::sync::OnceLock;

use arbitrary::{Arbitrary, Result, Unstructured};
use regalloc3::debug_utils::{self, GenericFunction, GenericRegInfo};

/// Example register descriptions that are parsed and validated once.
static EXAMPLE_REGINFOS: OnceLock<Vec<(&'static str, GenericRegInfo)>> = OnceLock::new();

enum TestCaseRegInfo {
    Example {
        path: &'static str,
        reginfo: &'static GenericRegInfo,
    },
    Arbitrary {
        reginfo: GenericRegInfo,
    },
}

impl TestCaseRegInfo {
    pub fn get(&self) -> &GenericRegInfo {
        match *self {
            TestCaseRegInfo::Example { path: _, reginfo } => reginfo,
            TestCaseRegInfo::Arbitrary { ref reginfo } => reginfo,
        }
    }
}

/// Common implementation of a test case used by all fuzz targets.
pub struct TestCase {
    reginfo: TestCaseRegInfo,
    pub func: GenericFunction,
}

impl TestCase {
    pub fn reginfo(&self) -> &GenericRegInfo {
        self.reginfo.get()
    }
}

impl Arbitrary<'_> for TestCase {
    fn arbitrary(u: &mut Unstructured) -> Result<Self> {
        // Ensure the logger is initialized.
        let _ = pretty_env_logger::try_init();

        let example_reginfos = EXAMPLE_REGINFOS.get_or_init(|| {
            let aarch64 =
                GenericRegInfo::parse(include_str!("../../example_reginfo/aarch64.reginfo"))
                    .unwrap();
            let riscv =
                GenericRegInfo::parse(include_str!("../../example_reginfo/riscv.reginfo")).unwrap();
            debug_utils::validate_reginfo(&aarch64).unwrap();
            debug_utils::validate_reginfo(&riscv).unwrap();
            vec![
                ("example_reginfo/aarch64.reginfo", aarch64),
                ("example_reginfo/riscv.reginfo", riscv),
            ]
        });
        let reginfo = if u.arbitrary()? {
            let (path, reginfo) = u.choose(example_reginfos)?;
            log::trace!("Using example reginfo: {path}");
            TestCaseRegInfo::Example { path, reginfo }
        } else {
            let reginfo = u.arbitrary()?;
            log::trace!("Using arbitrary reginfo:\n{reginfo}");
            TestCaseRegInfo::Arbitrary { reginfo }
        };
        let func = GenericFunction::arbitrary_with_config(reginfo.get(), u, Default::default())?;
        Ok(TestCase { reginfo, func })
    }
}

impl fmt::Debug for TestCase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.reginfo {
            TestCaseRegInfo::Example { path, reginfo: _ } => {
                writeln!(f, "Using example reginfo: {path}")?;
            }
            TestCaseRegInfo::Arbitrary { reginfo } => {
                writeln!(f, "{reginfo}")?;
            }
        }
        writeln!(f, "{}", self.func)
    }
}
