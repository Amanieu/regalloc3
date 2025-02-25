use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use arbitrary::Unstructured;
use clap::Parser;
use example_reginfo::Arch;
use rand::RngCore;
use regalloc3::debug_utils::{
    self, ArbitraryFunctionConfig, ArbitraryRegInfoConfig, GenericFunction, GenericRegInfo,
};
use regalloc3::{Options, RegisterAllocator};

mod example_reginfo;

#[derive(Parser)]
/// Tool for testing regalloc3.
enum Args {
    /// Run register allocation on a given function.
    Compile {
        /// Print the input function and the result of register allocation.
        #[clap(short = 'v')]
        verbose: bool,

        /// File containing the register description for the target.
        reginfo: PathBuf,

        /// File containing the function to register allocate.
        function: PathBuf,

        /// Register allocator options.
        #[clap(flatten)]
        options: Options,
    },

    /// Generate a random function.
    GenFunction {
        /// File containing the register description for the target.
        reginfo: PathBuf,

        /// Number of CFG edges. This also implicitly controls the number of blocks
        /// in a function since all blocks must be reachable from the entry block.
        #[clap(long, default_value_t = 10)]
        cfg_edges: usize,

        /// Number of block parameters for each block that is allowed to have them.
        #[clap(long, default_value_t = 10)]
        blockparams_per_block: usize,

        /// Number of instructions per block, excluding the terminator instruction.
        #[clap(long, default_value_t = 10)]
        insts_per_block: usize,

        /// Number of definition operands (`Def` and `EarlyDef`) per instruction.
        ///
        /// Some instructions may exceed this limit due to the way the algorithm
        /// works. This is because all used values need a definition, which may
        /// force extra definitions to be added.
        #[clap(long, default_value_t = 10)]
        defs_per_inst: usize,

        /// Number of non-definition operands (`Use` and `NonAllocatable`) per
        /// instruction.
        #[clap(long, default_value_t = 10)]
        uses_per_inst: usize,

        /// Number of clobbers per instruction.
        #[clap(long, default_value_t = 10)]
        clobbers_per_inst: usize,
    },

    /// Generate a random register description.
    GenReginfo {
        /// Number of register banks.
        #[clap(long, default_value_t = 5)]
        num_banks: usize,

        /// Number of registers in each register bank.
        #[clap(long, default_value_t = 20)]
        regs_per_bank: usize,

        /// Number of register classes in each register bank in addition to the
        /// top-level class and stack-to-stack class.
        #[clap(long, default_value_t = 5)]
        extra_classes_per_bank: usize,

        /// Number of register units per register.
        #[clap(long, default_value_t = 8)]
        units_per_reg: usize,
    },

    /// Parses the given function and re-dumps it with proper formatting.
    ///
    /// Note that this will strip all comments.
    FmtFunction {
        /// File containing the register description for the target.
        reginfo: PathBuf,

        /// File containing the function definition.
        function: PathBuf,
    },

    /// Parses the given register description and re-dumps it with proper formatting.
    ///
    /// Note that this will strip all comments.
    FmtReginfo {
        /// File containing the register description for the target.
        reginfo: PathBuf,
    },

    /// Generate a register description from a template.
    ExampleReginfo {
        /// Number of fixed stack slots to add to the register definition.
        #[clap(short = 'f', default_value_t = 0)]
        fixed_stack: usize,

        /// Architecture to generate the register definitions for.
        arch: Arch,
    },
}

fn load_reginfo(path: &Path) -> Result<GenericRegInfo> {
    let reginfo = fs::read(path).context("could not read reginfo input file")?;
    let reginfo = String::from_utf8(reginfo).context("reginfo input is not UTF-8")?;
    let reginfo = GenericRegInfo::parse(&reginfo).context("could not parse reginfo input file")?;
    debug_utils::validate_reginfo(&reginfo).context("reginfo validation failed")?;
    Ok(reginfo)
}

fn load_function(path: &Path, reginfo: &GenericRegInfo) -> Result<GenericFunction> {
    let function = fs::read(path).context("could not read function input file")?;
    let function = String::from_utf8(function).context("function input is not UTF-8")?;
    let function =
        GenericFunction::parse(&function).context("could not parse function input file")?;
    debug_utils::validate_function(&function, reginfo).context("function validation failed")?;
    Ok(function)
}

fn main() -> Result<()> {
    pretty_env_logger::init();
    let args = Args::parse();

    match args {
        Args::Compile {
            verbose,
            ref reginfo,
            ref function,
            ref options,
        } => {
            let reginfo = load_reginfo(reginfo)?;
            let function = load_function(function, &reginfo)?;

            if verbose {
                println!(
                    "================ Input function ================\n{}",
                    debug_utils::DisplayFunction(&function)
                );
            }

            let mut regalloc = RegisterAllocator::new();
            let output = regalloc
                .allocate_registers(&function, &reginfo, options)
                .unwrap();

            println!("================ Output ================\n{output}");

            println!(
                "Cost model score: {}",
                debug_utils::CostModel::default().evaluate(&output)
            );

            println!("{}", output.stats());

            debug_utils::check_output(&output)
                .context("register allocation result failed checker")?;
        }
        Args::GenFunction {
            ref reginfo,
            cfg_edges,
            blockparams_per_block,
            insts_per_block,
            defs_per_inst,
            uses_per_inst,
            clobbers_per_inst,
        } => {
            let reginfo = load_reginfo(reginfo)?;
            let config = ArbitraryFunctionConfig {
                cfg_edges: 0..=cfg_edges,
                blockparams_per_block: 0..=blockparams_per_block,
                insts_per_block: 0..=insts_per_block,
                defs_per_inst: 0..=defs_per_inst,
                uses_per_inst: 0..=uses_per_inst,
                clobbers_per_inst: 0..=clobbers_per_inst,
            };
            let mut bytes = [0; 4096];
            rand::rng().fill_bytes(&mut bytes);
            let function = GenericFunction::arbitrary_with_config(
                &reginfo,
                &mut Unstructured::new(&bytes),
                config,
            )
            .context("failed to generate arbitrary function")?;

            println!("{}", debug_utils::DisplayFunction(&function));
        }
        Args::GenReginfo {
            num_banks,
            regs_per_bank,
            extra_classes_per_bank,
            units_per_reg,
        } => {
            let config = ArbitraryRegInfoConfig {
                num_banks: 1..=num_banks,
                regs_per_bank: 1..=regs_per_bank,
                extra_classes_per_bank: 0..=extra_classes_per_bank,
                units_per_reg: 1..=units_per_reg,
            };
            let mut bytes = [0; 4096];
            rand::rng().fill_bytes(&mut bytes);
            let reginfo =
                GenericRegInfo::arbitrary_with_config(&mut Unstructured::new(&bytes), config)
                    .context("failed to generate arbitrary function")?;

            println!("{}", debug_utils::DisplayRegInfo(&reginfo));
        }
        Args::FmtFunction {
            ref reginfo,
            ref function,
        } => {
            let reginfo = load_reginfo(reginfo)?;
            let function = load_function(function, &reginfo)?;

            println!("{}", debug_utils::DisplayFunction(&function));
        }
        Args::FmtReginfo { ref reginfo } => {
            let reginfo = load_reginfo(reginfo)?;

            println!("{}", debug_utils::DisplayRegInfo(&reginfo));
        }
        Args::ExampleReginfo { fixed_stack, arch } => {
            let reginfo = arch.gen_reginfo(fixed_stack);
            let mut reginfo_text = String::new();
            reginfo.emit(&mut reginfo_text).unwrap();

            // Sanity-check
            let reginfo = GenericRegInfo::parse(&reginfo_text)
                .context("could not parse generated reginfo")?;
            debug_utils::validate_reginfo(&reginfo)
                .context("generated reginfo validation failed")?;

            print!("{reginfo_text}");
        }
    }
    Ok(())
}
