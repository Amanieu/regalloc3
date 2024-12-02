//! A flexible and fast register allocator implementation designed to be
//! embedded in an existing compiler code base.
//!
//! This crate is compatible with `#![no_std]` and only requires `alloc`.
//!
//! # Usage
//!
//! To use this register allocator in your project, you will need to define
//! types which implement 2 traits.
//!
//! You will need an implementation of [`RegInfo`] which describes the set of
//! CPU registers available to the allocator and their properties. See the
//! [`reginfo`] module documentation for more details. Generally, you will have
//! a register description for each target architecture supported by your
//! compiler.
//!
//! You will also need an implementation of [`Function`] which describes the
//! code on which the register allocator needs to run. See the [`function`]
//! module documentation for more details.
//!
//! The register allocator is invoked by creating an instance of
//! [`RegisterAllocator`] and then calling [`RegisterAllocator::allocate_registers`].
//!
//! Once register allocation succeeds, it will return an [`Output`] which
//! describes the registers that have been assigned to each instruction operand
//! and the sequence of move/spill/reload instructions that need to be inserted
//! between the original instructions. See the [`output`] module documentation
//! for more details.
//!
//! # Reusing allocations
//!
//! For performance reasons, the [`RegisterAllocator`] type don't free temporary
//! allocations after a call to [`RegisterAllocator::allocate_registers`].
//!
//! This allows register allocation to be run on multiple functions without the
//! need for further calls to the memory allocator, which can be slow. If the
//! memory usage is a concern or if you are finished compiling functions then
//! you can simply drop [`RegisterAllocator`] to free all temporary memory.
//!
//! # Validation
//!
//! When developing a new client of the register allocator, it is highly
//! recommended to use the validation functions in [`debug_utils`] to ensure
//! that the inputs to the register allocator are correct.
//!
//! If inputs fail validation then the register allocator may panic, get stuck
//! in an infinite loop or just produce garbage results.
//!
//! However validation is relatively expensive so you may not want to have it
//! enabled in production.
//!
//! Note that even code that passes validation may cause the register allocator
//! to return an error ([`RegAllocError`]). This is usually an indication of
//! impossible constraints on an instruction.

#![no_std]
#![warn(rust_2018_idioms, missing_docs)]
#![feature(btree_cursors)]
#![allow(
    clippy::too_many_arguments,
    clippy::collapsible_if,
    clippy::collapsible_else_if,
    clippy::single_char_add_str,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::doc_markdown
)]
#![warn(
    clippy::explicit_iter_loop,
    clippy::range_plus_one,
    clippy::map_unwrap_or,
    clippy::explicit_iter_loop,
    clippy::cloned_instead_of_copied,
    clippy::semicolon_if_nothing_returned,
    clippy::must_use_candidate,
    clippy::iter_without_into_iter,
    clippy::uninlined_format_args,
    clippy::ignored_unit_patterns
)]

extern crate alloc;

use core::fmt;

use function::Function;
use internal::allocations::Allocations;
use internal::allocator::Allocator;
use internal::coalescing::Coalescing;
use internal::hints::Hints;
use internal::move_optimizer::MoveOptimizer;
use internal::move_resolver::MoveResolver;
use internal::reg_matrix::RegMatrix;
use internal::spill_allocator::SpillAllocator;
use internal::split_placement::SplitPlacement;
use internal::uses::Uses;
use internal::value_live_ranges::ValueLiveRanges;
use internal::virt_regs::builder::VirtRegBuilder;
use internal::virt_regs::VirtRegs;
use output::Output;
use reginfo::RegInfo;

// Even when trace logging is disabled, the trace macro has a significant
// performance cost so we disable it in release builds.
macro_rules! trace {
    ($($tt:tt)*) => {
        if cfg!(feature = "trace-log") {
            ::log::trace!($($tt)*);
        }
    };
}
macro_rules! trace_enabled {
    () => {
        cfg!(feature = "trace-log") && ::log::log_enabled!(::log::Level::Trace)
    };
}

// Macro for collecting statistics.
//
// If this turns out to be too much overhead then we can put it under a cfg().
macro_rules! stat {
    ($stats:expr, $field:ident) => {
        $stats.$field += 1
    };
    ($stats:expr, $field:ident, $count:expr) => {
        $stats.$field += $count
    };
}

#[macro_use]
pub mod entity;

pub mod debug_utils;
pub mod function;
pub mod output;
pub mod reginfo;

mod allocation_unit;
mod internal;
mod union_find;

/// Exposed internal APIs for fuzzing.
#[cfg(feature = "fuzzing")]
pub use internal::parallel_moves;

/// Structure holding persistent memory allocations that can be reused across
/// multiple invocations of the register allocator.
///
/// This avoids repeated calls to the memory allocator when compiling multiple
/// functions.
pub struct RegisterAllocator {
    value_live_ranges: ValueLiveRanges,
    uses: Uses,
    hints: Hints,
    coalescing: Coalescing,
    virt_regs: VirtRegs,
    virt_reg_builder: VirtRegBuilder,
    allocations: Allocations,
    split_placement: SplitPlacement,
    reg_matrix: RegMatrix,
    allocator: Allocator,
    spill_allocator: SpillAllocator,
    move_resolver: MoveResolver,
    move_optimizer: MoveOptimizer,
    stats: Stats,
}

impl Default for RegisterAllocator {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl RegisterAllocator {
    /// Creates a new `RegisterAllocator` instance.
    #[must_use]
    pub fn new() -> Self {
        Self {
            value_live_ranges: ValueLiveRanges::new(),
            uses: Uses::new(),
            hints: Hints::new(),
            coalescing: Coalescing::new(),
            virt_regs: VirtRegs::new(),
            virt_reg_builder: VirtRegBuilder::new(),
            allocations: Allocations::new(),
            split_placement: SplitPlacement::new(),
            reg_matrix: RegMatrix::new(),
            allocator: Allocator::new(),
            spill_allocator: SpillAllocator::new(),
            move_resolver: MoveResolver::new(),
            move_optimizer: MoveOptimizer::new(),
            stats: Stats::default(),
        }
    }

    /// Runs the register allocator on the given function.
    pub fn allocate_registers<'a, F, R>(
        &'a mut self,
        func: &'a F,
        reginfo: &'a R,
        options: &Options,
    ) -> Result<Output<'a, F, R>, RegAllocError>
    where
        F: Function,
        R: RegInfo,
    {
        trace!("Input function:\n{}", debug_utils::DisplayFunction(func));

        // Reset stats and gather initial information.
        self.stats = Default::default();
        stat!(self.stats, blocks, func.num_blocks());
        stat!(self.stats, input_insts, func.num_insts());
        stat!(self.stats, values, func.num_values());
        stat!(self.stats, value_groups, func.num_value_groups());

        // Prepare data for computing optimal split placement.
        self.split_placement.prepare(func);

        // Reserve space for allocation results in the allocation map.
        self.allocations
            .compute_alloc_offsets(func, &mut self.stats)?;

        // Compute the live range for each SSA value.
        self.value_live_ranges.compute(
            &mut self.uses,
            &mut self.hints,
            &mut self.allocations,
            &mut self.reg_matrix,
            &mut self.stats,
            &mut self.allocator.empty_segments,
            func,
            reginfo,
        );

        // Coalesce SSA values into non-overlapping sets to eliminate
        // unnecessary move instructions.
        self.coalescing.run(
            func,
            &self.uses,
            &mut self.value_live_ranges,
            &mut self.stats,
        );

        // Build virtual registers from SSA values.
        self.virt_regs.build_initial_vregs(
            func,
            reginfo,
            &mut self.value_live_ranges,
            &mut self.coalescing,
            &mut self.uses,
            &self.hints,
            &self.split_placement,
            &mut self.spill_allocator,
            &mut self.virt_reg_builder,
            &mut self.stats,
        );

        // Allocate virtual registers to physical registers.
        self.allocator.run(
            &mut self.uses,
            &self.hints,
            &mut self.reg_matrix,
            &mut self.virt_regs,
            &mut self.virt_reg_builder,
            &mut self.spill_allocator,
            &mut self.coalescing,
            &mut self.stats,
            func,
            reginfo,
        )?;

        // Allocate spill slots.
        self.spill_allocator.allocate(&mut self.stats)?;

        // Generate move instructions between registers.
        self.move_resolver.generate_moves(
            &self.allocator,
            &self.virt_regs,
            &mut self.spill_allocator,
            &self.uses,
            &mut self.allocations,
            &self.reg_matrix,
            &mut self.stats,
            func,
            reginfo,
            options.move_optimization,
        );

        // Optimize generated moves.
        self.move_optimizer.run(
            &mut self.move_resolver,
            &self.spill_allocator,
            &mut self.coalescing,
            &mut self.allocations,
            &mut self.stats,
            func,
            reginfo,
            options.move_optimization,
        );

        let output = Output {
            regalloc: self,
            func,
            reginfo,
        };
        trace!("Output:\n{output}");
        trace!("{}", self.stats);
        Ok(output)
    }
}

/// Controls how much optimization to perform after register allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MoveOptimizationLevel {
    /// Don't do any optimizations.
    Off,

    /// Optimize moves within each block.
    Local,

    /// Optimize moves within blocks and across forward block edges.
    ///
    /// This is the default since it provides a good balance between
    /// optimization and runtime efficiency.
    #[default]
    Forward,

    /// Optimize moves across all blocks.
    ///
    /// This will find the most optimizations but is relatively slow since it
    /// requires several passes over the CFG.
    Global,
}

/// Configuration options for the register allocator.
#[derive(Debug, Clone, Default)]
pub struct Options {
    /// Controls how moves are optimized after register allocation.
    pub move_optimization: MoveOptimizationLevel,
}

#[cfg(feature = "arbitrary")]
impl<'a> arbitrary::Arbitrary<'a> for MoveOptimizationLevel {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        u.choose(&[Self::Off, Self::Local, Self::Forward, Self::Global])
            .copied()
    }
}

#[cfg(feature = "arbitrary")]
impl<'a> arbitrary::Arbitrary<'a> for Options {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(Self {
            move_optimization: u.arbitrary()?,
        })
    }
}

/// Error returned by the register allocator if allocation is impossible.
///
/// This does not cover errors which are returned by the register info and
/// function validators. If invalid inputs are given then register allocation
/// may panic or not terminate.
#[derive(Debug)]
#[non_exhaustive]
pub enum RegAllocError {
    /// More registers are needed for the operands instruction than there are
    /// available.
    ///
    /// Generally this can only occur due to excessive and/or invalid
    /// constraints on instruction operands, and should be considered a bug in
    /// the client.
    TooManyLiveRegs,

    /// The size of the function exceeded some internal limits in the allocator.
    ///
    /// E.g. number of virtual registers, total number of operands in the
    /// function, etc.
    FunctionTooBig,
}

impl fmt::Display for RegAllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegAllocError::TooManyLiveRegs => {
                write!(f, "too many live registers in a single instruction")
            }
            RegAllocError::FunctionTooBig => {
                write!(f, "function size exceeded implementation limits")
            }
        }
    }
}

/// Statistics collected by the register allocator.
///
/// This is an opaque type since the set of statistics may vary between
/// different versions of the register allocator, even across minor versions.
///
/// The only supported operations on this type are:
/// * Default initialization
/// * Printing with `Debug` or `Display`
#[derive(Debug, Default, Clone)]
pub struct Stats {
    // Stats from input function.
    blocks: usize,
    input_insts: usize,
    operands: usize,
    values: usize,
    value_groups: usize,

    // Stats from value live ranges.
    fixed_def: usize,
    class_def: usize,
    reuse_def: usize,
    reuse_group_def: usize,
    group_def: usize,
    fixed_use: usize,
    class_use: usize,
    group_use: usize,
    nonallocatable_operand: usize,
    blockparam_in: usize,
    blockparam_out: usize,
    local_values: usize,
    global_values: usize,
    value_segments: usize,

    // Stats from coalescing.
    value_sets: usize,
    coalesced_tied: usize,
    coalesced_tied_group: usize,
    coalesced_blockparam: usize,
    coalesced_group: usize,
    coalesced_failed_tied: usize,
    coalesced_failed_tied_group: usize,
    coalesced_failed_blockparam: usize,
    coalesced_failed_group: usize,
    coalesce_fast_path: usize,
    coalesce_slow_path: usize,

    // Stats from virtual register building.
    vreg_conflicts: usize,
    vreg_conflicts_on_same_inst: usize,
    initial_vregs: usize,
    initial_vreg_groups: usize,
    initial_vreg_segments: usize,

    // Stats from register allocation.
    dequeued_reg: usize,
    dequeued_group: usize,
    probe_for_free_reg: usize,
    found_free_reg: usize,
    try_evict_better_candidate: usize,
    evicted_better_candidate: usize,
    must_spill_vreg: usize,
    try_evict: usize,
    assigned_after_evict: usize,
    evicted_vregs: usize,
    evicted_groups: usize,
    spilled_vregs: usize,
    minimal_segments: usize,

    // Stats from spillslot allocation.
    spilled_sets: usize,
    spill_segments: usize,
    spillslots: usize,
    spill_area_size: usize,

    // Stats from move resolver.
    edits: usize,
    moves: usize,
    remats: usize,
    spills: usize,
    reloads: usize,

    // Stats from move optimizer.
    blocks_preprocessed_for_optimizer: usize,
    optimized_stack_use: usize,
    optimized_reload_to_move: usize,
    optimized_redundant_remat: usize,
    optimized_redundant_move: usize,
    optimized_redundant_spill: usize,
    optimized_redundant_reload: usize,
}

impl fmt::Display for Stats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:#?}")
    }
}
