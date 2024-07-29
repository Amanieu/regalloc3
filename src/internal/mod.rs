//! Internal implementation details of the allocator that are not part of the
//! public API.

pub(crate) mod allocations;
pub(crate) mod allocator;
pub(crate) mod coalescing;
pub(crate) mod dominator_tree;
pub(crate) mod live_range;
pub(crate) mod move_resolver;
pub(crate) mod postorder;
pub(crate) mod reg_matrix;
pub(crate) mod spill_allocator;
pub(crate) mod split_placement;
pub(crate) mod uses;
pub(crate) mod value_live_ranges;
pub(crate) mod virt_regs;

// Publicly exposed only for fuzzing.
#[allow(missing_docs)]
pub mod parallel_moves;
