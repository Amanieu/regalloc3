//! Utility functions and types for debugging register allocation.
//!
//! These are not needed for normal compilation, but are useful during
//! development of both the register allocator itself and users of the register
//! allocator.

mod checker;
mod display;
mod dominator_tree;
mod generic_function;
mod generic_reginfo;
mod postorder;
mod validate_func;
mod validate_reginfo;

pub use checker::*;
pub use display::*;
pub use generic_function::*;
pub use generic_reginfo::*;
pub use validate_func::*;
pub use validate_reginfo::*;
