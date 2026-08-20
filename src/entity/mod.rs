//! The public API of the register allocator make heavy use of "entities": these
//! are newtype wrappers around integers which represent an index into an array.
//!
//! These types implement the [`EntityRef`] trait which allows them to be
//! converted to and from `usize`, but these will generally contain a smaller
//! integer type internally to reduce memory usage.
//!
//! This module provides type-safe and efficient data structures for working
//! with entities:
//!
//! - [`PrimaryMap<K, V>`] is used to keep track of a vector of entities,
//!   assigning a unique entity reference to each. It is implemented as a
//!   wrapper around a `Vec<V>` indexed by `K`.
//! - [`SecondaryMap<K, V>`] is used to associate secondary information to an
//!   existing entity. It is also implemented as a wrapper around a `Vec<V>`
//!   indexed by `K`.
//! - [`SparseMap<K, V>`] is an alternative to `SecondaryMap` with different
//!   trade-offs.
//! - [`EntitySet<T>`] represents a set of entities, implemented using a
//!   resizable heap-allocated bitset. [`SmallEntitySet<T>`] is similar but
//!   includes an inline fixed-sized bitset.
//! - [`PackedOption<T>`] provides a compact representation of an `Option<T>`
//!   where `T` is an entity type by encoding `None` using the maximum integer
//!   value for the entity type.
//! - [`CompactList<T>`] allows a slice to be represented in just 8 bytes by
//!   referencing data in a [`CompactListPool<T>`].
//!
//! The design of these types is based on the `cranelift-entity` crate.

#[macro_use]
pub mod base;
pub mod compact_list;
pub mod iter;
pub mod packed_option;
pub mod primary_map;
pub mod secondary_map;
pub mod set;
pub mod small_set;
pub mod sparse;

pub use base::{EntityRange, EntityRef};
pub use compact_list::{CompactList, CompactListPool};
pub use packed_option::{PackedOption, ReservedValue};
pub use primary_map::PrimaryMap;
pub use secondary_map::SecondaryMap;
pub use set::EntitySet;
pub use small_set::SmallEntitySet;
pub use sparse::SparseMap;
