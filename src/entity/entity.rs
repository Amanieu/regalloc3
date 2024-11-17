//! Base definitions for entity types.

use core::fmt;

use super::packed_option::ReservedValue;

/// A typed wrapper around an integer index.
///
/// Types implementing this trait can be used as keys for collections like
/// [`PrimaryMap`] or [`SecondaryMap`].
///
/// [`PrimaryMap`]: super::PrimaryMap
/// [`SecondaryMap`]: super::SecondaryMap
pub trait EntityRef: Copy + Eq + ReservedValue {
    /// Creates a new entity reference from a raw index.
    fn new(index: usize) -> Self;

    /// Returns the index that was used to create this entity reference.
    fn index(self) -> usize;
}

/// A sequential range of entities.
///
/// A range is considered empty if the index of `from` is greater than or equal
/// to the index of `to`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EntityRange<T: EntityRef> {
    /// Inclusive lower bound of the range.
    pub from: T,

    /// Exclusive upper bound of the range.
    pub to: T,
}

impl<T: EntityRef + fmt::Display> fmt::Display for EntityRange<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.from, self.to)
    }
}

impl<T: EntityRef> EntityRange<T> {
    /// Creates a new range.
    #[inline]
    pub const fn new(from: T, to: T) -> Self {
        Self { from, to }
    }

    /// Returns the index of the last element in the range.
    ///
    /// Panics in debug builds if the range is empty.
    #[inline]
    pub fn last(self) -> T {
        debug_assert!(!self.is_empty());
        T::new(self.to.index() - 1)
    }

    /// Returns whether the range is empty.
    #[inline]
    pub fn is_empty(self) -> bool {
        self.to.index() <= self.from.index()
    }

    /// Number of elements in the range.
    #[inline]
    pub fn len(self) -> usize {
        self.to.index() - self.from.index()
    }

    /// Iterator over the elements in the range.
    #[inline]
    pub fn iter(self) -> impl DoubleEndedIterator<Item = T> + ExactSizeIterator {
        (self.from.index()..self.to.index()).map(|i| T::new(i))
    }
}

/// Internal helper macro to define a new entity type along with some trait
/// implementations.
macro_rules! entity_def {
    ($($(#[$attr:meta])* $vis:vis entity $name:ident($int:ident);)*) => {
        $(
            $(#[$attr])*
            #[derive(Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
            #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
            $vis struct $name($int);

            // Inherent copies of the EntityRef methods that are const.
            impl $name {
                /// Creates a new entity reference from a raw index.
                #[inline]
                #[allow(dead_code)]
                $vis const fn new(index: usize) -> Self {
                    debug_assert!(index < ($int::MAX as usize));
                    Self(index as $int)
                }

                /// Returns the index that was used to create this entity reference.
                #[inline]
                #[allow(dead_code)]
                $vis const fn index(self) -> usize {
                    self.0 as usize
                }
            }

            impl $crate::entity::EntityRef for $name {
                #[inline]
                fn new(index: usize) -> Self {
                    debug_assert!(index < ($int::MAX as usize));
                    $name(index as $int)
                }

                #[inline]
                fn index(self) -> usize {
                    self.0 as usize
                }
            }

            impl $crate::entity::ReservedValue for $name {
                #[inline]
                fn reserved_value() -> Self {
                    Self($int::MAX)
                }

                #[inline]
                fn is_reserved_value(&self) -> bool {
                    self.0 == $int::MAX
                }
            }
        )*
    };

    // Same as above but also provides Display/Debug impls.
    ($($(#[$attr:meta])* $vis:vis entity $name:ident($int:ident, $display_prefix:expr);)*) => {
        entity_def! {
            $($(#[$attr])* $vis entity $name($int);)*
        }
        $(
            impl core::fmt::Display for $name {
                fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                    write!(f, concat!($display_prefix, "{}"), self.0)
                }
            }

            impl core::fmt::Debug for $name {
                fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                    core::fmt::Display::fmt(self, f)
                }
            }
        )*
    };
}
