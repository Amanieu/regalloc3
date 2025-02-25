//! Space-efficient `Option` for entity references.

use core::{fmt, mem};

/// Types that have a reserved value which does not represent a valid value.
pub trait ReservedValue {
    /// Create an instance of the reserved value.
    fn reserved_value() -> Self;

    /// Checks whether value is the reserved one.
    fn is_reserved_value(&self) -> bool;
}

/// Packed representation of `Option<T>` for types implementing [`ReservedValue`].
///
/// `PackedOption<T>` instances can be created by converting an `Option<T>`
/// instance using `into`. They can be converted back to an `Option<T>` by using
/// the `expand` method.
///
/// ```
/// use regalloc3::entity::{ReservedValue, PackedOption};
///
/// #[derive(Debug, Copy, Clone, Eq, PartialEq)]
/// struct Entity(u32);
/// impl ReservedValue for Entity {
///     fn reserved_value() -> Self {
///         Self(u32::MAX)
///     }
///     fn is_reserved_value(&self) -> bool {
///         self.0 == u32::MAX
///     }
/// }
///
/// let packed_some: PackedOption<Entity> = Some(Entity(0)).into();
/// assert!(packed_some.is_some());
/// assert_eq!(packed_some.expand(), Some(Entity(0)));
/// let packed_none: PackedOption<Entity> = None.into();
/// assert!(packed_none.is_none());
/// assert_eq!(packed_none.expand(), None);
/// ```
#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(transparent)]
pub struct PackedOption<T: ReservedValue>(T);

impl<T: ReservedValue> PackedOption<T> {
    /// Returns `true` if the packed option is a `None` value.
    #[inline]
    pub fn is_none(&self) -> bool {
        self.0.is_reserved_value()
    }

    /// Returns `true` if the packed option is a `Some` value.
    #[inline]
    pub fn is_some(&self) -> bool {
        !self.0.is_reserved_value()
    }

    /// Expands the packed option into a normal `Option`.
    #[inline]
    pub fn expand(self) -> Option<T> {
        if self.is_none() { None } else { Some(self.0) }
    }

    /// Maps a `PackedOption<T>` to `Option<U>` by applying a function to a
    /// contained value.
    #[inline]
    pub fn map<U, F>(self, f: F) -> Option<U>
    where
        F: FnOnce(T) -> U,
    {
        self.expand().map(f)
    }

    /// Unwraps a packed `Some` value or panics.
    #[track_caller]
    #[inline]
    pub fn unwrap(self) -> T {
        self.expand().unwrap()
    }

    /// Unwraps a packed `Some` value or panics with the given message.
    #[track_caller]
    #[inline]
    pub fn expect(self, msg: &str) -> T {
        self.expand().expect(msg)
    }

    /// Takes the value out of the packed option, leaving a `None` in its place.
    #[inline]
    pub fn take(&mut self) -> Option<T> {
        mem::replace(self, None.into()).expand()
    }
}

impl<T: ReservedValue> Default for PackedOption<T> {
    /// Create a default packed option representing `None`.
    #[inline]
    fn default() -> Self {
        Self(T::reserved_value())
    }
}

impl<T: ReservedValue> From<T> for PackedOption<T> {
    /// Convert `t` into a packed `Some(x)`.
    #[inline]
    fn from(t: T) -> Self {
        debug_assert!(
            !t.is_reserved_value(),
            "Can't make a PackedOption from the reserved value."
        );
        Self(t)
    }
}

impl<T: ReservedValue> From<Option<T>> for PackedOption<T> {
    /// Convert an option into its packed equivalent.
    #[inline]
    fn from(opt: Option<T>) -> Self {
        match opt {
            None => Self::default(),
            Some(t) => t.into(),
        }
    }
}

impl<T: ReservedValue> From<PackedOption<T>> for Option<T> {
    /// Convert an option into its packed equivalent.
    #[inline]
    fn from(opt: PackedOption<T>) -> Self {
        opt.expand()
    }
}

impl<T> fmt::Debug for PackedOption<T>
where
    T: ReservedValue + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_none() {
            write!(f, "None")
        } else {
            write!(f, "Some({:?})", self.0)
        }
    }
}
