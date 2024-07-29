//! Helper types for working with entities from `cranelift-entity`.

use core::marker::PhantomData;
use core::{array, fmt, ops};

use cranelift_entity::EntityRef;

/// Variant of cranelift-entity's `entity_impl` which supports non-u32 types.
macro_rules! entity_impl {
    // Basic traits.
    ($entity:ident($int:ident)) => {
        impl cranelift_entity::EntityRef for $entity {
            #[inline]
            fn new(index: usize) -> Self {
                debug_assert!(index < ($int::MAX as usize));
                $entity(index as $int)
            }

            #[inline]
            fn index(self) -> usize {
                self.0 as usize
            }
        }

        impl cranelift_entity::packed_option::ReservedValue for $entity {
            #[inline]
            fn reserved_value() -> $entity {
                $entity($int::MAX)
            }

            #[inline]
            fn is_reserved_value(&self) -> bool {
                self.0 == $int::MAX
            }
        }
    };

    // Include basic `Display` impl using the given display prefix.
    // Display a `Block` reference as "block12".
    ($entity:ident($int:ident), $display_prefix:expr) => {
        entity_impl!($entity($int));

        impl core::fmt::Display for $entity {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(f, concat!($display_prefix, "{}"), self.0)
            }
        }

        impl core::fmt::Debug for $entity {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                core::fmt::Display::fmt(self, f)
            }
        }
    };
}

/// A sequential range of entities.
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
    pub fn new(from: T, to: T) -> Self {
        debug_assert!(from.index() <= to.index());
        Self { from, to }
    }

    /// Returns the index of the last element in the range.
    ///
    /// Panics if the range is empty.
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

/// Abstraction over an integer that can be used as a word in a [`SmallEntitySet`].
#[allow(missing_docs)]
pub trait SmallSetWord:
    Copy
    + ops::BitAnd<Output = Self>
    + ops::BitAndAssign
    + ops::BitOr<Output = Self>
    + ops::BitOrAssign
    + ops::Not<Output = Self>
    + Eq
{
    const ZERO: Self;
    const BITS: usize;
    fn bit(index: usize) -> Self;
    fn count_ones(self) -> usize;
    fn trailing_zeros(self) -> usize;
}

impl SmallSetWord for u64 {
    const ZERO: Self = 0;

    const BITS: usize = 64;

    #[inline]
    fn bit(index: usize) -> Self {
        1 << index
    }

    #[inline]
    fn count_ones(self) -> usize {
        self.count_ones() as usize
    }

    #[inline]
    fn trailing_zeros(self) -> usize {
        self.trailing_zeros() as usize
    }
}

impl SmallSetWord for u32 {
    const ZERO: Self = 0;

    const BITS: usize = 32;

    #[inline]
    fn bit(index: usize) -> Self {
        1 << index
    }

    #[inline]
    fn count_ones(self) -> usize {
        self.count_ones() as usize
    }

    #[inline]
    fn trailing_zeros(self) -> usize {
        self.trailing_zeros() as usize
    }
}

/// A set of entities implemented as a fixed-size bit set.
///
/// This should be sized according to the implementation limits for the
/// particular entity type.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SmallEntitySet<T, W, const N: usize> {
    storage: [W; N],
    marker: PhantomData<T>,
}

impl<T, W, const N: usize> SmallEntitySet<T, W, N>
where
    T: EntityRef,
    W: SmallSetWord,
{
    /// Creates an empty set.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            storage: [W::ZERO; N],
            marker: PhantomData,
        }
    }

    /// Internal function to convert an entity into an index.
    #[inline]
    fn index(entity: T) -> usize {
        // Only do bounds-checking in debug builds.
        debug_assert!(entity.index() < N * W::BITS);
        entity.index() % (N * W::BITS)
    }

    /// Returns whether the set contains the given entity.
    #[inline]
    pub fn contains(&self, entity: T) -> bool {
        let idx = Self::index(entity);
        self.storage[idx / W::BITS] & W::bit(idx % W::BITS) != W::ZERO
    }

    /// Inserts an element into the set.
    #[inline]
    pub fn insert(&mut self, entity: T) {
        let idx = Self::index(entity);
        self.storage[idx / W::BITS] |= W::bit(idx % W::BITS);
    }

    /// Removes an element from the set.
    #[inline]
    pub fn remove(&mut self, entity: T) {
        let idx = Self::index(entity);
        self.storage[idx / W::BITS] &= !W::bit(idx % W::BITS);
    }

    /// Removes all elements from the set.
    #[inline]
    pub fn clear(&mut self) {
        self.storage.iter_mut().for_each(|word| *word = W::ZERO);
    }

    /// Returns whether the set contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.storage.iter().all(|&word| word == W::ZERO)
    }

    /// Returns the number of elements in the set.
    #[inline]
    pub fn len(&self) -> usize {
        self.storage.iter().map(|word| word.count_ones()).sum()
    }

    /// Returns an iterator over all the elements in the set, starting from the
    /// lowest index.
    #[inline]
    pub fn iter(&self) -> SmallEntitySetIter<'_, T, W, N> {
        SmallEntitySetIter {
            current_word: W::ZERO,
            next_index: 0,
            set: self,
        }
    }
}

impl<'a, T, W, const N: usize> IntoIterator for &'a SmallEntitySet<T, W, N>
where
    T: EntityRef,
    W: SmallSetWord,
{
    type Item = T;

    type IntoIter = SmallEntitySetIter<'a, T, W, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T, W, const N: usize> Default for SmallEntitySet<T, W, N>
where
    T: EntityRef,
    W: SmallSetWord,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, W, const N: usize> ops::BitAnd for SmallEntitySet<T, W, N>
where
    T: EntityRef,
    W: SmallSetWord,
{
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        Self {
            storage: array::from_fn(|idx| self.storage[idx] & rhs.storage[idx]),
            marker: PhantomData,
        }
    }
}

impl<T, W, const N: usize> ops::BitOr for SmallEntitySet<T, W, N>
where
    T: EntityRef,
    W: SmallSetWord,
{
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        Self {
            storage: array::from_fn(|idx| self.storage[idx] | rhs.storage[idx]),
            marker: PhantomData,
        }
    }
}

impl<T, W, const N: usize> ops::Not for SmallEntitySet<T, W, N>
where
    T: EntityRef,
    W: SmallSetWord,
{
    type Output = Self;

    #[inline]
    fn not(self) -> Self::Output {
        Self {
            storage: array::from_fn(|idx| !self.storage[idx]),
            marker: PhantomData,
        }
    }
}

impl<T, W, const N: usize> ops::BitAndAssign for SmallEntitySet<T, W, N>
where
    T: EntityRef,
    W: SmallSetWord,
{
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.storage
            .iter_mut()
            .zip(&rhs.storage)
            .for_each(|(word, other)| *word &= *other);
    }
}

impl<T, W, const N: usize> ops::BitOrAssign for SmallEntitySet<T, W, N>
where
    T: EntityRef,
    W: SmallSetWord,
{
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.storage
            .iter_mut()
            .zip(&rhs.storage)
            .for_each(|(word, other)| *word |= *other);
    }
}

impl<T, W, const N: usize> Extend<T> for SmallEntitySet<T, W, N>
where
    T: EntityRef,
    W: SmallSetWord,
{
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for entity in iter {
            self.insert(entity);
        }
    }
}

impl<T, W, const N: usize> FromIterator<T> for SmallEntitySet<T, W, N>
where
    T: EntityRef,
    W: SmallSetWord,
{
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut set = Self::new();
        set.extend(iter);
        set
    }
}

impl<T, W, const N: usize> fmt::Debug for SmallEntitySet<T, W, N>
where
    T: EntityRef + fmt::Debug,
    W: SmallSetWord,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl<T, W, const N: usize> fmt::Display for SmallEntitySet<T, W, N>
where
    T: EntityRef + fmt::Debug,
    W: SmallSetWord,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

#[cfg(feature = "serde")]
impl<T, W, const N: usize> serde::Serialize for SmallEntitySet<T, W, N>
where
    T: EntityRef + serde::Serialize,
    W: SmallSetWord,
{
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.collect_seq(self.iter())
    }
}

#[cfg(feature = "serde")]
impl<'de, T, W, const N: usize> serde::Deserialize<'de> for SmallEntitySet<T, W, N>
where
    T: EntityRef + serde::Deserialize<'de>,
    W: SmallSetWord,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct SeqVisitor<T, W, const N: usize> {
            marker: PhantomData<SmallEntitySet<T, W, N>>,
        }

        impl<'de, T, W, const N: usize> serde::de::Visitor<'de> for SeqVisitor<T, W, N>
        where
            T: EntityRef + serde::Deserialize<'de>,
            W: SmallSetWord,
        {
            type Value = SmallEntitySet<T, W, N>;

            fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                formatter.write_str("a sequence")
            }

            #[inline]
            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let mut values = SmallEntitySet::new();

                while let Some(value) = seq.next_element()? {
                    values.insert(value);
                }

                Ok(values)
            }
        }

        let visitor = SeqVisitor {
            marker: PhantomData,
        };
        deserializer.deserialize_seq(visitor)
    }
}

/// Iterator over the elements in a [`SmallEntitySet`].
pub struct SmallEntitySetIter<'a, T, W, const N: usize> {
    current_word: W,
    next_index: usize,
    set: &'a SmallEntitySet<T, W, N>,
}

impl<T, W, const N: usize> Iterator for SmallEntitySetIter<'_, T, W, N>
where
    T: EntityRef,
    W: SmallSetWord,
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.current_word == W::ZERO {
            self.current_word = *self.set.storage.get(self.next_index)?;
            self.next_index += 1;
        }

        let low_bit = self.current_word.trailing_zeros();
        self.current_word &= !W::bit(low_bit % W::BITS);
        let bit = (self.next_index - 1) * W::BITS + low_bit;
        Some(T::new(bit))
    }
}
