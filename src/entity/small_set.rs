//! Densely numbered entity references as set keys.

use core::marker::PhantomData;
use core::{array, fmt, ops};

use super::EntityRef;

/// Abstraction over an integer that can be used as a word in a [`SmallEntitySet`].
#[allow(missing_docs)]
pub trait SmallSetWord:
    Copy
    + ops::BitAnd<Output = Self>
    + ops::BitAndAssign
    + ops::BitOr<Output = Self>
    + ops::BitOrAssign
    + ops::Sub<Output = Self>
    + ops::SubAssign
    + ops::Not<Output = Self>
    + Eq
{
    const ZERO: Self;
    const ONE: Self;
    const BITS: usize;
    fn bit(index: usize) -> Self;
    fn count_ones(self) -> usize;
    fn trailing_zeros(self) -> usize;
}

impl SmallSetWord for u64 {
    const ZERO: Self = 0;
    const ONE: Self = 1;

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
    const ONE: Self = 1;

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
pub struct SmallEntitySet<T, W, const N: usize>
where
    T: EntityRef,
    W: SmallSetWord,
{
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
        if let Some(&word) = self.storage.get(idx / W::BITS) {
            word & W::bit(idx % W::BITS) != W::ZERO
        } else {
            false
        }
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
        self.storage.fill(W::ZERO);
    }

    /// Returns whether the set contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.storage.iter().all(|&word| word == W::ZERO)
    }

    /// Returns the number of elements in the set.
    #[inline]
    pub fn count(&self) -> usize {
        self.storage.iter().map(|word| word.count_ones()).sum()
    }

    /// Returns an iterator over all the elements in the set, starting from the
    /// lowest index.
    #[inline]
    pub fn iter(&self) -> Iter<'_, T, W, N> {
        Iter {
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

    type IntoIter = Iter<'a, T, W, N>;

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
        struct SeqVisitor<T, W, const N: usize>
        where
            T: EntityRef,
            W: SmallSetWord,
        {
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
pub struct Iter<'a, T, W, const N: usize>
where
    T: EntityRef,
    W: SmallSetWord,
{
    current_word: W,
    next_index: usize,
    set: &'a SmallEntitySet<T, W, N>,
}

impl<T, W, const N: usize> Iterator for Iter<'_, T, W, N>
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
        self.current_word &= self.current_word - W::ONE;
        let bit = (self.next_index - 1) * W::BITS + low_bit;
        Some(T::new(bit))
    }
}
