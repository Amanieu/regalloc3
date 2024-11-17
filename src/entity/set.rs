//! Densely numbered entity references as set keys.

use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;

use super::EntityRef;

/// Word type used in the bit set.
type Word = usize;

/// A set of entities implemented as a bit vector.
///
/// This is conceptually equivalent to a `HashSet<K>` or a `SecondaryMap<K, bool>`
/// but is encoded much more efficiently as a bit vector.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct EntitySet<T>
where
    T: EntityRef,
{
    storage: Vec<Word>,
    marker: PhantomData<T>,
}

impl<T> EntitySet<T>
where
    T: EntityRef,
{
    /// Creates an empty set.
    ///
    /// The set must be grown with [`EntitySet::grow_to`] or [`EntitySet::clear_and_resize`]
    /// before any elements can be inserted.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            storage: vec![],
            marker: PhantomData,
        }
    }

    /// Create a new set large enough to hold entity references with an index
    /// below `max_index`.
    pub fn with_max_index(max_index: usize) -> Self {
        let mut set = Self::new();
        set.grow_to(max_index);
        set
    }

    /// Internal function to convert an entity into an index and a bit.
    #[inline]
    fn index(entity: T) -> (usize, u32) {
        (
            entity.index() / Word::BITS as usize,
            entity.index() as u32 % Word::BITS,
        )
    }

    /// Returns whether the set contains the given entity.
    #[inline]
    #[track_caller]
    pub fn contains(&self, entity: T) -> bool {
        let (idx, bit) = Self::index(entity);
        self.storage[idx] & (1 << bit) != 0
    }

    /// Inserts an element into the set.
    #[inline]
    #[track_caller]
    pub fn insert(&mut self, entity: T) {
        let (idx, bit) = Self::index(entity);
        self.storage[idx] |= 1 << bit;
    }

    /// Removes an element from the set.
    #[inline]
    #[track_caller]
    pub fn remove(&mut self, entity: T) {
        let (idx, bit) = Self::index(entity);
        self.storage[idx] &= !(1 << bit);
    }

    /// Removes all elements from the set and resizes it to be large enough to
    /// hold entity references with an index below `max_index`.
    #[inline]
    pub fn clear_and_resize(&mut self, max_index: usize) {
        self.storage.clear();
        self.grow_to(max_index);
    }

    /// Resizes the set to be large enough to hold entity references with an
    /// index below `max_index`.
    ///
    /// Any existing elements are not modified.
    #[inline]
    pub fn grow_to(&mut self, max_index: usize) {
        let words = (max_index + Word::BITS as usize - 1) / Word::BITS as usize;
        self.storage.resize(words, 0);
    }

    /// Returns whether the set contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.storage.iter().all(|&word| word == 0)
    }

    /// Returns the number of elements in the set.
    #[inline]
    pub fn count(&self) -> usize {
        self.storage
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    /// Returns an iterator over all the elements in the set, starting from the
    /// lowest index.
    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            current_word: 0,
            next_index: 0,
            set: self,
        }
    }
}

impl<'a, T> IntoIterator for &'a EntitySet<T>
where
    T: EntityRef,
{
    type Item = T;

    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T> Default for EntitySet<T>
where
    T: EntityRef,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Extend<T> for EntitySet<T>
where
    T: EntityRef,
{
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for entity in iter {
            self.insert(entity);
        }
    }
}

impl<T> FromIterator<T> for EntitySet<T>
where
    T: EntityRef,
{
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut set = Self::new();
        set.extend(iter);
        set
    }
}

impl<T> fmt::Debug for EntitySet<T>
where
    T: EntityRef + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

#[cfg(feature = "serde")]
impl<T> serde::Serialize for EntitySet<T>
where
    T: EntityRef + serde::Serialize,
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
impl<'de, T> serde::Deserialize<'de> for EntitySet<T>
where
    T: EntityRef + serde::Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct SeqVisitor<T>
        where
            T: EntityRef,
        {
            marker: PhantomData<EntitySet<T>>,
        }

        impl<'de, T> serde::de::Visitor<'de> for SeqVisitor<T>
        where
            T: EntityRef + serde::Deserialize<'de>,
        {
            type Value = EntitySet<T>;

            fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                formatter.write_str("a sequence")
            }

            #[inline]
            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let mut values = EntitySet::new();

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

/// Iterator over the elements in an [`EntitySet`].
pub struct Iter<'a, T>
where
    T: EntityRef,
{
    current_word: Word,
    next_index: usize,
    set: &'a EntitySet<T>,
}

impl<T> Iterator for Iter<'_, T>
where
    T: EntityRef,
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.current_word == 0 {
            self.current_word = *self.set.storage.get(self.next_index)?;
            self.next_index += 1;
        }

        let low_bit = self.current_word.trailing_zeros();
        self.current_word &= self.current_word - 1;
        let bit = (self.next_index - 1) * Word::BITS as usize + low_bit as usize;
        Some(T::new(bit))
    }
}
