//! Densely numbered entity references as mapping keys.

use alloc::vec::{self, Vec};
use core::marker::PhantomData;
use core::ops::{Index, IndexMut};
use core::{fmt, slice};

use super::iter::{IntoIter, Iter, IterMut, Keys};
use super::{EntityRange, EntityRef};

/// A primary mapping `K -> V` allocating dense entity references.
///
/// The `PrimaryMap` data structure uses the dense index space to implement a map with a vector.
///
/// A primary map contains the main definition of an entity, and it can be used to allocate new
/// entity references with the `push` method.
///
/// There should only be a single `PrimaryMap` instance for a given `EntityRef` type, otherwise
/// conflicting references will be created. Using unknown keys for indexing may cause a panic.
#[derive(Clone, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PrimaryMap<K, V>
where
    K: EntityRef,
{
    elems: Vec<V>,
    marker: PhantomData<K>,
}

impl<K, V> PrimaryMap<K, V>
where
    K: EntityRef,
{
    /// Create a new empty map.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            elems: Vec::new(),
            marker: PhantomData,
        }
    }

    /// Create a new empty map with the given capacity.
    #[inline]
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            elems: Vec::with_capacity(capacity),
            marker: PhantomData,
        }
    }

    /// Returns the number of elements the map can hold without reallocating.
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.elems.capacity()
    }

    /// Check if `k` is a valid key in the map.
    pub fn is_valid(&self, k: K) -> bool {
        k.index() < self.elems.len()
    }

    /// Get the element at `k` if it exists.
    pub fn get(&self, k: K) -> Option<&V> {
        self.elems.get(k.index())
    }

    /// Get the element at `k` if it exists, mutable version.
    pub fn get_mut(&mut self, k: K) -> Option<&mut V> {
        self.elems.get_mut(k.index())
    }

    /// Is this map completely empty?
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.elems.is_empty()
    }

    /// Get the total number of entity references created.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.elems.len()
    }

    /// Iterate over all the keys in this map.
    #[inline]
    #[must_use]
    pub fn keys(&self) -> Keys<K> {
        Keys::with_len(self.elems.len())
    }

    /// Iterate over all the values in this map.
    #[inline]
    pub fn values(&self) -> slice::Iter<'_, V> {
        self.elems.iter()
    }

    /// Iterate over all the values in this map, mutable edition.
    #[inline]
    pub fn values_mut(&mut self) -> slice::IterMut<'_, V> {
        self.elems.iter_mut()
    }

    /// Return an owning iterator over the values of the map.
    #[inline]
    #[must_use]
    pub fn into_values(self) -> vec::IntoIter<V> {
        self.elems.into_iter()
    }

    /// Iterate over all the keys and values in this map.
    #[inline]
    #[must_use]
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter::new(self.elems.iter())
    }

    /// Iterate over all the keys and values in this map, mutable edition.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut::new(self.elems.iter_mut())
    }

    /// Remove all entries from this map.
    #[inline]
    pub fn clear(&mut self) {
        self.elems.clear();
    }

    /// Get the key that will be assigned to the next pushed value.
    #[inline]
    #[must_use]
    pub fn next_key(&self) -> K {
        K::new(self.elems.len())
    }

    /// Append `v` to the mapping, assigning a new key which is returned.
    #[inline]
    pub fn push(&mut self, v: V) -> K {
        let k = self.next_key();
        self.elems.push(v);
        k
    }

    /// Appends multiple values from the given iterator, returning a range of
    /// keys for the newly added elements.
    #[inline]
    pub fn extend(&mut self, iter: impl IntoIterator<Item = V>) -> EntityRange<K> {
        let from = self.next_key();
        self.elems.extend(iter);
        let to = self.next_key();
        EntityRange { from, to }
    }

    /// Returns the last element that was inserted in the map.
    #[inline]
    #[must_use]
    pub fn last(&self) -> Option<(K, &V)> {
        let len = self.elems.len();
        let last = self.elems.last()?;
        Some((K::new(len - 1), last))
    }

    /// Returns the last element that was inserted in the map.
    #[inline]
    pub fn last_mut(&mut self) -> Option<(K, &mut V)> {
        let len = self.elems.len();
        let last = self.elems.last_mut()?;
        Some((K::new(len - 1), last))
    }

    /// Reserves capacity for at least `additional` more elements to be inserted.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.elems.reserve(additional);
    }

    /// Reserves the minimum capacity for exactly `additional` more elements to be inserted.
    #[inline]
    pub fn reserve_exact(&mut self, additional: usize) {
        self.elems.reserve_exact(additional);
    }

    /// Shrinks the capacity of the `PrimaryMap` as much as possible.
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.elems.shrink_to_fit();
    }
}

impl<K, V> Default for PrimaryMap<K, V>
where
    K: EntityRef,
{
    #[inline]
    fn default() -> PrimaryMap<K, V> {
        PrimaryMap::new()
    }
}

impl<K, V> Index<K> for PrimaryMap<K, V>
where
    K: EntityRef,
{
    type Output = V;

    #[inline]
    #[track_caller]
    fn index(&self, k: K) -> &V {
        &self.elems[k.index()]
    }
}

impl<K, V> IndexMut<K> for PrimaryMap<K, V>
where
    K: EntityRef,
{
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, k: K) -> &mut V {
        &mut self.elems[k.index()]
    }
}

impl<K, V> Index<EntityRange<K>> for PrimaryMap<K, V>
where
    K: EntityRef,
{
    type Output = [V];

    #[inline]
    #[track_caller]
    fn index(&self, range: EntityRange<K>) -> &[V] {
        &self.elems[range.from.index()..range.to.index()]
    }
}

impl<K, V> IndexMut<EntityRange<K>> for PrimaryMap<K, V>
where
    K: EntityRef,
{
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, range: EntityRange<K>) -> &mut [V] {
        &mut self.elems[range.from.index()..range.to.index()]
    }
}

impl<K, V> IntoIterator for PrimaryMap<K, V>
where
    K: EntityRef,
{
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self.elems.into_iter())
    }
}

impl<'a, K, V> IntoIterator for &'a PrimaryMap<K, V>
where
    K: EntityRef,
{
    type Item = (K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self.elems.iter())
    }
}

impl<'a, K, V> IntoIterator for &'a mut PrimaryMap<K, V>
where
    K: EntityRef,
{
    type Item = (K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IterMut::new(self.elems.iter_mut())
    }
}

impl<K, V> FromIterator<V> for PrimaryMap<K, V>
where
    K: EntityRef,
{
    #[inline]
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = V>,
    {
        Self {
            elems: Vec::from_iter(iter),
            marker: PhantomData,
        }
    }
}

impl<K, V> From<Vec<V>> for PrimaryMap<K, V>
where
    K: EntityRef,
{
    #[inline]
    fn from(elems: Vec<V>) -> Self {
        Self {
            elems,
            marker: PhantomData,
        }
    }
}

impl<K, V> fmt::Debug for PrimaryMap<K, V>
where
    K: EntityRef + fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}
