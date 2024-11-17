//! Densely numbered entity references as mapping keys.

use alloc::vec::{self, Vec};
use core::marker::PhantomData;
use core::ops::{Index, IndexMut};
use core::{fmt, slice};

use super::iter::{IntoIter, Iter, IterMut, Keys};
use super::{EntityRange, EntityRef};

/// A mapping `K -> V` for densely indexed entity references.
///
/// The `SecondaryMap` data structure uses the dense index space to implement a map with a vector.
/// Unlike `PrimaryMap`, an `SecondaryMap` can't be used to allocate entity references. It is used
/// to associate secondary information with entities.
///
/// The map does not track if an entry for a key has been inserted or not. Instead, you should
/// resize it in advance with a default value for the maximum entity index that will be used.
#[derive(Clone, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SecondaryMap<K, V>
where
    K: EntityRef,
{
    elems: Vec<V>,
    marker: PhantomData<K>,
}

/// Shared `SecondaryMap` implementation for all value types.
impl<K, V> SecondaryMap<K, V>
where
    K: EntityRef,
{
    /// Create a new empty map.
    ///
    /// The map must be grown with [`SecondaryMap::grow_to`] or [`SecondaryMap::clear_and_resize`]
    /// before any elements can be inserted.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            elems: Vec::new(),
            marker: PhantomData,
        }
    }

    /// Create a new map large enough to hold entity references with an index
    /// below `max_index`.
    ///
    /// All values are initialized with the [`Default`] trait.
    #[inline]
    #[must_use]
    pub fn with_max_index(max_index: usize) -> Self
    where
        V: Default,
    {
        let mut map = Self::new();
        map.grow_to(max_index);
        map
    }

    /// Removes all elements from the map and resizes it to be large enough to
    /// hold entity references with an index below `max_index`.
    ///
    /// All values are initialized with the [`Default`] trait.
    #[inline]
    pub fn clear_and_resize(&mut self, max_index: usize)
    where
        V: Default,
    {
        self.clear_and_resize_with(max_index, || Default::default());
    }

    /// Removes all elements from the map and resizes it to be large enough to
    /// hold entity references with an index below `max_index`.
    ///
    /// All values are initialized by calling `f`.
    #[inline]
    pub fn clear_and_resize_with(&mut self, max_index: usize, f: impl FnMut() -> V) {
        self.elems.clear();
        self.elems.resize_with(max_index, f);
    }

    /// Resizes the map to be large enough to hold entity references with an
    /// index below `max_index`.
    ///
    /// Existing values are not modified. New values are constructed with the
    /// [`Default`] trait.
    #[inline]
    pub fn grow_to(&mut self, max_index: usize)
    where
        V: Default,
    {
        self.grow_to_with(max_index, || Default::default());
    }

    /// Resizes the map to be large enough to hold entity references with an
    /// index below `max_index`.
    ///
    /// Existing values are not modified. New values are constructed by calling
    /// `f``.
    #[inline]
    pub fn grow_to_with(&mut self, max_index: usize, f: impl FnMut() -> V) {
        if self.elems.len() < max_index {
            self.elems.resize_with(max_index, f);
        }
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
}

impl<K, V> Default for SecondaryMap<K, V>
where
    K: EntityRef,
    V: Default,
{
    #[inline]
    fn default() -> SecondaryMap<K, V> {
        SecondaryMap::new()
    }
}

impl<K, V> Index<K> for SecondaryMap<K, V>
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

impl<K, V> IndexMut<K> for SecondaryMap<K, V>
where
    K: EntityRef,
{
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, k: K) -> &mut V {
        &mut self.elems[k.index()]
    }
}

impl<K, V> Index<EntityRange<K>> for SecondaryMap<K, V>
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

impl<K, V> IndexMut<EntityRange<K>> for SecondaryMap<K, V>
where
    K: EntityRef,
{
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, range: EntityRange<K>) -> &mut [V] {
        &mut self.elems[range.from.index()..range.to.index()]
    }
}

impl<K, V> IntoIterator for SecondaryMap<K, V>
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

impl<'a, K, V> IntoIterator for &'a SecondaryMap<K, V>
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

impl<'a, K, V> IntoIterator for &'a mut SecondaryMap<K, V>
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

impl<K, V> fmt::Debug for SecondaryMap<K, V>
where
    K: EntityRef + fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}
