//! Sparse mapping of entity references to larger value types.
//!
//! This module provides a `SparseMap` data structure which implements a sparse mapping from an
//! `EntityRef` key to a value type that may be on the larger side. This implementation is based on
//! the paper:
//!
//! > Briggs, Torczon, *An efficient representation for sparse sets*,
//! > ACM Letters on Programming Languages and Systems, Volume 2, Issue 1-4, March-Dec. 1993.

use alloc::vec;
use alloc::vec::Vec;
use core::{fmt, mem, slice};

use crate::entity::{EntityRef, SecondaryMap};

/// A sparse mapping of entity references, logically equivalent to `IndexMap<K, V>`.
///
/// A `SparseMap<K, V>` map provides:
///
/// - Memory usage equivalent to `SecondaryMap<K, u32>` + `Vec<(K, V)>`, so much
///   smaller than `SecondaryMap<K, V>` for sparse mappings of larger `V` types.
/// - Constant time lookup, slightly slower than `SecondaryMap`.
/// - A very fast, constant time `clear()` operation.
/// - Fast insert and erase operations.
/// - Stable insertion order that is as fast as a `Vec<(K, V)>`.
/// - Tracking of unmapped keys without needing a default value.
///
/// # Compared to `SecondaryMap`
///
/// When should we use a `SparseMap` instead of a secondary `SecondaryMap`? First of all,
/// `SparseMap` does not provide the functionality of a `PrimaryMap` which can allocate and assign
/// entity references to objects as they are pushed onto the map. It is only the secondary entity
/// maps that can be replaced with a `SparseMap`.
///
/// - A secondary entity map assigns a default mapping to all keys. It doesn't distinguish between
///   an unmapped key and one that maps to the default value. `SparseMap` does not require
///   `Default` values, and it tracks accurately if a key has been mapped or not.
/// - Iterating over the contents of an `SecondaryMap` is linear in the size of the *key space*,
///   while iterating over a `SparseMap` is linear in the number of elements in the mapping. This
///   is an advantage precisely when the mapping is sparse.
/// - `SparseMap::clear()` is constant time and super-fast. `SecondaryMap::clear_and_resize()` is
///   linear inthe size of the key space.
#[derive(Clone)]
pub struct SparseMap<K, V>
where
    K: EntityRef,
{
    sparse: SecondaryMap<K, u32>,
    dense: Vec<(K, V)>,
}

impl<K, V> SparseMap<K, V>
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
            sparse: SecondaryMap::new(),
            dense: vec![],
        }
    }

    /// Create a new map large enough to hold entity references with an index
    /// below `max_index`.
    #[inline]
    #[must_use]
    pub fn with_max_index(max_index: usize) -> Self {
        let mut map = Self::new();
        map.grow_to(max_index);
        map
    }

    /// Resizes the map to be large enough to hold entity references with an
    /// index below `max_index`.
    #[inline]
    pub fn grow_to(&mut self, max_index: usize) {
        self.sparse.grow_to(max_index);
    }

    /// Returns the number of elements in the map.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.dense.len()
    }

    /// Returns true if the map contains no elements.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.dense.is_empty()
    }

    /// Remove all elements from the map.
    #[inline]
    pub fn clear(&mut self) {
        self.dense.clear();
    }

    /// Returns a reference to the value corresponding to the key.
    #[inline]
    #[track_caller]
    pub fn get(&self, key: K) -> Option<&V> {
        let entry = self.dense.get(self.sparse[key] as usize)?;
        if entry.0 == key {
            return Some(&entry.1);
        }
        None
    }

    /// Returns a mutable reference to the value corresponding to the key.
    ///
    /// Note that the returned value must not be mutated in a way that would change its key. This
    /// would invalidate the sparse set data structure.
    #[inline]
    #[track_caller]
    pub fn get_mut(&mut self, key: K) -> Option<&mut V> {
        let entry = self.dense.get_mut(self.sparse[key] as usize)?;
        if entry.0 == key {
            return Some(&mut entry.1);
        }
        None
    }

    /// Return the index into `dense` of the value corresponding to `key`.
    #[inline]
    #[track_caller]
    pub fn get_index_of(&self, key: K) -> Option<usize> {
        let index = self.sparse[key] as usize;
        let entry = self.dense.get(index)?;
        if entry.0 == key {
            return Some(index);
        }
        None
    }

    /// Return `true` if the map contains a value corresponding to `key`.
    #[inline]
    #[track_caller]
    pub fn contains_key(&self, key: K) -> bool {
        self.get(key).is_some()
    }

    /// Get the given key’s corresponding entry in the map for insertion and/or
    /// in-place manipulation.
    #[inline]
    #[track_caller]
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        let index = self.sparse[key] as usize;
        let entry = self.dense.get(index);
        if entry.is_some_and(|entry| entry.0 == key) {
            Entry::Occupied(OccupiedEntry { map: self, index })
        } else {
            Entry::Vacant(VacantEntry { map: self, key })
        }
    }

    /// Insert a value into the map.
    ///
    /// If the map did not have this key present, `None` is returned.
    ///
    /// If the map did have this key present, the value is updated, and the old value is returned.
    #[inline]
    #[track_caller]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if let Some(entry) = self.get_mut(key) {
            Some(mem::replace(entry, value))
        } else {
            self.insert_unique(key, value);
            None
        }
    }

    /// Insert a value into the map without checking for an existing element.
    ///
    /// This function panics in debug builds if the map already contained `key`.
    ///
    /// A reference to the inserted value is returned.
    #[inline]
    #[track_caller]
    pub fn insert_unique(&mut self, key: K, value: V) -> &mut V {
        debug_assert!(!self.contains_key(key));
        let idx = self.dense.len();
        debug_assert!(idx <= u32::MAX as usize, "SparseMap overflow");
        self.dense.push((key, value));
        self.sparse[key] = idx as u32;
        &mut self.dense.last_mut().unwrap().1
    }

    /// Remove a value from the map and return it.
    ///
    /// Like [`Vec::swap_remove`], the pair is removed by swapping it with
    /// the last element of the map and popping it off.
    #[inline]
    #[track_caller]
    pub fn remove(&mut self, key: K) -> Option<V> {
        match self.entry(key) {
            Entry::Occupied(entry) => Some(entry.remove()),
            Entry::Vacant(_) => None,
        }
    }

    /// Remove the last value from the map.
    #[inline]
    pub fn pop(&mut self) -> Option<(K, V)> {
        self.dense.pop()
    }

    /// Get the values as a slice.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[(K, V)] {
        self.dense.as_slice()
    }

    /// Get the values as a mutable vector.
    ///
    /// If any keys are modified, you must call [`SparseMap::rebuild_mapping`]
    /// afterwards.
    #[inline]
    pub fn as_mut_vec(&mut self) -> &mut Vec<(K, V)> {
        &mut self.dense
    }

    /// Re-builds the map of keys to values.
    ///
    /// This must be called after any keys are modified, such as with
    /// [`SparseMap::as_mut_vec`].
    #[inline]
    pub fn rebuild_mapping(&mut self) {
        for (idx, &(k, _)) in self.dense.iter().enumerate() {
            self.sparse[k] = idx as u32;
        }
    }

    /// Iterate over all the keys and values in this map.
    #[inline]
    pub fn iter(&self) -> slice::Iter<'_, (K, V)> {
        self.dense.iter()
    }

    /// Iterate over all the keys and values in this map, mutable edition.
    #[inline]
    pub fn iter_mut(&mut self) -> slice::IterMut<'_, (K, V)> {
        self.dense.iter_mut()
    }

    /// Iterate over all the keys in this map.
    #[inline]
    #[must_use]
    pub fn keys(&self) -> impl DoubleEndedIterator<Item = K> + ExactSizeIterator + '_ {
        self.dense.iter().map(|(k, _v)| *k)
    }

    /// Iterate over all the values in this map.
    #[inline]
    #[must_use]
    pub fn values(&self) -> impl DoubleEndedIterator<Item = &V> + ExactSizeIterator + '_ {
        self.dense.iter().map(|(_k, v)| v)
    }

    /// Iterate over all the values in this map, mutable edition.
    #[inline]
    pub fn values_mut(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = &mut V> + ExactSizeIterator + '_ {
        self.dense.iter_mut().map(|(_k, v)| v)
    }

    /// Return an owning iterator over the values of the map.
    #[inline]
    #[must_use]
    pub fn into_values(self) -> impl DoubleEndedIterator<Item = V> + ExactSizeIterator {
        self.dense.into_iter().map(|(_k, v)| v)
    }

    /// Scan through each key-value pair in the map and keep those where the
    /// closure `keep` returns `true`.
    ///
    /// The elements are visited in order, and remaining elements keep their
    /// order.
    #[inline]
    pub fn retain<F>(&mut self, mut keep: F)
    where
        F: FnMut(K, &mut V) -> bool,
    {
        let prev_len = self.dense.len();
        self.dense.retain_mut(|&mut (k, ref mut v)| keep(k, v));
        if prev_len != self.dense.len() {
            self.rebuild_mapping();
        }
    }
}

impl<K, V> Default for SparseMap<K, V>
where
    K: EntityRef,
{
    #[inline]
    fn default() -> Self {
        Self {
            sparse: Default::default(),
            dense: Default::default(),
        }
    }
}

impl<K, V> IntoIterator for SparseMap<K, V>
where
    K: EntityRef,
{
    type Item = (K, V);
    type IntoIter = vec::IntoIter<(K, V)>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.dense.into_iter()
    }
}

impl<'a, K, V> IntoIterator for &'a SparseMap<K, V>
where
    K: EntityRef,
{
    type Item = &'a (K, V);
    type IntoIter = slice::Iter<'a, (K, V)>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.dense.iter()
    }
}

impl<'a, K, V> IntoIterator for &'a mut SparseMap<K, V>
where
    K: EntityRef,
{
    type Item = &'a mut (K, V);
    type IntoIter = slice::IterMut<'a, (K, V)>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.dense.iter_mut()
    }
}

impl<K, V> fmt::Debug for SparseMap<K, V>
where
    K: EntityRef + fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(self.iter().map(|(k, v)| (k, v)))
            .finish()
    }
}

/// Entry for an existing key-value pair in an [`SparseMap`]
/// or a vacant location to insert one.
pub enum Entry<'a, K, V>
where
    K: EntityRef,
{
    /// Existing slot with equivalent key.
    Occupied(OccupiedEntry<'a, K, V>),
    /// Vacant slot (no equivalent key in the map).
    Vacant(VacantEntry<'a, K, V>),
}

impl<'a, K, V> Entry<'a, K, V>
where
    K: EntityRef,
{
    /// Inserts the given default value in the entry if it is vacant and returns a mutable
    /// reference to it. Otherwise a mutable reference to an already existent value is returned.
    #[inline]
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(default),
        }
    }

    /// Inserts the result of the `call` function in the entry if it is vacant and returns a mutable
    /// reference to it. Otherwise a mutable reference to an already existent value is returned.
    #[inline]
    pub fn or_insert_with<F>(self, call: F) -> &'a mut V
    where
        F: FnOnce() -> V,
    {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(call()),
        }
    }

    /// Inserts the result of the `call` function with a reference to the entry's key if it is
    /// vacant, and returns a mutable reference to the new value. Otherwise a mutable reference to
    /// an already existent value is returned.
    #[inline]
    pub fn or_insert_with_key<F>(self, call: F) -> &'a mut V
    where
        F: FnOnce(&K) -> V,
    {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let value = call(&entry.key);
                entry.insert(value)
            }
        }
    }

    /// Gets the entry's key.
    #[inline]
    pub fn key(&self) -> K {
        match *self {
            Entry::Occupied(ref entry) => entry.key(),
            Entry::Vacant(ref entry) => entry.key(),
        }
    }

    /// Modifies the entry if it is occupied.
    #[inline]
    pub fn and_modify<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut V),
    {
        if let Entry::Occupied(entry) = &mut self {
            f(entry.get_mut());
        }
        self
    }

    /// Inserts a default-constructed value in the entry if it is vacant and returns a mutable
    /// reference to it. Otherwise a mutable reference to an already existent value is returned.
    #[inline]
    pub fn or_default(self) -> &'a mut V
    where
        V: Default,
    {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(V::default()),
        }
    }
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for Entry<'_, K, V>
where
    K: EntityRef,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut tuple = f.debug_tuple("Entry");
        match self {
            Entry::Vacant(v) => tuple.field(v),
            Entry::Occupied(o) => tuple.field(o),
        };
        tuple.finish()
    }
}

/// A view into an occupied entry in an [`SparseMap`].
/// It is part of the [`Entry`] enum.
pub struct OccupiedEntry<'a, K, V>
where
    K: EntityRef,
{
    map: &'a mut SparseMap<K, V>,
    index: usize,
}

impl<'a, K, V> OccupiedEntry<'a, K, V>
where
    K: EntityRef,
{
    /// Return the index of the key-value pair
    #[inline]
    #[must_use]
    pub fn index(&self) -> usize {
        self.index
    }

    /// Gets the entry's key in the map.
    #[inline]
    #[must_use]
    pub fn key(&self) -> K {
        self.map.dense[self.index].0
    }

    /// Gets a reference to the entry's value in the map.
    #[inline]
    #[must_use]
    pub fn get(&self) -> &V {
        &self.map.dense[self.index].1
    }

    /// Gets a mutable reference to the entry's value in the map.
    ///
    /// If you need a reference which may outlive the destruction of the
    /// [`Entry`] value, see [`into_mut`][Self::into_mut].
    #[inline]
    pub fn get_mut(&mut self) -> &mut V {
        &mut self.map.dense[self.index].1
    }

    /// Converts into a mutable reference to the entry's value in the map,
    /// with a lifetime bound to the map itself.
    #[inline]
    #[must_use]
    pub fn into_mut(self) -> &'a mut V {
        &mut self.map.dense[self.index].1
    }

    /// Sets the value of the entry to `value`, and returns the entry's old value.
    #[inline]
    pub fn insert(&mut self, value: V) -> V {
        mem::replace(self.get_mut(), value)
    }

    /// Remove the key, value pair stored in the map for this entry, and return the value.
    ///
    /// Like [`Vec::swap_remove`], the pair is removed by swapping it with
    /// the last element of the map and popping it off.
    #[inline]
    #[allow(clippy::must_use_candidate)] // False positive
    pub fn remove(self) -> V {
        let back = self.map.dense.pop().unwrap();

        // Are we popping the back of `dense`?
        if self.index == self.map.dense.len() {
            return back.1;
        }

        // We're removing an element from the middle of `dense`.
        // Replace the element at `idx` with the back of `dense`.
        // Repair `sparse` first.
        self.map.sparse[back.0] = self.index as u32;
        mem::replace(&mut self.map.dense[self.index], back).1
    }
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for OccupiedEntry<'_, K, V>
where
    K: EntityRef,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OccupiedEntry")
            .field("key", &self.key())
            .field("value", self.get())
            .finish()
    }
}

/// A view into a vacant entry in a [`SparseMap`].
/// It is part of the [`Entry`] enum.
pub struct VacantEntry<'a, K, V>
where
    K: EntityRef,
{
    map: &'a mut SparseMap<K, V>,
    key: K,
}

impl<'a, K, V> VacantEntry<'a, K, V>
where
    K: EntityRef,
{
    /// Gets the key that was used to find the entry.
    #[inline]
    pub fn key(&self) -> K {
        self.key
    }

    /// Inserts the entry's key and the given value into the map, and returns a mutable reference
    /// to the value.
    #[inline]
    pub fn insert(self, value: V) -> &'a mut V {
        self.map.insert_unique(self.key, value)
    }
}

impl<K: fmt::Debug, V> fmt::Debug for VacantEntry<'_, K, V>
where
    K: EntityRef,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("VacantEntry").field(&self.key()).finish()
    }
}
