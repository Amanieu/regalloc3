use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

/// A small list of values allocated from a pool.
///
/// This is similar to a `Box<[T]>` but with the following differences:
///
/// 1. Memory is allocated from a `CompactListPool<T>` instead of the global heap.
/// 2. The footprint of a list is 8 bytes, compared with 16 bytes for `Box<[T]>`.
/// 3. A list doesn't implement `Drop`, leaving it to the pool to manage memory.
#[derive(Debug)]
pub struct CompactList<T> {
    start: u32,
    end: u32,
    marker: PhantomData<T>,
}

impl<T> Clone for CompactList<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for CompactList<T> {}

impl<T> CompactList<T> {
    /// Creates a new empty list.
    pub const fn new() -> Self {
        Self {
            start: 0,
            end: 0,
            marker: PhantomData,
        }
    }

    /// Creates a new list from an iterator.
    pub fn from_iter<I>(iter: I, pool: &mut CompactListPool<T>) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let start = pool.elems.len() as u32;
        pool.elems.extend(iter);
        let end = pool.elems.len() as u32;
        Self {
            start,
            end,
            marker: PhantomData,
        }
    }

    /// Get the number of elements in the list.
    pub fn len(self) -> usize {
        (self.end - self.start) as usize
    }

    /// Get the list as a slice.
    pub fn as_slice(self, pool: &CompactListPool<T>) -> &[T] {
        &pool.elems[self.start as usize..self.end as usize]
    }

    /// Get the list as a mutable slice.
    pub fn as_mut_slice(self, pool: &mut CompactListPool<T>) -> &mut [T] {
        &mut pool.elems[self.start as usize..self.end as usize]
    }

    /// Extends the given list with the elements from an iterator, inserting
    /// them at the given index of the original list, returning a new
    /// `CompactList`.
    ///
    /// The original list is not modified.
    #[must_use]
    pub fn insert_iter_at<I>(self, index: usize, iter: I, pool: &mut CompactListPool<T>) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Clone,
    {
        let start = pool.elems.len() as u32;
        pool.elems
            .extend_from_within(self.start as usize..self.start as usize + index);
        pool.elems.extend(iter);
        pool.elems
            .extend_from_within(self.start as usize + index..self.end as usize);
        let end = pool.elems.len() as u32;
        Self {
            start,
            end,
            marker: PhantomData,
        }
    }
}

impl<T> Default for CompactList<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// A memory pool for storing lists of `T`.
#[derive(Clone, Debug, Default)]
pub struct CompactListPool<T> {
    elems: Vec<T>,
}

impl<T> CompactListPool<T> {
    /// Create a new list pool.
    pub const fn new() -> Self {
        Self { elems: vec![] }
    }

    /// Clear the pool, forgetting about all lists that use it.
    ///
    /// This invalidates any existing entity lists that used this pool to allocate memory.
    ///
    /// The pool's memory is not released to the operating system, but kept around for faster
    /// allocation in the future.
    pub fn clear(&mut self) {
        self.elems.clear();
    }
}
