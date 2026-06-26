//! Iterator types shared by `PrimaryMap` and `SecondaryMap`.

use alloc::vec;
use core::iter::Enumerate;
use core::marker::PhantomData;
use core::ops::Range;
use core::slice;

use super::EntityRef;

/// Iterate over all keys in order.
pub struct Keys<K: EntityRef> {
    range: Range<usize>,
    marker: PhantomData<K>,
}

impl<K: EntityRef> Keys<K> {
    /// Create a `Keys` iterator that visits `len` entities starting from 0.
    #[inline]
    pub(crate) fn with_len(len: usize) -> Self {
        Self {
            range: 0..len,
            marker: PhantomData,
        }
    }
}

impl<K: EntityRef> Iterator for Keys<K> {
    type Item = K;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.range.next().map(|idx| K::new(idx))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}

impl<K: EntityRef> DoubleEndedIterator for Keys<K> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.range.next_back().map(|idx| K::new(idx))
    }
}

impl<K: EntityRef> ExactSizeIterator for Keys<K> {}

/// Iterator over all the keys and values of an entity map.
pub struct Iter<'a, K: EntityRef, V> {
    enumerate: Enumerate<slice::Iter<'a, V>>,
    marker: PhantomData<K>,
}

impl<'a, K: EntityRef, V> Iter<'a, K, V> {
    /// Create an `Iter` iterator that visits the keys and values of `iter`.
    #[inline]
    pub(crate) fn new(iter: slice::Iter<'a, V>) -> Self {
        Self {
            enumerate: iter.enumerate(),
            marker: PhantomData,
        }
    }
}

impl<'a, K: EntityRef, V> Iterator for Iter<'a, K, V> {
    type Item = (K, &'a V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.enumerate.next().map(|(i, v)| (K::new(i), v))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.enumerate.size_hint()
    }

    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        self.enumerate
            .fold(init, |accum, (idx, val)| f(accum, (K::new(idx), val)))
    }
}

impl<K: EntityRef, V> DoubleEndedIterator for Iter<'_, K, V> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.enumerate.next_back().map(|(i, v)| (K::new(i), v))
    }

    #[inline]
    fn rfold<B, F>(self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        self.enumerate
            .rfold(init, |accum, (idx, val)| f(accum, (K::new(idx), val)))
    }
}

impl<K: EntityRef, V> ExactSizeIterator for Iter<'_, K, V> {}

/// Iterator over all the keys and values of an entity map.
pub struct IterMut<'a, K: EntityRef, V> {
    enumerate: Enumerate<slice::IterMut<'a, V>>,
    marker: PhantomData<K>,
}

impl<'a, K: EntityRef, V> IterMut<'a, K, V> {
    /// Create an `IterMut` iterator that visits the keys and values of `iter`.
    #[inline]
    pub(crate) fn new(iter: slice::IterMut<'a, V>) -> Self {
        Self {
            enumerate: iter.enumerate(),
            marker: PhantomData,
        }
    }
}

impl<'a, K: EntityRef, V> Iterator for IterMut<'a, K, V> {
    type Item = (K, &'a mut V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.enumerate.next().map(|(i, v)| (K::new(i), v))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.enumerate.size_hint()
    }

    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        self.enumerate
            .fold(init, |accum, (i, v)| f(accum, (K::new(i), v)))
    }
}

impl<K: EntityRef, V> DoubleEndedIterator for IterMut<'_, K, V> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.enumerate.next_back().map(|(i, v)| (K::new(i), v))
    }

    #[inline]
    fn rfold<B, F>(self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        self.enumerate
            .rfold(init, |accum, (i, v)| f(accum, (K::new(i), v)))
    }
}

impl<K: EntityRef, V> ExactSizeIterator for IterMut<'_, K, V> {}

/// Iterator over all the keys and values of an entity map.
pub struct IntoIter<K: EntityRef, V> {
    enumerate: Enumerate<vec::IntoIter<V>>,
    marker: PhantomData<K>,
}

impl<K: EntityRef, V> IntoIter<K, V> {
    /// Create an `IntoIter` iterator that visits the keys and values of `iter`.
    #[inline]
    pub(crate) fn new(iter: vec::IntoIter<V>) -> Self {
        Self {
            enumerate: iter.enumerate(),
            marker: PhantomData,
        }
    }
}

impl<K: EntityRef, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.enumerate.next().map(|(i, v)| (K::new(i), v))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.enumerate.size_hint()
    }

    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        self.enumerate
            .fold(init, |accum, (i, v)| f(accum, (K::new(i), v)))
    }
}

impl<K: EntityRef, V> DoubleEndedIterator for IntoIter<K, V> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.enumerate.next_back().map(|(i, v)| (K::new(i), v))
    }

    #[inline]
    fn rfold<B, F>(self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        self.enumerate
            .rfold(init, |accum, (i, v)| f(accum, (K::new(i), v)))
    }
}

impl<K: EntityRef, V> ExactSizeIterator for IntoIter<K, V> {}
