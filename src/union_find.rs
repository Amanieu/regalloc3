//! Union-find algorithm with entities as key types.
//!
//! Implementation based on <https://en.wikipedia.org/wiki/Disjoint-set_data_structure>

use alloc::vec;
use alloc::vec::Vec;

use crate::entity::EntityRef;

/// A node in the union-find data structure.
struct UnionFindEntry<K> {
    /// The parent of this node. Roots are their own parent.
    parent: K,

    /// The rank is the upper bound of the
    rank: u32,
}

/// Fast union-find implementation which uses an [`EntityRef`] as a key type.
///
/// This supports the following operations:
/// - Merging 2 keys into the same equivalence class.
/// - Finding the "leader" of the equivalence class a value is in.
pub struct UnionFind<K: EntityRef> {
    table: Vec<UnionFindEntry<K>>,
}

impl<K: EntityRef> UnionFind<K> {
    pub fn new() -> Self {
        Self { table: vec![] }
    }

    /// Resets the table to its initial state where all keys are in their own
    /// singleton equivalence class.
    pub fn reset(&mut self, num_keys: usize) {
        self.table.clear();
        self.table.extend((0..num_keys).map(|i| UnionFindEntry {
            parent: K::new(i),
            rank: 0,
        }));
    }

    /// Returns the leader of the set containing the given key.
    ///
    /// This takes a mutable reference to self because it performs path
    /// compression for improved performance.
    pub fn find(&mut self, mut k: K) -> K {
        // Instead of the usual recursive path compression algorithm, we use
        // path halving which updates every other node to point to its
        // grandparent.
        //
        // Tarjan, Robert E.; van Leeuwen, Jan (1984). "Worst-case analysis of set union algorithms".
        // https://doi.org/10.1145/62.2160
        while self.table[k.index()].parent != k {
            let parent = self.table[k.index()].parent;
            let grand_parent = self.table[parent.index()].parent;
            self.table[k.index()].parent = grand_parent;
            k = grand_parent;
        }
        k
    }

    /// Merges the two sets containing the given keys, but only if the `unify`
    /// function returns true.
    ///
    /// The first argument to the `unify` function is the leader of one set and
    /// will remain as the leader of the unified set. The second argument is the
    /// leader of the other set that will be merged into the first set.
    ///
    /// Returns whether the sets were successfully merged (or if both keys were
    /// already in the same set).
    pub fn try_union(&mut self, a: K, b: K, unify: impl FnOnce(K, K) -> bool) {
        let a = self.find(a);
        let b = self.find(b);
        if a == b {
            return;
        }

        // Merge into the set with the higher rank.
        let (leader, follower) = if self.table[a.index()].rank >= self.table[b.index()].rank {
            (a, b)
        } else {
            (b, a)
        };

        if unify(leader, follower) {
            // Increment the rank if both sets have the same rank.
            if self.table[leader.index()].rank == self.table[follower.index()].rank {
                self.table[leader.index()].rank += 1;
            }

            // Merge the follower into the leader's tree.
            self.table[follower.index()].parent = leader;
        }
    }
}
