//! Dominator tree.
//!
//! The code in this file is based on the dominator tree implementation from
//! Cranelift.

use alloc::vec;
use alloc::vec::Vec;
use core::cmp::{self, Ordering};
use core::mem;

use super::postorder::PostOrder;
use crate::entity::packed_option::PackedOption;
use crate::entity::{EntitySet, SecondaryMap};
use crate::function::{Block, Function};

#[derive(Default, Clone)]
struct DominatorTreeNode {
    /// Parent node in the dominator tree.
    parent: PackedOption<Block>,

    /// First child node in the dominator tree.
    child: PackedOption<Block>,

    /// Next sibling node in the dominator tree.
    sibling: PackedOption<Block>,

    /// Sequence number in a pre-order traversal of the dominator tree.
    pre_number: u32,

    /// Maximum sequence number of this node and all its children.
    pre_max: u32,
}

/// Dominator tree of the control flow graph.
pub struct DominatorTree {
    nodes: SecondaryMap<Block, DominatorTreeNode>,

    /// First child node of the internal virtual root.
    virtual_root_child: PackedOption<Block>,

    /// Blocks whose immediate dominator has been computed.
    processed: EntitySet<Block>,

    /// Stack for DFS traversal.
    stack: Vec<Block>,
}

impl DominatorTree {
    /// Creates a new `DominatorTree`.
    pub fn new() -> Self {
        Self {
            nodes: SecondaryMap::new(),
            virtual_root_child: None.into(),
            processed: EntitySet::new(),
            stack: vec![],
        }
    }

    /// Computes the dominator tree.
    pub fn compute(&mut self, func: &impl Function, po: &PostOrder) {
        // Reset all immediate dominators to None.
        self.nodes.clear_and_resize(func.num_blocks());

        // 1. Compute immediate dominators for each basic block.
        self.compute_idoms(func, po);

        // 2. Populate child and sibling links for each node, including
        // children of the internal virtual root.
        self.virtual_root_child = None.into();
        for block in po.cfg_postorder() {
            let sibling = if let Some(idom) = self.nodes[block].parent.expand() {
                mem::replace(&mut self.nodes[idom].child, Some(block).into())
            } else {
                mem::replace(&mut self.virtual_root_child, Some(block).into())
            };
            self.nodes[block].sibling = sibling;
        }

        // 3. Assign pre-order numbers from a DFS of the dominator tree.
        self.stack.clear();
        if let Some(child) = self.virtual_root_child.expand() {
            self.stack.push(child);
        }
        let mut n = 0;
        while let Some(block) = self.stack.pop() {
            n += 1;
            let node = &mut self.nodes[block];
            node.pre_number = n;
            node.pre_max = n;
            if let Some(n) = node.sibling.expand() {
                self.stack.push(n);
            }
            if let Some(n) = node.child.expand() {
                self.stack.push(n);
            }
        }

        // 4. Propagate the `pre_max` numbers up the tree.
        // The CFG post-order is topologically ordered w.r.t. dominance so a node comes after all
        // its dominator tree children.
        for block in po.cfg_postorder() {
            if let Some(idom) = self.nodes[block].parent.expand() {
                let pre_max = cmp::max(self.nodes[block].pre_max, self.nodes[idom].pre_max);
                self.nodes[idom].pre_max = pre_max;
            }
        }
    }

    /// Returns the immediate dominator of the given basic block.
    ///
    /// Returns `None` for entry points, unreachable blocks, and blocks whose
    /// only immediate dominator is the virtual root of a multi-entry CFG.
    pub fn immediate_dominator(&self, block: Block) -> Option<Block> {
        self.nodes[block].parent.into()
    }

    /// Determines whether `a` dominates `b`.
    ///
    /// Both blocks must be reachable from an entry point.
    ///
    /// Returns true if `a == b`.
    pub fn dominates(&self, a: Block, b: Block) -> bool {
        let na = &self.nodes[a];
        let nb = &self.nodes[b];
        debug_assert_ne!(na.pre_number, 0);
        debug_assert_ne!(nb.pre_number, 0);
        na.pre_number <= nb.pre_number && na.pre_max >= nb.pre_max
    }

    /// Computes the immediate dominator of each basic block.
    ///
    /// The algorithm is based on https://www.cs.rice.edu/~keith/EMBED/dom.pdf.
    fn compute_idoms(&mut self, func: &impl Function, po: &PostOrder) {
        self.processed.clear_and_resize(func.num_blocks());

        // Initialize the entry points as children of an internal virtual root.
        // This also makes them valid predecessors for the first iteration.
        for &entry in func.entry_points() {
            if po.is_reachable(entry) {
                self.processed.insert(entry);
            }
        }

        // Iterate to convergence
        let mut changed = true;
        while changed {
            changed = false;

            for block in po.cfg_postorder().rev() {
                // Filter out predecessors which haven't been processed yet.
                let mut valid_preds = func
                    .block_preds(block)
                    .iter()
                    .copied()
                    .filter(|&pred| self.processed.contains(pred));

                // Entry points have no predecessors and have already been
                // initialized above. Other blocks may need to wait until a
                // predecessor has been processed.
                let Some(first_pred) = valid_preds.next() else {
                    continue;
                };
                let mut new_idom = Some(first_pred);

                // The immediate dominator is the common dominator of all our
                // predecessors.
                for pred in valid_preds {
                    let Some(idom) = new_idom else {
                        break;
                    };
                    new_idom = self.compute_common_dominator(po, pred, idom);
                }

                // If anything changed, we need to perform another iteration.
                if !self.processed.contains(block) || self.nodes[block].parent.expand() != new_idom
                {
                    self.nodes[block].parent = new_idom.into();
                    self.processed.insert(block);
                    changed = true;
                }
            }
        }
    }

    /// Computes the common dominator of two basic blocks using only the
    /// parent links in the tree.
    fn compute_common_dominator(
        &self,
        po: &PostOrder,
        mut a: Block,
        mut b: Block,
    ) -> Option<Block> {
        loop {
            match po.rpo_cmp(a, b) {
                Ordering::Less => b = self.nodes[b].parent.expand()?,
                Ordering::Greater => a = self.nodes[a].parent.expand()?,
                Ordering::Equal => return Some(a),
            }
        }
    }
}
