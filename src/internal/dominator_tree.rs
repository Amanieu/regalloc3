//! Dominator tree.
//!
//! The code in this file is based on the dominator tree implementation from
//! Cranelift.

use alloc::vec;
use alloc::vec::Vec;
use core::cmp::{self, Ordering};
use core::mem;

use cranelift_entity::packed_option::PackedOption;
use cranelift_entity::SecondaryMap;

use super::postorder::PostOrder;
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

    /// Stack for DFS traversal.
    stack: Vec<Block>,
}

impl DominatorTree {
    /// Creates a new `DominatorTree`.
    pub fn new() -> Self {
        Self {
            nodes: SecondaryMap::new(),
            stack: vec![],
        }
    }

    /// Computes the dominator tree.
    pub fn compute(&mut self, func: &impl Function, po: &PostOrder) {
        // Reset all immediate dominators to None.
        self.nodes.clear();

        // 1. Compute immediate dominators for each basic block.
        self.compute_idoms(func, po);

        // 2. Populate child and sibling links for each node.
        for block in po.cfg_postorder() {
            // The entry block has no parent.
            if let Some(idom) = self.nodes[block].parent.expand() {
                let sibling = mem::replace(&mut self.nodes[idom].child, Some(block).into());
                self.nodes[block].sibling = sibling;
            }
        }

        // 3. Assign pre-order numbers from a DFS of the dominator tree.
        self.stack.push(Block::ENTRY_BLOCK);
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
    /// Returns `None` for the entry block and unreachable blocks.
    pub fn immediate_dominator(&self, block: Block) -> Option<Block> {
        self.nodes[block].parent.into()
    }

    /// Determines whether `a` dominates `b`.
    ///
    /// Both blocks must be reachable from the entry point.
    ///
    /// Returns true if `a == b`.
    pub fn dominates(&self, a: Block, b: Block) -> bool {
        let na = &self.nodes[a];
        let nb = &self.nodes[b];
        na.pre_number <= nb.pre_number && na.pre_max >= nb.pre_max
    }

    /// Computes the immediate dominator of each basic block.
    ///
    /// The algorithm is based on https://www.cs.rice.edu/~keith/EMBED/dom.pdf.
    fn compute_idoms(&mut self, func: &impl Function, po: &PostOrder) {
        // Initialize the immediate dominator of the entry block to itself. This
        // is necessary so that valid_preds includes the root block even though
        // it has no predecessor.
        self.nodes[Block::ENTRY_BLOCK].parent = Some(Block::ENTRY_BLOCK).into();

        // Iterate to convergence
        let mut changed = true;
        while changed {
            changed = false;

            // Skip the entry block which has no predecessors.
            for block in po.cfg_postorder().rev().skip(1) {
                // Filter out predecessors which haven't been processed yet.
                let mut valid_preds = func
                    .block_preds(block)
                    .iter()
                    .copied()
                    .filter(|&pred| self.nodes[pred].parent.is_some());

                // At least one predecessor must have been processed already
                // since we are iterating in reverse post-order.
                let mut new_idom = valid_preds.next().unwrap();

                // The immediate dominator is the common dominator of all our
                // predecessors.
                for pred in valid_preds {
                    new_idom = self.compute_common_dominator(po, pred, new_idom);
                }

                // If anything changed, we need to perform another iteration.
                if self.nodes[block].parent != Some(new_idom).into() {
                    self.nodes[block].parent = Some(new_idom).into();
                    changed = true;
                }
            }
        }

        // Now fix up the entry block: it has no immediate dominator since it
        // has no predecessors.
        self.nodes[Block::ENTRY_BLOCK].parent = None.into();
    }

    /// Computes the common dominator of two basic blocks using only the
    /// parent links in the tree.
    fn compute_common_dominator(&self, po: &PostOrder, mut a: Block, mut b: Block) -> Block {
        loop {
            match po.rpo_cmp(a, b) {
                Ordering::Less => b = self.nodes[b].parent.expect("unreachable block"),
                Ordering::Greater => a = self.nodes[a].parent.expect("unreachable block"),
                Ordering::Equal => return a,
            }
        }
    }
}
