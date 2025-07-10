//! Post-order traversal of the control-flow graph.
//!
//! The code in this file is based on the post-order implementation from
//! Cranelift.

use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;

use crate::entity::SecondaryMap;
use crate::function::{Block, Function};

/// Post-order traversal of the control flow graph.
///
/// This also doubles as a reachability check to determine whether a basic block
/// is reachable from the root block.
pub struct PostOrder {
    /// Stack used to compute the post-order.
    stack: Vec<(Visit, Block)>,

    /// List of basic blocks in CFG post-order.
    postorder: Vec<Block>,

    /// Number of a basic block in the postorder traversal of the control
    /// flow graph. This number is guranteed to be monotonically increasing for
    /// each node of the graph but may not be contiguous.
    ///
    /// Unreahcable blocks get a value of `UNREACHABLE` (0). `SEEN` is used as a
    /// marker during processing, but this is later replaced with the actual
    /// number.
    po_number: SecondaryMap<Block, u32>,
}

/// Special value for `po_number` which indicates an unreachable block.
///
/// Within `PostOrder::compute` this is also used to indicate a block that has
/// not yet been processed.
const UNREACHABLE: u32 = 0;

/// Special value for `po_number` only used within `PostOrder::compute` which
/// indicates that the block has been processed and its successors have been
/// pushed onto the stack.
const SEEN: u32 = 1;

/// DFS stack state marker for computing the cfg postorder.
enum Visit {
    First,
    Last,
}

impl PostOrder {
    /// Creates a new `PostOrder`.
    pub fn new() -> Self {
        PostOrder {
            stack: vec![],
            postorder: vec![],
            po_number: SecondaryMap::new(),
        }
    }

    /// Creates a new `PostOrder` for the given function.
    pub fn for_function(func: &impl Function) -> Self {
        let mut postorder = Self::new();
        postorder.compute(func);
        postorder
    }

    /// Returns whether a basic block is reachable from the root block.
    pub fn is_reachable(&self, block: Block) -> bool {
        self.po_number[block] != UNREACHABLE
    }

    /// Returns the list of basic blocks in control-flow postorder.
    ///
    /// The returned iterator only includes basic blocks reachable from the root
    /// block.
    ///
    /// Use `rev` on the returned iterator for reverse post-order iteration.
    pub fn cfg_postorder(&self) -> impl DoubleEndedIterator<Item = Block> + ExactSizeIterator + '_ {
        self.postorder.iter().copied()
    }

    /// Compares the position of two blocks in the CFG reverse-postorder.
    pub fn rpo_cmp(&self, a: Block, b: Block) -> Ordering {
        self.po_number[a].cmp(&self.po_number[b]).reverse()
    }

    /// Computes the control-flow post-order.
    pub fn compute(&mut self, func: &impl Function) {
        // Clear all the internal data structures and initialize po_number with
        // zeroes (UNREACHABLE).
        self.stack.clear();
        self.postorder.clear();
        self.postorder.reserve(func.num_blocks());
        self.po_number.clear_and_resize(func.num_blocks());

        // This algorithm is a depth first traversal (DFT) of the control flow graph, computing a
        // post-order of the blocks that are reachable from the entry block. A DFT post-order is not
        // unique. The specific order we get is controlled by the order each node's children are
        // visited.
        //
        // During this algorithm only, use `po_number` to hold the following state:
        //
        //   0:    block has not yet had its first visit
        //   SEEN: block has been visited at least once, implying that all of its successors are on
        //         the stack

        // Traverse the control flow graph in depth-first order starting from
        // the root. Post-ordering is assigned on the way back up the graph.
        self.stack.push((Visit::First, Block::ENTRY_BLOCK));
        while let Some((visit, block)) = self.stack.pop() {
            match visit {
                Visit::First => {
                    if self.po_number[block] == UNREACHABLE {
                        // This is the first time we pop the block, so we need to scan its
                        // successors and then revisit it.
                        self.po_number[block] = SEEN;
                        self.stack.push((Visit::Last, block));

                        // Process the successors in order: we treat the first
                        // successor as the most likely one, which want to to
                        // appear last in the post-order.
                        for &succ in func.block_succs(block) {
                            // This is purely an optimization to avoid additional iterations of
                            // the loop, and is not required; it's merely inlining the check
                            // from the outer conditional of this case to avoid the extra loop
                            // iteration.
                            if self.po_number[succ] == UNREACHABLE {
                                self.stack.push((Visit::First, succ));
                            }
                        }
                    }
                }
                Visit::Last => {
                    // We've finished all this node's successors.
                    self.postorder.push(block);
                    self.po_number[block] = 2 + self.postorder.len() as u32;
                }
            }
        }
    }
}
