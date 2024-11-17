//! Optimal placement of live range splits based on block execution frequency.
//!
//! The basic algorithm is fairly straightfoward:
//! - We pre-process the CFG by computing, for each block, the previous and next
//!   blocks with a lower execution frequency.
//! - When a split is requested, a range of instructions is provided: these are
//!   the valid split points. Each split also comes with a preference to keep
//!   the split point towards either the start or end of this range.
//! - Starting from the preferred split point, we move the split point down to
//!   the next block with a lower execution frequency. This is repeated until
//!   the split point is outside the requested range.
//! - We return the start or end of the block depending on the preference.
//!
//! This has the downside of being very sensitive to the linear ordering of
//! basic blocks, but works well if proper ordering is applied during lowering.
//! Specifically, this works well if blocks are in reverse post-order and loops
//! are properly nested: any loop exit blocks should be after any loop body
//! blocks.

use alloc::vec;
use alloc::vec::Vec;

use crate::entity::packed_option::PackedOption;
use crate::entity::SecondaryMap;
use crate::function::{Block, Function, Inst};

/// Optimal placement of live range split points.
pub struct SplitPlacement {
    /// For each block, gives the first following block with a lower execution
    /// frequency.
    next_lower_freq: SecondaryMap<Block, PackedOption<Block>>,

    /// For each block, gives the first preceding block with a lower execution
    /// frequency.
    prev_lower_freq: SecondaryMap<Block, PackedOption<Block>>,

    /// Stack used for computing the above 2 vectors.
    stack: Vec<(Block, f32)>,
}

impl SplitPlacement {
    pub fn new() -> Self {
        Self {
            next_lower_freq: SecondaryMap::new(),
            prev_lower_freq: SecondaryMap::new(),
            stack: vec![],
        }
    }

    /// Prepares the metadata for split placement from the function CFG.
    pub fn prepare(&mut self, func: &impl Function) {
        self.next_lower_freq.clear_and_resize(func.num_blocks());
        self.prev_lower_freq.clear_and_resize(func.num_blocks());

        // Scan through the blocks in linear order.
        self.stack.clear();
        for block in func.blocks() {
            // Pop entries from the stack that have a higher or equal frequency
            // compared to the current block.
            let freq = func.block_frequency(block);
            while let Some(&(prev_block, prev_freq)) = self.stack.last() {
                if prev_freq >= func.block_frequency(block) {
                    self.stack.pop();
                } else {
                    // Found a previous block with a lower frequency.
                    self.prev_lower_freq[block] = Some(prev_block).into();
                    break;
                }
            }
            self.stack.push((block, freq));
        }

        // Same as above, but in reverse.
        self.stack.clear();
        for block in func.blocks().rev() {
            let freq = func.block_frequency(block);
            while let Some(&(next_block, next_freq)) = self.stack.last() {
                if next_freq >= func.block_frequency(block) {
                    self.stack.pop();
                } else {
                    self.next_lower_freq[block] = Some(next_block).into();
                    break;
                }
            }
            self.stack.push((block, freq));
        }
    }

    /// Finds an optimal split point which is between the two instuctions
    /// given, based on basic block frequencies.
    ///
    /// `prefer_early` indicates whether to prefer an early or late split point
    /// when block frequencies are equal.
    ///
    /// The return value indicates a chosen split point before the given `Inst`.
    pub fn find_optimal_split_point(
        &self,
        after: Inst,
        before: Inst,
        prefer_early: bool,
        func: &impl Function,
    ) -> Inst {
        debug_assert!(after < before);
        if prefer_early {
            let mut split = after.next();
            let limit = before;
            while split != limit {
                let block = func.inst_block(split);
                let Some(next_lower_freq) = self.next_lower_freq[block].into() else {
                    break;
                };
                debug_assert!(func.block_frequency(next_lower_freq) < func.block_frequency(block));

                let new_split = func.block_insts(next_lower_freq).from;
                debug_assert!(new_split > split);
                if new_split <= limit {
                    split = new_split;
                } else {
                    break;
                }
            }
            trace!(
                "Selecting split point at {split}-pre between {after} and {before} (prefer start)"
            );
            split
        } else {
            let mut split = before;
            let limit = after.next();
            while split != limit {
                let block = func.inst_block(split.prev());
                let Some(prev_lower_freq) = self.prev_lower_freq[block].into() else {
                    break;
                };
                debug_assert!(func.block_frequency(prev_lower_freq) < func.block_frequency(block));

                let new_split = func.block_insts(prev_lower_freq).to;
                debug_assert!(new_split < split);
                if new_split >= limit {
                    split = new_split;
                } else {
                    break;
                }
            }
            trace!(
                "Selecting split point at {split}-pre between {after} and {before} (prefer end)"
            );
            split
        }
    }
}
