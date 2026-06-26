//! Dead spill elimination pass.
//!
//! Move optimization will modify instruction operands and edit to source values
//! from registers instead of spill slots, which may in turn cause writes to
//! these spill slots to become dead.
//!
//! This pass performs a backwards dataflow liveness analysis to find all
//! spill slots that are never read. Any edits writing to these locations are
//! then removed.

use alloc::collections::BinaryHeap;
use core::debug_assert_matches;

use super::allocations::Allocations;
use super::move_resolver::{Edit, MoveResolver};
use super::spill_allocator::SpillAllocator;
use crate::Stats;
use crate::entity::{EntitySet, SecondaryMap};
use crate::function::{Block, Function, Inst, OperandKind};
use crate::output::{Allocation, AllocationKind, SpillSlot};

pub struct DeadSpillElimination {
    /// Spill slots that are live at the end of each block.
    block_live_out: SecondaryMap<Block, EntitySet<SpillSlot>>,

    /// Current liveness used by `process_block` as it scans backwards.
    live: EntitySet<SpillSlot>,

    /// Queue of blocks that need to be pre-processed.
    ///
    /// For faster convergence, we always pre-process the higher-index blocks
    /// first.
    blocks_to_process: BinaryHeap<Block>,

    /// Set of blocks in `blocks_to_process`.
    blocks_in_queue: EntitySet<Block>,
}

impl DeadSpillElimination {
    pub fn new() -> Self {
        Self {
            block_live_out: SecondaryMap::new(),
            blocks_to_process: BinaryHeap::new(),
            blocks_in_queue: EntitySet::new(),
            live: EntitySet::new(),
        }
    }

    pub fn run(
        &mut self,
        move_resolver: &mut MoveResolver,
        spill_allocator: &SpillAllocator,
        allocations: &Allocations,
        stats: &mut Stats,
        func: &impl Function,
    ) {
        // Initialize all spillslots to dead while preserving existing
        // allocations if available.
        let num_spillslots = spill_allocator.stack_layout.num_spillslots();
        self.block_live_out
            .grow_to_with(func.num_blocks(), EntitySet::new);
        for live_out in self.block_live_out.values_mut() {
            live_out.clear_and_resize(num_spillslots);
        }
        self.live.grow_to(num_spillslots);

        // Start by inserting all blocks into the queue.
        self.blocks_to_process.clear();
        self.blocks_in_queue.clear_and_resize(func.num_blocks());
        self.blocks_to_process.extend(func.blocks());
        self.blocks_in_queue.extend(func.blocks());

        // Propagate liveness through blocks until it converges.
        while let Some(block) = self.blocks_to_process.pop() {
            self.blocks_in_queue.remove(block);
            trace!("Pre-processing {block}...");
            self.process_block(block, move_resolver, allocations, stats, func, false);
            self.propagate_to_preds(block, func);
        }

        // Then perform a final pass which actually removes dead edits.
        for block in func.blocks().rev() {
            trace!("Optimizing {block}...");
            self.process_block(block, move_resolver, allocations, stats, func, true);
        }
    }

    /// Unions our live-in into predecessors' live-out and re-queues any blocks
    /// whose live-out has changed.
    fn propagate_to_preds(&mut self, block: Block, func: &impl Function) {
        let live_count = self.live.count();
        for &pred in func.block_preds(block) {
            if self.block_live_out[pred].union(&self.live) {
                trace!("Propagating {live_count} live spill slots from {block} to {pred}");
                if !self.blocks_in_queue.contains(pred) {
                    trace!("Queueing {pred} for dead spill liveness");
                    self.blocks_to_process.push(pred);
                    self.blocks_in_queue.insert(pred);
                }
            }
        }
    }

    fn process_block(
        &mut self,
        block: Block,
        move_resolver: &mut MoveResolver,
        allocations: &Allocations,
        stats: &mut Stats,
        func: &impl Function,
        delete_dead_edits: bool,
    ) {
        self.live.clone_from(&self.block_live_out[block]);
        trace!("Live-out at end of {block}: {:?}", self.live);

        let insts = func.block_insts(block);
        let mut edits = move_resolver.edits_to_mut(insts.to);

        // Process instructions within the block in reverse.
        for inst in insts.iter().rev() {
            self.process_inst(inst, allocations, func);

            // Process any edits located before this instruction.
            while let Some(last) = edits.last_mut() {
                if last.0 < inst {
                    break;
                }

                if let Some(edit) = last.1 {
                    if !self.process_edit(edit, allocations) && delete_dead_edits {
                        trace!("Eliminating dead spill at {}: {edit}", last.0);

                        // Edits are only removed on the final pass when we are
                        // sure of the liveness.
                        last.1 = None;

                        match edit {
                            Edit::Move { .. } => stat!(stats, eliminated_dead_spill),
                            Edit::EmergencySpill { .. } | Edit::EmergencyReload { .. } => {}
                            Edit::Rematerialize { .. } | Edit::IndirectRematerialize { .. } => {
                                stat!(stats, eliminated_dead_remat);
                            }
                        }
                    }
                }

                edits.split_off_last_mut();
            }
        }

        trace!("Live-in at start of {block}: {:?}", self.live);
    }

    fn process_inst(&mut self, inst: Inst, allocations: &Allocations, func: &impl Function) {
        trace!("Processing {inst}");
        let operands = func.inst_operands(inst);
        let operand_allocs = allocations.inst_allocations(inst);

        for (&op, &alloc) in operands.iter().zip(operand_allocs) {
            if let AllocationKind::SpillSlot(slot) = alloc.kind() {
                match op.kind() {
                    OperandKind::Def(_) | OperandKind::EarlyDef(_) => self.live.remove(slot),
                    OperandKind::DefGroup(_) | OperandKind::EarlyDefGroup(_) => {
                        debug_assert_matches!(alloc.kind(), AllocationKind::PhysReg(_));
                    }
                    OperandKind::Use(_)
                    | OperandKind::UseGroup(_)
                    | OperandKind::NonAllocatable => {}
                }
            }
        }

        for (&op, &alloc) in operands.iter().zip(operand_allocs) {
            if let AllocationKind::SpillSlot(slot) = alloc.kind() {
                match op.kind() {
                    OperandKind::Use(_) => self.live.insert(slot),
                    OperandKind::UseGroup(_) => {
                        debug_assert_matches!(alloc.kind(), AllocationKind::PhysReg(_));
                    }
                    OperandKind::Def(_)
                    | OperandKind::EarlyDef(_)
                    | OperandKind::DefGroup(_)
                    | OperandKind::EarlyDefGroup(_)
                    | OperandKind::NonAllocatable => {}
                }
            }
        }
    }

    /// Returns false if the edit is dead and can be removed.
    fn process_edit(&mut self, edit: Edit, allocations: &Allocations) -> bool {
        trace!("Processing edit {edit}");
        match edit {
            Edit::Move { to, from, value: _ } => {
                if self.is_dead_spillslot_write(to) {
                    return false;
                }
                self.clear_spillslot(to);
                self.mark_spillslot(from);
            }
            Edit::Rematerialize { to, value: _ } => {
                if self.is_dead_spillslot_write(to) {
                    return false;
                }
                self.clear_spillslot(to);
            }
            Edit::IndirectRematerialize {
                to,
                value: _,
                inputs,
            } => {
                if self.is_dead_spillslot_write(to) {
                    return false;
                }
                self.clear_spillslot(to);
                for &input in inputs.as_slice(&allocations.remat_inputs) {
                    // Inputs can only have Use/UseGroup/NonAllocatable inputs,
                    // and only the first can resolve to spillslots.
                    self.mark_spillslot(input);
                }
            }
            Edit::EmergencyReload { to: _, from } => {
                self.live.insert(from);
            }
            Edit::EmergencySpill { to, from: _ } => {
                self.live.remove(to);
            }
        }
        true
    }

    fn is_dead_spillslot_write(&self, alloc: Allocation) -> bool {
        matches!(alloc.kind(), AllocationKind::SpillSlot(slot) if !self.live.contains(slot))
    }

    fn mark_spillslot(&mut self, alloc: Allocation) {
        if let AllocationKind::SpillSlot(slot) = alloc.kind() {
            self.live.insert(slot);
        }
    }

    fn clear_spillslot(&mut self, alloc: Allocation) {
        if let AllocationKind::SpillSlot(slot) = alloc.kind() {
            self.live.remove(slot);
        }
    }
}
