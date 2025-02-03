//! Move optimization pass.
//!
//! Throughout the allocation process, we make a simplifying assumption that
//! each `Value` only lives at a single `Allocation` (register or spill slot) at
//! any one time. This is necessary to make the allocation problem tractable,
//! but can lead to inefficient code because we "forget" that a value is already
//! present in a register or spill slot.
//!
//! This is particularly severe when a value is repeatedly spilled and reloaded
//! from the stack: quite often, the stack slot will already have a copy of the
//! value so we can skip storing the value in that case.
//!
//! To address this, we run a general move optimization pass. The pass aims to
//! make the following optimizations:
//! - Eliminate moves if the destination of the move already holds the expected
//!   value.
//! - Change `Use` operands that read from stack locations to read from a
//!   register if the required value is available in one.
//!
//! To be able to do this, we need to know which registers contain which values
//! at each instruction boundary. We get this information in 2 steps:
//! - First we need to determine what values are available in registers and
//!   spill slots at the start of each block. This is obtained by simulating
//!   execution of each block and propagating the state to successor block,
//!   until a fixed point is reached.
//! - Then, once the set of registers on block entry is definitely known, we
//!   can go through each block and optimize moves and uses.
//!
//! However since this is rather expensive, by default we only track values
//! across forward edges in the CFG. This allows move optimization to be done in
//! a single pass. Any blocks that have incoming back edges are simply assumed
//! to not have any live values on entry.

use alloc::collections::binary_heap::BinaryHeap;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;
use core::fmt;

use smallvec::{smallvec, SmallVec};

use super::allocations::Allocations;
use super::coalescing::Coalescing;
use super::move_resolver::{Edit, MoveResolver};
use super::spill_allocator::SpillAllocator;
use crate::entity::packed_option::{PackedOption, ReservedValue};
use crate::entity::sparse::Entry;
use crate::entity::{EntitySet, PrimaryMap, SecondaryMap, SparseMap};
use crate::function::{
    Block, Function, Inst, Operand, OperandConstraint, OperandKind, Value, ValueGroup,
};
use crate::output::{Allocation, AllocationKind, SpillSlot};
use crate::reginfo::{PhysReg, PhysRegSet, RegInfo, RegUnit, RegUnitSet, MAX_REG_UNITS};
use crate::{MoveOptimizationLevel, Stats};

entity_def! {
    /// Multiple blocks may share the same entry state if they all have the same
    /// single predecessor block. We avoid duplication by adding one level of
    /// indirection.
    entity EntryState(u32);
}

pub struct MoveOptimizer {
    /// State tracker used for processing blocks.
    state_tracker: StateTracker,

    /// Set of values and their allocations on entry to a block.
    ///
    /// A single entry state may be shared by multiple blocks if they have the
    /// same predecessor block.
    entry_states: PrimaryMap<EntryState, SmallVec<[(Allocation, Value); 15]>>,

    /// For each block, the state that it has on entry when processing it.
    ///
    /// This must be some for blocks that are in the queue or have been
    /// processed.
    block_entry_states: SecondaryMap<Block, PackedOption<EntryState>>,

    /// Queue of blocks that need to be pre-processed.
    ///
    /// For faster convergence, we always pre-process the lower-index blocks
    /// first.
    blocks_to_preprocess: BinaryHeap<Reverse<Block>>,

    /// Set of blocks in `blocks_to_process`.
    blocks_in_queue: EntitySet<Block>,
}

impl MoveOptimizer {
    pub fn new() -> Self {
        Self {
            state_tracker: StateTracker::new(),
            entry_states: PrimaryMap::new(),
            block_entry_states: SecondaryMap::new(),
            blocks_to_preprocess: BinaryHeap::new(),
            blocks_in_queue: EntitySet::new(),
        }
    }

    /// Entry point for the move optimization pass.
    pub fn run(
        &mut self,
        move_resolver: &mut MoveResolver,
        spill_allocator: &SpillAllocator,
        coalescing: &mut Coalescing,
        allocations: &mut Allocations,
        stats: &mut Stats,
        func: &impl Function,
        reginfo: &impl RegInfo,
        move_optimization: MoveOptimizationLevel,
    ) {
        // Nothing to do if move optimization is disabled.
        if move_optimization == MoveOptimizationLevel::Off {
            return;
        }

        self.state_tracker
            .prepare(spill_allocator, coalescing, func);

        // For global optimization, we need to compute entry states for each
        // block.
        if move_optimization == MoveOptimizationLevel::Global {
            self.compute_entry_states(allocations, move_resolver, stats, func, reginfo);
        }

        // For forward-only optimization, identify loop headers (any block that
        // has a back-edge into it) and force their entry state to be empty.
        if move_optimization == MoveOptimizationLevel::Forward {
            self.prepare_forward_pass(func);
        }

        for block in func.blocks() {
            self.state_tracker.new_block();

            if move_optimization == MoveOptimizationLevel::Global
                || move_optimization == MoveOptimizationLevel::Forward
            {
                // Set the initial state for this block from the pre-computed
                // states.
                let entry_state = &self.entry_states[self.block_entry_states[block].unwrap()];
                for &(alloc, value) in entry_state {
                    self.state_tracker.initial_value(value, alloc, reginfo);
                }
            }

            // Optimize the block.
            self.state_tracker.optimize_block(
                block,
                allocations,
                move_resolver,
                stats,
                func,
                reginfo,
            );

            if move_optimization == MoveOptimizationLevel::Forward {
                self.propagate_to_succs(block, false, func);
            }
        }
    }

    /// Prepares the block entry states for forward-edge-only optimization.
    fn prepare_forward_pass(&mut self, func: &impl Function) {
        self.entry_states.clear();
        self.block_entry_states.clear_and_resize(func.num_blocks());

        // Set the entry state of the entry block to empty.
        let initial_state = self.entry_states.push(smallvec![]);
        self.block_entry_states[Block::ENTRY_BLOCK] = Some(initial_state).into();

        // Copy that state for any blocks that have a predecessor that later or
        // equal to itself.
        for block in func.blocks().skip(1) {
            let preds = func.block_preds(block);
            if preds.len() > 1 {
                if preds.iter().any(|&pred| pred >= block) {
                    self.block_entry_states[block] = Some(initial_state).into();
                }
            }
        }
    }

    /// Computes the entry state for all blocks in the function.
    fn compute_entry_states(
        &mut self,
        allocations: &Allocations,
        move_resolver: &MoveResolver,
        stats: &mut Stats,
        func: &impl Function,
        reginfo: &impl RegInfo,
    ) {
        self.entry_states.clear();
        self.block_entry_states.clear_and_resize(func.num_blocks());
        self.blocks_to_preprocess.clear();
        self.blocks_in_queue.clear_and_resize(func.num_blocks());

        // Initialize the state for the entry block and add it to the queue.
        let initial_state = self.entry_states.push(smallvec![]);
        self.block_entry_states[Block::ENTRY_BLOCK] = Some(initial_state).into();
        self.blocks_to_preprocess.push(Reverse(Block::ENTRY_BLOCK));
        self.blocks_in_queue.insert(Block::ENTRY_BLOCK);

        while let Some(Reverse(block)) = self.blocks_to_preprocess.pop() {
            // Skip blocks with no successors, they don't have any state to
            // propagate.
            if func.block_succs(block).is_empty() {
                continue;
            }
            stat!(stats, blocks_preprocessed_for_optimizer);
            self.blocks_in_queue.remove(block);
            self.state_tracker.new_block();

            // Set the initial state for this block. This must have been
            // initialized from a predecessor block.
            let entry_state = &self.entry_states[self.block_entry_states[block].unwrap()];
            for &(alloc, value) in entry_state {
                self.state_tracker.initial_value(value, alloc, reginfo);
            }

            // Pre-process the block to get the allocation state at the end of
            // the block.
            self.state_tracker
                .preprocess_block(block, allocations, move_resolver, func, reginfo);

            // Propagate the end state of the block to successor blocks. Blocks
            // whose entry state has been modified are re-queued for processing.
            self.propagate_to_succs(block, true, func);
        }
    }

    /// Propagates the current `StateTracker` state to successor blocks.
    fn propagate_to_succs(&mut self, block: Block, use_queue: bool, func: &impl Function) {
        let succs = func.block_succs(block);
        if succs.is_empty() {
            return;
        }
        match self.block_entry_states[succs[0]].expand() {
            Some(state) => {
                // All of our successors must have the same EntryState.
                for &succ in succs {
                    debug_assert!(
                        self.block_entry_states[succ] == self.block_entry_states[succs[0]]
                    );
                }

                let state = &mut self.entry_states[state];
                let prev_len = state.len();

                // Merge our state into the existing state by only keeping
                // allocations which have the same value in our state and the
                // existing state.
                state.retain(|&mut (alloc, value)| match alloc.kind() {
                    AllocationKind::PhysReg(reg) => {
                        if let Some(regs) = self.state_tracker.value_regs.get(value) {
                            if regs.contains(reg) {
                                return true;
                            }
                        }
                        false
                    }
                    AllocationKind::SpillSlot(slot) => {
                        if let Some(&slot_value) = self.state_tracker.spillslot_values.get(slot) {
                            if slot_value == value {
                                return true;
                            }
                        }
                        false
                    }
                });

                // If values were removed from the state then we need to
                // re-process successor blocks.
                if use_queue && state.len() != prev_len {
                    for &succ in succs {
                        trace!("State changed, propagating to {succ}");
                        if !self.blocks_in_queue.contains(succ) {
                            self.blocks_to_preprocess.push(Reverse(succ));
                            self.blocks_in_queue.insert(succ);
                        }
                    }
                }
            }
            None => {
                // Construct a new EntryState from our current state.
                //
                // As an optimization, we avoid propagating value through
                // back-edges since the value can't possibly be used in the
                // target block. The definition of `Function` requires that
                // dominating blocks appear before any blocks they dominate.
                //
                // For the same reason, back-edges can only occur from blocks
                // with a single predecessor so we only need to check those.
                let mut state = smallvec![];
                for &(value, regs) in &self.state_tracker.value_regs {
                    if succs.len() == 1 && self.state_tracker.value_def_block[value] >= succs[0] {
                        continue;
                    }
                    for reg in regs {
                        state.push((Allocation::reg(reg), value));
                    }
                }
                for &(slot, value) in &self.state_tracker.spillslot_values {
                    if succs.len() == 1 && self.state_tracker.value_def_block[value] >= succs[0] {
                        continue;
                    }
                    state.push((Allocation::spillslot(slot), value));
                }

                let state = self.entry_states.push(state);
                for &succ in succs {
                    // Assign the state to each successor.
                    debug_assert!(self.block_entry_states[succ].is_none());
                    self.block_entry_states[succ] = Some(state).into();

                    // And queue the successors for processing.
                    if use_queue {
                        trace!("Queuing fresh successor {succ}");
                        if !self.blocks_in_queue.contains(succ) {
                            self.blocks_to_preprocess.push(Reverse(succ));
                            self.blocks_in_queue.insert(succ);
                        }
                    }
                }
            }
        }
    }
}

/// The state tracker tracks the location of each live value as we go through a
/// block and also performs the move optimizations based on this state.
struct StateTracker {
    /// Spill slot assigned to each value.
    ///
    /// The alloctor guarantees that each value is only ever spilled to a
    /// single spill slot.
    slot_for_value: SecondaryMap<Value, PackedOption<SpillSlot>>,

    /// Block in which a value is defined.
    ///
    /// This is used to limit propagation of values across back-edges.
    value_def_block: SecondaryMap<Value, Block>,

    /// Set of `PhysReg` that a `Value` is located in at the current point in a
    /// block that is being processed.
    ///
    /// Empty `PhysRegSet` are omitted from the map to keep the size down.
    value_regs: SparseMap<Value, PhysRegSet>,

    /// The last `Value` that was written to a `RegUnit` and the `PhysReg` used
    /// for that write.
    ///
    /// This does not necessarily mean that `Value` is located in `PhysReg`
    /// since other units of the `PhysReg` may have been overwritten.
    last_unit_write: SparseMap<RegUnit, (PhysReg, Value)>,

    /// `Value` currently held in each `SpillSlot`.
    spillslot_values: SparseMap<SpillSlot, Value>,

    /// List of operands whose allocation is reused in the current instruction.
    reused_operands: Vec<usize>,

    /// Set of register units the were defined by the current instruction.
    def_units: RegUnitSet,

    /// Register units spilled to an emergency spill slot, along with the data
    /// that was in `last_unit_write` for that unit.
    ///
    /// Since the allocator only tracks a single emergency spill at any one
    /// time, we don't need to track the contents of multiple spill slots.
    emergency_spill: Vec<(RegUnit, PhysReg, Value)>,

    /// Block in which a value was last spilled to its spillslot.
    ///
    /// If that block dominates the current block then all spills in the current
    /// block are redundant and can be eliminated.
    last_spilled_in: SecondaryMap<Value, PackedOption<Block>>,
}

impl fmt::Display for StateTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut alloc_values = vec![];
        for &(value, regs) in &self.value_regs {
            for reg in regs {
                alloc_values.push((Allocation::reg(reg), value));
            }
        }
        alloc_values.sort_unstable_by_key(|&(alloc, _)| alloc);
        for &(alloc, value) in &alloc_values {
            write!(f, "{alloc}({value}) ")?;
        }
        alloc_values.clear();
        for &(slot, value) in &self.spillslot_values {
            alloc_values.push((Allocation::spillslot(slot), value));
        }
        alloc_values.sort_unstable_by_key(|&(alloc, _)| alloc);
        for &(alloc, value) in &alloc_values {
            write!(f, "{alloc}({value}) ")?;
        }
        Ok(())
    }
}

impl StateTracker {
    fn new() -> Self {
        Self {
            slot_for_value: SecondaryMap::new(),
            value_def_block: SecondaryMap::new(),
            value_regs: SparseMap::new(),
            last_unit_write: SparseMap::new(),
            spillslot_values: SparseMap::new(),
            reused_operands: vec![],
            def_units: RegUnitSet::new(),
            emergency_spill: vec![],
            last_spilled_in: SecondaryMap::new(),
        }
    }

    /// Prepares several data structures that will be used by the state tracker.
    fn prepare(
        &mut self,
        spill_allocator: &SpillAllocator,
        coalescing: &mut Coalescing,
        func: &impl Function,
    ) {
        self.value_def_block
            .grow_to_with(func.num_values(), Block::reserved_value);
        self.slot_for_value.grow_to(func.num_values());
        for value in func.values() {
            self.slot_for_value[value] = spill_allocator.value_spillslot(value, coalescing).into();
        }
        self.last_spilled_in.clear_and_resize(func.num_values());
        self.value_regs.grow_to(func.num_values());
        self.last_unit_write.grow_to(MAX_REG_UNITS);
        self.spillslot_values
            .grow_to(spill_allocator.stack_layout.num_spillslots());
    }

    /// Resets the state tracker to a clean slate with no live values.
    fn new_block(&mut self) {
        self.value_regs.clear();
        self.last_unit_write.clear();
        self.spillslot_values.clear();
    }

    /// Clobbers any existing value on a register that contains the given unit.
    fn clobber_unit(&mut self, unit: RegUnit) {
        if let Some((reg, value)) = self.last_unit_write.remove(unit) {
            if let Entry::Occupied(mut e) = self.value_regs.entry(value) {
                let set = e.get_mut();
                set.remove(reg);
                if set.is_empty() {
                    e.remove();
                }
            }
        }
    }

    /// Handles the definition of a value.
    ///
    /// This consists of 3 parts:
    /// 1) Invalidating any previous mapping for the same value, since those are
    ///    from the previous loop iteration and no longer valid.
    /// 2) Invalidating any previous value at the destination location.
    /// 3) Recording the location of the new value.
    fn def_value(&mut self, value: Value, alloc: Allocation, block: Block, reginfo: &impl RegInfo) {
        self.value_def_block[value] = block;
        match alloc.kind() {
            AllocationKind::PhysReg(reg) => {
                for unit in reginfo.reg_units(reg) {
                    self.def_units.insert(unit);
                    self.clobber_unit(unit);
                    self.last_unit_write.insert(unit, (reg, value));
                }
                let set = [reg].into_iter().collect();
                self.value_regs.insert(value, set);
                if let Some(slot) = self.slot_for_value[value].expand() {
                    self.spillslot_values.remove(slot);
                }
            }
            AllocationKind::SpillSlot(slot) => {
                self.value_regs.remove(value);
                if self.slot_for_value[value].expand() == Some(slot) {
                    self.spillslot_values.insert(slot, value);
                }
            }
        }
    }

    /// Simpler version of `def_value` used for to set up the initial block
    /// entry state.
    ///
    /// This is always called on a clean state and therefore doesn't need to
    /// worry about overlaps or clobbering existing values.
    ///
    /// However it does need to handle the case where a value is in multiple
    /// allocations at once, which `def_value` doesn't need to.
    fn initial_value(&mut self, value: Value, alloc: Allocation, reginfo: &impl RegInfo) {
        match alloc.kind() {
            AllocationKind::PhysReg(reg) => {
                for unit in reginfo.reg_units(reg) {
                    self.last_unit_write.insert_unique(unit, (reg, value));
                }
                self.value_regs.entry(value).or_default().insert(reg);
            }
            AllocationKind::SpillSlot(slot) => {
                self.spillslot_values.insert_unique(slot, value);
            }
        }
    }

    /// Calls `def_value` for each member of a group.
    fn def_value_group(
        &mut self,
        inst: Inst,
        op: Operand,
        value_group: ValueGroup,
        first_alloc: Allocation,
        block: Block,
        func: &impl Function,
        reginfo: &impl RegInfo,
    ) {
        // The output only gives us the allocation for the first group member,
        // use that to recover the register group and get the other members.
        let AllocationKind::PhysReg(reg) = first_alloc.kind() else {
            unreachable!()
        };
        let class = match op.constraint() {
            OperandConstraint::Class(class) => class,
            OperandConstraint::Fixed(_) => unreachable!(),
            OperandConstraint::Reuse(idx) => {
                let OperandConstraint::Class(class) = func.inst_operands(inst)[idx].constraint()
                else {
                    unreachable!();
                };
                class
            }
        };
        let reg_group = reginfo.group_for_reg(reg, 0, class).unwrap();
        for (&value, &reg) in func
            .value_group_members(value_group)
            .iter()
            .zip(reginfo.reg_group_members(reg_group))
        {
            self.def_value(value, Allocation::reg(reg), block, reginfo);
        }
    }

    /// Applies the effects of an edit on the state.
    fn process_edit(&mut self, edit: Edit, reginfo: &impl RegInfo) {
        // Nothing to do if the edit has been optimized away.
        let Some(to) = edit.to.expand() else {
            return;
        };

        if let Some(value) = edit.value.expand() {
            // For normal moves, clobber the destination and then add it as
            // a location for the given value.
            match to.kind() {
                AllocationKind::PhysReg(reg) => {
                    for unit in reginfo.reg_units(reg) {
                        self.clobber_unit(unit);
                        self.last_unit_write.insert(unit, (reg, value));
                    }
                    self.value_regs.entry(value).or_default().insert(reg);
                }
                AllocationKind::SpillSlot(slot) => {
                    if self.slot_for_value[value].expand() == Some(slot) {
                        self.spillslot_values.insert(slot, value);
                    }
                }
            }
        } else {
            match (edit.from.unwrap().kind(), to.kind()) {
                // For emergency spills, save the contents of `last_unit_write`
                // for the affected units.
                //
                // Note that there can only be one emergency spill at any time.
                // Also, emergency spill slots are not used to hold normal
                // values and therefore don't need to be tracked.
                (AllocationKind::PhysReg(reg), AllocationKind::SpillSlot(slot)) => {
                    debug_assert!(!self.spillslot_values.contains_key(slot));
                    self.emergency_spill.clear();
                    for unit in reginfo.reg_units(reg) {
                        if let Some(&(reg, value)) = self.last_unit_write.get(unit) {
                            self.emergency_spill.push((unit, reg, value));
                        }
                    }
                }

                // For emergency reloads, clobber the destination units and then
                // restore the saved `last_unit_write` contents.
                (AllocationKind::SpillSlot(_), AllocationKind::PhysReg(reg)) => {
                    for unit in reginfo.reg_units(reg) {
                        self.clobber_unit(unit);
                    }
                    for &(unit, reg, value) in &self.emergency_spill {
                        self.last_unit_write.insert(unit, (reg, value));

                        // Mark the value as being located in a register only if
                        // all units have the same value.
                        if reginfo.reg_units(reg).len() == 1
                            || reginfo.reg_units(reg).all(|unit| {
                                self.last_unit_write
                                    .get(unit)
                                    .is_some_and(|&(_, value2)| value2 == value)
                            })
                        {
                            self.value_regs.entry(value).or_default().insert(reg);
                        }
                    }
                }
                (AllocationKind::PhysReg(_), AllocationKind::PhysReg(_))
                | (AllocationKind::SpillSlot(_), AllocationKind::SpillSlot(_)) => unreachable!(),
            }
        }
    }

    /// Pre-processes a block by simulating the execution of all edits and
    /// instructions while updating the state accordingly.
    ///
    /// This function don't optimize the block but will produce the values that
    /// are available in registers and spill slots at the end of the block.
    fn preprocess_block(
        &mut self,
        block: Block,
        allocations: &Allocations,
        move_resolver: &MoveResolver,
        func: &impl Function,
        reginfo: &impl RegInfo,
    ) {
        trace!("Pre-processing {block}...");

        // Define blockparam values. The corresponding moves are in predecessor
        // blocks, but tagged with the outgoing value.
        for (value, alloc) in move_resolver.blockparam_allocs(block) {
            self.def_value(value, alloc, block, reginfo);
        }
        let mut edits = move_resolver.edits_from(func.block_insts(block).from);

        for inst in func.block_insts(block).iter() {
            // Process any edits before the current instruction.
            while let Some((first, rest)) = edits.split_first() {
                if first.0 > inst {
                    break;
                }
                trace!("Values: {self}");
                trace!("Pre-processing edit: {}", first.1);
                self.process_edit(first.1, reginfo);
                edits = rest;
            }

            trace!("Values: {self}");
            trace!("Pre-processing {inst}");

            // Process def operands.
            self.def_units.clear();
            for (&op, &alloc) in func
                .inst_operands(inst)
                .iter()
                .zip(allocations.inst_allocations(inst))
            {
                match op.kind() {
                    OperandKind::Def(value) | OperandKind::EarlyDef(value) => {
                        self.def_value(value, alloc, block, reginfo);
                    }
                    OperandKind::DefGroup(value_group)
                    | OperandKind::EarlyDefGroup(value_group) => {
                        self.def_value_group(inst, op, value_group, alloc, block, func, reginfo);
                    }
                    OperandKind::Use(_)
                    | OperandKind::UseGroup(_)
                    | OperandKind::NonAllocatable => {}
                }
            }

            // Process clobbers.
            for unit in func.inst_clobbers(inst) {
                if !self.def_units.contains(unit) {
                    self.clobber_unit(unit);
                }
            }
        }

        trace!("Values: {self}");
    }

    /// Attempts to optimize an `Edit` by eliminating it or making it cheaper.
    fn optimize_edit(
        &mut self,
        edit: &mut Edit,
        block: Block,
        stats: &mut Stats,
        func: &impl Function,
        reginfo: &impl RegInfo,
    ) {
        if let Some(value) = edit.value.expand() {
            // First, see if the destination already contains the desired value.
            // If that is the case then we can turn the edit into a `nop` by
            // setting its destination to `None`.
            match edit.to.unwrap().kind() {
                AllocationKind::PhysReg(reg) => {
                    if let Some(set) = self.value_regs.get(value) {
                        if set.contains(reg) {
                            match edit.from.expand() {
                                Some(from) => match from.kind() {
                                    AllocationKind::PhysReg(_) => {
                                        stat!(stats, optimized_redundant_move);
                                    }
                                    AllocationKind::SpillSlot(_) => {
                                        stat!(stats, optimized_redundant_reload);
                                    }
                                },
                                None => stat!(stats, optimized_redundant_remat),
                            }
                            trace!("Eliminated redundant edit");
                            edit.to = None.into();
                            return;
                        }
                    }
                }
                AllocationKind::SpillSlot(slot) => {
                    if self.spillslot_values.get(slot) == Some(&value) {
                        stat!(stats, optimized_redundant_spill);
                        trace!("Eliminated redundant spill");
                        edit.to = None.into();
                        return;
                    }

                    // We may not know that the value is already in the spill
                    // slot if we are using limited propagation (`Local` or
                    // `Forward`). However we can still determine that a value
                    // is already in the spill slot if it has been spilled in
                    // a block that dominates the current block.
                    //
                    // This works because of several factors:
                    // - spill slot allocation will allocate the entire live
                    //   range of the value, not just the spilled portions.
                    // - each value is only ever spilled to a single spill slot.
                    // - we require that blocks be topologically ordered with
                    //   regards to dominance.
                    if self.slot_for_value[value].expand() == Some(slot) {
                        if let Some(prev_block) = self.last_spilled_in[value].expand() {
                            if func.block_dominates(prev_block, block) {
                                stat!(stats, optimized_redundant_spill);
                                edit.to = None.into();
                                trace!(
                                    "Eliminated redundant spill: already spilled in dominating \
                                     block"
                                );
                                return;
                            }
                        }
                        self.last_spilled_in[value] = Some(block).into();
                    }
                }
            }

            // We couldn't eliminate the move entirely, but we may be able to
            // turn a load from memory into a register move if the desired value
            // is already present in another register.
            if let Some(from) = edit.from.expand() {
                if from.is_memory(reginfo) {
                    if let Some(&regs_with_value) = self.value_regs.get(value) {
                        if let Some(reg) = regs_with_value
                            .into_iter()
                            .find(|&reg| !reginfo.is_memory(reg))
                        {
                            trace!("Optimizing reload to use {reg}");
                            stat!(stats, optimized_reload_to_move);
                            edit.from = Some(Allocation::reg(reg)).into();
                        }
                    }
                }
            }
        }
    }

    /// This does the same processing as `process_block`, but additionally tries
    /// to optimize instructions and edits in the block.
    fn optimize_block(
        &mut self,
        block: Block,
        allocations: &mut Allocations,
        move_resolver: &mut MoveResolver,
        stats: &mut Stats,
        func: &impl Function,
        reginfo: &impl RegInfo,
    ) {
        trace!("Optimizing {block}...");

        // Define blockparam values. The corresponding moves are in predecessor
        // blocks, but tagged with the outgoing value.
        for (value, alloc) in move_resolver.blockparam_allocs(block) {
            self.def_value(value, alloc, block, reginfo);
        }
        let mut edits = move_resolver.edits_from_mut(func.block_insts(block).from);

        for inst in func.block_insts(block).iter() {
            // Process and optimize any edits before the current instruction.
            while let Some(first) = edits.first_mut() {
                if first.0 > inst {
                    break;
                }
                trace!("Values: {self}");
                trace!("Optimizing edit: {}", first.1);
                self.optimize_edit(&mut first.1, block, stats, func, reginfo);
                self.process_edit(first.1, reginfo);
                edits = &mut edits[1..];
            }

            trace!("Values: {self}");
            trace!("Optimizing {inst}");

            // Process early def operands.
            self.def_units.clear();
            self.reused_operands.clear();
            for (&op, &alloc) in func
                .inst_operands(inst)
                .iter()
                .zip(allocations.inst_allocations(inst))
            {
                if let OperandConstraint::Reuse(idx) = op.constraint() {
                    self.reused_operands.push(idx);
                }
                match op.kind() {
                    OperandKind::EarlyDef(value) => {
                        self.def_value(value, alloc, block, reginfo);
                    }
                    OperandKind::EarlyDefGroup(value_group) => {
                        self.def_value_group(inst, op, value_group, alloc, block, func, reginfo);
                    }
                    _ => {}
                }
            }

            // Search for Use operands that have been assigned to memory and try
            // to replace them with a register that already contains the value.
            for (idx, (&op, alloc)) in func
                .inst_operands(inst)
                .iter()
                .zip(allocations.inst_allocations_mut(inst))
                .enumerate()
            {
                if let OperandKind::Use(value) = op.kind() {
                    if !alloc.is_memory(reginfo) {
                        continue;
                    }
                    if self.reused_operands.contains(&idx) {
                        continue;
                    }
                    if let OperandConstraint::Class(class) = op.constraint() {
                        if let Some(&regs_with_value) = self.value_regs.get(value) {
                            let class_members = reginfo.class_members(class);
                            if let Some(reg) = (class_members & regs_with_value)
                                .into_iter()
                                .find(|&reg| !reginfo.is_memory(reg))
                            {
                                stat!(stats, optimized_stack_use);
                                *alloc = Allocation::reg(reg);
                            }
                        }
                    }
                }
            }

            // Process normal def operands.
            for (&op, &alloc) in func
                .inst_operands(inst)
                .iter()
                .zip(allocations.inst_allocations(inst))
            {
                match op.kind() {
                    OperandKind::Def(value) => {
                        self.def_value(value, alloc, block, reginfo);
                    }
                    OperandKind::DefGroup(value_group) => {
                        self.def_value_group(inst, op, value_group, alloc, block, func, reginfo);
                    }
                    _ => {}
                }
            }

            // Process clobbers.
            for unit in func.inst_clobbers(inst) {
                if !self.def_units.contains(unit) {
                    self.clobber_unit(unit);
                }
            }
        }

        trace!("Values: {self}");
    }
}
