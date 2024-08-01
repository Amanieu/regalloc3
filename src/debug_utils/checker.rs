//! Checker which verifies that the output produced by the register allocator
//! actually matches the constraints in the input function.

use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use anyhow::{bail, ensure, Result};
use cranelift_entity::{EntityRef, SecondaryMap};
use hashbrown::hash_map::DefaultHashBuilder;
use hashbrown::{HashMap, HashSet};
use indexmap::IndexSet;
use smallvec::SmallVec;

use crate::allocation_unit::{AllocationUnit, AllocationUnitSet};
use crate::debug_utils::DisplayOutputInst;
use crate::function::{Block, Function, Inst, Operand, OperandConstraint, OperandKind, Value};
use crate::output::{Allocation, AllocationKind, Output, OutputInst, SpillSlot};
use crate::reginfo::{PhysReg, RegBank, RegClass, RegGroup, RegInfo, RegOrRegGroup, RegUnitSet};

/// Type representing a set of values. A `SmallVec` is used instead of a
/// `HashSet` for efficiency since sets tend to be small (1-2 elements).
type CheckerValueSet = SmallVec<[Value; 3]>;

/// The state represents, for each `RegUnit` or `SpillSlot`, the set of values
/// that the unit could have.
///
/// In straight-line code, this set only has at most 1 value which is the one
/// set by a `Def` operand or copied by a move/rematerialization.
///
/// However at control flow join points we may end up with different values in a
/// unit from each of the predecessor blocks. We handle this using the lattice
/// meet operation from the [regalloc2 checker]: for each unit, we only keep the
/// intersection of the sets of possible values from each predecessor.
///
/// The final tricky part is handling block parameters which take different
/// incoming values from each predecessor. This is handled by, at the end of
/// each predecessor block, adding the destination block parameter value to the
/// value set of each unit that already contains the source value for the block
/// parameter. This will result in the lattice join operation discarding the
/// conflicting source values and only keeping the destination value.
///
/// [regalloc2 checker]: https://github.com/bytecodealliance/regalloc2/blob/main/src/checker.rs
#[derive(Debug, Clone, Default)]
struct CheckerState {
    /// This is internally represented by a set of values for each
    /// `AllocationUnit`.
    ///
    /// A vacant entry in the `HashMap` is logically equivalent to a present
    /// entry with an empty value set.
    unit_values: HashMap<AllocationUnit, CheckerValueSet>,
}

impl CheckerState {
    /// Lattice meet operation when merging initial block states from multiple
    /// predecessors. This only keeps the common subset of values between the
    /// two incoming states.
    ///
    /// This also records whether any changes occurs, which helps determine when
    /// a fixed point has been reached.
    fn meet(&mut self, other: &Self, changed: &mut bool) {
        self.unit_values.retain(|&unit, values| {
            if let Some(other_values) = other.unit_values.get(&unit) {
                values.retain(|&mut value| {
                    if !other_values.contains(&value) {
                        *changed = true;
                        false
                    } else {
                        true
                    }
                });
                !values.is_empty()
            } else {
                *changed = !values.is_empty();
                false
            }
        });
    }

    /// Removes the given value from all `AllocationUnit`s.
    ///
    /// This is used at value definition points to remove any stale versions of
    /// a value (e.g. from a previous loop iteration).
    fn remove_value(&mut self, value: Value) {
        for values in self.unit_values.values_mut() {
            if let Some(idx) = values.iter().position(|&v| v == value) {
                values.remove(idx);
            }
        }
    }

    /// Adds the given value to the set of values for an `AllocationUnit`.
    fn add_value(&mut self, unit: AllocationUnit, value: Value) {
        let values = self.unit_values.entry(unit).or_default();
        values.push(value);
    }

    /// Sets the `AllocationUnit` to contain only the given value.
    fn set_value(&mut self, unit: AllocationUnit, value: Value) {
        let values = self.unit_values.entry(unit).or_default();
        values.clear();
        values.push(value);
    }

    /// Checks whether an `AllocationUnit` contains the given value.
    fn unit_contains(&self, unit: AllocationUnit, value: Value) -> bool {
        if let Some(values) = self.unit_values.get(&unit) {
            values.contains(&value)
        } else {
            false
        }
    }

    /// Returns an iterator over all the values in an `AllocationUnit`.
    fn unit_values(&mut self, unit: AllocationUnit) -> impl Iterator<Item = Value> + '_ {
        self.unit_values.entry(unit).or_default().iter().copied()
    }

    /// In all units matching the given filter, removes all values for which
    /// `keep_value` returns `false`.
    fn retain_in_units(
        &mut self,
        mut unit_filter: impl FnMut(AllocationUnit) -> bool,
        mut keep_value: impl FnMut(Value) -> bool,
    ) {
        for (&unit, values) in &mut self.unit_values {
            if !unit_filter(unit) {
                continue;
            }
            values.retain(|&mut value| keep_value(value));
        }
    }

    /// Returns an iterator over all `AllocationUnit`s containing the given
    /// value.
    fn units_containing_value(&self, value: Value) -> impl Iterator<Item = AllocationUnit> + '_ {
        self.unit_values
            .iter()
            .filter(move |(_, values)| values.contains(&value))
            .map(|(&unit, _)| unit)
    }

    /// Updates the state of an `AllocationUnit` to indicate it no longer holds
    /// a value.
    fn clobber_unit(&mut self, unit: AllocationUnit) {
        self.unit_values.remove(&unit);
    }

    /// Saves the set of values held in an `AllocationUnit` in a way that can be
    /// losslessly restored later.
    fn save_values(&self, unit: AllocationUnit) -> CheckerValueSet {
        self.unit_values.get(&unit).cloned().unwrap_or_default()
    }

    /// Restores the set of values held in an `AllocationUnit` that were
    /// previously saved by `save_values`.
    fn restore_values(&mut self, unit: AllocationUnit, values: &CheckerValueSet) {
        self.unit_values.entry(unit).or_default().clone_from(values);
    }
}

impl fmt::Display for CheckerState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut unit_values: Vec<_> = self.unit_values.iter().collect();
        unit_values.sort_unstable_by_key(|&(&unit, _)| unit);
        for (&unit, values) in &unit_values {
            write!(f, "{unit}: {values:?} ")?;
        }
        Ok(())
    }
}

/// Process instruction operands in 3 separate passes to properly model their
/// effects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pass {
    EarlyDef,
    Use,
    Def,
}

/// Saved state for a register that has been temporarily evicted to an
/// emergency spill slot.
struct EvictedReg {
    reg: PhysReg,
    saved_values: SmallVec<[CheckerValueSet; 2]>,
}

struct Context<'a, F, R> {
    output: &'a Output<'a, F, R>,
    blocks_to_check: IndexSet<Block, DefaultHashBuilder>,
    block_entry_state: SecondaryMap<Block, Option<CheckerState>>,
    state: CheckerState,
    evicted: HashMap<SpillSlot, EvictedReg>,
    blockparams_to_insert: Vec<(AllocationUnit, Value)>,
    def_units: AllocationUnitSet,
    fixed_def_units: RegUnitSet,
    early_reused_operands: Vec<usize>,
    next_inst: Inst,
    stack_map_units: AllocationUnitSet,
    stack_map_values: HashSet<Value>,
    terminated: bool,
    can_have_move: bool,
}

impl<F: Function, R: RegInfo> Context<'_, F, R> {
    /// Top-level function for the checker.
    fn check_function(&mut self) -> Result<()> {
        trace!("Checking register allocator output...");

        self.check_stack()?;

        // Start with an empty state on the entry block and keep visiting blocks
        // as long as their initial state has changed.
        //
        // The lattice meet() function guarantees that the initial state of a
        // block can only get smaller so this will eventually settle to a fixed
        // point at which point checking is complete.
        self.blocks_to_check.clear();
        self.blocks_to_check.insert(Block::ENTRY_BLOCK);
        self.block_entry_state[Block::ENTRY_BLOCK] = Some(CheckerState::default());
        while let Some(block) = self.blocks_to_check.pop() {
            self.check_block(block)?;
        }

        trace!("Check complete");

        Ok(())
    }

    /// Checks the stack layout and spill slot definitions.
    fn check_stack(&self) -> Result<()> {
        let stack_layout = self.output.stack_layout();
        for slot in stack_layout.spillslots() {
            let size = stack_layout.spillslot_size(slot);
            let offset = stack_layout.spillslot_offset(slot);
            ensure!(
                offset % size.bytes() == 0,
                "{slot} offset {offset} is not aligned to size {size}"
            );
            let end = offset + size.bytes();
            ensure!(
                end <= stack_layout.spillslot_area_size(),
                "{slot} ends at {end} which exceeds spill area size {}",
                stack_layout.spillslot_area_size()
            );
        }
        Ok(())
    }

    /// Checks the allocation result for a block and then propagates the end
    /// state to any successor blocks.
    fn check_block(&mut self, block: Block) -> Result<()> {
        let func = self.output.function();

        // Set up the initial state for the block. We don't preserve information
        // about evicted registers across blocks: these should only be local to
        // a single parallel move anyways.
        self.evicted.clear();
        self.state = self.block_entry_state[block].clone().unwrap();
        self.next_inst = func.block_insts(block).from;
        self.terminated = false;
        self.can_have_move = func.block_preds(block).len() == 1;
        trace!("Checking {block}...");
        trace!("Values: {}", self.state);

        // Update the state based on the instructions in the block.
        for inst in self.output.output_insts(block) {
            self.check_inst(inst, block)?;
            trace!("Values: {}", self.state);
        }
        self.check_skipped_inst(func.block_insts(block).to)?;
        debug_assert!(self.terminated);

        // If there are any outgoing block parameters, add the corresponding
        // incoming block parameter value to any unit that holds the outgoing
        // value at the end of the block.
        if let &[succ] = func.block_succs(block) {
            // Find units which hold a value that is an outoing block parameter.
            self.blockparams_to_insert.clear();
            for (&blockparam_out, &blockparam_in) in self
                .output
                .function()
                .jump_blockparams(block)
                .iter()
                .zip(func.block_params(succ))
            {
                self.blockparams_to_insert.extend(
                    self.state
                        .units_containing_value(blockparam_out)
                        .map(|unit| (unit, blockparam_in)),
                );
            }

            // Remove any stale block parameter values from previous loop
            // iterations.
            for &blockparam_in in func.block_params(succ) {
                self.state.remove_value(blockparam_in);
            }

            // Then add the incoming block parameters to units which held the
            // corresponding outgoing value.
            for &(unit, blockparam_in) in &self.blockparams_to_insert {
                self.state.add_value(unit, blockparam_in);
            }

            trace!("Values after blockparams: {}", self.state);
        }

        // Propagate the end state to successor blocks. If a successor has not
        // been visited yet or the merge changes their state, queue up that
        // block for another pass.
        for &succ in func.block_succs(block) {
            let mut changed = false;
            match self.block_entry_state[succ] {
                Some(ref mut succ_state) => succ_state.meet(&self.state, &mut changed),
                None => {
                    changed = true;
                    self.block_entry_state[succ] = Some(self.state.clone());
                }
            }
            if changed {
                trace!("Propagation changed state in {succ}");
                self.blocks_to_check.insert(succ);
            }
        }

        Ok(())
    }

    /// Checks an `OutputInst` and updates the checker state to reflect that
    /// instruction.
    fn check_inst(&mut self, inst: OutputInst<'_>, block: Block) -> Result<()> {
        let func = self.output.function();
        let reginfo = self.output.reginfo();

        trace!(
            "{}",
            DisplayOutputInst {
                inst,
                block,
                output: self.output
            }
        );

        ensure!(!self.terminated, "Output instruction after terminator");

        // Moves before the first instruction in a block are not allowed in
        // blocks with multiple (or no) predecessors, except if the first
        // instruction of that block has "use" operands or is a terminator.
        if !self.can_have_move {
            let has_use =
                func.inst_operands(func.block_insts(block).from)
                    .iter()
                    .any(|op| match op.kind() {
                        OperandKind::Use(_) | OperandKind::UseGroup(_) => true,
                        OperandKind::Def(_)
                        | OperandKind::EarlyDef(_)
                        | OperandKind::DefGroup(_)
                        | OperandKind::EarlyDefGroup(_)
                        | OperandKind::NonAllocatable => false,
                    });
            if func.block_insts(block).len() == 1 || has_use {
                self.can_have_move = true;
            }
        }
        match inst {
            OutputInst::Inst { .. } => self.can_have_move = true,
            OutputInst::Rematerialize { .. } => ensure!(
                self.can_have_move,
                "Cannot have remat before first instruction"
            ),
            OutputInst::Move { .. } => ensure!(
                self.can_have_move,
                "Cannot have move before first instruction"
            ),
        }

        match inst {
            OutputInst::Inst {
                inst,
                operand_allocs,
                stack_map,
            } => {
                self.check_skipped_inst(inst)?;

                if func.inst_is_terminator(inst) {
                    self.terminated = true;
                }

                let operands = func.inst_operands(inst);
                ensure!(
                    operands.len() == operand_allocs.len(),
                    "{inst}: expected {} allocations, got {}",
                    operands.len(),
                    operand_allocs.len()
                );

                // Process instruction operands in order. We also track which
                // units are written to detect conflicting outputs.
                self.def_units.clear();
                self.fixed_def_units.clear();
                self.early_reused_operands.clear();
                for pass in [Pass::EarlyDef, Pass::Use, Pass::Def] {
                    for (idx, (&op, &alloc)) in operands.iter().zip(operand_allocs).enumerate() {
                        self.check_operand(pass, inst, idx, op, alloc, operand_allocs)?;
                    }

                    if pass == Pass::Use && func.safepoint_insts().binary_search(&inst).is_ok() {
                        self.check_stack_map(inst, stack_map)?;
                    }
                }

                // Ensure that all live reftype values are covered by the stack
                // map.
                if func.safepoint_insts().binary_search(&inst).is_ok() {
                    self.clean_stale_reftype_values();
                }

                // Clear any clobbers, except when the corresponding unit has
                // been written to by a fixed def.
                for &clobber in func.inst_clobbers(inst) {
                    if !self.fixed_def_units.contains(clobber) {
                        let unit = AllocationUnit::Reg(clobber);
                        ensure!(
                            !self.def_units.contains(unit),
                            "Def operand conflicts with clobber {unit}"
                        );
                        self.state.clobber_unit(unit);
                    }
                }
            }
            OutputInst::Rematerialize { value, to } => {
                let Some((_cost, class)) = func.can_rematerialize(value) else {
                    bail!("{value} is not rematerializable");
                };
                self.check_class(to, class)?;

                for unit in to.units(reginfo) {
                    self.state.set_value(unit, value);
                }
                if let AllocationKind::SpillSlot(slot) = to.kind() {
                    self.evicted.remove(&slot);
                }
            }
            OutputInst::Move { from, to, value } => {
                if let Some(value) = value {
                    let bank = func.value_bank(value);
                    self.check_bank(from, bank)?;
                    self.check_bank(to, bank)?;
                    ensure!(
                        !from.is_memory(reginfo) || !to.is_memory(reginfo),
                        "Stack to stack move between {from} and {to}"
                    );
                    for unit in from.units(reginfo) {
                        ensure!(
                            self.state.unit_contains(unit, value),
                            "{unit} in {from} does not contain {value}"
                        );
                    }
                    for unit in to.units(reginfo) {
                        self.state.set_value(unit, value);
                    }
                    if let AllocationKind::SpillSlot(slot) = to.kind() {
                        self.evicted.remove(&slot);
                    }
                } else {
                    // Register evictions (used when the parallel move resolver
                    // can't find a scratch register) requires special handling.
                    //
                    // The problem is that it creates moves that don't carry a
                    // value and instead blindly preserve what is already in a
                    // register. This may cover several units each of which
                    // must be preserved.
                    //
                    // Thankfully, they follow a very predictable pattern and
                    // are local to a single parallel move. This means we can
                    // handle them by preserving the value sets for each
                    // affected unit in a side table and then restoring them
                    // later.
                    match (from.kind(), to.kind()) {
                        (AllocationKind::PhysReg(reg), AllocationKind::SpillSlot(slot))
                        | (AllocationKind::SpillSlot(slot), AllocationKind::PhysReg(reg)) => {
                            let Some(bank) = reginfo.bank_for_reg(reg) else {
                                bail!("{reg} is non-allocatable")
                            };
                            ensure!(
                                !self.output.reginfo().is_memory(reg),
                                "Stack to stack move between {from} and {to}"
                            );
                            ensure!(
                                reginfo.spillslot_size(bank)
                                    == self.output.stack_layout().spillslot_size(slot),
                                "{slot} has wrong size for {bank}: expected {}, got {}",
                                reginfo.spillslot_size(bank),
                                self.output.stack_layout().spillslot_size(slot)
                            );
                            match to.kind() {
                                AllocationKind::PhysReg(_) => {
                                    // Restore a previously evicted register.
                                    let Some(evicted) = self.evicted.get(&slot) else {
                                        bail!(
                                            "Emergency eviction slot {slot} doesn't contain an \
                                             evicted register"
                                        );
                                    };
                                    ensure!(
                                        reg == evicted.reg,
                                        "Emergency eviction slot {slot} restored to different \
                                         register: expected {}, got {reg}",
                                        evicted.reg
                                    );
                                    for (unit, values) in
                                        to.units(reginfo).zip(&evicted.saved_values)
                                    {
                                        self.state.restore_values(unit, values);
                                    }
                                }
                                AllocationKind::SpillSlot(_) => {
                                    // Save the state of a register to an
                                    // emergency spill slot.
                                    let saved_values = from
                                        .units(reginfo)
                                        .map(|unit| self.state.save_values(unit))
                                        .collect();
                                    self.evicted.insert(slot, EvictedReg { reg, saved_values });
                                    self.state.clobber_unit(AllocationUnit::Slot(slot));
                                }
                            }
                        }
                        (AllocationKind::SpillSlot(_), AllocationKind::SpillSlot(_))
                        | (AllocationKind::PhysReg(_), AllocationKind::PhysReg(_)) => {
                            bail!("Emergency eviction must be between a register and a spill slot");
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Checks that no instructions were skipped and that instructions are
    /// processed in order.
    ///
    /// An exception is made for pure instructions which are allowed to be
    /// eliminated if their outputs are unused.
    fn check_skipped_inst(&mut self, current_inst: Inst) -> Result<()> {
        ensure!(
            self.next_inst <= current_inst,
            "Output instructions in wrong order: expected {}, got {current_inst}",
            self.next_inst
        );
        while self.next_inst != current_inst {
            ensure!(
                self.output
                    .function()
                    .can_eliminate_dead_inst(self.next_inst),
                "Output skipped non-pure {}",
                self.next_inst
            );
            self.next_inst = self.next_inst.next();
        }
        self.next_inst = current_inst.next();
        Ok(())
    }

    /// Checks that the allocation can hold values of the given bank.
    fn check_bank(&self, alloc: Allocation, bank: RegBank) -> Result<()> {
        let reginfo = self.output.reginfo();
        match alloc.kind() {
            AllocationKind::PhysReg(reg) => ensure!(
                reginfo.bank_for_reg(reg) == Some(bank),
                "{bank} doesn't contain {reg}"
            ),
            AllocationKind::SpillSlot(slot) => {
                ensure!(
                    reginfo.spillslot_size(bank) == self.output.stack_layout().spillslot_size(slot),
                    "{slot} has wrong size for {bank}: expected {}, got {}",
                    reginfo.spillslot_size(bank),
                    self.output.stack_layout().spillslot_size(slot)
                );
            }
        }
        Ok(())
    }

    /// Checks that the allocation is compatible with the given register class.
    fn check_class(&self, alloc: Allocation, class: RegClass) -> Result<()> {
        let reginfo = self.output.reginfo();
        match alloc.kind() {
            AllocationKind::PhysReg(reg) => ensure!(
                reginfo
                    .class_members(class)
                    .contains(RegOrRegGroup::single(reg)),
                "{class} doesn't contain {reg}"
            ),
            AllocationKind::SpillSlot(slot) => {
                ensure!(
                    reginfo.class_includes_spillslots(class),
                    "{class} doesn't allow spillslots"
                );
                let bank = reginfo.bank_for_class(class);
                ensure!(
                    reginfo.spillslot_size(bank) == self.output.stack_layout().spillslot_size(slot),
                    "{slot} has wrong size for {bank}: expected {}, got {}",
                    reginfo.spillslot_size(bank),
                    self.output.stack_layout().spillslot_size(slot)
                );
            }
        }
        Ok(())
    }

    /// Checks that the allocation is the first member of a group in the given
    /// register class, and returns the register group it is part of.
    fn check_group_class(&self, alloc: Allocation, class: RegClass) -> Result<RegGroup> {
        let reginfo = self.output.reginfo();
        match alloc.kind() {
            AllocationKind::PhysReg(reg) => {
                let Some(group) = reginfo.group_for_reg(reg, 0, class) else {
                    bail!("{reg} is not the first member of a group in {class}");
                };
                Ok(group)
            }
            AllocationKind::SpillSlot(slot) => {
                bail!("Spillslot {slot} cannot be used in register group");
            }
        }
    }

    /// Checks the allocation assigned to an instruction operand.
    ///
    /// Also updates the checker state for def operands.
    ///
    /// `pass` indicates which set of operands to process. Operands for a
    /// different pass are ignored.
    fn check_operand(
        &mut self,
        pass: Pass,
        inst: Inst,
        idx: usize,
        op: Operand,
        alloc: Allocation,
        operand_allocs: &[Allocation],
    ) -> Result<()> {
        let func = self.output.function();
        let reginfo = self.output.reginfo();

        // We need to process operands in a specific order: EarlyDef -> Use -> Def
        let expected_pass = match op.kind() {
            OperandKind::Def(_) | OperandKind::DefGroup(_) => Pass::Def,
            OperandKind::Use(_) | OperandKind::UseGroup(_) => Pass::Use,
            OperandKind::EarlyDef(_) | OperandKind::EarlyDefGroup(_) => Pass::EarlyDef,
            // It doesn't matter which pass we process these in, just pick one.
            OperandKind::NonAllocatable => Pass::EarlyDef,
        };
        if pass != expected_pass {
            return Ok(());
        }

        // If this is an early def operand that reuses an input, process the
        // input operand first before we overwrite its value.
        if let (
            OperandKind::EarlyDef(_) | OperandKind::EarlyDefGroup(_),
            OperandConstraint::Reuse(target),
        ) = (op.kind(), op.constraint())
        {
            self.check_operand(
                Pass::Use,
                inst,
                target,
                func.inst_operands(inst)[target],
                operand_allocs[target],
                operand_allocs,
            )?;
            self.early_reused_operands.push(target);
        }

        // Check that the selected allocation is suitable for the constraint.
        self.check_constraint(alloc, op, operand_allocs)?;

        // Skip inputs tied to EarlyDef Reuse operands that have already been
        // checked. The values in the registers have by this point already been
        // overwritten by the early def.
        if self.early_reused_operands.contains(&idx) {
            return Ok(());
        }

        // Check/update values in the state.
        match op.kind() {
            OperandKind::Def(value) | OperandKind::EarlyDef(value) => {
                self.state.remove_value(value);
                for unit in alloc.units(reginfo) {
                    ensure!(!self.def_units.contains(unit), "Conflicting def on {unit}");
                    self.def_units.insert(unit);
                    self.state.set_value(unit, value);
                }
                if let AllocationKind::SpillSlot(slot) = alloc.kind() {
                    self.evicted.remove(&slot);
                }
            }
            OperandKind::Use(value) => {
                for unit in alloc.units(reginfo) {
                    ensure!(
                        self.state.unit_contains(unit, value),
                        "{unit} in {alloc} does not contain {value}"
                    );
                }
            }
            OperandKind::DefGroup(value_group) | OperandKind::EarlyDefGroup(value_group) => {
                let class = match op.constraint() {
                    OperandConstraint::Class(class) => class,
                    OperandConstraint::Fixed(_) => unreachable!(),
                    OperandConstraint::Reuse(idx) => {
                        let OperandConstraint::Class(class) =
                            func.inst_operands(inst)[idx].constraint()
                        else {
                            unreachable!();
                        };
                        class
                    }
                };
                let reg_group = self.check_group_class(alloc, class)?;
                for (&value, &reg) in func
                    .value_group_members(value_group)
                    .iter()
                    .zip(reginfo.reg_group_members(reg_group))
                {
                    self.state.remove_value(value);
                    for unit in Allocation::reg(reg).units(reginfo) {
                        ensure!(!self.def_units.contains(unit), "Conflicting def on {unit}");
                        self.def_units.insert(unit);
                        self.state.set_value(unit, value);
                    }
                }
            }
            OperandKind::UseGroup(value_group) => {
                let OperandConstraint::Class(class) = op.constraint() else {
                    unreachable!();
                };
                let reg_group = self.check_group_class(alloc, class)?;
                for (&value, &reg) in func
                    .value_group_members(value_group)
                    .iter()
                    .zip(reginfo.reg_group_members(reg_group))
                {
                    for unit in Allocation::reg(reg).units(reginfo) {
                        ensure!(
                            self.state.unit_contains(unit, value),
                            "{unit} in {reg} does not contain {value}"
                        );
                    }
                }
            }
            OperandKind::NonAllocatable => {
                // Nothing to do, everything is handled by check_constraint.
            }
        }

        Ok(())
    }

    /// Checks that the allocation for an operand matches the operand
    /// constraints.
    fn check_constraint(
        &mut self,
        alloc: Allocation,
        op: Operand,
        operand_allocs: &[Allocation],
    ) -> Result<()> {
        match op.constraint() {
            OperandConstraint::Class(class) => {
                let is_group = match op.kind() {
                    OperandKind::Def(_)
                    | OperandKind::Use(_)
                    | OperandKind::EarlyDef(_)
                    | OperandKind::NonAllocatable => false,
                    OperandKind::DefGroup(_)
                    | OperandKind::UseGroup(_)
                    | OperandKind::EarlyDefGroup(_) => true,
                };
                if is_group {
                    self.check_group_class(alloc, class)?;
                } else {
                    self.check_class(alloc, class)?;
                }
            }
            OperandConstraint::Fixed(reg) => {
                ensure!(
                    alloc.kind() == AllocationKind::PhysReg(reg),
                    "Expected {reg} for fixed constraint, got {alloc}"
                );

                // Track fixed definitions so we ignore clobbers for those units.
                for &unit in self.output.reginfo().reg_units(reg) {
                    self.fixed_def_units.insert(unit);
                }
            }
            OperandConstraint::Reuse(idx) => {
                ensure!(
                    alloc == operand_allocs[idx],
                    "Expected reused allocation {}, got {alloc}",
                    operand_allocs[idx]
                );
            }
        }
        Ok(())
    }

    /// Checks the stack map entries in an instruction.
    fn check_stack_map(&mut self, inst: Inst, stack_map: &[Allocation]) -> Result<()> {
        let func = self.output.function();
        let reginfo = self.output.reginfo();

        self.stack_map_units.clear();
        for &alloc in stack_map {
            for unit in alloc.units(reginfo) {
                // Check that stack map entries don't overlap.
                ensure!(
                    !self.stack_map_units.contains(unit),
                    "{inst}: stack map entries overlap on {unit}"
                );
                self.stack_map_units.insert(unit);

                // All units in the stack map must hold a reftype value.
                let Some(value) = self
                    .state
                    .unit_values(unit)
                    .find(|&value| func.reftype_values().binary_search(&value).is_ok())
                else {
                    bail!("{alloc} ({unit}) does not hold a reftype value at safepoint {inst}");
                };

                // All stack map entries must be in reftype_class.
                let bank = func.value_bank(value);
                let Some(class) = reginfo.reftype_class(bank) else {
                    bail!("Stack map contains {value} in {bank} which has no reftype_class");
                };
                self.check_class(alloc, class)?;
            }
        }

        // Find all reftype values that are preserved in the stack map.
        self.stack_map_values.clear();
        for &alloc in stack_map {
            for unit in alloc.units(reginfo) {
                for value in self.state.unit_values(unit) {
                    trace!("in stack map: {value} {unit} {alloc}");
                    if func.reftype_values().binary_search(&value).is_ok() {
                        trace!("add to stack_map_values: {value} {unit} {alloc}");
                        self.stack_map_values.insert(value);
                    }
                }
            }
        }

        Ok(())
    }

    /// All live reftype values at this point must be in the stack map. To
    /// catch any cases where a live reftype value is missing from the stack
    /// map we simply remove all reftype values from allocations not in the
    /// stack map.
    fn clean_stale_reftype_values(&mut self) {
        self.state.retain_in_units(
            // Exclude units that were defined in the current instruction.
            |unit| !self.def_units.contains(unit),
            // Remove all reftype values that were not saved in the stack map.
            |value| {
                let retain = self
                    .output
                    .function()
                    .reftype_values()
                    .binary_search(&value)
                    .is_err()
                    || self.stack_map_values.contains(&value);
                if !retain {
                    trace!("Removing {value} from unit");
                }
                retain
            },
        );
    }
}

/// Verifies the output of the register allocator.
///
/// If this fails then it indicates a bug in the register allocator, assuming
/// the `Function` and `RegInfo` have passed validation.
pub fn check_output(output: &Output<'_, impl Function, impl RegInfo>) -> Result<()> {
    let mut context = Context {
        output,
        blocks_to_check: IndexSet::with_hasher(DefaultHashBuilder::default()),
        block_entry_state: SecondaryMap::new(),
        state: CheckerState::default(),
        evicted: HashMap::new(),
        blockparams_to_insert: vec![],
        next_inst: Inst::new(0),
        early_reused_operands: vec![],
        def_units: AllocationUnitSet::new(),
        fixed_def_units: RegUnitSet::new(),
        stack_map_units: AllocationUnitSet::new(),
        stack_map_values: HashSet::new(),
        terminated: false,
        can_have_move: false,
    };
    context.check_function()
}
