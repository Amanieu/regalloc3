//! Calculation of live ranges for SSA values.
//!
//! This takes place in 2 stages:
//! - First, we collect all of the places where a [`Value`] is used or defined
//!   and sort them into a single array (held in [`Uses`]). The sorting allows
//!   us to retrieve all of the locations where a value is used in a slice.
//! - Second, for each [`Value`] we perform liveness propagation to determine
//!   which blocks it is live-in and live-out of. We then use this to produce
//!   a set of continuous segments ([`ValueSegment`]) of a value's live range.
//!
//! The algorithm is inspired by the "var-by-var" algorithm in this paper:
//!
//! Brandner, Florian, et al. Computing Liveness Sets for SSA-Form Programs. Diss. INRIA, 2011.
//!
//! The main difference is that we only need to track the block liveness of on
//! value at a time so we can just use a simple bitset.

use alloc::vec;
use alloc::vec::Vec;
use core::ops::{Index, IndexMut};

use smallvec::SmallVec;

use super::allocations::Allocations;
use super::live_range::{LiveRangeSegment, Slot};
use super::reg_matrix::RegMatrix;
use super::uses::{UseIndex, UseKind, Uses};
use crate::entity::{EntitySet, SecondaryMap};
use crate::function::{
    Block, Function, Inst, Operand, OperandConstraint, OperandKind, Value, ValueGroup,
};
use crate::internal::uses::{UseList, UsePosition};
use crate::output::Allocation;
use crate::reginfo::{RegClass, RegInfo};
use crate::Stats;

/// A continuous segment of a value's live range.
#[derive(Debug, Clone, Copy)]
pub struct ValueSegment {
    /// Live range covered by this segment.
    pub live_range: LiveRangeSegment,

    /// Set of `Use`s associated with this live range segment.
    pub use_list: UseList,

    /// SSA value associated with this live range segment.
    ///
    /// Coalescing may produce virtual registers which cover multiple SSA values
    /// but each segment will only come from a single SSA value.
    pub value: Value,
}

entity_def! {
    /// A value set corresponds to a set of [`Value`]s whose live range do not
    /// overlap.
    pub entity ValueSet(u32, "valueset");
}

impl ValueSet {
    /// Returns the initial `ValueSet` for the given `Value`.
    ///
    /// This is only valid in this pass before coalescing.
    fn from_value(value: Value) -> Self {
        ValueSet::new(value.index())
    }
}

/// A value set is represented as the set of `ValueSegment`s of all its
/// constituent `Value`s.
#[derive(Default, Clone)]
struct ValueSetData {
    /// Sorted list of live range segments for this value set.
    ///
    /// The live ranges of these segments are guaranteed not to overlap.
    segments: SmallVec<[ValueSegment; 4]>,
}

/// Information about an input value to an instruction whose register is reused
/// by an output operand.
///
/// Where there are multiple inputs with the same value as a reused input, we
/// can treat all of them as reused inputs. This allows coalescing to succeed
/// where it would normally fail because the other inputs conflict with the
/// def (which has its start point extended to the previous instruction
/// boundary).
#[derive(Debug)]
struct ReusedValue {
    /// Input value that will be reused.
    use_value: Value,

    /// Operand slot for the def in the current instruction.
    def_slot: u16,

    /// Operand slot for the tied use in the current instruction.
    use_slot: u16,

    /// Register class that will be used for the reused operand.
    class: RegClass,

    /// If `class` is a group class, the index of the value in its group.
    group_index: Option<u8>,

    /// Whether this reuse is from an EarlyDef operand. Such operands cannot
    /// share the same register as any input other than the reused one.
    is_early_def: bool,
}

/// Value live range calculation pass.
pub struct ValueLiveRanges {
    /// `ValueSegment`s for each `ValueSet`.
    ///
    /// Initially there is a 1-to-1 mapping of `Value` to `ValueSet`, but then
    /// coalescing will merge sets together.
    ///
    /// Any sets with an empty segment list should be ignored.
    value_sets: SecondaryMap<ValueSet, ValueSetData>,

    /// The live range that a value definition covers.
    def_range: SecondaryMap<Value, LiveRangeSegment>,

    /// Set of blocks into which a value is known to be live-in, used by
    /// `build_segments`.
    live_in: EntitySet<Block>,

    /// Set of blocks into which a value is known to be live-out, used by
    /// `build_segments`.
    live_out: EntitySet<Block>,

    /// Stack used by `build_segments` when propagating live-in/live-out bits
    /// through blocks.
    worklist: Vec<Block>,

    /// Input values to an instruction whose register is reused by an output
    /// operand.
    reused_values: Vec<ReusedValue>,
}

impl Index<ValueSet> for ValueLiveRanges {
    type Output = SmallVec<[ValueSegment; 4]>;

    fn index(&self, set: ValueSet) -> &Self::Output {
        &self.value_sets[set].segments
    }
}

impl IndexMut<ValueSet> for ValueLiveRanges {
    fn index_mut(&mut self, set: ValueSet) -> &mut Self::Output {
        &mut self.value_sets[set].segments
    }
}

impl ValueLiveRanges {
    pub fn new() -> Self {
        Self {
            value_sets: SecondaryMap::new(),
            def_range: SecondaryMap::new(),
            live_in: EntitySet::new(),
            live_out: EntitySet::new(),
            worklist: vec![],
            reused_values: vec![],
        }
    }

    /// Returns all value sets and their associated `ValueSegment`s.
    pub fn all_value_sets(&self) -> impl Iterator<Item = (ValueSet, &[ValueSegment])> {
        self.value_sets
            .iter()
            .filter(|(_set, data)| !data.segments.is_empty())
            .map(|(set, data)| (set, &data.segments[..]))
    }

    /// Returns all value sets and their associated `ValueSegment`s.
    ///
    /// This gives clobbers the data, but it is only used by the virtual
    /// register builder. This is fine since the data is no longer used
    /// afterwards.
    pub fn take_all_value_sets(
        &mut self,
    ) -> impl Iterator<Item = (ValueSet, SmallVec<[ValueSegment; 4]>)> + '_ {
        self.value_sets
            .iter_mut()
            .filter(|(_set, data)| !data.segments.is_empty())
            .map(|(set, data)| (set, core::mem::take(&mut data.segments)))
    }

    /// Computes live ranges and uses for all SSA values in the function.
    pub fn compute(
        &mut self,
        uses: &mut Uses,
        allocations: &mut Allocations,
        reg_matrix: &mut RegMatrix,
        stats: &mut Stats,
        func: &impl Function,
        reginfo: &impl RegInfo,
    ) {
        uses.clear();
        reg_matrix.clear();
        self.def_range.clear_and_resize_with(func.num_values(), || {
            let zero_point = Inst::new(0).slot(Slot::Boundary);
            LiveRangeSegment::new(zero_point, zero_point)
        });
        self.value_sets.clear_and_resize(func.num_values());

        let mut ctx = Context {
            func,
            reginfo,
            uses,
            allocations,
            reg_matrix,
            stats,
            value_live_ranges: self,
        };

        // Compute live ranges for values and collect uses from instruction
        // operands.
        ctx.collect_uses();

        // All uses have been added at this point, we can sort the use vector.
        let mut use_list_end = ctx.uses.sort_uses();

        // Builds `ValueSegment`s for each value from the collected uses.
        for value in func.values() {
            ctx.build_segments(value, &mut use_list_end);
        }

        self.dump(uses);
    }

    /// Dumps the value live ranges to the log.
    pub fn dump(&self, uses: &Uses) {
        if !trace_enabled!() {
            return;
        }

        trace!("Value live ranges:");
        for (set, segments) in self.all_value_sets() {
            trace!("  {set}:");
            for segment in segments {
                trace!("    {}: {}", segment.value, segment.live_range);
                if segment.use_list.has_livein() {
                    trace!("    - livein");
                }
                for u in &uses[segment.use_list] {
                    trace!("    - {}: {}", u.pos(), u.kind);
                }
                if segment.use_list.has_liveout() {
                    trace!("    - liveout");
                }
            }
        }
    }
}

/// Context for value live range computation.
struct Context<'a, F, R> {
    func: &'a F,
    reginfo: &'a R,
    uses: &'a mut Uses,
    allocations: &'a mut Allocations,
    reg_matrix: &'a mut RegMatrix,
    stats: &'a mut Stats,
    value_live_ranges: &'a mut ValueLiveRanges,
}

impl<F: Function, R: RegInfo> Context<'_, F, R> {
    /// Iterate over all blocks and instructions to collect value uses.
    fn collect_uses(&mut self) {
        // Walk over all value uses and definitions.
        for block in self.func.blocks() {
            let block_insts = self.func.block_insts(block);

            // Create uses for incoming block parameters.
            for (idx, &value) in self.func.block_params(block).iter().enumerate() {
                trace!("Processing incoming blockparam {value} in {block}");
                stat!(self.stats, blockparam_in);
                self.value_def(
                    value,
                    block_insts.from,
                    LiveRangeSegment::new(
                        block_insts.from.slot(Slot::Boundary),
                        block_insts.from.slot(Slot::Boundary),
                    ),
                    UseKind::BlockparamIn {
                        blockparam_idx: idx as u32,
                    },
                );
            }

            // Create uses and definitions for all instruction operands.
            for inst in block_insts.iter() {
                // Do an initial scan to find reused operands.
                self.find_reused_values(inst);

                let operands = self.func.inst_operands(inst);
                for (slot, &operand) in operands.iter().enumerate() {
                    self.process_operand(inst, slot as u16, operand);
                }

                // Reserve fixed ranges for instruction clobbers.
                for &unit in self.func.inst_clobbers(inst) {
                    self.reg_matrix.reserve_clobber(
                        unit,
                        LiveRangeSegment::new(
                            inst.slot(Slot::Normal),
                            inst.next().slot(Slot::Boundary),
                        ),
                    );
                }
            }

            // Create uses for outgoing block parameters.
            if let &[_succ] = self.func.block_succs(block) {
                for &value in self.func.jump_blockparams(block) {
                    trace!("Processing outgoing blockparam {value} in {block}");
                    stat!(self.stats, blockparam_out);
                    self.value_use(value, block_insts.last(), UseKind::BlockparamOut {});
                }
            }
        }
    }

    /// Scans the operand list of an instruction to find input values whose
    /// register is reused for an output.
    fn find_reused_values(&mut self, inst: Inst) {
        self.value_live_ranges.reused_values.clear();
        let operands = self.func.inst_operands(inst);
        for (idx, &def_op) in operands.iter().enumerate() {
            if let OperandConstraint::Reuse(target) = def_op.constraint() {
                let is_early_def = matches!(
                    def_op.kind(),
                    OperandKind::EarlyDef(_) | OperandKind::EarlyDefGroup(_)
                );
                let use_op = operands[target];
                let OperandConstraint::Class(class) = use_op.constraint() else {
                    unreachable!();
                };
                match use_op.kind() {
                    OperandKind::Use(use_value) => {
                        trace!("Reused operand: {use_op} -> {def_op}");
                        self.value_live_ranges.reused_values.push(ReusedValue {
                            use_value,
                            def_slot: idx as u16,
                            use_slot: target as u16,
                            class,
                            group_index: None,
                            is_early_def,
                        });
                    }
                    OperandKind::UseGroup(use_value_group) => {
                        trace!("Reused group operand: {use_op} -> {def_op}");
                        for (group_idx, &use_value) in self
                            .func
                            .value_group_members(use_value_group)
                            .iter()
                            .enumerate()
                        {
                            self.value_live_ranges.reused_values.push(ReusedValue {
                                use_value,
                                def_slot: idx as u16,
                                use_slot: target as u16,
                                class,
                                group_index: Some(group_idx as u8),
                                is_early_def,
                            });
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    /// Check if the given value is reused by a def operand in the same
    /// instruction.
    ///
    /// This also checks if the class of the def operand is compatible with that
    /// of the current operand.
    ///
    /// If both are true then we can replace the use operand with the reused
    /// operand.
    fn try_reuse_value(
        &self,
        value: Value,
        use_slot: u16,
        class: RegClass,
        group_index: Option<u8>,
        def_slot: Option<u16>,
    ) -> Option<&ReusedValue> {
        let mut found = None;
        for reused in &self.value_live_ranges.reused_values {
            // If we are reusing values for other members of a group, this must
            // precisely match the reused slot of the first member.
            if let Some(def_slot) = def_slot {
                if reused.def_slot != def_slot {
                    continue;
                }
            }

            // The class for the use operand must be a strict subset of the
            // class that will be used for the reused operand. We also need the
            // group index to match exactly.
            if !(reused.use_value == value
                && self.reginfo.common_subclass(reused.class, class) == Some(reused.class)
                && reused.group_index == group_index)
            {
                continue;
            }

            // If we find an exact match for the desired use slot then we must
            // return it. This is necessary to ensure that the proper allocation
            // is copied to our slot.
            if reused.use_slot == use_slot && reused.group_index == group_index {
                return Some(reused);
            }

            // Otherwise this is a non-reused operand that we may still want to
            // allocate to the same register as a reused operand. This allows
            // for things like `add rax, rax` where the second input would
            // otherwise conflict with the reused operand.
            if !reused.is_early_def {
                found = Some(reused);
            }
        }

        if let Some(found) = found {
            if group_index.is_none() {
                trace!(
                    "-> tying {value} in slot {use_slot} to def in slot {}",
                    found.def_slot
                );
            }
        }
        found
    }

    /// For group uses, we can only fold them into a group reuse def if we do so
    /// for *all* members of the group.
    fn try_reuse_value_group(
        &self,
        value_group: ValueGroup,
        use_slot: u16,
        class: RegClass,
    ) -> Option<u16> {
        let members = self.func.value_group_members(value_group);
        let reused = self.try_reuse_value(members[0], use_slot, class, Some(0), None)?;
        for (idx, &value) in members.iter().enumerate().skip(1) {
            let reused_idx = self.try_reuse_value(
                value,
                use_slot,
                class,
                Some(idx as u8),
                Some(reused.def_slot),
            )?;
            if reused.def_slot != reused_idx.def_slot {
                trace!("-> cannot reuse def for {value_group}");
                return None;
            }
        }

        trace!(
            "-> tying {value_group} in slot {use_slot} to def in slot {}",
            reused.def_slot
        );
        Some(reused.def_slot)
    }

    /// For each instruction operand, either initialize a new live range for a
    /// definition or extend an existing live range for a use.
    ///
    /// For operands that are fixed to a particular register, we also:
    /// - Set the relevant allocation in the allocation map in advance.
    /// - Mark the fixed portion of the live range as reserved in the
    ///   `RegMatrix`.
    fn process_operand(&mut self, inst: Inst, slot: u16, operand: Operand) {
        trace!("Processing {inst}[{slot}]: {operand}");
        match (operand.kind(), operand.constraint()) {
            (OperandKind::Def(value), OperandConstraint::Class(class)) => {
                stat!(self.stats, class_def);
                self.value_def(
                    value,
                    inst,
                    LiveRangeSegment::new(
                        inst.slot(Slot::Normal),
                        inst.next().slot(Slot::Boundary),
                    ),
                    UseKind::ClassDef { slot, class },
                );
            }
            (OperandKind::Def(value), OperandConstraint::Fixed(reg)) => {
                stat!(self.stats, fixed_def);
                self.allocations
                    .set_allocation(inst, slot, Allocation::reg(reg));
                for &unit in self.reginfo.reg_units(reg) {
                    self.reg_matrix.reserve_fixed(
                        unit,
                        LiveRangeSegment::new(
                            inst.slot(Slot::Normal),
                            inst.next().slot(Slot::Boundary),
                        ),
                        value,
                    );
                }
                self.value_def(
                    value,
                    inst,
                    LiveRangeSegment::new(
                        inst.next().slot(Slot::Boundary),
                        inst.next().slot(Slot::Boundary),
                    ),
                    UseKind::FixedDef { reg },
                );
            }
            (OperandKind::EarlyDef(value), OperandConstraint::Class(class)) => {
                stat!(self.stats, class_def);
                self.value_def(
                    value,
                    inst,
                    LiveRangeSegment::new(inst.slot(Slot::Early), inst.next().slot(Slot::Boundary)),
                    UseKind::ClassDef { slot, class },
                );
            }
            (OperandKind::EarlyDef(value), OperandConstraint::Fixed(reg)) => {
                stat!(self.stats, fixed_def);
                self.allocations
                    .set_allocation(inst, slot, Allocation::reg(reg));
                for &unit in self.reginfo.reg_units(reg) {
                    self.reg_matrix.reserve_fixed(
                        unit,
                        LiveRangeSegment::new(
                            inst.slot(Slot::Early),
                            inst.next().slot(Slot::Boundary),
                        ),
                        value,
                    );
                }
                self.value_def(
                    value,
                    inst,
                    LiveRangeSegment::new(
                        inst.next().slot(Slot::Boundary),
                        inst.next().slot(Slot::Boundary),
                    ),
                    UseKind::FixedDef { reg },
                );
            }
            (OperandKind::Use(value), OperandConstraint::Class(class)) => {
                stat!(self.stats, class_use);
                if let Some(reused) = self.try_reuse_value(value, slot, class, None, None) {
                    self.value_use(
                        value,
                        inst,
                        UseKind::TiedUse {
                            use_slot: slot,
                            def_slot: reused.def_slot,
                            class,
                            group_index: 0,
                        },
                    );
                } else {
                    self.value_use(value, inst, UseKind::ClassUse { slot, class });
                }
            }
            (OperandKind::Use(value), OperandConstraint::Fixed(reg)) => {
                stat!(self.stats, fixed_use);
                self.allocations
                    .set_allocation(inst, slot, Allocation::reg(reg));
                for &unit in self.reginfo.reg_units(reg) {
                    self.reg_matrix.reserve_fixed(
                        unit,
                        LiveRangeSegment::new(inst.slot(Slot::Boundary), inst.slot(Slot::Normal)),
                        value,
                    );
                }
                self.value_use(value, inst, UseKind::FixedUse { reg });
            }
            (
                OperandKind::Def(value) | OperandKind::EarlyDef(value),
                OperandConstraint::Reuse(target),
            ) => {
                // At this point we just copy the constraint of the target
                // operand. Tying is handled by the use operands.
                stat!(self.stats, reuse_def);
                let target_operand = self.func.inst_operands(inst)[target];
                let OperandConstraint::Class(class) = target_operand.constraint() else {
                    unreachable!("Reuse target operand must have class constraint")
                };
                self.value_def(
                    value,
                    inst,
                    LiveRangeSegment::new(
                        inst.slot(Slot::Boundary),
                        inst.next().slot(Slot::Boundary),
                    ),
                    UseKind::ClassDef { slot, class },
                );
            }
            (OperandKind::DefGroup(group), OperandConstraint::Class(class)) => {
                stat!(self.stats, group_def);
                for (idx, &value) in self.func.value_group_members(group).iter().enumerate() {
                    let group_index = idx as u8;
                    self.value_def(
                        value,
                        inst,
                        LiveRangeSegment::new(
                            inst.slot(Slot::Normal),
                            inst.next().slot(Slot::Boundary),
                        ),
                        UseKind::GroupClassDef {
                            slot,
                            class,
                            group_index,
                        },
                    );
                }
            }
            (OperandKind::EarlyDefGroup(group), OperandConstraint::Class(class)) => {
                stat!(self.stats, group_def);
                for (idx, &value) in self.func.value_group_members(group).iter().enumerate() {
                    let group_index = idx as u8;
                    self.value_def(
                        value,
                        inst,
                        LiveRangeSegment::new(
                            inst.slot(Slot::Early),
                            inst.next().slot(Slot::Boundary),
                        ),
                        UseKind::GroupClassDef {
                            slot,
                            class,
                            group_index,
                        },
                    );
                }
            }
            (OperandKind::UseGroup(group), OperandConstraint::Class(class)) => {
                stat!(self.stats, group_use);
                let def_slot = self.try_reuse_value_group(group, slot, class);
                for (idx, &value) in self.func.value_group_members(group).iter().enumerate() {
                    if let Some(def_slot) = def_slot {
                        self.value_use(
                            value,
                            inst,
                            UseKind::TiedUse {
                                use_slot: slot,
                                def_slot,
                                class,
                                group_index: idx as u8,
                            },
                        );
                    } else {
                        self.value_use(
                            value,
                            inst,
                            UseKind::GroupClassUse {
                                slot,
                                class,
                                group_index: idx as u8,
                            },
                        );
                    }
                }
            }
            (
                OperandKind::DefGroup(group) | OperandKind::EarlyDefGroup(group),
                OperandConstraint::Reuse(target),
            ) => {
                // At this point we just copy the constraint of the target
                // operand. Tying is handled by the use operands.
                stat!(self.stats, reuse_group_def);
                let target_operand = self.func.inst_operands(inst)[target];
                let OperandConstraint::Class(class) = target_operand.constraint() else {
                    unreachable!("Reuse target operand must have class constraint")
                };
                for (idx, &value) in self.func.value_group_members(group).iter().enumerate() {
                    let group_index = idx as u8;
                    self.value_def(
                        value,
                        inst,
                        LiveRangeSegment::new(
                            inst.slot(Slot::Boundary),
                            inst.next().slot(Slot::Boundary),
                        ),
                        UseKind::GroupClassDef {
                            slot,
                            class,
                            group_index,
                        },
                    );
                }
            }
            (OperandKind::NonAllocatable, OperandConstraint::Fixed(reg)) => {
                stat!(self.stats, nonallocatable_operand);
                self.allocations
                    .set_allocation(inst, slot, Allocation::reg(reg));
            }

            // Invalid kind/constraint combinations.
            (OperandKind::Use(_) | OperandKind::UseGroup(_), OperandConstraint::Reuse(_)) => {
                unreachable!()
            }
            (
                OperandKind::UseGroup(_) | OperandKind::DefGroup(_) | OperandKind::EarlyDefGroup(_),
                OperandConstraint::Fixed(_),
            ) => unreachable!(),
            (
                OperandKind::NonAllocatable,
                OperandConstraint::Class(_) | OperandConstraint::Reuse(_),
            ) => unreachable!(),
        }
    }

    /// Visits the definition of a value.
    fn value_def(
        &mut self,
        value: Value,
        inst: Inst,
        live_range: LiveRangeSegment,
        use_kind: UseKind,
    ) {
        trace!("{value} definition at {live_range} ({inst}) with {use_kind}");

        // Save the range of the defining instruction. This is needed when
        // generating live ranges since `ClassDef` doesn't record at which
        // point the def started.
        self.value_live_ranges.def_range[value] = live_range;

        // Add the use to the list of uses. This will later be sorted by value
        // and position so that we can get a linear range of all uses for each
        // live range segment.
        self.uses
            .add_unsorted_use(UsePosition::at_def(inst), value, use_kind);
    }

    /// Visits a use of a value.
    fn value_use(&mut self, value: Value, inst: Inst, use_kind: UseKind) {
        trace!("{value} use at {inst} with {use_kind}");

        self.uses
            .add_unsorted_use(UsePosition::at_use(inst), value, use_kind);
    }

    /// Calculates the live-in/live-out bitsets for each block of the value's
    /// live range.
    ///
    /// This returns the highest numbered block found by the search.
    fn calc_block_live_in_out(&mut self, use_list: UseList, def_block: Block) -> Block {
        // Mark the definition block as live-in so that the search stops there.
        // The live-in bit on the definition block isn't read later so it
        // doesn't matter if it is incorrect.
        self.value_live_ranges.live_in.insert(def_block);
        trace!("Defined in {def_block}");

        // For all other uses, propagate liveness up through predecessor blocks
        // until a block that is already live-in is reached.
        let mut last_block = def_block;
        for &u in self.uses[use_list].iter().skip(1).rev() {
            let block = self.func.inst_block(u.pos());
            if !self.value_live_ranges.live_in.contains(block) {
                self.value_live_ranges.worklist.push(block);
                while let Some(block) = self.value_live_ranges.worklist.pop() {
                    if self.value_live_ranges.live_in.contains(block) {
                        continue;
                    }

                    self.value_live_ranges.live_in.insert(block);

                    if block > last_block {
                        last_block = block;
                    }

                    for &pred in self.func.block_preds(block) {
                        if !self.value_live_ranges.live_out.contains(pred) {
                            self.value_live_ranges.live_out.insert(pred);
                            self.value_live_ranges.worklist.push(pred);
                        }
                    }
                }
            }
        }

        if last_block == def_block {
            stat!(self.stats, local_values);
        } else {
            stat!(self.stats, global_values);
        }

        last_block
    }

    /// Computes the live range for the given value.
    fn build_segments(&mut self, value: Value, prev_use_list_end: &mut UseIndex) {
        trace!("Building live range segments for {value}");

        self.value_live_ranges
            .live_in
            .clear_and_resize(self.func.num_blocks());
        self.value_live_ranges
            .live_out
            .clear_and_resize(self.func.num_blocks());

        // Get the sorted list of all uses for this value.
        let (full_use_list, use_list_end) = self.uses.resolve_use_list(value, *prev_use_list_end);
        *prev_use_list_end = use_list_end;

        // The first use in the list is always the one which defines the value.
        let def = &self.uses[full_use_list][0];
        debug_assert!(def.is_def());
        let def_block = self.func.inst_block(def.pos());

        // Calculate the set of blocks in which the value is live-in or
        // live-out.
        let last_block = self.calc_block_live_in_out(full_use_list, def_block);

        // Start with an initial segment containing just the definition.
        let mut segment = ValueSegment {
            live_range: self.value_live_ranges.def_range[value],
            use_list: full_use_list,
            value,
        };
        let segments = &mut self.value_live_ranges.value_sets[ValueSet::from_value(value)].segments;
        debug_assert!(segments.is_empty());

        // Iterate over all uses and try to add them to the segment. Split up
        // segments where there is a gap in the live range.
        let mut next_dead_block = self
            .value_live_ranges
            .live_in
            .next_absent_from(def_block.next());
        for use_idx in full_use_list.iter().skip(1) {
            let u = self.uses[use_idx];
            debug_assert!(u.is_use());
            let use_block = self.func.inst_block(u.pos());
            debug_assert!(use_block <= last_block);

            // If there is a live range gap until this use then we need to split
            // the segment.
            while use_block > next_dead_block {
                // If we encounter a dead block then we need to end the current
                // segment. Any uses up to that point are added to the segment.
                let (prev_list, next_list) = segment.use_list.split_at_index(use_idx);
                segment.use_list = prev_list;

                // If the last live block is live-out then extend the segment
                // to the end of the block and mark it as a live-out.
                let end_block = next_dead_block.prev();
                if self.value_live_ranges.live_out.contains(end_block) {
                    segment.live_range.to =
                        self.func.block_insts(end_block).to.slot(Slot::Boundary);
                    segment.use_list.set_liveout(true);
                }

                // Add the segment to the list.
                segments.push(segment);

                // Skip over any dead blocks to find the next live block.
                let next_live_block = self
                    .value_live_ranges
                    .live_in
                    .next_present_from(next_dead_block.next())
                    .unwrap();
                debug_assert!(next_live_block <= use_block);
                debug_assert!(next_live_block <= last_block);

                // Then find the next dead block.
                next_dead_block = self
                    .value_live_ranges
                    .live_in
                    .next_absent_from(next_live_block.next());

                // Start a new segment at the beginning of the next live block,
                // marking it as having a live-in value.
                segment.live_range.from = self
                    .func
                    .block_insts(next_live_block)
                    .from
                    .slot(Slot::Boundary);
                segment.use_list = next_list;
                segment.use_list.set_livein(true);
            }

            // Extend the segment to the point of the next use.
            let use_point = u.end_point();
            if use_point > segment.live_range.to {
                segment.live_range.to = use_point;
            }
        }

        // Handle any remaining live range after the last use.
        loop {
            // If the last live block is live-out then extend the segment
            // to the end of the block and mark it as a live-out.
            let end_block = next_dead_block.prev();
            if self.value_live_ranges.live_out.contains(end_block) {
                segment.live_range.to = self.func.block_insts(end_block).to.slot(Slot::Boundary);
                segment.use_list.set_liveout(true);
            }

            // Add the segment to the list.
            segments.push(segment);

            // If this was the last block, we're done.
            if end_block == last_block {
                break;
            }

            // Skip over any dead blocks to find the next live block.
            let next_live_block = self
                .value_live_ranges
                .live_in
                .next_present_from(next_dead_block.next())
                .unwrap();
            debug_assert!(next_live_block <= last_block);

            // Then find the next dead block.
            next_dead_block = self
                .value_live_ranges
                .live_in
                .next_absent_from(next_live_block.next());

            // Start a new segment at the beginning of the next live block,
            // marking it as having a live-in value.
            segment.live_range.from = self
                .func
                .block_insts(next_live_block)
                .from
                .slot(Slot::Boundary);
            segment.use_list = UseList::empty();
            segment.use_list.set_livein(true);
        }

        stat!(self.stats, value_segments, segments.len());
    }
}
