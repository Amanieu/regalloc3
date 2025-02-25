//! Virtual registers produced from SSA values and later through coalescing may
//! have conflicting constraints. This pass splits these virtual registers
//! until each new virtual register has a valid constraint.
//!
//! Conflicting constraints can come from 2 sources:
//! - If there are 2 register class uses which don't share a common sub-class.
//!   For example if one only accepts stack locations but the other requires a
//!   register.
//! - If there are 2 constraints for register group classes where the virtual
//!   register has a different index within the group. Register groups can only
//!   be merged if there is a precise match between element indices and group
//!   size (the group size is already checked at validation time).
//!
//! The normal way to handle conflicts is to find a split point between the two
//! conflicting uses and split the virtual register at that point. However this
//! is not possible when the conflicts happen in the same instruction. Such
//! cases need to be handled specially by "deleting" just the conflicting
//! constraint from the use list and moving it to a separate virtual register.

use alloc::vec;
use alloc::vec::Vec;
use core::array;
use core::cmp::Ordering;

use crate::Stats;
use crate::entity::packed_option::{PackedOption, ReservedValue};
use crate::entity::{CompactList, SecondaryMap};
use crate::function::{Function, OperandKind, Value, ValueGroup};
use crate::internal::coalescing::Coalescing;
use crate::internal::hints::Hints;
use crate::internal::live_range::{LiveRangeSegment, Slot, ValueSegment};
use crate::internal::split_placement::SplitPlacement;
use crate::internal::uses::{Use, UseIndex, UseKind, Uses};
use crate::internal::value_live_ranges::ValueSet;
use crate::internal::virt_regs::{VirtReg, VirtRegData, VirtRegGroup, VirtRegs};
use crate::reginfo::{MAX_GROUP_SIZE, RegBank, RegClass, RegInfo};

/// Utility type for building a virtual register from value live ranges.
pub struct VirtRegBuilder {
    /// Mapping of [`ValueGroup`] to [`VirtRegGroup`].
    value_group_mapping: SecondaryMap<ValueGroup, PackedOption<VirtRegGroup>>,

    /// `Use`s that could not be merged into the current virtual register due to
    /// a conflicting `Use` at the same instruction. These must be processed
    /// separately after normal uses are processed.
    conflicting_uses: Vec<(Value, Use)>,
}

impl VirtRegBuilder {
    pub fn new() -> Self {
        Self {
            value_group_mapping: SecondaryMap::new(),
            conflicting_uses: vec![],
        }
    }

    pub fn clear(&mut self, func: &impl Function) {
        self.value_group_mapping
            .clear_and_resize(func.num_value_groups());
    }

    /// Invalidates a `ValueGroup` to `VirtRegGroup` mapping.
    ///
    /// This is used when a virtual register is split or spilled into a new set
    /// of virtual registers.
    pub fn invalidate_value_group_mapping(&mut self, value_group: ValueGroup) {
        trace!(
            "Invalidating {value_group} -> {:?} mapping",
            self.value_group_mapping[value_group]
        );
        self.value_group_mapping[value_group] = None.into();
    }

    /// Constructs virtual registers from the given live range segments.
    ///
    /// If `split_placement` is `None` then there must be no constraint
    /// conflicts.
    pub fn build(
        &mut self,
        bank: RegBank,
        func: &impl Function,
        reginfo: &impl RegInfo,
        virt_regs: &mut VirtRegs,
        uses: &mut Uses,
        hints: &Hints,
        coalescing: &mut Coalescing,
        stats: &mut Stats,
        split_placement: Option<&SplitPlacement>,
        new_vregs: Option<&mut Vec<VirtReg>>,
        segments: &mut [ValueSegment],
    ) {
        self.conflicting_uses.clear();
        let top_level_class = reginfo.top_level_class(bank);
        trace!("Building new vregs in {bank}");

        let mut ctx = Context {
            func,
            reginfo,
            virt_regs,
            coalescing,
            stats,
            hints,
            value_group_mapping: &mut self.value_group_mapping,
            conflicting_uses: &mut self.conflicting_uses,
            new_vregs,
            uses,
            split_placement,
            top_level_class,
            constraints: VirtRegBuilderConstraints::new(top_level_class),
        };

        ctx.compute_constraints(segments, split_placement.is_some());
        if !ctx.conflicting_uses.is_empty() {
            ctx.emit_vregs_for_conflicts();
        }
    }
}

/// Information about a register group being built by `VirtRegBuilder` if the
/// register class of the virtual register is a group class.
struct VirtRegBuilderGroup {
    /// The index of that vreg within its group.
    index: u8,

    /// The group that this vreg should join if it is joining an existing
    /// register group. If this is `None` then a new virtual register group will
    /// be created for this vreg.
    existing_group: Option<VirtRegGroup>,

    /// The set of values that each group member can take.
    ///
    /// This is used when building virtual registers to ensure that values added
    /// as register group members don't interfere with each other.
    value_sets: [ValueSet; MAX_GROUP_SIZE],
}

/// Utility type for computing constraints on a virtual register.
struct VirtRegBuilderConstraints {
    /// Register class constraint for this vreg. As `Use`s are processes, this
    /// can be further constrained into a subclass of itself.
    class: RegClass,

    /// Group information if this vreg is in a register group class.
    group: Option<VirtRegBuilderGroup>,

    /// Whether we need to add the uses to an existing virtual register instead
    /// of creating a new one.
    merge_into_existing_vreg: Option<VirtReg>,
}

impl VirtRegBuilderConstraints {
    /// Initializes the constraints with the top-level class for the register
    /// bank.
    fn new(top_level_class: RegClass) -> Self {
        Self {
            class: top_level_class,
            group: None,
            merge_into_existing_vreg: None,
        }
    }

    /// Updates the constraints to accept the given `Use`, or returns false
    /// if this is not possible.
    ///
    /// Constraints are not updated if this function returns false.
    fn merge_use(
        &mut self,
        u: Use,
        virt_regs: &VirtRegs,
        value_group_mapping: &SecondaryMap<ValueGroup, PackedOption<VirtRegGroup>>,
        coalescing: &mut Coalescing,
        func: &impl Function,
        reginfo: &impl RegInfo,
    ) -> bool {
        match u.kind {
            // The simple case: we just need to ensure there exists a common
            // sub-class that covers all constraints used in this vreg.
            UseKind::ClassUse { slot: _, class } | UseKind::ClassDef { slot: _, class } => {
                if let Some(new_class) = reginfo.common_subclass(self.class, class) {
                    self.class = new_class;
                    true
                } else {
                    trace!("-> No common subclass between {} and {class}", self.class);
                    false
                }
            }

            // Groups are more complex because there are more restrictions on
            // the constraints that need to match with a vreg and its group.
            UseKind::GroupClassUse {
                slot,
                class,
                group_index,
            }
            | UseKind::GroupClassDef {
                slot,
                class,
                group_index,
            } => {
                // This implicitly checks for any group size conflicts.
                let Some(mut class) = reginfo.common_subclass(self.class, class) else {
                    trace!("-> No common subclass between {} and {class}", self.class);
                    return false;
                };

                // Each vreg can only be associated with a single index within a
                // register group.
                if let Some(group) = &self.group {
                    if group.index != group_index {
                        trace!("-> Group index mismatch: {} vs {group_index}", group.index);
                        return false;
                    }
                }

                // Retrieve the `ValueGroup` that for this operand. This can be
                // used as a unique identifier because each `ValueGroup` can
                // only be used once.
                let value_group = match func.inst_operands(u.pos)[slot as usize].kind() {
                    OperandKind::DefGroup(group)
                    | OperandKind::UseGroup(group)
                    | OperandKind::EarlyDefGroup(group) => group,
                    OperandKind::Def(_)
                    | OperandKind::Use(_)
                    | OperandKind::EarlyDef(_)
                    | OperandKind::NonAllocatable => unreachable!(),
                };
                let value_group_members = func.value_group_members(value_group);

                // Has a `VirtRegGroup` already been created for this operand?
                let existing_group = value_group_mapping[value_group].expand();
                if let Some(existing_group) = existing_group {
                    trace!(
                        "Trying to join existing register group {existing_group} for {value_group}"
                    );
                    if let Some(group) = &self.group {
                        // If we already got assigned a group from a previous
                        // group operand that we visited, this must match exactly.
                        if let Some(prev_existing_group) = group.existing_group {
                            if existing_group != prev_existing_group {
                                trace!(
                                    "-> We already joined a different group {prev_existing_group}"
                                );
                                return false;
                            }
                        }

                        // Ensure that values in the new group don't overlap with
                        // those in the existing group. We do this by checking that
                        // the values are the same.
                        for (&value, &set) in value_group_members.iter().zip(&group.value_sets) {
                            if coalescing.set_for_value(value) != set {
                                trace!(
                                    "-> Incompatible valueset for group member: {value} not in \
                                     {set}"
                                );
                                return false;
                            }
                        }
                    }

                    // We need all group uses of the same value for the same
                    // index to be merged in the same virtual register. If there
                    // is already a virtual register for this then this use must
                    // be added to that virtual register.
                    let members = virt_regs.group_members(existing_group);
                    if members[group_index as usize] != VirtReg::reserved_value() {
                        let vreg = members[group_index as usize];
                        trace!("-> Existing group already has {vreg} at index {group_index}");
                        if let Some(existing_vreg) = self.merge_into_existing_vreg {
                            debug_assert_eq!(existing_vreg, vreg);
                        }

                        // Ensure that our current class is compatible with the
                        // vreg that we need to merge into.
                        match reginfo.common_subclass(self.class, virt_regs[vreg].class) {
                            Some(new_class) => {
                                class = new_class;
                                self.merge_into_existing_vreg = Some(vreg);
                            }
                            None => {
                                trace!(
                                    "-> No common subclass between {} and {class} when merging \
                                     with group member {vreg}",
                                    self.class
                                );
                                return false;
                            }
                        }
                    } else {
                        // Otherwise, try to join this group by unifying our class
                        // with those of other group members for which vregs have
                        // been created so far.
                        for &vreg in members {
                            if vreg.is_reserved_value() {
                                continue;
                            }

                            let member_class = virt_regs[vreg].class;
                            match reginfo.common_subclass(class, member_class) {
                                Some(new_class) => class = new_class,
                                None => {
                                    trace!(
                                        "-> No common subclass between {} and {class} when \
                                         merging with group member {vreg}",
                                        self.class
                                    );
                                    return false;
                                }
                            }
                        }
                    }
                } else {
                    if let Some(group) = &self.group {
                        // If not then we must abort if we have already joined a
                        // group: it must have mismatching vregs since it would
                        // have been assigned a group already otherwise.
                        if group.existing_group.is_some() {
                            trace!(
                                "-> {value_group} has no group mapping even though we are already \
                                 in a group"
                            );
                            return false;
                        }

                        // Ensure that values in the new group don't overlap with
                        // those in the existing group. We do this by checking that
                        // the values are the same.
                        for (&value, &set) in value_group_members.iter().zip(&group.value_sets) {
                            if coalescing.set_for_value(value) != set {
                                trace!(
                                    "-> Incompatible valueset for group member: {value} not in \
                                     {set}"
                                );
                                return false;
                            }
                        }
                    }
                }

                // We are committed to accepting this use, update the
                // constraints for the vreg we are building.
                let value_sets = array::from_fn(|i| {
                    value_group_members
                        .get(i)
                        .map_or(ValueSet::reserved_value(), |&value| {
                            coalescing.set_for_value(value)
                        })
                });
                self.class = class;
                self.group = Some(VirtRegBuilderGroup {
                    index: group_index,
                    value_sets,
                    existing_group,
                });
                true
            }

            // These don't affect the allocation constraint: we will
            // automatically insert a move as necessary.
            UseKind::FixedDef { .. }
            | UseKind::FixedUse { .. }
            | UseKind::TiedUse { .. }
            | UseKind::ConstraintConflict { .. }
            | UseKind::BlockparamIn { .. }
            | UseKind::BlockparamOut { .. } => true,
        }
    }
}

struct Context<'a, F, R> {
    func: &'a F,
    reginfo: &'a R,
    virt_regs: &'a mut VirtRegs,
    uses: &'a mut Uses,
    hints: &'a Hints,
    split_placement: Option<&'a SplitPlacement>,
    coalescing: &'a mut Coalescing,
    stats: &'a mut Stats,
    value_group_mapping: &'a mut SecondaryMap<ValueGroup, PackedOption<VirtRegGroup>>,
    conflicting_uses: &'a mut Vec<(Value, Use)>,
    new_vregs: Option<&'a mut Vec<VirtReg>>,

    /// Top-level class, used by `reset_constraints`.
    top_level_class: RegClass,

    /// Constraints on the virtual register being built.
    constraints: VirtRegBuilderConstraints,
}

impl<F: Function, R: RegInfo> Context<'_, F, R> {
    /// Resets the current constraints to their most generic form.
    fn reset_constraints(&mut self) {
        self.constraints = VirtRegBuilderConstraints::new(self.top_level_class);
    }

    /// Computes register class constraints for the live range uses in the
    /// current set of segments.
    ///
    /// If a conflict is found, the segments are split and the non-conflicting
    /// half is commited to a vreg. This is repeated until all segments have
    /// been processed and committed to vregs.
    ///
    /// Once all segments are processed, the remaining uses are packaged into a
    /// virtual register.
    fn compute_constraints(&mut self, mut segments: &mut [ValueSegment], may_conflict: bool) {
        trace!("Computing vreg constraints");
        'outer: loop {
            self.reset_constraints();

            // Iterate over all the uses in our segments.
            for seg_idx in 0..segments.len() {
                let value = segments[seg_idx].value;
                let use_list = segments[seg_idx].use_list;
                debug_assert!(!segments[seg_idx].live_range.is_empty());
                for idx in 0..use_list.len() {
                    // Attempt to adjust our constraints to include the use.
                    // In most cases the use is already covered by our register
                    // class, but in some cases we may have to restrict
                    // ourselves to a sub-class. If no common sub-class exists
                    // then we need to split the vreg.
                    let u = self.uses[use_list.index(idx)];
                    trace!("Processing use of {} at {}: {}", value, u.pos, u.kind);
                    if self.constraints.merge_use(
                        u,
                        self.virt_regs,
                        self.value_group_mapping,
                        self.coalescing,
                        self.func,
                        self.reginfo,
                    ) {
                        if let Some(group) = &self.constraints.group {
                            trace!(
                                "-> {} group_index={} existing_group={:?}",
                                self.constraints.class, group.index, group.existing_group,
                            );
                        } else {
                            trace!("-> {}", self.constraints.class);
                        }
                    } else {
                        trace!("-> conflict!");
                        stat!(self.stats, vreg_conflicts);
                        debug_assert!(may_conflict, "should not conflict after split");

                        // We have conflicting uses, which we must place in
                        // separate vregs. First, scan backwards to find the
                        // previous use with which this new use conflicts.
                        let end_use_idx = use_list.index(idx);
                        let end_class = self.constraints.class;
                        let end_pos = self.uses[end_use_idx].pos;
                        let (start_use_idx, start_value) =
                            self.find_conflict_start_point(segments, seg_idx, idx);
                        let start_pos = self.uses[start_use_idx].pos;
                        let start_class = self.constraints.class;

                        if start_pos == end_pos {
                            // The conflicting uses are on the same instruction,
                            // so we need to remove one of the conflicting uses
                            // from our vreg and place it in its own vreg.
                            trace!(
                                "Conflicting constraints on same instruction, evicting a single \
                                 use to a vreg"
                            );
                            stat!(self.stats, vreg_conflicts_on_same_inst);

                            // We can only replace a use with a
                            // `ConstraintConflict`, not a definition. It's
                            // impossible for both to be definitions since live
                            // ranges would overlap.
                            //
                            // If both are uses, evict the new incoming one.
                            // This is necessary to properly handle register
                            // groups: we need to attempt to merge a use with
                            // all other potentially conflicting uses in the
                            // same instruction.
                            //
                            // The 2 conflicting uses may have different values
                            // if the conflict is from 2 tied operands with
                            // incompatible register classes.
                            let (conflict_idx, conflict_value) =
                                if !self.uses[end_use_idx].kind.is_def() {
                                    (end_use_idx, value)
                                } else {
                                    (start_use_idx, start_value)
                                };
                            let conflict_use = self.uses[conflict_idx];
                            debug_assert!(!conflict_use.kind.is_def());

                            // Remove the `Use` from the segment and split it
                            // off into a separate virtual register.
                            self.conflicting_uses.push((conflict_value, conflict_use));
                            self.uses[conflict_idx].kind = UseKind::ConstraintConflict {};
                        } else {
                            // Prefer to align the split close to the use that
                            // is harder to allocate a register for. We estimate
                            // this by using the class spill cost, and break
                            // ties by preferring the earlier use.
                            let prefer_early_split = self.reginfo.class_spill_cost(start_class)
                                >= self.reginfo.class_spill_cost(end_class);

                            // Pick a split point in a cold block if possible
                            // and split the segments into 2 halves.
                            trace!(
                                "Conflicting constraints between {start_pos} and {end_pos}, \
                                 splitting vreg"
                            );
                            let split_point =
                                self.split_placement.unwrap().find_optimal_split_point(
                                    start_pos,
                                    end_pos,
                                    prefer_early_split,
                                    self.func,
                                );

                            // Create a vreg for the portion before the split.
                            // This should never conflict and therefore can't
                            // recurse.
                            let mut split = ValueSegment::split_segments_at(
                                segments,
                                self.uses,
                                self.hints,
                                split_point,
                            );
                            self.compute_constraints(split.first_half(), false);

                            // Then continue vreg creation as normal for the
                            // portion after the split, which may still have
                            // conflicts.
                            segments = split.into_second_half();
                        }

                        // After a conflict has been resolved, we need to
                        // re-calculate constraints from the beginning.
                        trace!("Restarting constraint calculation after conflict resolution");
                        continue 'outer;
                    }
                }
            }

            self.emit_vreg(segments);
            break;
        }
    }

    /// Emits a set of vregs containing single `Use`s which were removed from
    /// another vreg due to a constraint conflict on the same instruction.
    fn emit_vregs_for_conflicts(&mut self) {
        // Iterate over groups of conflicting uses at each instruction. We need
        // to attempt to merge each pair of uses so that group uses at the same
        // index for the same group are assigned to the same vreg.
        let mut conflicting_uses = core::mem::take(self.conflicting_uses);
        for uses in conflicting_uses.chunk_by_mut(|a, b| a.1.pos == b.1.pos) {
            // Conflicts can only happen due to multiple uses of the same value
            // at the same instruction. It's impossible to have conflicts with
            // multiple values since the values would not have been coalesced >
            // that case.
            let (value, Use { pos, kind: _ }) = uses[0];
            debug_assert!(uses.iter().all(|&u| u.0 == value));

            trace!("Processing conflicting uses of {value} at {pos}");

            let mut vreg_start = 0;
            while vreg_start != uses.len() {
                // Collect as many uses as possible at this position which have
                // compatible constraints.
                let mut vreg_end = vreg_start;
                let mut conflict_start = uses.len();
                self.reset_constraints();
                while vreg_end != conflict_start {
                    trace!("Attempting to merge use: {}", uses[vreg_end].1.kind);
                    let merged = self.constraints.merge_use(
                        uses[vreg_end].1,
                        self.virt_regs,
                        self.value_group_mapping,
                        self.coalescing,
                        self.func,
                        self.reginfo,
                    );
                    if merged {
                        trace!("-> success!");
                        vreg_end += 1;
                    } else {
                        trace!("-> conflict!");
                        conflict_start -= 1;
                        uses.swap(vreg_end, conflict_start);
                    }
                }

                // Add an implicit live-in to the new segment.
                let mut use_list = self
                    .uses
                    .add_use_list(uses[vreg_start..vreg_end].iter().map(|&(_value, u)| u));
                use_list.set_livein(true);

                // Emit a vreg with the uses we managed to merge.
                let conflict_segment = ValueSegment {
                    live_range: LiveRangeSegment::new(
                        pos.slot(Slot::Boundary),
                        pos.slot(Slot::Normal),
                    ),
                    use_list,
                    value,
                };
                self.emit_vreg(&mut [conflict_segment]);

                // Continue with any uses that we weren't able to merge into
                // this vreg.
                vreg_start = vreg_end;
            }
        }

        *self.conflicting_uses = conflicting_uses;
    }

    /// Starting from the `Use` that caused a conflict, scan backwards to find
    /// the corresponding `Use` that actually conflicts with the other `Use`.
    ///
    /// This will give us a range in between the two uses where we can split
    /// the virtual register into 2 separate vregs.
    fn find_conflict_start_point(
        &mut self,
        segments: &[ValueSegment],
        end_seg_idx: usize,
        end_idx: usize,
    ) -> (UseIndex, Value) {
        trace!("Finding conflict start point...");
        let mut constraints = VirtRegBuilderConstraints::new(self.top_level_class);

        // Scan the segment in which the conflict first occurred.
        let use_list = segments[end_seg_idx].use_list;
        for idx in (0..=end_idx).rev() {
            if !constraints.merge_use(
                self.uses[use_list.index(idx)],
                self.virt_regs,
                self.value_group_mapping,
                self.coalescing,
                self.func,
                self.reginfo,
            ) {
                return (use_list.index(idx), segments[end_seg_idx].value);
            }
        }

        // Scan the remaining segments in reverse.
        for seg_idx in (0..end_seg_idx).rev() {
            for idx in segments[seg_idx].use_list.iter().rev() {
                if !constraints.merge_use(
                    self.uses[idx],
                    self.virt_regs,
                    self.value_group_mapping,
                    self.coalescing,
                    self.func,
                    self.reginfo,
                ) {
                    return (idx, segments[seg_idx].value);
                }
            }
        }

        unreachable!("No conflicting use found in reverse scan");
    }

    /// Calculates the spill weight of the virtual register currently being
    /// built.
    ///
    /// This also detects cases where a virtual register only spans a single
    /// instruction, in which case it cannot be split further. This case is
    /// represented by giving that virtual register an infinite spill weight.
    fn calc_spill_weight(&self, segments: &[ValueSegment]) -> f32 {
        let num_insts = ValueSegment::live_insts(segments);
        trace!("Computing spill weight with {num_insts} instructions");

        // Register classes that allow spillslots are always spillable.
        debug_assert_ne!(num_insts, 0);
        let mut spill_weight = if num_insts <= 1
            && !self
                .reginfo
                .class_includes_spillslots(self.constraints.class)
        {
            trace!("-> Infinite spill weight");
            f32::INFINITY
        } else {
            // Accumulate the spill cost weighed by the block frequency.
            let spill_cost: f32 = segments
                .iter()
                .map(|seg| {
                    // Add up the spill weights of all uses.
                    self.uses[seg.use_list]
                        .iter()
                        .map(|u| {
                            let spill_cost = u.spill_cost(self.reginfo);
                            let block_freq = self.func.block_frequency(self.func.inst_block(u.pos));
                            trace!(
                                "Use of {} at {} ({}) has spill cost {} ({spill_cost} * \
                                 {block_freq})",
                                seg.value,
                                u.pos,
                                u.kind,
                                spill_cost * block_freq
                            );
                            spill_cost * block_freq
                        })
                        .sum::<f32>()
                })
                .sum();

            // Cap the spill weight at f32::MAX. Infinite spill weights are only
            // for unspillable virtual registers.
            let spill_weight = (spill_cost / num_insts as f32).min(f32::MAX);
            trace!("-> Spill weight of {spill_weight} ({spill_cost} / {num_insts})");
            spill_weight
        };

        // If another register in a group we are joining has a lower spill
        // weight then use that instead.
        if let Some(group) = &self.constraints.group {
            if let Some(existing_group) = group.existing_group {
                for (idx, &member) in self
                    .virt_regs
                    .group_members(existing_group)
                    .iter()
                    .enumerate()
                {
                    if idx != group.index as usize && !member.is_reserved_value() {
                        spill_weight = spill_weight.min(self.virt_regs[member].spill_weight);
                    }
                }
            }
        }

        spill_weight
    }

    /// Commits the current contents of `segments` and the current constraints
    /// to a virtual register.
    fn emit_vreg(&mut self, segments: &mut [ValueSegment]) {
        debug_assert!(!segments.is_empty());
        debug_assert!(segments.iter().all(|seg| !seg.live_range.is_empty()));
        let has_fixed_hint = segments.iter().any(|seg| seg.use_list.has_fixedhint());

        // Special handling if we need to insert our segments into an existing
        // virtual register. This only happens when we are joining an existing
        // group.
        if let Some(vreg) = self.constraints.merge_into_existing_vreg {
            debug_assert!(self.constraints.group.is_some());
            debug_assert!(
                self.constraints
                    .group
                    .as_ref()
                    .unwrap()
                    .existing_group
                    .is_some()
            );

            // Add the segments to the virtual register.
            //
            // The new segments should not conflict with any existing segments
            // for that register group. We just need to find the correct point
            // in the segment list in which to insert the new segments.
            let idx = self
                .virt_regs
                .segments(vreg)
                .binary_search_by(|seg| {
                    if seg.live_range.to <= segments[0].live_range.from {
                        Ordering::Less
                    } else {
                        debug_assert!(
                            seg.live_range.from >= segments.last().unwrap().live_range.to
                        );
                        Ordering::Greater
                    }
                })
                .unwrap_err();
            self.virt_regs.virt_regs[vreg].segments =
                self.virt_regs.virt_regs[vreg].segments.insert_iter_at(
                    idx,
                    segments.iter().copied(),
                    &mut self.virt_regs.segment_pool,
                );

            // Update spill_weight, class and has_fixed_use for the virtual
            // register.
            let spill_weight = self.calc_spill_weight(self.virt_regs.segments(vreg));
            self.virt_regs.virt_regs[vreg].has_fixed_hint |= has_fixed_hint;
            self.virt_regs.virt_regs[vreg].class = self.constraints.class;

            // Propagate the class and spill weight to all members of the
            // virtual register group.
            let vreg_group = self.virt_regs[vreg].group.unwrap();
            for &member in self.virt_regs.groups[vreg_group].as_slice(&self.virt_regs.group_pool) {
                if !member.is_reserved_value() {
                    self.virt_regs.virt_regs[member].class = self.constraints.class;
                    self.virt_regs.virt_regs[member].spill_weight = spill_weight;
                }
            }

            return;
        }

        // Allocate a virtual register.
        let spill_weight = self.calc_spill_weight(segments);
        let vreg = self.virt_regs.virt_regs.push(VirtRegData {
            segments: CompactList::from_iter(
                segments.iter().copied(),
                &mut self.virt_regs.segment_pool,
            ),
            class: self.constraints.class,
            group_index: 0,
            group: None.into(),
            has_fixed_hint,
            spill_weight,
        });

        // Special handling for vregs that are part of a register group.
        if let Some(group) = &self.constraints.group {
            let vreg_group = match group.existing_group {
                // We are the first vreg for the group operands we visited,
                // create a new group.
                None => {
                    let group_size = self.reginfo.class_group_size(self.constraints.class);
                    let vreg_group = self.virt_regs.groups.push(CompactList::from_iter(
                        (0..group_size).map(|_| VirtReg::reserved_value()),
                        &mut self.virt_regs.group_pool,
                    ));
                    trace!("Created new vreg group {vreg_group}");

                    // Update the ValueGroup mappings to point to our new group.
                    for seg in self.virt_regs.segments(vreg) {
                        for u in &self.uses[seg.use_list] {
                            let slot = match u.kind {
                                UseKind::GroupClassUse {
                                    slot,
                                    class: _,
                                    group_index: _,
                                }
                                | UseKind::GroupClassDef {
                                    slot,
                                    class: _,
                                    group_index: _,
                                } => slot,
                                UseKind::FixedDef { .. }
                                | UseKind::FixedUse { .. }
                                | UseKind::TiedUse { .. }
                                | UseKind::ConstraintConflict { .. }
                                | UseKind::ClassUse { .. }
                                | UseKind::ClassDef { .. }
                                | UseKind::BlockparamIn { .. }
                                | UseKind::BlockparamOut { .. } => continue,
                            };
                            let value_group =
                                match self.func.inst_operands(u.pos)[slot as usize].kind() {
                                    OperandKind::DefGroup(group)
                                    | OperandKind::UseGroup(group)
                                    | OperandKind::EarlyDefGroup(group) => group,
                                    OperandKind::Def(_)
                                    | OperandKind::Use(_)
                                    | OperandKind::EarlyDef(_)
                                    | OperandKind::NonAllocatable => unreachable!(),
                                };
                            debug_assert!(self.value_group_mapping[value_group].is_none());
                            self.value_group_mapping[value_group] = Some(vreg_group).into();
                        }
                    }

                    vreg_group
                }

                // We joined an existing group, propagate our class constraint
                // and spill weight to all members since it may be smaller than
                // the existing constraint and spill weight we inherited from
                // the group.
                Some(vreg_group) => {
                    for &member in
                        self.virt_regs.groups[vreg_group].as_slice(&self.virt_regs.group_pool)
                    {
                        if !member.is_reserved_value() {
                            self.virt_regs.virt_regs[member].class = self.constraints.class;
                            self.virt_regs.virt_regs[member].spill_weight = spill_weight;
                        }
                    }
                    vreg_group
                }
            };

            // Insert the vreg into its group.
            self.virt_regs.virt_regs[vreg].group_index = group.index;
            self.virt_regs.virt_regs[vreg].group = Some(vreg_group).into();
            let slot = &mut self.virt_regs.groups[vreg_group]
                .as_mut_slice(&mut self.virt_regs.group_pool)[group.index as usize];
            debug_assert_eq!(*slot, VirtReg::reserved_value());
            *slot = vreg;

            trace!("Emitting {vreg} in {vreg_group}");
        } else {
            trace!("Emitting {vreg}");
        }
        if let Some(new_vregs) = &mut self.new_vregs {
            new_vregs.push(vreg);
        }
    }
}
