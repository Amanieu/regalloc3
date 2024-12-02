//! Code related to spilling virtual registers to the stack.

use alloc::vec;
use alloc::vec::Vec;

use super::queue::VirtRegOrGroup;
use super::{AbstractVirtRegGroup, Assignment, Context, Stage};
use crate::function::{Function, OperandKind};
use crate::internal::live_range::ValueSegment;
use crate::internal::uses::UseKind;
use crate::internal::virt_regs::VirtReg;
use crate::reginfo::RegInfo;

/// Temporary state used when spilling.
pub struct Spiller {
    /// Scratch space for collecting minimal live ranges.
    minimal_segments: Vec<ValueSegment>,

    /// Newly created virtual register from the minimal live ranges.
    new_vregs: Vec<VirtReg>,
}

impl Spiller {
    pub fn new() -> Self {
        Self {
            minimal_segments: vec![],
            new_vregs: vec![],
        }
    }
}

impl<F: Function, R: RegInfo> Context<'_, F, R> {
    /// Spills the given virtual register to a spillslot.
    ///
    /// Any uses which must be in a register are split off into separate virtual
    /// registers which only cover a single instruction and therefore are
    /// unspillable.
    pub(super) fn spill(&mut self, vreg: impl AbstractVirtRegGroup) {
        self.allocator.spiller.minimal_segments.clear();

        // TODO(perf): Fast path if class allows spillslots?

        // Invalidate any existing ValueGroup mappings: the group they point to
        // will no longer be valid after it is spilled. New mappings will be
        // created when building the new virtual registers.
        if vreg.is_group() {
            let vreg = vreg.first_vreg(self.virt_regs);
            for segment in self.virt_regs.segments(vreg) {
                for &u in &self.uses[segment.use_list] {
                    if let UseKind::GroupClassUse {
                        slot,
                        class: _,
                        group_index: _,
                    }
                    | UseKind::GroupClassDef {
                        slot,
                        class: _,
                        group_index: _,
                    } = u.kind
                    {
                        let value_group = match self.func.inst_operands(u.pos)[slot as usize].kind()
                        {
                            OperandKind::DefGroup(group)
                            | OperandKind::UseGroup(group)
                            | OperandKind::EarlyDefGroup(group) => group,
                            OperandKind::Def(_)
                            | OperandKind::Use(_)
                            | OperandKind::EarlyDef(_)
                            | OperandKind::NonAllocatable => unreachable!(),
                        };
                        self.virt_reg_builder
                            .invalidate_value_group_mapping(value_group);
                    }
                }
            }
        }

        // Find any uses that can't be spilled and split them into minimal
        // segments that only cover a single instruction (or part of one).
        //
        // Any remaining live range between these minimal segments is spilled to
        // the stack.
        for (group_index, vreg) in vreg.vregs(self.virt_regs).enumerate() {
            trace!("Spilling {vreg}");
            stat!(self.stats, spilled_vregs);
            for &segment in self.virt_regs.segments(vreg) {
                // If the value of that segment is rematerializable then we
                // don't need to spill it. This will result in it having no
                // allocation, which the move resolver will handle by
                // rematerializing the value.
                let can_remat = self.func.can_rematerialize(segment.value).is_some();

                let mut segment = segment;
                'outer: loop {
                    let mut must_spill = false;
                    for &u in &self.uses[segment.use_list] {
                        // Ignore uses that can be assigned to spill slots.
                        let can_spill = match u.kind {
                            UseKind::ClassUse { slot: _, class }
                            | UseKind::ClassDef { slot: _, class } => {
                                if self.reginfo.class_includes_spillslots(class) {
                                    // Even if the value is rematerializable, we
                                    // must *still* spill it because ClassUse
                                    // and ClassDef constraints need an
                                    // allocation.
                                    must_spill = true;
                                    true
                                } else {
                                    false
                                }
                            }
                            UseKind::GroupClassUse {
                                slot: _,
                                class,
                                group_index: use_group_index,
                            }
                            | UseKind::GroupClassDef {
                                slot: _,
                                class,
                                group_index: use_group_index,
                            } => {
                                debug_assert!(!self.reginfo.class_includes_spillslots(class));
                                debug_assert_eq!(use_group_index as usize, group_index);
                                false
                            }
                            UseKind::FixedDef { .. }
                            | UseKind::FixedUse { .. }
                            | UseKind::TiedUse { .. }
                            | UseKind::ConstraintConflict { .. }
                            | UseKind::BlockparamIn { .. }
                            | UseKind::BlockparamOut { .. } => true,
                        };
                        if can_spill {
                            trace!(
                                "Spillable use of {} at {}: {}",
                                segment.value,
                                u.pos,
                                u.kind
                            );
                            continue;
                        }

                        trace!("Splitting around unspillable use {}: {}", u.pos, u.kind);

                        // If there is a live range segment with no uses before
                        // this use, spill that segment.
                        if u.pos != segment.live_range.from.round_to_prev_inst().inst() {
                            let (before, after) = segment.split_at(u.pos, self.uses, self.hints);
                            if must_spill || !can_remat {
                                let set = self.coalescing.set_for_value(before.value);
                                self.spill_allocator.spill_segment(set, before);
                            } else {
                                trace!(
                                    "Rematerializing segment for {} at {}",
                                    before.value,
                                    before.live_range
                                );
                                self.allocator.remat_segments.push(before);
                            }
                            segment = after;
                        }

                        if u.pos.next() == segment.live_range.to.round_to_next_inst().inst() {
                            // If this use is on the last instruction of the segment
                            // then split off the rest of the segment into a
                            // minimal segment.
                            trace!(
                                "Generating minimal segment for {} at {}",
                                segment.value,
                                segment.live_range
                            );
                            self.allocator.spiller.minimal_segments.push(segment);
                            break 'outer;
                        } else {
                            // Otherwise split the segment at the next
                            // instruction boundary after the use. The first
                            // half is split as a minimal segment and continue
                            // processing the remaining half.
                            let (before, after) =
                                segment.split_at(u.pos.next(), self.uses, self.hints);
                            trace!(
                                "Generating minimal segment for {} at {}",
                                before.value,
                                before.live_range
                            );
                            self.allocator.spiller.minimal_segments.push(before);
                            segment = after;
                            continue 'outer;
                        }
                    }

                    // If there are no uses left in the segment, spill it. The
                    // live range must be non-empty at this point, this is
                    // checked in spill_segment.
                    if must_spill || !can_remat {
                        let set = self.coalescing.set_for_value(segment.value);
                        self.spill_allocator.spill_segment(set, segment);
                    } else {
                        trace!(
                            "Rematerializing segment for {} at {}",
                            segment.value,
                            segment.live_range
                        );
                        self.allocator.remat_segments.push(segment);
                    }
                    break;
                }
            }

            // The original vreg is no longer used after this point.
            self.allocator.assignments[vreg] = Assignment::Dead;
        }

        // Create a new virtual register for each minimal segment that was
        // created.
        self.allocator.spiller.new_vregs.clear();
        self.allocator
            .spiller
            .minimal_segments
            .iter()
            .for_each(|&segment| {
                stat!(self.stats, minimal_segments);
                self.virt_regs.create_vreg_from_segments(
                    &mut [segment],
                    self.func,
                    self.reginfo,
                    self.uses,
                    self.hints,
                    self.virt_reg_builder,
                    self.coalescing,
                    self.stats,
                    &mut self.allocator.spiller.new_vregs,
                );
            });

        // Initialize assignments for the new virtual registers.
        self.allocator
            .assignments
            .grow_to(self.virt_regs.num_virt_regs());

        // Then queue the newly created virtual registers. This needs to be done
        // after all spill products are created so that virtual register groups
        // are complete.
        for &vreg in &self.allocator.spiller.new_vregs {
            trace!("Created spill product {vreg}");

            // If the virtual register is part of a group, enqueue the whole
            // group instead.
            let vreg_or_group = if let Some(group) = self.virt_regs[vreg].group.expand() {
                // Only do this once for the group leader.
                if self.virt_regs[vreg].group_index != 0 {
                    continue;
                }
                VirtRegOrGroup::Group(group)
            } else {
                VirtRegOrGroup::Reg(vreg)
            };

            trace!("Queuing spill product {vreg_or_group}");
            self.allocator
                .queue
                .enqueue(vreg_or_group, Stage::Evict, self.virt_regs);
        }
    }
}
