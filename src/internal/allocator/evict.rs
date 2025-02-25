//! Code related to evicting interference.

use core::mem;
use core::ops::ControlFlow;

use super::order::CandidateReg;
use super::{AbstractVirtRegGroup, Assignment, Context};
use crate::function::Function;
use crate::internal::allocator::Stage;
use crate::internal::allocator::queue::VirtRegOrGroup;
use crate::internal::live_range::ValueSegment;
use crate::internal::reg_matrix::{Interference, InterferenceKind};
use crate::reginfo::{PhysReg, RegInfo};

impl<F: Function, R: RegInfo> Context<'_, F, R> {
    /// Searches for a better candidate register by potentially evicting
    /// interference.
    pub(super) fn try_evict_for_preferred_reg(
        &mut self,
        vreg: impl AbstractVirtRegGroup,
        candidate: CandidateReg,
    ) -> Option<CandidateReg> {
        'outer: for new_candidate in self.allocator.allocation_order.order() {
            // Only evict if this is strictly more profitable then our
            // existing candidate. We can stop otherwise since candidates are
            // sorted by weight.
            if new_candidate.preference_weight <= candidate.preference_weight {
                break;
            }

            // Check if we have a stronger claim to the register than all the
            // interfering virtual registers currently allocated to this
            // register.
            let mut interference_weight = 0.0;
            self.allocator.interfering_vregs.clear();
            for (vreg, reg) in
                vreg.zip_with_reg_group(new_candidate.reg, self.virt_regs, self.reginfo)
            {
                let result = self.reg_matrix.check_interference(
                    self.virt_regs.segments(vreg),
                    reg,
                    self.reginfo,
                    self.stats,
                    false,
                    |interference| {
                        // Can't evict fixed interference.
                        let InterferenceKind::VirtReg(interfering_vreg) = interference.kind else {
                            trace!("Found fixed interference, cannot evict");
                            return ControlFlow::Break(());
                        };

                        // Stop if the cost of the eviction exceeds what we would gain
                        // from using this register ourselves.
                        interference_weight +=
                            self.allocator.assignments[interfering_vreg].preference_weight();
                        if interference_weight >= new_candidate.preference_weight {
                            return ControlFlow::Break(());
                        }

                        self.allocator.interfering_vregs.push(interfering_vreg);
                        ControlFlow::Continue(())
                    },
                );

                if result.is_break() {
                    continue 'outer;
                }
            }

            return Some(new_candidate);
        }

        None
    }

    pub(super) fn try_evict(&mut self, vreg: impl AbstractVirtRegGroup) -> bool {
        // Estimate of the cost of an eviction, which we want to minimize.
        #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
        struct EvictCost {
            /// Sum of the register preference weights of all the virtual
            /// registers that we would need to evict.
            ///
            /// This comes first because we don't want to evict a value from its
            /// preferred register unless we really have to.
            preference_weight: f32,

            //// Maximum spill weight of the virtual registers that we would
            //// need to evict.
            spill_weight: f32,
        }
        let mut best_cost = EvictCost {
            preference_weight: f32::INFINITY,
            spill_weight: f32::INFINITY,
        };
        let mut best_candidate = None;

        // By default (unless we have a register preference) we only want to
        // evict virtual registers with a lower spill weight than us.
        let max_spill_weight = self.virt_regs[vreg.first_vreg(self.virt_regs)].spill_weight;

        // Only allow evicting virtual registers with a higher spill weight
        // once per virtual register. This can happen if we have a higher
        // preference for this register than the evictees.
        let strict_max_weight =
            self.allocator.assignments[vreg.first_vreg(self.virt_regs)].evicted_for_preference();

        trace!(
            "Searching for candidate to evict interference from \
             (max_spill_weight={max_spill_weight}, strict={strict_max_weight})"
        );
        'outer: for candidate in self.allocator.allocation_order.order() {
            trace!("Candidate: {candidate}");

            let mut cost = EvictCost {
                preference_weight: 0.0,
                spill_weight: 0.0,
            };
            self.allocator.candidate_interfering_vregs.clear();

            for (vreg, reg) in vreg.zip_with_reg_group(candidate.reg, self.virt_regs, self.reginfo)
            {
                let f = |interference: Interference<ValueSegment>| {
                    // Can't evict fixed interference.
                    let InterferenceKind::VirtReg(interfering_vreg) = interference.kind else {
                        trace!("Found fixed interference, cannot evict");
                        return ControlFlow::Break(());
                    };

                    let spill_weight = self.virt_regs[interfering_vreg].spill_weight;
                    let preference_weight =
                        self.allocator.assignments[interfering_vreg].preference_weight();
                    trace!(
                        "Interfering vreg {interfering_vreg} at {} in {} has cost {:?}",
                        interference.range,
                        interference.unit,
                        EvictCost {
                            preference_weight,
                            spill_weight
                        }
                    );

                    // Don't double-count preference weight when we have
                    // multiple interfering segments with the same register.
                    if preference_weight != 0.0
                        && self
                            .allocator
                            .candidate_interfering_vregs
                            .contains(&interfering_vreg)
                    {
                        trace!(
                            "Skipping preference weight from {interfering_vreg} which has already \
                            been counted"
                        );
                        return ControlFlow::Continue(());
                    }

                    cost.preference_weight += preference_weight;
                    cost.spill_weight = cost.spill_weight.max(spill_weight);
                    trace!("Total cost so far: {cost:?}");

                    // Don't bother continuing with this candidate if we exceeded
                    // the current best cost.
                    if cost >= best_cost {
                        trace!("Exceeded best cost {best_cost:?}");
                        return ControlFlow::Break(());
                    }

                    // We can't evict virtual registers with a higher spill weight
                    // than ours, *except* if our preference for the candidate is
                    // higher than the total of those of all the evictees.
                    if cost.spill_weight >= max_spill_weight
                        && (candidate.preference_weight == 0.0 || strict_max_weight)
                    {
                        if strict_max_weight {
                            trace!(
                                "Exceeded maximum spill weight {max_spill_weight} and we already \
                                evicted for preference"
                            );
                            return ControlFlow::Break(());
                        }
                        if cost.preference_weight >= candidate.preference_weight {
                            trace!(
                                "Exceeded maximum spill weight {max_spill_weight} and evictees have a \
                                higher preference for this register"
                            );
                            return ControlFlow::Break(());
                        }
                    }

                    // Build up the list of virtual registers to evict in
                    // candidate_interfering_vregs.
                    self.allocator
                        .candidate_interfering_vregs
                        .push(interfering_vreg);

                    ControlFlow::Continue(())
                };
                let result = self.reg_matrix.check_interference(
                    self.virt_regs.segments(vreg),
                    reg,
                    self.reginfo,
                    self.stats,
                    false,
                    f,
                );

                if result.is_break() {
                    continue 'outer;
                }
            }

            // Promote candidate_interfering_vregs to interfering_vregs, which
            // is used by evict_interfering_vregs.
            trace!("Best candidate for evicition is now {candidate} with cost {cost:?}");
            best_candidate = Some(candidate);
            best_cost = cost;
            mem::swap(
                &mut self.allocator.candidate_interfering_vregs,
                &mut self.allocator.interfering_vregs,
            );
        }

        if let Some(best_candidate) = best_candidate {
            trace!("Evicting interference from {best_candidate}");
            self.evict_interfering_vregs(None);
            self.assign(
                vreg,
                best_candidate,
                best_cost.spill_weight >= max_spill_weight,
            );
            true
        } else {
            false
        }
    }

    /// Evicts all the virtual registers in `interfering_vregs` from their
    /// current assignment.
    pub(super) fn evict_interfering_vregs(&mut self, hint: Option<PhysReg>) {
        let assignments = &mut self.allocator.assignments;
        while let Some(vreg) = self.allocator.interfering_vregs.pop() {
            // There may be duplicates in the collected interferring vregs.
            let Assignment::Assigned {
                evicted_for_preference,
                reg,
                preference_weight: _,
            } = assignments[vreg]
            else {
                continue;
            };

            // All members of a register group need to be evicted together and
            // the whole group needs to be re-enqueued.
            if let Some(group) = self.virt_regs[vreg].group.expand() {
                trace!("Evicting {group}");
                stat!(self.stats, evicted_groups);
                for &vreg in self.virt_regs.group_members(group) {
                    let Assignment::Assigned {
                        evicted_for_preference,
                        reg,
                        preference_weight: _,
                    } = assignments[vreg]
                    else {
                        unreachable!();
                    };
                    assignments[vreg] = Assignment::Unassigned {
                        evicted_for_preference,
                        hint: None.into(),
                    };
                    self.reg_matrix
                        .evict(vreg, reg, self.virt_regs, self.reginfo);
                }
                self.allocator.queue.enqueue(
                    VirtRegOrGroup::Group(group),
                    Stage::Evict,
                    self.virt_regs,
                );
            } else {
                trace!("Evicting {vreg}");
                stat!(self.stats, evicted_vregs);
                self.reg_matrix
                    .evict(vreg, reg, self.virt_regs, self.reginfo);
                assignments[vreg] = Assignment::Unassigned {
                    evicted_for_preference,
                    hint: hint.into(),
                };
                self.allocator.queue.enqueue(
                    VirtRegOrGroup::Reg(vreg),
                    Stage::Evict,
                    self.virt_regs,
                );
            }
        }
    }
}
