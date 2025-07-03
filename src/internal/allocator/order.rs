//! Order in which to probe physical registers for allocating a virtual
//! register.
//!
//! We want to consider several factors here:
//! - If the virtual register has fixed-register uses, we want to try those
//!   first. Give more priority to more frequent uses.
//! - Otherwise defer to the register class for its allocation order.

use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;
use core::fmt;

use ordered_float::OrderedFloat;

use super::AbstractVirtRegGroup;
use crate::entity::SparseMap;
use crate::internal::hints::Hints;
use crate::internal::virt_regs::{VirtReg, VirtRegs};
use crate::reginfo::{RegClass, RegInfo};

/// A candidate physical register to which a virtual register can be assigned.
#[derive(Debug, Clone, Copy)]
pub struct CandidateReg<V: AbstractVirtRegGroup> {
    /// The register to try allocating into.
    pub reg: V::Phys,

    /// Estimate of the cost if this register is not chosen.
    ///
    /// The cost comes from move instructions that will need to be emitted to
    /// move the value into the fixed registers required by the constraints.
    pub preference_weight: f32,
}

impl<V: AbstractVirtRegGroup> fmt::Display for CandidateReg<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.preference_weight != 0.0 {
            write!(f, "{} (preference = {})", self.reg, self.preference_weight)
        } else {
            write!(f, "{}", self.reg)
        }
    }
}

pub struct AllocationOrder<V: AbstractVirtRegGroup> {
    /// Map of registers with a preference weight.
    hinted_regs: SparseMap<V::Phys, f32>,

    /// Physical register candidates for the current virtual register, with
    /// associated preference weight.
    ///
    /// Entries are sorted by preference weight.
    candidates: Vec<CandidateReg<V>>,
}

impl<V: AbstractVirtRegGroup> AllocationOrder<V> {
    pub fn new() -> Self {
        Self {
            hinted_regs: SparseMap::new(),
            candidates: vec![],
        }
    }

    pub fn prepare(&mut self, reginfo: &impl RegInfo) {
        if V::is_group() {
            self.hinted_regs.grow_to(reginfo.num_reg_groups());
            self.candidates.reserve(reginfo.num_reg_groups());
        } else {
            self.hinted_regs.grow_to(reginfo.num_regs());
            self.candidates.reserve(reginfo.num_regs());
        }
    }

    /// Computes the allocation order for the given virtual register.
    pub fn compute(
        &mut self,
        vreg: V,
        virt_regs: &VirtRegs,
        hints: &Hints,
        reginfo: &impl RegInfo,
    ) {
        let class = virt_regs[vreg.first_vreg(virt_regs)].class;

        // If this virtual register has fixed-register constraints, collect them
        // and assign them weights based on their use frequency.
        self.hinted_regs.clear();
        for (group_index, vreg) in vreg.vregs(virt_regs).enumerate() {
            if virt_regs[vreg].has_fixed_hint {
                self.collect_fixed_preferences(vreg, group_index, class, virt_regs, hints, reginfo);
            }
        }

        // Initialize the candidate list from the hinted registers.
        self.candidates.clear();
        if !self.hinted_regs.is_empty() {
            self.candidates
                .extend(
                    self.hinted_regs
                        .iter()
                        .map(|&(reg, preference_weight)| CandidateReg {
                            reg,
                            preference_weight,
                        }),
                );

            // If there are multiple candidates, they need to be sorted in order
            // of decreasing weight.
            if self.candidates.len() > 1 {
                self.candidates.sort_unstable_by_key(|candidate| {
                    Reverse(OrderedFloat(candidate.preference_weight))
                });
            }
        }

        // Add the remaining candidates from the register class's allocation
        // order with a preference weight of 0.
        if self.candidates.is_empty() {
            // Fast path if there are no hints: we don't need to check against
            // the map and can just copy the allocation order as-is.
            self.candidates
                .extend(
                    V::allocation_order(class, reginfo)
                        .iter()
                        .map(|&reg| CandidateReg {
                            reg,
                            preference_weight: 0.0,
                        }),
                );
        } else {
            self.candidates.extend(
                V::allocation_order(class, reginfo)
                    .iter()
                    .filter(|&&reg| !self.hinted_regs.contains_key(reg))
                    .map(|&reg| CandidateReg {
                        reg,
                        preference_weight: 0.0,
                    }),
            );
        }
    }

    /// Returns an iterator over all the registers in the allocation order.
    pub fn order(&self) -> impl Iterator<Item = CandidateReg<V>> + '_ {
        self.candidates.iter().copied()
    }

    /// Indicates whether the allocation order is empty and the virtual register
    /// can only be spilled to the stack.
    pub fn must_spill(&mut self) -> bool {
        self.candidates.is_empty()
    }

    /// Returns the highest preferrence weight in the available candidates.
    pub fn highest_preferrence_weight(&self) -> f32 {
        self.candidates
            .first()
            .map_or(0.0, |candidate| candidate.preference_weight)
    }

    /// Scans the uses of the given virtual register to find any preferences for
    /// a particular register due to fixed-register constraints.
    fn collect_fixed_preferences(
        &mut self,
        vreg: VirtReg,
        group_index: usize,
        class: RegClass,
        virt_regs: &VirtRegs,
        hints: &Hints,
        reginfo: &impl RegInfo,
    ) {
        // Iterate over all uses in this vreg and collect fixed register uses.
        trace!("Collecting fixed register hints for {vreg}:");
        for seg in virt_regs.segments(vreg) {
            if seg.use_list.has_fixedhint() {
                for hint in hints.hints_for_segment(seg.value, seg.live_range) {
                    trace!("- {hint}");
                    if let Some(reg) = V::group_for_reg(hint.reg, group_index, class, reginfo) {
                        *self.hinted_regs.entry(reg).or_default() += hint.weight;
                    }
                }
            }
        }
    }
}
