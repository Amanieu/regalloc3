//! Order in which to probe physical registers for allocating a virtual
//! register.
//!
//! We want to consider several factors here:
//! - If the virtual register has fixed-register uses, we want to try those
//!   first. Give more priority to more frequent uses.
//! - Otherwise defer to the register class for its allocation order.

use core::cmp::Reverse;
use core::fmt;

use ordered_float::OrderedFloat;

use super::AbstractVirtRegGroup;
use crate::entity::{PackedOption, SecondaryMap, SparseMap};
use crate::internal::hints::Hints;
use crate::internal::value_live_ranges::ValueSet;
use crate::internal::virt_regs::{VirtReg, VirtRegs};
use crate::reginfo::{PhysReg, RegClass, RegInfo};

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
    /// Physical register candidates for the current virtual register, with
    /// associated preference weight.
    ///
    /// Entries are sorted by preference weight.
    hinted_regs: SparseMap<V::Phys, f32>,
}

impl<V: AbstractVirtRegGroup> AllocationOrder<V> {
    pub fn new() -> Self {
        Self {
            hinted_regs: SparseMap::new(),
        }
    }

    pub fn prepare(&mut self, reginfo: &impl RegInfo) {
        if V::is_group() {
            self.hinted_regs.grow_to(reginfo.num_reg_groups());
        } else {
            self.hinted_regs.grow_to(reginfo.num_regs());
        }
    }

    /// Computes the allocation order for the given virtual register.
    pub fn compute(
        &mut self,
        vreg: V,
        virt_regs: &VirtRegs,
        hints: &Hints,
        last_allocated_reg: &SecondaryMap<ValueSet, PackedOption<PhysReg>>,
        reginfo: &impl RegInfo,
    ) {
        // If this virtual register has fixed-register constraints, collect them
        // and assign them weights based on their use frequency.
        self.hinted_regs.clear();
        for (group_index, vreg) in vreg.vregs(virt_regs).enumerate() {
            if virt_regs[vreg].has_fixed_hint {
                let class = virt_regs[vreg].class;
                self.collect_fixed_preferences(vreg, group_index, class, virt_regs, hints, reginfo);
            }
        }

        // If there are hinted registers, they need to be sorted in order
        // of decreasing weight.
        if self.hinted_regs.len() > 1 {
            self.hinted_regs
                .as_mut_vec()
                .sort_unstable_by_key(|&(_reg, preference_weight)| {
                    Reverse(OrderedFloat(preference_weight))
                });
            self.hinted_regs.rebuild_mapping();
        }

        // If another virtual register in the same value set was assigned, try
        // to reuse the same physical register. This can help with move
        // elimination in the later stages.
        //
        // We don't assign it a preference though since it's not worth evicting
        // other registers over.
        for (group_index, vreg) in vreg.vregs(virt_regs).enumerate() {
            let set = virt_regs[vreg].value_set;
            if let Some(hint) = last_allocated_reg[set].expand() {
                let class = virt_regs[vreg.first_vreg(virt_regs)].class;
                if let Some(reg) = V::group_for_reg(hint, group_index, class, reginfo) {
                    self.hinted_regs.entry(reg).or_insert(0.0);
                }
            }
        }
    }

    /// Returns an iterator over all the registers in the allocation order.
    pub fn order<'a>(
        &'a self,
        vreg: V,
        virt_regs: &VirtRegs,
        reginfo: &'a impl RegInfo,
    ) -> impl Iterator<Item = CandidateReg<V>> + 'a {
        let class = virt_regs[vreg.first_vreg(virt_regs)].class;
        self.hinted_regs
            .iter()
            .map(|&(reg, preference_weight)| CandidateReg {
                reg,
                preference_weight,
            })
            .chain(
                V::allocation_order(class, reginfo)
                    .iter()
                    .filter(|&&reg| {
                        // Fast path if there are no hinted registers.
                        self.hinted_regs.is_empty() || !self.hinted_regs.contains_key(reg)
                    })
                    .map(|&reg| CandidateReg {
                        reg,
                        preference_weight: 0.0,
                    }),
            )
    }

    /// Returns the subset of the allocation order that comes from hints rather
    /// than the class allocation order.
    pub fn hinted_order<'a>(&'a self) -> impl Iterator<Item = CandidateReg<V>> + 'a {
        self.hinted_regs
            .iter()
            .map(|&(reg, preference_weight)| CandidateReg {
                reg,
                preference_weight,
            })
    }

    /// Indicates whether the allocation order is empty and the virtual register
    /// can only be spilled to the stack.
    pub fn must_spill(&mut self, vreg: V, virt_regs: &VirtRegs, reginfo: &impl RegInfo) -> bool {
        // Register groups cannot have an empty allocation order.
        if V::is_group() {
            return false;
        }

        // If we have a hint then we can allocate to it, even if the allocation
        // order is empty.
        if !self.hinted_regs.is_empty() {
            return false;
        }

        let class = virt_regs[vreg.first_vreg(virt_regs)].class;
        reginfo.allocation_order(class).is_empty()
    }

    /// Returns the highest preferrence weight in the available candidates.
    pub fn highest_preferrence_weight(&self) -> f32 {
        self.hinted_regs
            .iter()
            .next()
            .map_or(0.0, |&(_reg, preference_weight)| preference_weight)
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
