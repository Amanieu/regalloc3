//! Order in which to probe physical registers for allocating a virtual
//! register.
//!
//! We want to consider several factors here:
//! - If the virtual register has fixed-register uses, we want to try those
//!   first. Give more priority to more frequent uses.
//! - If the virtual register was recently split around the interferences in a
//!   physical register then it is likely to be allocatable in that register:
//!   try that one first.
//! - Otherwise defer to the register class for its allocation order.
//!
//! Finally, we want to randomize the register probing a bit to maximize the
//! chance of successfully allocating with as few probes as possible. This helps
//! improve allocation times.

use core::cmp::Reverse;
use core::fmt;

use ordered_float::OrderedFloat;

use super::{AbstractVirtRegGroup, RegOrRegGroup};
use crate::entity::SparseMap;
use crate::internal::hints::Hints;
use crate::internal::virt_regs::{VirtReg, VirtRegs};
use crate::reginfo::{AllocationOrderSet, PhysReg, RegClass, RegInfo};

/// Returns a single allocation order from [`RegInfo::allocation_order`] which
/// combines all [`AllocationOrderSet`]s.
///
/// `random_seed` perturbs the order in a deterministic way to increase the
/// likelyhood of finding a free register on the first try.
pub fn combined_allocation_order<'a, T: Copy + 'a>(
    get_slice: impl Fn(AllocationOrderSet) -> &'a [T] + Copy,
    random_seed: usize,
) -> impl DoubleEndedIterator<Item = T> + 'a {
    let iter = move |set| {
        let slice = get_slice(set);
        let (a, b) = slice.split_at(random_seed.checked_rem(slice.len()).unwrap_or(0));
        b.iter().copied().chain(a.iter().copied())
    };
    iter(AllocationOrderSet::Preferred).chain(iter(AllocationOrderSet::NonPreferred))
}

/// A candidate physical register to which a virtual register can be assigned.
#[derive(Debug, Clone, Copy)]
pub struct CandidateReg {
    /// The register to try allocating into.
    pub reg: RegOrRegGroup,

    /// Estimate of the cost if this register is not chosen.
    ///
    /// The cost comes from move instructions that will need to be emitted to
    /// move the value into the fixed registers required by the constraints.
    pub preference_weight: f32,
}

impl fmt::Display for CandidateReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.preference_weight != 0.0 {
            write!(f, "{} (preference = {})", self.reg, self.preference_weight)
        } else {
            write!(f, "{}", self.reg)
        }
    }
}

pub struct AllocationOrder {
    /// Physical register candidates for the current virtual register, with
    /// associated preference weight.
    ///
    /// Entries are sorted by preference weight.
    candidates: SparseMap<RegOrRegGroup, f32>,
}

impl AllocationOrder {
    pub fn new() -> Self {
        Self {
            candidates: SparseMap::new(),
        }
    }

    pub fn prepare(&mut self, reginfo: &impl RegInfo) {
        self.candidates.grow_to(reginfo.num_regs());
        self.candidates.grow_to(reginfo.num_reg_groups());
    }

    /// Computes the allocation order for the given virtual register.
    pub fn compute(
        &mut self,
        vreg: impl AbstractVirtRegGroup,
        virt_regs: &VirtRegs,
        hints: &Hints,
        reginfo: &impl RegInfo,
        hint: Option<PhysReg>,
        const_alloc_order: bool,
    ) {
        self.candidates.clear();
        let class = virt_regs[vreg.first_vreg(virt_regs)].class;

        // If this virtual register has fixed-register constraints, collect them
        // and assign them weights based on their use frequency.
        let is_group = vreg.is_group();
        for (group_index, vreg) in vreg.vregs(virt_regs).enumerate() {
            if virt_regs[vreg].has_fixed_hint {
                self.collect_fixed_preferences(
                    vreg,
                    is_group.then_some(group_index),
                    class,
                    virt_regs,
                    hints,
                    reginfo,
                );
            }
        }

        // If there are multiple candidates, they need to be sorted in order
        // of decreasing weight.
        if self.candidates.len() > 1 {
            self.candidates
                .as_mut_slice()
                .sort_unstable_by_key(|&(_, preference_weight)| {
                    Reverse(OrderedFloat(preference_weight))
                });
            self.candidates.rebuild_mapping();
        }

        // If a previous split or eviction produced a hint when this register
        // was pushed back onto the allocation queue, use that.
        if let Some(hint) = hint {
            if !vreg.is_group() {
                if reginfo.class_members(class).contains(hint) {
                    self.candidates
                        .entry(RegOrRegGroup::single(hint))
                        .or_insert(0.0);
                }
            }
        }

        // Random seed algorithm copied from regalloc2: this helps spread
        // allocations around and increases the chance that we find a free
        // register on the first try.
        //
        // This is optional as non-random allocation orders may produce
        // better register allocation at the cost of compile time.
        let random_seed = if const_alloc_order {
            0
        } else {
            virt_regs.segments(vreg.first_vreg(virt_regs))[0]
                .live_range
                .from
                .inst()
                .index()
                + vreg.first_vreg(virt_regs).index()
        };

        // Add the remaining candidates from the register class's allocation
        // order.
        if vreg.is_group() {
            for group in combined_allocation_order(
                |set| reginfo.group_allocation_order(class, set),
                random_seed,
            ) {
                // Insert remaining candidates with a preference weight of 0.
                self.candidates
                    .entry(RegOrRegGroup::multi(group))
                    .or_insert(0.0);
            }
        } else {
            for reg in
                combined_allocation_order(|set| reginfo.allocation_order(class, set), random_seed)
            {
                // Insert remaining candidates with a preference weight of 0.
                self.candidates
                    .entry(RegOrRegGroup::single(reg))
                    .or_insert(0.0);
            }
        }
    }

    /// Returns an iterator over all the registers in the allocation order.
    pub fn order(&self) -> impl Iterator<Item = CandidateReg> + '_ {
        self.candidates
            .iter()
            .map(|&(reg, preference_weight)| CandidateReg {
                reg,
                preference_weight,
            })
    }

    /// Indicates whether the allocation order is empty and the virtual register
    /// can only be spilled to the stack.
    pub fn must_spill(&mut self) -> bool {
        self.candidates.is_empty()
    }

    /// Returns the highest preferrence weight in the available candidates.
    pub fn highest_preferrence_weight(&self) -> f32 {
        self.candidates
            .as_slice()
            .first()
            .map_or(0.0, |&(_reg, preference_weight)| preference_weight)
    }

    /// Scans the uses of the given virtual register to find any preferences for
    /// a particular register due to fixed-register constraints.
    fn collect_fixed_preferences(
        &mut self,
        vreg: VirtReg,
        group_index: Option<usize>,
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
                    let reg_group = if let Some(group_index) = group_index {
                        // If this is a register group then we need to find the register
                        // group in this class which contains this register.
                        let Some(reg_group) = reginfo.group_for_reg(hint.reg, group_index, class)
                        else {
                            continue;
                        };
                        debug_assert!(reginfo.class_group_members(class).contains(reg_group));
                        RegOrRegGroup::multi(reg_group)
                    } else {
                        // Ignore preferences that conflict with our register class
                        // constraint.
                        if !reginfo.class_members(class).contains(hint.reg) {
                            continue;
                        }
                        RegOrRegGroup::single(hint.reg)
                    };

                    *self.candidates.entry(reg_group).or_default() += hint.weight;
                }
            }
        }
    }
}
