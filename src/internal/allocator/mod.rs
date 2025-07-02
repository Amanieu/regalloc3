//! Main allocation loop which assigns virtual registers to physical registers.
//!
//! The general approach here follows the one used in LLVM's greedy register
//! allocator. [This video] gives a good overview of the algorithm as used in
//! LLVM.
//!
//! [This video]: https://www.youtube.com/watch?v=IK8TMJf3G6U
//!
//! The general approach involves running each virtual register through 3
//! stages. At each stage, we first attempt to assign the virtual register to
//! a suitable physical register in the register class of the virtual register.
//!
//! If this fails then, depending on the stage the virtual register is at, we:
//!
//! 1. Attempt to evict conflicting virtual registers from a physical register,
//!    if they have a lower spill weight than the incoming virtual register.
//!
//! 2. Attempt to split the incoming virtual register into smaller pieces, where
//!    these smaller pieces have a chance to be allocated separately.
//!
//! 3. If the virtual register's constraint allows it to be spilled to the stack
//!    then do so if splitting is unprofitable.

mod evict;
mod order;
mod queue;
mod split;

use alloc::vec;
use alloc::vec::Vec;
use core::ops::ControlFlow;
use core::{fmt, iter};

pub use order::combined_allocation_order;

use self::order::{AllocationOrder, CandidateReg};
use self::queue::{AllocationQueue, VirtRegOrGroup};
use self::split::Splitter;
use super::coalescing::Coalescing;
use super::hints::Hints;
use super::live_range::ValueSegment;
use super::reg_matrix::RegMatrix;
use super::spill_allocator::SpillAllocator;
use super::split_placement::SplitPlacement;
use super::uses::Uses;
use super::virt_regs::builder::VirtRegBuilder;
use super::virt_regs::{VirtReg, VirtRegGroup, VirtRegs};
use crate::entity::SecondaryMap;
use crate::entity::packed_option::PackedOption;
use crate::function::Function;
use crate::internal::reg_matrix::InterferenceKind;
use crate::reginfo::{PhysReg, RegGroup, RegInfo};
use crate::{Options, RegAllocError, SplitStrategy, Stats};

entity_def! {
    /// This type represents either a [`PhysReg`] for groups of size 1 or a
    /// [`RegGroup`] for larger group sizes. The group size is not encoded in
    /// this type itself: it must instead be inferred from context.
    entity RegOrRegGroup(u16, "r/rg");
}

impl RegOrRegGroup {
    /// Creates a [`RegOrRegGroup`] representing a single [`PhysReg`].
    #[inline]
    #[must_use]
    fn single(reg: PhysReg) -> Self {
        Self::new(reg.index())
    }

    /// Creates a [`RegOrRegGroup`] representing a sequence of more than one
    /// register through a [`RegGroup`].
    #[inline]
    #[must_use]
    fn multi(group: RegGroup) -> Self {
        Self::new(group.index())
    }

    /// For single registers, returns the [`PhysReg`].
    ///
    /// This should only be called for register groups created using
    /// [`RegOrRegGroup::single`].
    #[inline]
    #[must_use]
    fn as_single(self) -> PhysReg {
        PhysReg::new(self.index())
    }

    /// For register groups, returns the [`RegGroup`] which describes
    /// the sequence of registers in this group.
    ///
    /// This should only be called for register groups created using
    /// [`RegOrRegGroup::multi`].
    #[inline]
    #[must_use]
    fn as_multi(self) -> RegGroup {
        RegGroup::new(self.index())
    }
}

/// Abstraction over a virtual register group.
///
/// We want to optimize allocation speed for the case of single virtual
/// registers since these are far more common than register groups.
///
/// This is achieved by emitting separate specialized implementations of the
/// core allocation loop for virtual registers and virtual register groups.
///
/// This trait effectively treats `VirtReg` as a group with a size of 1, but
/// the compiler can produce better code for this case.
trait AbstractVirtRegGroup: Copy + fmt::Debug + fmt::Display + Into<VirtRegOrGroup> {
    /// Iterator over all the virtual registers in this group.
    fn vregs(self, virt_regs: &VirtRegs) -> impl ExactSizeIterator<Item = VirtReg>;

    /// Whether this is a virtual register group with more than one member.
    fn is_group(self) -> bool;

    /// First virtual register in this group.
    fn first_vreg(self, virt_regs: &VirtRegs) -> VirtReg {
        self.vregs(virt_regs).next().unwrap()
    }

    /// Iterates over the members of this virtual register group with those of a
    /// physical register group.
    fn zip_with_reg_group(
        self,
        reg: RegOrRegGroup,
        virt_regs: &VirtRegs,
        reginfo: &impl RegInfo,
    ) -> impl ExactSizeIterator<Item = (VirtReg, PhysReg)>;

    /// Dumps the virtual register to the log.
    fn dump(self, virt_regs: &VirtRegs, uses: &Uses);
}

impl AbstractVirtRegGroup for VirtReg {
    fn vregs(self, _virt_regs: &VirtRegs) -> impl ExactSizeIterator<Item = VirtReg> {
        iter::once(self)
    }

    fn is_group(self) -> bool {
        false
    }

    fn zip_with_reg_group(
        self,
        reg: RegOrRegGroup,
        _virt_regs: &VirtRegs,
        _reginfo: &impl RegInfo,
    ) -> impl ExactSizeIterator<Item = (VirtReg, PhysReg)> {
        iter::once((self, reg.as_single()))
    }

    fn dump(self, virt_regs: &VirtRegs, uses: &Uses) {
        let vreg_data = &virt_regs[self];
        trace!(
            "  {self} ({}, spill_weight={}):",
            vreg_data.class, vreg_data.spill_weight,
        );
        for segment in virt_regs.segments(self) {
            segment.dump(uses);
        }
    }
}

impl From<VirtReg> for VirtRegOrGroup {
    fn from(vreg: VirtReg) -> Self {
        VirtRegOrGroup::Reg(vreg)
    }
}

impl AbstractVirtRegGroup for VirtRegGroup {
    fn vregs(self, virt_regs: &VirtRegs) -> impl ExactSizeIterator<Item = VirtReg> {
        virt_regs.group_members(self).iter().copied()
    }

    fn is_group(self) -> bool {
        true
    }

    fn zip_with_reg_group(
        self,
        reg: RegOrRegGroup,
        virt_regs: &VirtRegs,
        reginfo: &impl RegInfo,
    ) -> impl ExactSizeIterator<Item = (VirtReg, PhysReg)> {
        let members = reginfo.reg_group_members(reg.as_multi());
        debug_assert_eq!(members.len(), self.vregs(virt_regs).len());
        self.vregs(virt_regs).zip(members.iter().copied())
    }

    fn dump(self, virt_regs: &VirtRegs, uses: &Uses) {
        for &vreg in virt_regs.group_members(self) {
            vreg.dump(virt_regs, uses);
        }
    }
}

impl From<VirtRegGroup> for VirtRegOrGroup {
    fn from(group: VirtRegGroup) -> Self {
        VirtRegOrGroup::Group(group)
    }
}

/// Assignments for each virtual register produced by this pass.
enum Assignment {
    /// The virtual register has been assigned to a physical register.
    Assigned {
        /// Whether this virtual register has evicted another virtual register
        /// with a higher spill weight to steal a physical register for which it
        /// has a higher preference.
        ///
        /// To avoid infinite eviction loops, we only allow this to happen once
        /// per virtual register.
        evicted_for_preference: bool,

        /// The physical register that this virtual register has been assigned to.
        reg: PhysReg,

        /// The preference weight represents how much we like the register that we
        /// are currently assigned to.
        ///
        /// This is calculated as the number of move instructions (weighed by block
        /// frequency) that would be needed to satisfy our fixed-register
        /// constraints if we were assigned to a different register.
        preference_weight: f32,
    },

    /// The virtual register has not been assigned to a physical register yet,
    /// and is currently in the allocation queue.
    Unassigned {
        /// Whether this virtual register has evicted another virtual register
        /// with a higher spill weight to steal a physical register for which it
        /// has a higher preference.
        ///
        /// To avoid infinite eviction loops, we only allow this to happen once
        /// per virtual register.
        evicted_for_preference: bool,

        /// A hint for a physical register that should be probed first on the
        /// next attempt to allocate this virtual register.
        ///
        /// The goal of this hint is to avoid unnecessary probes by quickly
        /// finding a physical register that is likely to be available.
        hint: PackedOption<PhysReg>,
    },

    /// The virtual register is dead as a result of live range splitting: its
    /// contents are now covered by new virtual registers produced by splitting.
    Dead,
}

impl Default for Assignment {
    fn default() -> Self {
        Assignment::Unassigned {
            evicted_for_preference: false,
            hint: None.into(),
        }
    }
}

impl Assignment {
    /// Whether this virtual register has evicted another virtual register
    /// with a higher spill weight to steal a physical register for which it
    /// has a higher preference.
    ///
    /// To avoid infinite eviction loops, we only allow this to happen once
    /// per virtual register.
    fn evicted_for_preference(&self) -> bool {
        match *self {
            Assignment::Assigned {
                evicted_for_preference,
                reg: _,
                preference_weight: _,
            } => evicted_for_preference,
            Assignment::Unassigned {
                evicted_for_preference,
                hint: _,
            } => evicted_for_preference,
            Assignment::Dead => false,
        }
    }

    /// Returns the register hint associated with a virtual register, if any.
    ///
    /// The assignment must currently be unassigned.
    fn allocation_hint(&self) -> Option<PhysReg> {
        match self {
            Assignment::Assigned { .. } => {
                unreachable!("assigning already assigned virtual register")
            }
            Assignment::Unassigned {
                evicted_for_preference: _,
                hint,
            } => hint.expand(),
            Assignment::Dead => unreachable!("assigning dead virtual register"),
        }
    }

    /// Returns the preference weight associated with a virtual register.
    ///
    /// The assignment must currently be assigned.
    fn preference_weight(&self) -> f32 {
        match *self {
            Assignment::Assigned {
                evicted_for_preference: _,
                reg: _,
                preference_weight,
            } => preference_weight,
            Assignment::Unassigned { .. } => unreachable!("unassigned virtual register"),
            Assignment::Dead => unreachable!("dead virtual register"),
        }
    }
}

/// Allocation stage for a virtual register.
///
/// This is a simplified version of the allocation stages in LLVM's greedy
/// allocator.
#[derive(Clone, Copy, Debug)]
enum Stage {
    /// Allow this virtual register to evict virtual registers with a lower
    /// spill weight.
    Evict,

    /// This virtual register failed to evict virtual registers because its
    /// spill weight was too low. It must now be split into smaller segments or
    /// be spilled to the stack.
    Split,
}

/// Data used by the main allocation stage.
pub struct Allocator {
    /// Priority queue of virtual registers to allocate.
    queue: AllocationQueue,

    /// Order in which to probe registers in a register class, taking hints and
    /// preferences into account.
    allocation_order: AllocationOrder,

    /// Result of allocation for each virtual register.
    assignments: SecondaryMap<VirtReg, Assignment>,

    /// List of interfering virtual registers for `evict_interfering_vregs` to
    /// evict.
    interfering_vregs: Vec<VirtReg>,

    /// Scratch space for building the list of interfering virtual registers in
    /// `try_evict`.
    candidate_interfering_vregs: Vec<VirtReg>,

    /// Temporary state used by live range splitting.
    splitter: Splitter,

    /// Segments with an empty live range that are not part of a virtual
    /// register.
    ///
    /// These don't need to be allocated to a register and only hold metadata
    /// needed for move resolution.
    pub empty_segments: Vec<ValueSegment>,

    /// Segments with a non-empty live range which are not part of a virtual
    /// register and which can be rematerialized.
    ///
    /// These don't need to be allocated to a register and only hold metadata
    /// needed for move resolution.
    pub remat_segments: Vec<ValueSegment>,
}

impl Allocator {
    pub fn new() -> Self {
        Self {
            queue: AllocationQueue::new(),
            allocation_order: AllocationOrder::new(),
            assignments: SecondaryMap::new(),
            interfering_vregs: vec![],
            candidate_interfering_vregs: vec![],
            splitter: Splitter::new(),
            empty_segments: vec![],
            remat_segments: vec![],
        }
    }

    /// Assigns a physical register (or spill index) to every virtual register
    /// in the function.
    ///
    /// This will split unallocatable virtual registers into smaller pieces or
    /// spill them as needed.
    pub fn run(
        &mut self,
        uses: &mut Uses,
        hints: &Hints,
        reg_matrix: &mut RegMatrix,
        virt_regs: &mut VirtRegs,
        virt_reg_builder: &mut VirtRegBuilder,
        spill_allocator: &mut SpillAllocator,
        split_placement: &SplitPlacement,
        coalescing: &mut Coalescing,
        stats: &mut Stats,
        func: &impl Function,
        reginfo: &impl RegInfo,
        options: &Options,
    ) -> Result<(), RegAllocError> {
        self.assignments.clear_and_resize(virt_regs.num_virt_regs());
        self.remat_segments.clear();
        self.allocation_order.prepare(reginfo);
        let mut context = Context {
            func,
            reginfo,
            allocator: self,
            uses,
            hints,
            reg_matrix,
            virt_regs,
            virt_reg_builder,
            spill_allocator,
            split_placement,
            coalescing,
            stats,
            split_strategy: options.split_strategy,
        };

        // Populate the queue with the initial set of virtual registers.
        context.allocator.queue.init(context.virt_regs);

        // Allocate each virtual register in priority order.
        // TODO(perf): Optimize the case where we dequeue the same vreg twice in a row
        while let Some((vreg, stage)) = context.allocator.queue.dequeue() {
            match vreg {
                VirtRegOrGroup::Reg(vreg) => {
                    context.allocate(vreg, stage, options.const_allocation_order)?
                }
                VirtRegOrGroup::Group(group) => {
                    context.allocate(group, stage, options.const_allocation_order)?
                }
            };
        }

        if trace_enabled!() {
            virt_regs.dump(uses, |vreg| {
                !matches!(self.assignments[vreg], Assignment::Dead)
            });
            trace!("Virtual register assignments:");
            for (vreg, reg) in self.assignments() {
                trace!("  {vreg} -> {reg}");
            }
            trace!("Empty segments:");
            for segment in &self.empty_segments {
                trace!("  {} ({})", segment.live_range, segment.value);
                for u in &uses[segment.use_list] {
                    trace!("  - {}: {}", u.pos, u.kind);
                }
            }
            trace!("Rematerialized segments:");
            for segment in &self.remat_segments {
                trace!("  {} ({})", segment.live_range, segment.value);
                for u in &uses[segment.use_list] {
                    trace!("  - {}: {}", u.pos, u.kind);
                }
            }
            trace!("Spilled segments:");
            for (_, segment) in spill_allocator.spilled_segments() {
                trace!("  {} ({})", segment.live_range, segment.value);
                for u in &uses[segment.use_list] {
                    trace!("  - {}: {}", u.pos, u.kind);
                }
            }
        }

        // Ensure all virtual registers are dead or assigned.
        for vreg in virt_regs.virt_regs() {
            debug_assert!(matches!(
                self.assignments[vreg],
                Assignment::Assigned { .. } | Assignment::Dead
            ));
        }

        Ok(())
    }

    /// Returns the mapping of all `VirtReg` to `PhysReg` for which an
    /// assignment exists.
    ///
    /// This does not include any virtual registers that have been spilled.
    pub fn assignments(&self) -> impl Iterator<Item = (VirtReg, PhysReg)> + '_ {
        self.assignments
            .iter()
            .filter_map(|(vreg, assignment)| match *assignment {
                Assignment::Assigned {
                    evicted_for_preference: _,
                    reg,
                    preference_weight: _,
                } => Some((vreg, reg)),
                Assignment::Unassigned { .. } => None,
                Assignment::Dead => None,
            })
    }
}

struct Context<'a, F, R> {
    func: &'a F,
    reginfo: &'a R,
    allocator: &'a mut Allocator,
    uses: &'a mut Uses,
    hints: &'a Hints,
    reg_matrix: &'a mut RegMatrix,
    virt_regs: &'a mut VirtRegs,
    virt_reg_builder: &'a mut VirtRegBuilder,
    spill_allocator: &'a mut SpillAllocator,
    split_placement: &'a SplitPlacement,
    coalescing: &'a mut Coalescing,
    stats: &'a mut Stats,
    split_strategy: SplitStrategy,
}

impl<F: Function, R: RegInfo> Context<'_, F, R> {
    /// Attempts to assign the given virtual register to a physical register.
    fn allocate(
        &mut self,
        vreg: impl AbstractVirtRegGroup,
        stage: Stage,
        const_alloc_order: bool,
    ) -> Result<(), RegAllocError> {
        trace!("Allocating {vreg} in stage {stage:?}");
        vreg.dump(self.virt_regs, self.uses);
        let first_vreg = vreg.first_vreg(self.virt_regs);
        if vreg.is_group() {
            stat!(self.stats, dequeued_group);
        } else {
            stat!(self.stats, dequeued_reg);
        }

        // Determine the order in which to probe for available registers.
        let hint = self.allocator.assignments[first_vreg].allocation_hint();
        self.allocator.allocation_order.compute(
            vreg,
            self.virt_regs,
            self.hints,
            self.reginfo,
            hint,
            const_alloc_order,
        );
        if trace_enabled!() {
            trace!("Allocation order:");
            for candidate in self.allocator.allocation_order.order() {
                trace!("  {}", candidate);
            }
        }

        // If the allocation order is empty then skip straight to spilling.
        if self.allocator.allocation_order.must_spill() {
            trace!("Empty allocation order, spilling immediately");
            stat!(self.stats, must_spill_vreg);
            self.spill(vreg);
            return Ok(());
        }

        // Try to find a physical register that can fit our virtual register
        // without any interference.
        trace!("Attempting direct assignment");
        if let Some(candidate) = self.find_available_reg(vreg) {
            trace!("-> Got candidate {candidate}");
            stat!(self.stats, found_free_reg);

            // We found a free register in which to allocate. However it may not
            // be the *best* register in which to allocate this virtual
            // register. If the virtual register has a fixed-register
            // constraint then it may be more profitable to evict an existing
            // virtual register from our preferred register.
            if candidate.preference_weight
                < self.allocator.allocation_order.highest_preferrence_weight()
                && !self.allocator.assignments[first_vreg].evicted_for_preference()
            {
                stat!(self.stats, try_evict_better_candidate);
                trace!("Searching for a better candidate by evicting from a preferred register");
                if let Some(better_candidate) = self.try_evict_for_preferred_reg(vreg, candidate) {
                    trace!("-> Found better candidate {better_candidate}");
                    stat!(self.stats, evicted_better_candidate);

                    // The initial candidate that we found is likely to fit any
                    // virtual registers that we evict. Use it as a hint.
                    let hint = (!vreg.is_group()).then_some(candidate.reg.as_single());
                    self.evict_interfering_vregs(hint);
                    self.assign(vreg, better_candidate, true);
                    return Ok(());
                }
                trace!("-> No better candidate found");
            }

            self.assign(vreg, candidate, false);
            return Ok(());
        }

        match stage {
            // First, try to evict any interfering virtual registers if they
            // have a lower spill weight.
            Stage::Evict => {
                trace!("Attempting to evict interfering registers");
                stat!(self.stats, try_evict);
                if self.try_evict(vreg) {
                    stat!(self.stats, assigned_after_evict);
                    return Ok(());
                }

                trace!("Failed to evict interference for {vreg}, re-queuing for splitting");

                // If the virtual register has an infinite spill weight (meaning
                // that it covers only a single instruction and cannot be
                // split further) then it means that the allocation problem is
                // fundamentally unsatisfiable.
                //
                // Note that this doesn't apply if the register class allows
                // allocation into a spillslot. This case is handled in the
                // splitting stage.
                if self.virt_regs[vreg.first_vreg(self.virt_regs)]
                    .spill_weight
                    .is_infinite()
                {
                    trace!("Allocation failed: could not allocate unspillable {vreg}");

                    if trace_enabled!() {
                        self.virt_regs.dump(self.uses, |vreg| {
                            !matches!(self.allocator.assignments[vreg], Assignment::Dead)
                        });
                        trace!("Virtual register assignments:");
                        for (vreg, reg) in self.allocator.assignments() {
                            trace!("  {vreg} -> {reg}");
                        }
                    }
                    return Err(RegAllocError::TooManyLiveRegs);
                }

                // If we failed to evict, re-queue for splitting after all
                // evictions have been processed so we have a clearer picture of
                // the interference we need to split around.
                self.allocator
                    .queue
                    .enqueue(vreg.into(), Stage::Split, self.virt_regs);
            }

            // If the virtual register failed to evict interference, then we
            // need to split it.
            Stage::Split => {
                trace!("Splitting {vreg} into smaller pieces");
                stat!(self.stats, try_split_or_spill);

                self.split_or_spill(vreg);
            }
        }

        Ok(())
    }

    /// Searches for a register that has no interference with the given virtual
    /// register.
    fn find_available_reg(&mut self, vreg: impl AbstractVirtRegGroup) -> Option<CandidateReg> {
        for cand in self.allocator.allocation_order.order() {
            trace!("Attempting to assign to {cand}");
            if vreg
                .zip_with_reg_group(cand.reg, self.virt_regs, self.reginfo)
                .all(|(vreg, reg)| {
                    stat!(self.stats, probe_for_free_reg);
                    self.reg_matrix
                        .check_interference(
                            self.virt_regs.segments(vreg),
                            reg,
                            self.reginfo,
                            self.stats,
                            false,
                            |_| ControlFlow::Break(()),
                        )
                        .is_continue()
                })
            {
                return Some(cand);
            }

            if trace_enabled!() {
                trace!("Interference found:");
                for (vreg, reg) in vreg.zip_with_reg_group(cand.reg, self.virt_regs, self.reginfo) {
                    let mut first = true;
                    _ = self.reg_matrix.check_interference(
                        self.virt_regs.segments(vreg),
                        reg,
                        self.reginfo,
                        &mut Default::default(),
                        true,
                        |interference| {
                            if first {
                                trace!("- For {vreg} in {reg}:");
                                first = false;
                            }
                            match interference.kind {
                                InterferenceKind::Fixed => {
                                    trace!(
                                        "  - Fixed interference at {} in {}",
                                        interference.range, interference.unit
                                    );
                                }
                                InterferenceKind::VirtReg(vreg) => {
                                    trace!(
                                        "  - Interference with {vreg} at {} in {} (weight={})",
                                        interference.range,
                                        interference.unit,
                                        self.virt_regs[vreg].spill_weight,
                                    );
                                }
                            }
                            ControlFlow::<()>::Continue(())
                        },
                    );
                }
            }
        }

        None
    }

    /// Assigns `vreg` to the chosen register.
    ///
    /// `evicted_for_preference` is true if we evicted a virtual register with a
    /// higher spill weight due to our preference for the register it was
    /// occupying.
    fn assign(
        &mut self,
        vreg: impl AbstractVirtRegGroup,
        candidate: CandidateReg,
        evicted_for_preference: bool,
    ) {
        trace!("Assigning {vreg} to {candidate} (evicted_for_preference={evicted_for_preference})");

        for (vreg, reg) in vreg.zip_with_reg_group(candidate.reg, self.virt_regs, self.reginfo) {
            debug_assert!(matches!(
                self.allocator.assignments[vreg],
                Assignment::Unassigned { .. }
            ));
            let evicted_for_preference =
                evicted_for_preference || self.allocator.assignments[vreg].evicted_for_preference();
            self.allocator.assignments[vreg] = Assignment::Assigned {
                evicted_for_preference,
                reg,
                preference_weight: candidate.preference_weight,
            };
            self.reg_matrix
                .assign(vreg, reg, self.virt_regs, self.reginfo);
        }
    }
}
