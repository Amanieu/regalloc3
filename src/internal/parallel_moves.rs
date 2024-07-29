//! Resolution of parallel moves into a sequences of move operations, possibly
//! using a scratch register to resolve cycles.

use alloc::vec;
use alloc::vec::Vec;

use cranelift_entity::packed_option::{PackedOption, ReservedValue};
use cranelift_entity::{PrimaryMap, SecondaryMap};
use smallvec::{smallvec, SmallVec};

use super::move_resolver::Edit;
use crate::allocation_unit::AllocationUnitMap;
use crate::function::{Function, RematCost, Value};
use crate::internal::allocator::combined_allocation_order;
use crate::output::{Allocation, AllocationKind, SpillSlot};
use crate::reginfo::{
    PhysReg, RegBank, RegClass, RegInfo, RegOrRegGroup, RegUnit, RegUnitSet, SpillSlotSize,
};

/// Cache for reusing emergency spill slots.
///
/// This avoids unbounded stack growth if multiple parallel moves in a function
/// all need emergency spill slots.
///
/// This is the only part of the parallel move resolver that is preserved across
/// multiple parallel moves.
struct EmergencySpillSlotCache {
    slots_by_size: [Vec<SpillSlot>; 32],
}

impl EmergencySpillSlotCache {
    fn new() -> Self {
        EmergencySpillSlotCache {
            slots_by_size: [const { vec![] }; 32],
        }
    }

    fn clear(&mut self) {
        self.slots_by_size.iter_mut().for_each(|v| v.clear());
    }

    fn acquire(
        &mut self,
        size: SpillSlotSize,
        alloc_emergency_spillslot: &mut impl FnMut(SpillSlotSize) -> SpillSlot,
    ) -> SpillSlot {
        if let Some(slot) = self.slots_by_size[size.log2_bytes() as usize].pop() {
            slot
        } else {
            alloc_emergency_spillslot(size)
        }
    }

    fn release(&mut self, slot: SpillSlot, size: SpillSlotSize) {
        self.slots_by_size[size.log2_bytes() as usize].push(slot);
    }
}

/// An index into the set of parallel moves that is currently being considered.
#[derive(Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
struct MoveIndex(u16);
entity_impl!(MoveIndex(u16), "move");

/// Information about a parallel move.
struct Move {
    /// Source of the move.
    source: Allocation,

    /// Destination of the move.
    dest: Allocation,

    /// Value being moved.
    ///
    /// If the value is rematerializable then the move can be eliminated.
    value: Value,

    /// Register bank for this move.
    bank: RegBank,

    /// State tracking for DFS.
    state: State,

    /// Values which have been temporarily diverted to scratch registers
    /// due to a cycle conflict with this move.
    ///
    /// These are released after this move is processed.
    diverted_values: SmallVec<[Value; 4]>,
}

/// Information about a value that has been diverted to a scratch register to
/// resolve a move cycle.
#[derive(Debug, Clone, Copy)]
struct Diversion {
    /// Number of pending moves that make this diversion required because they
    /// would overwrite the diverted value.
    ///
    /// The other fields below should be ignored if this is 0.
    ref_count: u32,

    /// Allocation that was originally holding the value.
    alloc: Allocation,

    /// Allocation to which the value has been diverted.
    scratch: Allocation,
}

impl Default for Diversion {
    fn default() -> Self {
        Self {
            ref_count: 0,
            alloc: Allocation::reserved_value(),
            scratch: Allocation::reserved_value(),
        }
    }
}

/// Allocator for scratch registers which can be used in move resolution.
///
/// This also keeps track of values diverted to scratch registers.
struct ScratchAllocator {
    /// Register units available for use as scratch registers.
    available: RegUnitSet,

    /// Register units which have been checked to be available for use as
    /// scratch registers. The RegMatrix needs to be queried to check if these
    /// register units are actually unused.
    probed: RegUnitSet,

    /// Move cycles are resolved by using a scratch register to hold the
    /// original value of a move destination before it is overwritten.
    ///
    /// All accesses to the original move destination need to be diverted to the
    /// scratch register instead. Additionally, once the diversion is no longer
    /// needed then the scratch register needs to be released.
    ///
    /// This map tracks currently active diversions, and a reference count for
    /// how many pending moves still conflict with the register.
    diverted: SecondaryMap<Value, Diversion>,

    /// If we need a scratch register but none are available then we need to
    /// evict an existing value in a register to an emergency spillslot.
    ///
    /// However we need to later restore this register if its value is needed
    /// for another move.
    evicted_reg: Option<(PhysReg, SpillSlot, SpillSlotSize)>,

    /// Cache for emergency spill slots.
    ///
    /// This is preserved across multiple parallel moves.
    emergency_spillslot_cache: EmergencySpillSlotCache,
}

impl ScratchAllocator {
    fn new() -> Self {
        Self {
            available: RegUnitSet::new(),
            probed: RegUnitSet::new(),
            diverted: SecondaryMap::new(),
            evicted_reg: None,
            emergency_spillslot_cache: EmergencySpillSlotCache::new(),
        }
    }

    fn clear(&mut self) {
        self.available.clear();
        self.probed.clear();
        self.diverted.clear();
        self.evicted_reg = None;
    }

    /// Makes the given register available for use as a scratch register.
    ///
    /// This should be called after writing to a move destination which, since
    /// we are emitting moves in reverse order, means that the destination
    /// register will not be read by any moves earlier in program order.
    fn make_available(&mut self, reg: PhysReg, reginfo: &impl RegInfo) {
        for &unit in reginfo.reg_units(reg) {
            self.available.insert(unit);
            self.probed.insert(unit);
        }
    }

    /// Makes the given register unavailable for use as a scratch register.
    ///
    /// This should be called after reading from a move source which, since
    /// we are emitting moves in reverse order, means that the source
    /// register will not be overwritten by any moves earlier in program order.
    fn make_unavailable(&mut self, reg: PhysReg, reginfo: &impl RegInfo) {
        for &unit in reginfo.reg_units(reg) {
            self.available.remove(unit);
            self.probed.insert(unit);
        }
    }

    /// Restore all evicted registers.
    fn unevict_all(&mut self, edits: &mut Vec<Edit>) {
        if let Some((reg, spillslot, size)) = self.evicted_reg {
            // We are emitting moves in reverse so this comes before all uses of
            // the scratch register in program order. Save the scratch register
            // value into the emergency spill slot.
            trace!("-> Restoring evicted {reg} from {spillslot}");
            edits.push(Edit {
                value: None.into(),
                from: Some(Allocation::reg(reg)).into(),
                to: Allocation::spillslot(spillslot),
            });
            self.evicted_reg = None;
            self.emergency_spillslot_cache.release(spillslot, size);
        }
    }

    /// Restore all evicted registers that conflict with the given register.
    fn unevict_for_reg(
        &mut self,
        alloc: Allocation,
        edits: &mut Vec<Edit>,
        reginfo: &impl RegInfo,
    ) {
        // Spillslots are never evicted.
        let AllocationKind::PhysReg(reg) = alloc.kind() else {
            return;
        };

        if let Some((evicted_reg, spillslot, size)) = self.evicted_reg {
            // Only do this if the given register overlaps with the evicted
            // register.
            if !reginfo
                .reg_units(reg)
                .iter()
                .any(|&unit| reginfo.reg_units(evicted_reg).contains(&unit))
            {
                return;
            }

            // We are emitting moves in reverse so this comes before all uses of
            // the scratch register in program order. Save the scratch register
            // value into the emergency spill slot.
            trace!("-> Restoring evicted {evicted_reg} from {spillslot}");
            edits.push(Edit {
                value: None.into(),
                from: Some(Allocation::reg(evicted_reg)).into(),
                to: Allocation::spillslot(spillslot),
            });
            self.evicted_reg = None;
            self.emergency_spillslot_cache.release(spillslot, size);
        }
    }

    /// Returns a scratch `Allocation` in the given `RegClass` which is
    /// currently unused.
    ///
    /// This may return a `SpillSlot` if the register class allows spillslots.
    ///
    /// If `cycle_move_sources` is set then this contains the set of registers
    /// which should be excluded from consideration as a scratch register.
    fn get_scratch_reg(
        &mut self,
        bank: RegBank,
        class: RegClass,
        cycle_move_sources: Option<&RegUnitSet>,
        edits: &mut Vec<Edit>,
        reginfo: &impl RegInfo,
        is_unit_free: &impl Fn(RegUnit) -> bool,
        alloc_emergency_spillslot: &mut impl FnMut(SpillSlotSize) -> SpillSlot,
    ) -> Allocation {
        trace!("Searching for scratch register in {class}");
        for reg in
            combined_allocation_order(reginfo, class, 0, |_| false).map(RegOrRegGroup::as_single)
        {
            if reginfo.reg_units(reg).iter().all(|&unit| {
                // If we need the scratch register for resolving a cycle,
                // don't select a move source involved in the cycle.
                if let Some(cycle_move_sources) = cycle_move_sources {
                    if cycle_move_sources.contains(unit) {
                        return false;
                    }
                }

                // Lazily probe the allocation matrix for free registers only
                // when needed.
                if !self.probed.contains(unit) {
                    debug_assert!(!self.available.contains(unit));
                    self.probed.insert(unit);

                    if is_unit_free(unit) {
                        self.available.insert(unit);
                    }
                }

                self.available.contains(unit)
            }) {
                trace!("-> got scratch register {reg}");
                return Allocation::reg(reg);
            }
        }

        // If the register class allows spillslots, return an emergency
        // spillslot directly.
        if reginfo.class_includes_spillslots(class) {
            let spillslot = self
                .emergency_spillslot_cache
                .acquire(reginfo.spillslot_size(bank), alloc_emergency_spillslot);
            trace!("-> got emergency spill slot {spillslot}");
            return Allocation::spillslot(spillslot);
        }

        // Try to reuse an already-evicted register if it is in the correct
        // class. We *only* do this if the class doesn't allow spillslots: this
        // is important since evicted register should only be used for a single
        // move like a rematerialization or for resolving a stack-to-stack move.
        //
        // We don't want register diversions used for cycle resolution to use
        // evicted registers since other moves in the cycle could require
        // un-evicting that register.
        if let Some((reg, _spillslot, _size)) = self.evicted_reg {
            if reginfo.class_contains(class, RegOrRegGroup::single(reg)) {
                trace!("-> re-using previously evicted {reg}");
                return Allocation::reg(reg);
            }

            // We only track one evicted register at a time.
            self.unevict_all(edits);
        }

        // Otherwise we need to evict a register from the class into the
        // emergency spill slot. We pick the *last* register of the class since
        // it is least likely to hold a hot value.
        let size = reginfo.spillslot_size(bank);
        let spillslot = self
            .emergency_spillslot_cache
            .acquire(size, alloc_emergency_spillslot);
        let reg = combined_allocation_order(reginfo, class, 0, |_| false)
            .map(RegOrRegGroup::as_single)
            .next_back()
            .unwrap();
        trace!("-> evicted {reg} to emergency spillslot {spillslot}");

        // Since we are emitting moves in reverse order, this is after the
        // *last* use of the scratch register. We need to restore the scratch
        // register contents from the emergency spill slot.
        edits.push(Edit {
            value: None.into(),
            from: Some(Allocation::spillslot(spillslot)).into(),
            to: Allocation::reg(reg),
        });

        self.evicted_reg = Some((reg, spillslot, size));

        Allocation::reg(reg)
    }

    /// Diverts the given value to a scratch register to resolve cyclic
    /// conflicts in the parallel moves. A scratch register is selected and
    /// returned.
    fn divert(
        &mut self,
        ref_count: u32,
        value: Value,
        alloc: Allocation,
        bank: RegBank,
        cycle_move_sources: &RegUnitSet,
        edits: &mut Vec<Edit>,
        reginfo: &impl RegInfo,
        is_unit_free: &impl Fn(RegUnit) -> bool,
        alloc_emergency_spillslot: &mut impl FnMut(SpillSlotSize) -> SpillSlot,
    ) -> Allocation {
        let mut scratch = self.diverted[value].scratch;
        let prev_count = self.diverted[value].ref_count;
        self.diverted[value].ref_count += ref_count;
        if ref_count == 0 {
            debug_assert_ne!(prev_count, 0);
        }

        // If there is no existing scratch register for this value, allocate a
        // new one. If no scratch registers are available then we fall back to
        // an emergency spill slot.
        if prev_count == 0 {
            scratch = self.get_scratch_reg(
                bank,
                reginfo.top_level_class(bank),
                Some(cycle_move_sources),
                edits,
                reginfo,
                is_unit_free,
                alloc_emergency_spillslot,
            );

            if let AllocationKind::PhysReg(reg) = scratch.kind() {
                for &unit in reginfo.reg_units(reg) {
                    self.available.remove(unit);
                }
            }

            self.diverted[value].scratch = scratch;
            self.diverted[value].alloc = alloc;
        } else {
            debug_assert_eq!(self.diverted[value].scratch, scratch);
        }

        scratch
    }

    /// Restores a diverted value to its original location after the move that
    /// would have overwritten it is processed.
    fn undivert(
        &mut self,
        value: Value,
        bank: RegBank,
        edits: &mut Vec<Edit>,
        reginfo: &impl RegInfo,
        is_unit_free: &impl Fn(RegUnit) -> bool,
        alloc_emergency_spillslot: &mut impl FnMut(SpillSlotSize) -> SpillSlot,
    ) {
        self.diverted[value].ref_count -= 1;
        if self.diverted[value].ref_count == 0 {
            trace!("Ending diversion of {value}");

            self.do_move(
                self.diverted[value].alloc,
                self.diverted[value].scratch,
                value,
                bank,
                edits,
                reginfo,
                is_unit_free,
                alloc_emergency_spillslot,
            );

            // If get_scratch_reg gave us an emergency spillslot, release it so
            // that is can later be reused.
            if let AllocationKind::SpillSlot(slot) = self.diverted[value].scratch.kind() {
                self.emergency_spillslot_cache
                    .release(slot, reginfo.spillslot_size(bank));
            }
        }
    }

    /// Performs a move operation from one allocation to another.
    ///
    /// This handles details like stack-to-stack moves and conflicts with
    /// evicted registers.
    fn do_move(
        &mut self,
        from: Allocation,
        to: Allocation,
        value: Value,
        bank: RegBank,
        edits: &mut Vec<Edit>,
        reginfo: &impl RegInfo,
        is_unit_free: &impl Fn(RegUnit) -> bool,
        alloc_emergency_spillslot: &mut impl FnMut(SpillSlotSize) -> SpillSlot,
    ) {
        trace!("Emitting move {to} <- {from}");

        // Ensure that neither the source nor destination of the move conflict
        // with currently evicted registers. If so then restore these registers.
        self.unevict_for_reg(from, edits, reginfo);
        self.unevict_for_reg(to, edits, reginfo);

        // Check if this is a stack-to-stack move, which requires an
        // intermediate scratch register.
        if from.is_memory(reginfo) && to.is_memory(reginfo) {
            let class = reginfo.stack_to_stack_class(bank);
            trace!("-> need scratch register for stack-to-stack move ({bank}:{class})");
            let scratch = self.get_scratch_reg(
                bank,
                class,
                None,
                edits,
                reginfo,
                is_unit_free,
                alloc_emergency_spillslot,
            );
            edits.push(Edit {
                value: Some(value).into(),
                from: Some(scratch).into(),
                to,
            });
            edits.push(Edit {
                value: Some(value).into(),
                from: Some(from).into(),
                to: scratch,
            });
        } else {
            edits.push(Edit {
                value: Some(value).into(),
                from: Some(from).into(),
                to,
            });
        }

        // Make the destination register available as a scratch
        // register: earlier moves in program order are free to
        // use it since we will overwrite its contents.
        if let AllocationKind::PhysReg(reg) = to.kind() {
            self.make_available(reg, reginfo);
        }

        // Make the source register unavailable as a scratch
        // register: earlier moves in program order must not
        // clobber its value.
        //
        // In the case of scratch registers this will be made
        // available again once the diversion ends.
        if let AllocationKind::PhysReg(reg) = from.kind() {
            self.make_unavailable(reg, reginfo);
        }
    }
}

/// State tracking for DFS.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum State {
    /// Not on stack, not visited
    New,
    /// On stack, not yet visited
    Pending,
    /// Visited
    Done,
}

/// DFS stack state marker.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Visit {
    First,
    Last,
}

/// Resolver which turns a set of parallel move operations into a linear
/// sequence of move operations.
///
/// This is also responsible for inserting the necessary scratch registers for
/// things like stack-to-stack moves and rematerialization into a stack slot.
pub struct ParallelMoves {
    /// Final list of moves generated by the parallel move resolver.
    ///
    /// The edits are stored in *reverse order* due to the way we generate them.
    edits: Vec<Edit>,

    /// Set of parallel move operations that need to be resolved.
    moves: PrimaryMap<MoveIndex, Move>,

    /// List of values that will be re-materialized separately after all moves
    /// have been performed.
    remat: Vec<(Value, RegClass, Allocation)>,

    /// Same as `remat` but for rematerializations where an intermediate scratch
    /// register is needed.
    remat_with_scratch: Vec<(Value, RegClass, Allocation)>,

    /// For each allocation unit that is written to by a move, this holds the
    /// index of the move writing to it.
    writes_to_unit: AllocationUnitMap<PackedOption<MoveIndex>>,

    /// Allocator for scratch registers that may be needed to resolve cycles and
    /// memory-to-memory moves.
    scratch: ScratchAllocator,

    /// Stack for DFS.
    stack: Vec<(Visit, MoveIndex)>,
}

impl Default for ParallelMoves {
    fn default() -> Self {
        Self::new()
    }
}

impl ParallelMoves {
    #[must_use]
    pub fn new() -> Self {
        Self {
            edits: vec![],
            moves: PrimaryMap::new(),
            remat: vec![],
            remat_with_scratch: vec![],
            writes_to_unit: AllocationUnitMap::new(),
            scratch: ScratchAllocator::new(),
            stack: vec![],
        }
    }

    pub fn clear(&mut self) {
        self.scratch.emergency_spillslot_cache.clear();
    }

    pub fn new_parallel_move(&mut self) {
        self.moves.clear();
        self.remat.clear();
        self.remat_with_scratch.clear();
        self.writes_to_unit.clear();
    }

    pub fn add_remat(
        &mut self,
        dest: Allocation,
        value: Value,
        func: &impl Function,
        reginfo: &impl RegInfo,
    ) {
        let (_cost, class) = func
            .can_rematerialize(value)
            .expect("add_remat called with non-rematerializable value");

        let need_scratch = match dest.kind() {
            AllocationKind::PhysReg(reg) => {
                !reginfo.class_contains(class, RegOrRegGroup::single(reg))
            }
            AllocationKind::SpillSlot(_) => !reginfo.class_includes_spillslots(class),
        };
        if need_scratch {
            self.remat_with_scratch.push((value, class, dest));
        } else {
            self.remat.push((value, class, dest));
        }
    }

    pub fn add_move(
        &mut self,
        source: Allocation,
        dest: Allocation,
        value: Value,
        func: &impl Function,
        reginfo: &impl RegInfo,
    ) {
        // Self-moves will confuse our cycle detection algorithm.
        debug_assert_ne!(source, dest);

        // Handle cases where rematerialization is always better than a move.
        if let Some((cost, class)) = func.can_rematerialize(value) {
            if cost == RematCost::CheaperThanMove || source.is_memory(reginfo) {
                let need_scratch = match dest.kind() {
                    AllocationKind::PhysReg(reg) => {
                        !reginfo.class_contains(class, RegOrRegGroup::single(reg))
                    }
                    AllocationKind::SpillSlot(_) => !reginfo.class_includes_spillslots(class),
                };
                if need_scratch {
                    self.remat_with_scratch.push((value, class, dest));
                } else {
                    self.remat.push((value, class, dest));
                }
                return;
            }
        }

        // Ignore duplicate moves where the source and destination are the same.
        let unit = dest.units(reginfo).next().unwrap();
        if let Some(index) = self.writes_to_unit[unit].into() {
            debug_assert_eq!(self.moves[index].source, source);
            debug_assert_eq!(self.moves[index].dest, dest);
            debug_assert_eq!(self.moves[index].value, value);
            return;
        }

        let index = self.moves.push(Move {
            source,
            dest,
            value,
            bank: func.value_bank(value),
            state: State::New,
            diverted_values: smallvec![],
        });

        // Track which moves writes to which register unit. This is needed to
        // correctly order moves and resolve cycles.
        for unit in dest.units(reginfo) {
            debug_assert!(self.writes_to_unit[unit].is_none());
            self.writes_to_unit[unit] = Some(index).into();
        }
    }

    pub fn resolve(
        &mut self,
        reginfo: &impl RegInfo,
        func: &impl Function,
        is_unit_free: impl Fn(RegUnit) -> bool,
        mut alloc_emergency_spillslot: impl FnMut(SpillSlotSize) -> SpillSlot,
    ) {
        // Fast path if no moves are needed (all moves were resolved as
        // self-moves).
        self.edits.clear();
        if self.moves.is_empty() && self.remat.is_empty() && self.remat_with_scratch.is_empty() {
            return;
        }

        trace!("Resolving parallel moves");

        // Process rematerializations first: this will free up scratch registers
        // which can be used later. Since we emit moves in reverse, these will
        // happen after all normal moves.
        //
        // We process rematerializations that don't need a scratch register
        // first since they can free up a scratch register for later
        // rematerializations.
        self.scratch.clear();
        for &(value, class, dest) in &self.remat {
            trace!("Processing remat of {value} into {dest} with {class}");

            self.edits.push(Edit {
                value: Some(value).into(),
                from: None.into(),
                to: dest,
            });

            // Make the destination register available as a scratch register.
            if let AllocationKind::PhysReg(reg) = dest.kind() {
                self.scratch.make_available(reg, reginfo);
            }
        }
        for &(value, class, dest) in &self.remat_with_scratch {
            trace!("Processing remat of {value} into {dest} with {class} with scratch register");

            // Ensure that our destination register hasn't been evicted into an
            // emergency spill slot. This can happen if a previous
            // rematerialization had to evict for a scratch register.
            self.scratch.unevict_for_reg(dest, &mut self.edits, reginfo);

            // Get a free register in the given class as a scratch register.
            let bank = reginfo.bank_for_class(class);
            let scratch = self.scratch.get_scratch_reg(
                bank,
                class,
                None,
                &mut self.edits,
                reginfo,
                &is_unit_free,
                &mut alloc_emergency_spillslot,
            );

            // Copy the scratch register to the destination.
            //
            // If the first call to `get_scratch_reg` gave us a spill slot then
            // this may be a stack-to-stack move that needs another
            // intermediate scratch register.
            if scratch.is_memory(reginfo) && dest.is_memory(reginfo) {
                let class = reginfo.stack_to_stack_class(bank);
                trace!("-> need scratch register for stack-to-stack move ({bank}:{class})");
                let scratch2 = self.scratch.get_scratch_reg(
                    bank,
                    class,
                    None,
                    &mut self.edits,
                    reginfo,
                    &is_unit_free,
                    &mut alloc_emergency_spillslot,
                );
                self.edits.push(Edit {
                    value: Some(value).into(),
                    from: Some(scratch2).into(),
                    to: dest,
                });
                self.edits.push(Edit {
                    value: Some(value).into(),
                    from: Some(scratch).into(),
                    to: scratch2,
                });
            } else {
                self.edits.push(Edit {
                    value: Some(value).into(),
                    from: Some(scratch).into(),
                    to: dest,
                });
            }

            // Rematerialize into the scratch register.
            self.edits.push(Edit {
                value: Some(value).into(),
                from: None.into(),
                to: scratch,
            });

            // If get_scratch_reg gave us an emergency spillslot, release it so
            // that is can later be reused.
            if let AllocationKind::SpillSlot(slot) = scratch.kind() {
                self.scratch
                    .emergency_spillslot_cache
                    .release(slot, reginfo.spillslot_size(bank));
            }

            // Make the destination register available as a scratch register.
            if let AllocationKind::PhysReg(reg) = dest.kind() {
                self.scratch.make_available(reg, reginfo);
            }
        }

        // Compute a topological ordering of the move graph using depth-first
        // search. Moves are emitted in post-order as we unwind back out of the
        // stack.
        //
        // Since a topological ordering is only possible for acyclic graphs, we
        // break cycles as they are discovered by using a scratch register.
        self.stack.clear();
        for pass in 0..2 {
            for m in self.moves.keys() {
                // Nothing to do if this move has already been processed.
                if self.moves[m].state != State::New {
                    debug_assert_eq!(self.moves[m].state, State::Done);
                    continue;
                }

                // Skip stack-to-stack moves in the first pass. Process other
                // moves first to make scratch registers available.
                if pass == 0
                    && (self.moves[m].source.is_memory(reginfo)
                        || self.moves[m].dest.is_memory(reginfo))
                {
                    continue;
                }

                self.stack.push((Visit::First, m));
                while let Some((visit, m)) = self.stack.pop() {
                    let value = self.moves[m].value;
                    let bank = self.moves[m].bank;
                    let source = self.moves[m].source;
                    let dest = self.moves[m].dest;
                    match visit {
                        Visit::First => {
                            // Nothing to do if this move has already been
                            // processed.
                            if self.moves[m].state != State::New {
                                debug_assert_eq!(self.moves[m].state, State::Done);
                                continue;
                            }

                            trace!("First visit of move of {value} from {source} to {dest}");

                            // Visit any moves that may overwrite our source and
                            // that haven't been visited yet. This is necessary
                            // for proper cycle detection when register many
                            // span multiple units.
                            //
                            // The actual move is emitted on Visit::Last.
                            self.moves[m].state = State::Pending;
                            self.stack.push((Visit::Last, m));
                            for m2 in source
                                .units(reginfo)
                                .filter_map(|unit| self.writes_to_unit[unit].expand())
                            {
                                if self.moves[m2].state == State::New {
                                    self.stack.push((Visit::First, m2));
                                }
                            }
                        }
                        Visit::Last => {
                            debug_assert_eq!(self.moves[m].state, State::Pending);
                            self.moves[m].state = State::Done;

                            trace!("Second visit of move of {value} from {source} to {dest}");

                            // There is a cycle if the source of this move would
                            // be overwritten by a prior move on the stack.
                            let cycle = source
                                .units(reginfo)
                                .filter_map(|unit| self.writes_to_unit[unit].expand())
                                .any(|m2| self.moves[m2].state == State::Pending);

                            let adjusted_source = if cycle {
                                // Break the cycle by using a scratch register
                                // as the source of the move instead. The
                                // scratch register will remain reserved until
                                // the DFS unwinds back to the move that
                                // overwrites our source. At that point, we can
                                // copy the source to the scratch register
                                // before it is overwritten.

                                // For each move that would overwrite our source,
                                // record the diversion so that it is undone
                                // after all such moves are processed. This is
                                // done by assigning a reference count to each
                                // diversion.
                                let mut count = 0;
                                for m2 in source
                                    .units(reginfo)
                                    .filter_map(|unit| self.writes_to_unit[unit].expand())
                                {
                                    if self.moves[m2].state == State::Pending {
                                        if !self.moves[m2].diverted_values.contains(&value) {
                                            self.moves[m2].diverted_values.push(value);
                                            count += 1;
                                        }
                                    }
                                }

                                // Then, allocate a scratch register to hold the
                                // diverted value.
                                //
                                // We need to select a scratch register that
                                // does not clobber any move source involved in
                                // the cycle.
                                let mut cycle_move_sources = RegUnitSet::new();
                                for move_ in self.moves.values() {
                                    if let AllocationKind::PhysReg(reg) = move_.source.kind() {
                                        for &unit in reginfo.reg_units(reg) {
                                            cycle_move_sources.insert(unit);
                                        }
                                    }
                                }
                                let scratch = self.scratch.divert(
                                    count,
                                    value,
                                    source,
                                    bank,
                                    &cycle_move_sources,
                                    &mut self.edits,
                                    reginfo,
                                    &is_unit_free,
                                    &mut alloc_emergency_spillslot,
                                );
                                trace!(
                                    "-> cycle detected! Diverting {value} in {source} to {scratch}"
                                );

                                scratch
                            } else {
                                source
                            };

                            // After moves that write to our source have been
                            // emitted (which will happen *after* this move once
                            // the order is reversed) then we can emit the
                            // current move.
                            self.scratch.do_move(
                                adjusted_source,
                                dest,
                                value,
                                bank,
                                &mut self.edits,
                                reginfo,
                                &is_unit_free,
                                &mut alloc_emergency_spillslot,
                            );

                            // Release any scratch registers used for
                            // diversions.
                            //
                            // This will emit moves to initialize scratch
                            // registers with values before this move overwrites
                            // them.
                            for &value in &self.moves[m].diverted_values {
                                self.scratch.undivert(
                                    value,
                                    func.value_bank(value),
                                    &mut self.edits,
                                    reginfo,
                                    &is_unit_free,
                                    &mut alloc_emergency_spillslot,
                                );
                            }
                        }
                    }
                }
            }
        }

        // Restore all evicted registers at the end of the parallel moves.
        //
        // Since moves are inserted in reverse order, this actually emits the
        // code to save the evicted registers to emergency spill slots.
        self.scratch.unevict_all(&mut self.edits);
    }

    /// Returns the linear move sequence that was resolved by `resolve`.
    pub fn edits(&self) -> impl Iterator<Item = Edit> + '_ {
        self.edits.iter().rev().copied()
    }
}
