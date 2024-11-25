//! Allocation of spill slots for values which could not be placed in registers.

use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;

use super::coalescing::Coalescing;
use super::live_range::{LiveRangeSegment, Slot, ValueSegment};
use super::value_live_ranges::ValueSet;
use crate::entity::{PrimaryMap, SecondaryMap};
use crate::function::{Function, Inst, Value};
use crate::output::{SpillSlot, StackLayout};
use crate::reginfo::SpillSlotSize;
use crate::{RegAllocError, Stats};

/// All values in `ValueSet` (whose live ranges are therefore guaranteed not to
/// overlap) are assigned to the same spill slot in order to avoid unnecessary
/// stack-to-stack moves on block transitions.
///
/// Any spilled virtual registers in the same set will be assigned to the
/// same spill slot, which helps avoid unnecessary stack-to-stack moves.
struct SpillData {
    /// Size of the spill slot needed by this `ValueSet`.
    size: SpillSlotSize,

    /// Union of the live ranges of all segments in this `ValueSet`.
    live_range_union: LiveRangeSegment,

    /// Spill slot assigned to this value set by `allocate`.
    ///
    /// This is only valid if `spilled` is true.
    slot: SpillSlot,

    /// Whether this `ValueSet` requires a spill slot to be allocated.
    spilled: bool,
}

pub struct SpillAllocator {
    /// Spill data for all value sets, including those which don't contain any
    /// spilled segments yet.
    sets: SecondaryMap<ValueSet, SpillData>,

    /// Unsorted list of live ranges that have been spilled.
    spilled_segments: Vec<(ValueSet, ValueSegment)>,

    /// Stack frame layout.
    pub stack_layout: StackLayout,

    // Everything below this point is temporary storage used in the linear scan
    // allocation algorithm.
    //
    /// List of `ValueSet` for which a spill slot needs to be allocated. This is
    /// grouped by spill slot size, and within each group the value sets are
    /// sorted by the start point of their live range.
    sets_to_allocate: Vec<ValueSet>,

    /// Set of `ValueSet`s that are currently allocated to a spill slot at the
    /// current point in the scan.
    active_sets: Vec<ValueSet>,

    /// Set of `SpillSlot`s that are free for allocation at this point in the
    /// scan.
    available_slots: Vec<SpillSlot>,
}

impl SpillAllocator {
    pub fn new() -> Self {
        Self {
            sets: SecondaryMap::new(),
            stack_layout: StackLayout {
                slots: PrimaryMap::new(),
                spillslot_area_size: 0,
            },
            spilled_segments: vec![],
            sets_to_allocate: vec![],
            active_sets: vec![],
            available_slots: vec![],
        }
    }

    pub fn clear(&mut self, func: &impl Function) {
        // We don't need to clear self.sets, entries are initialized by
        // set_range and the uninitialized entries can be ignored.
        self.sets.grow_to_with(func.num_values(), || {
            let zero_point = Inst::new(0).slot(Slot::Boundary);
            SpillData {
                size: SpillSlotSize::from_log2_bytes(0),
                live_range_union: LiveRangeSegment::new(zero_point, zero_point),
                slot: SpillSlot::new(0),
                spilled: false,
            }
        });
        self.spilled_segments.clear();
        self.sets_to_allocate.clear();
    }

    /// Records the total live range of a `ValueSet` and its required spillslot
    /// size.
    ///
    /// The move optimizer relies on the entire range being reserved instead of
    /// just the range containing spilled segments.
    pub fn set_range(
        &mut self,
        set: ValueSet,
        size: SpillSlotSize,
        live_range_union: LiveRangeSegment,
    ) {
        self.sets[set].size = size;
        self.sets[set].live_range_union = live_range_union;
        self.sets[set].spilled = false;
    }

    /// Spills the given `ValueSegment` to a spill slot.
    pub fn spill_segment(&mut self, set: ValueSet, segment: ValueSegment) {
        debug_assert_eq!(
            self.sets[set]
                .live_range_union
                .intersection(segment.live_range),
            Some(segment.live_range)
        );
        trace!(
            "Spilling segment for {} at {} to {set}",
            segment.value,
            segment.live_range
        );
        self.spilled_segments.push((set, segment));
        if !self.sets[set].spilled {
            self.sets[set].spilled = true;
            self.sets_to_allocate.push(set);
        }
    }

    /// Returns all segments that have been spilled along with the `SpillSlot`
    /// that they have been spilled to.
    ///
    /// The spill slot is only valid after `allocate` has been called.
    pub fn spilled_segments(&self) -> impl Iterator<Item = (SpillSlot, &ValueSegment)> {
        self.spilled_segments
            .iter()
            .map(|&(set, ref segment)| (self.sets[set].slot, segment))
    }

    /// Returns the spill slot that the given value will be spilled to, or
    /// `None` if the value has not been spilled.
    pub fn value_spillslot(&self, value: Value, coalescing: &mut Coalescing) -> Option<SpillSlot> {
        let set = coalescing.set_for_value(value);
        self.sets[set].spilled.then_some(self.sets[set].slot)
    }

    /// Allocates a `SpillSlot` of the given size after stack allocation has
    /// already finished.
    ///
    /// This is used in the move resolver when a scratch register is needed but
    /// none is available.
    pub fn alloc_emergency_spillslot(&mut self, size: SpillSlotSize) -> SpillSlot {
        // Ensure the new slot is properly aligned.
        self.stack_layout.spillslot_area_size += size.bytes() - 1;
        self.stack_layout.spillslot_area_size &= !(size.bytes() - 1);

        // Allocate a new slot.
        let offset = self.stack_layout.spillslot_area_size;
        self.stack_layout.spillslot_area_size += size.bytes();
        self.stack_layout.slots.push((offset, size))
    }

    /// Assigns a `SpillSlot` to each `ValueSet` that has segments spilled into
    /// it.
    ///
    /// The basic algorithm here is based on linear scan allocation from
    /// <https://doi.org/10.1145/330249.330250> where each `ValueSet` is treated
    /// as a single live range segment. This has the advantage of being fast,
    /// but at the cost of not being able to allocate value sets in the live
    /// range gaps of another value set. This is less of an issue than for
    /// registers though since spill slots are effectively unlimited.
    pub fn allocate(&mut self, stats: &mut Stats) -> Result<(), RegAllocError> {
        self.stack_layout.slots.clear();
        self.stack_layout.spillslot_area_size = 0;
        self.active_sets.clear();
        self.available_slots.clear();

        trace!("Allocating spill slots:");

        // Gather the value sets that need to be allocated and sort them by
        // spill slot size first, and then by start position.
        stat!(stats, spilled_sets, self.sets_to_allocate.len());
        stat!(stats, spill_segments, self.spilled_segments.len());
        self.sets_to_allocate.sort_unstable_by_key(|&set| {
            (
                Reverse(self.sets[set].size),
                self.sets[set].live_range_union.from,
            )
        });

        for &set in &self.sets_to_allocate {
            trace!(
                "- {set}: {} {}",
                self.sets[set].size,
                self.sets[set].live_range_union
            );
        }

        let mut current_size = SpillSlotSize::new(1);
        for &set in &self.sets_to_allocate {
            // Restart linear scan if the spill slot size changes. We don't
            // mix spill slots of different sizes.
            if self.sets[set].size != current_size {
                self.available_slots.clear();
                self.active_sets.clear();
            }
            current_size = self.sets[set].size;

            // Remove any value sets whose live range ended before the current
            // set from the list of active sets. These are no longer active at
            // this point and their spill slot is now available for allocation.
            self.active_sets.retain(|&active_set| {
                if self.sets[active_set].live_range_union.to <= self.sets[set].live_range_union.from
                {
                    self.available_slots.push(self.sets[active_set].slot);
                    false
                } else {
                    true
                }
            });

            // Assign the set to an available slot or allocate a new slot.
            let slot = match self.available_slots.pop() {
                Some(slot) => slot,
                None => {
                    let slot = self
                        .stack_layout
                        .slots
                        .push((self.stack_layout.spillslot_area_size, current_size));

                    // This is guaranteed to be properly aligned because we start
                    // allocating from larger sizes first, and all sizes are
                    // powers of 2.
                    debug_assert_eq!(
                        self.stack_layout.spillslot_area_size % current_size.bytes(),
                        0
                    );
                    self.stack_layout.spillslot_area_size = self
                        .stack_layout
                        .spillslot_area_size
                        .checked_add(current_size.bytes())
                        .ok_or(RegAllocError::FunctionTooBig)?;
                    slot
                }
            };
            self.sets[set].slot = slot;
            trace!("Assigned {set} to {slot}");
            self.active_sets.push(set);
        }
        stat!(stats, spillslots, self.stack_layout.slots.len());
        stat!(
            stats,
            spill_area_size,
            self.stack_layout.spillslot_area_size as usize
        );

        Ok(())
    }
}
