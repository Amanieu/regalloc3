//! Allocation of spill slots for values which could not be placed in registers.

use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;

use cranelift_entity::packed_option::ReservedValue;
use cranelift_entity::PrimaryMap;

use super::live_range::LiveRangeSegment;
use super::value_live_ranges::ValueSegment;
use crate::output::{SpillSlot, StackLayout};
use crate::reginfo::SpillSlotSize;
use crate::{RegAllocError, Stats};

/// A spill set is used to assign spill slots to spilled virtual registers.
///
/// A single spill set is created for each group of coalesced SSA values (whose
/// live ranges are therefore guaranteed not to overlap). This spill set is
/// preserved by virtual registers constructed from these SSA values, even
/// through splitting.
///
/// Any spilled virtual registers in the same spill set will be assigned to the
/// same spill slot, which helps avoid unnecessary stack-to-stack moves.
#[derive(Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub struct SpillSet(u32);
entity_impl!(SpillSet(u32), "spillset");

struct SpillSetData {
    /// Size of the spill slot needed by this `SpillSet`.
    size: SpillSlotSize,

    /// Union of the live ranges of all segments spilled to this `SpillSet`.
    live_range_union: LiveRangeSegment,

    /// Spill slot assigned to this spill set by `allocate`.
    ///
    /// This is only valid if `range` is non-empty.
    slot: SpillSlot,
}

pub struct SpillAllocator {
    /// All spill sets, including those which don't contain any spills yet.
    sets: PrimaryMap<SpillSet, SpillSetData>,

    /// Unsorted list of live ranges that have been spilled to a `SpillSet`.
    spilled_segments: Vec<(SpillSet, ValueSegment)>,

    /// Stack frame layout.
    pub stack_layout: StackLayout,

    // Everything below this point is temporary storage used in the linear scan
    // allocation algorithm.
    //
    /// List of `SpillSet` for which a spill slot needs to be allocated. This is
    /// grouped by spill slot size, and within each group the spill sets are
    /// sorted by the start point of their live range.
    sets_to_allocate: Vec<SpillSet>,

    /// Set of `SpillSet`s that are currently allocated to a spill slot at the
    /// current point in the scan.
    active_sets: Vec<SpillSet>,

    /// Set of `SpillSlot`s that are free for allocation at this point in the
    /// scan.
    available_slots: Vec<SpillSlot>,
}

impl SpillAllocator {
    pub fn new() -> Self {
        Self {
            sets: PrimaryMap::new(),
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

    /// Clears any existing `SpillSet` definitions.
    pub fn clear(&mut self) {
        self.sets.clear();
        self.spilled_segments.clear();
    }

    /// Creates a new `SpillSet` with the given spill slot size.
    pub fn new_spillset(&mut self, size: SpillSlotSize) -> SpillSet {
        self.sets.push(SpillSetData {
            size,
            live_range_union: LiveRangeSegment::EMPTY,
            slot: SpillSlot::reserved_value(),
        })
    }

    /// Spills the given `ValueSegment` to a spill set.
    pub fn spill_segment(&mut self, set: SpillSet, segment: ValueSegment) {
        debug_assert!(!segment.live_range.is_empty());
        trace!(
            "Spilling segment for {} at {} to {set}",
            segment.value,
            segment.live_range
        );
        self.sets[set].live_range_union = self.sets[set].live_range_union.union(segment.live_range);
        self.spilled_segments.push((set, segment));
    }

    pub fn spilled_segments(&self) -> impl Iterator<Item = (SpillSlot, &ValueSegment)> {
        self.spilled_segments
            .iter()
            .map(|&(set, ref segment)| (self.sets[set].slot, segment))
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

    /// Assigns a `SpillSlot` to each `SpillSet` that has segments spilled into
    /// it.
    ///
    /// The basic algorithm here is based on linear scan allocation from
    /// <https://doi.org/10.1145/330249.330250> where each `SpillSet` is treated
    /// as a single live range segment. This has the advantage of being fast,
    /// but at the cost of not being able to allocate spill sets in the live
    /// range gaps of another spill set. This is less of an issue than for
    /// registers though since spill slots are effectively unlimited.
    pub fn allocate(&mut self, stats: &mut Stats) -> Result<(), RegAllocError> {
        self.stack_layout.slots.clear();
        self.stack_layout.spillslot_area_size = 0;
        self.sets_to_allocate.clear();
        self.active_sets.clear();
        self.available_slots.clear();

        trace!("Allocating spill slots:");

        // Gather the spill sets that need to be allocated and sort them by
        // spill slot size first, and then by start position.
        self.sets_to_allocate.extend(
            self.sets
                .iter()
                .filter(|(_, data)| !data.live_range_union.is_empty())
                .map(|(set, _)| set),
        );
        stat!(stats, spillsets, self.sets_to_allocate.len());
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

            // Remove any spill sets whose live range ended before the current
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
