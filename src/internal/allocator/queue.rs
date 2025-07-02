//! Priority queue of virtual registers for which an allocation decision
//! (assign or spill) has not been made yet. These can also be registers which
//! have previously been assigned but have been evicted due to another virtual
//! register with a higher spill weight.

use alloc::collections::BinaryHeap;
use core::{fmt, mem};

use super::Stage;
use crate::internal::live_range::ValueSegment;
use crate::internal::virt_regs::{VirtReg, VirtRegGroup, VirtRegs};
use crate::reginfo::MAX_GROUP_SIZE;

/// The allocation queue can hold either individual virtual registers or
/// virtual register groups which must be allocated together as unit.
#[derive(Debug, Clone, Copy)]
pub enum VirtRegOrGroup {
    Reg(VirtReg),
    Group(VirtRegGroup),
}

impl fmt::Display for VirtRegOrGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VirtRegOrGroup::Reg(reg) => reg.fmt(f),
            VirtRegOrGroup::Group(group) => group.fmt(f),
        }
    }
}

/// Entry in the priority queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Entry {
    /// The priority and index are encoded in a single `u64` for performance.
    ///
    /// The bit encoding is designed to prioritize virtual registers as follows:
    /// - Earlier allocation stages are processed first.
    /// - Virtual registers with a fixed-register hint are prioritized.
    /// - Larger groups are harder to allocate, and so are prioritized.
    /// - Large live ranges are harder to allocate, and so are prioritized.
    /// - The virtual register index is used as a tiebreaker. It is negated to
    ///   prefer lower-indexed virtual registers when the size is the same.
    ///
    /// stage:1 has_fixed_hint:1 group_size:3 size:27 index:32
    bits: u64,
}

// The above encoding assumes a maximum group size of 8.
const _: () = assert!(MAX_GROUP_SIZE == 8);

impl Entry {
    /// Encodes an entry for a virtual register.
    fn encode(vreg: VirtReg, stage: Stage, virt_regs: &VirtRegs) -> Self {
        let stage = match stage {
            Stage::Evict => 1,
            Stage::Split => 0,
        };
        let has_fixed_hint = virt_regs[vreg].has_fixed_hint as u64;
        let group_size = 0;
        let size = ValueSegment::live_insts(virt_regs.segments(vreg)) as u64;
        let size = size.min((1 << 27) - 1);
        let index = vreg.index() as u64;
        Entry {
            bits: (stage << 63)
                | (has_fixed_hint << 62)
                | (group_size << 59)
                | (size << 32)
                | index,
        }
    }

    /// Encodes an entry for a virtual register group.
    fn encode_group(group: VirtRegGroup, stage: Stage, virt_regs: &VirtRegs) -> Self {
        let stage = match stage {
            Stage::Evict => 1,
            Stage::Split => 0,
        };
        let members = virt_regs.group_members(group);
        let has_fixed_hint = members.iter().any(|&vreg| virt_regs[vreg].has_fixed_hint) as u64;
        let group_size = members.len() as u64 - 1;
        let size: u64 = members
            .iter()
            .map(|&vreg| ValueSegment::live_insts(virt_regs.segments(vreg)) as u64)
            .sum();
        let size = size.min((1 << 27) - 1);
        let index = group.index() as u64;
        Entry {
            bits: (stage << 63)
                | (has_fixed_hint << 62)
                | (group_size << 59)
                | (size << 32)
                | index,
        }
    }

    /// Decodes an entry.
    fn decode(self) -> (VirtRegOrGroup, Stage) {
        // Group size has 1 subtracted from it in the encoding to fit in 3 bits.
        let group_size = ((self.bits >> 59) & 0b111) + 1;
        let stage = if self.bits >> 63 != 0 {
            Stage::Evict
        } else {
            Stage::Split
        };
        let index = self.bits as u32 as usize;
        let vreg_or_group = if group_size == 1 {
            let vreg = VirtReg::new(index);
            VirtRegOrGroup::Reg(vreg)
        } else {
            let group = VirtRegGroup::new(index);
            VirtRegOrGroup::Group(group)
        };
        (vreg_or_group, stage)
    }
}

/// Priority queue of virtual registers and virtual register groups that need
/// to be allocated.
pub struct AllocationQueue {
    queue: BinaryHeap<Entry>,
}

impl AllocationQueue {
    pub fn new() -> Self {
        Self {
            queue: BinaryHeap::new(),
        }
    }

    /// Initializes the allocation queue from the set of existing virtual
    /// register and virtual register groups.
    pub fn init(&mut self, virt_regs: &VirtRegs) {
        let mut vec = mem::take(&mut self.queue).into_vec();
        vec.clear();

        // Add virtual registers that are not part of a group.
        vec.extend(
            virt_regs
                .virt_regs()
                .filter(|&vreg| virt_regs[vreg].group.is_none())
                .map(|vreg| Entry::encode(vreg, Stage::Evict, virt_regs)),
        );

        // Add virtual register groups.
        vec.extend(
            virt_regs
                .groups()
                .map(|group| Entry::encode_group(group, Stage::Evict, virt_regs)),
        );

        // O(n) heap construction, which is much faster than inserting entries
        // one by one.
        self.queue = vec.into();
    }

    /// Dequeues the entry with the highest priority from the queue.
    pub fn dequeue(&mut self) -> Option<(VirtRegOrGroup, Stage)> {
        self.queue.pop().map(Entry::decode)
    }

    /// Enqueues an entry into the priority queue.
    pub fn enqueue(&mut self, vreg_or_group: VirtRegOrGroup, stage: Stage, virt_regs: &VirtRegs) {
        let entry = match vreg_or_group {
            VirtRegOrGroup::Reg(vreg) => Entry::encode(vreg, stage, virt_regs),
            VirtRegOrGroup::Group(group) => Entry::encode_group(group, stage, virt_regs),
        };
        self.queue.push(entry);
    }
}
