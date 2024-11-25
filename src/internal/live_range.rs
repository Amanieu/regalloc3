//! Data structure representing the live range of a value or  virtual register.
//!
//! The live range describes the set of program points during which the value of
//! a virtual register is live. It is represented as a sorted array of
//! [`LiveRangeSegment`] which each cover a range of program point in the linear
//! instruction space. Segments may cover multiple basic blocks.
//!
//! Each [`LiveRangeSegment`] is also associated with a list of [`Use`]s, which
//! represent ways in which a value is used in this segment. [`Use`]s are
//! primarily used to map virtual register allocations back to

use core::fmt;

use crate::function::{Inst, Value};

use super::uses::{UseList, Uses};

/// A slot within an instruction at which a live range starts or end.
///
/// To properly model the intricacies of live ranges, we need more granularity
/// than just knowing whether a live range start or ends at a particular
/// instruction.
///
/// This design is inspired by LLVM's `SlotIndex`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Slot {
    /// This is the boundary between 2 instructions where move instructions are
    /// inserted by the register allocator to move values between live ranges
    /// with different allocations.
    ///
    /// A live range starting at a `Boundary` indicates that it is receiving an
    /// existing value from a preceding live range. A live range ending at a
    /// `Boundary` indicates that it is passing its value on to another live
    /// range.
    ///
    /// This is also the only point at which a live range can be split, since
    /// it's not possible to split in the middle of an instruction.
    ///
    /// The `Boundary` associated with a particular instruction is the one
    /// *before* that instruction.
    Boundary = 0,

    /// This is the point at which live ranges for `EarlyDef`s start.
    ///
    /// Since `Use`s end their live range at the `Normal` slot, this ensures
    /// that a `Use` and an `EarlyDef` can never share the same register.
    Early = 1,

    /// This is the point at which live ranges for `Def`s start and those for
    /// `Use`s end.
    ///
    /// This allows a `Def` and `Use` to be assigned to the same register since
    /// their live ranges don't overlap (assuming the `Use` is not
    /// live-through).
    Normal = 2,
}

impl fmt::Display for Slot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Slot::Boundary => f.write_str("B"),
            Slot::Early => f.write_str("e"),
            Slot::Normal => f.write_str("n"),
        }
    }
}

/// A `LiveRangePoint` is logically a `(Inst, LiveRangeSlot)` tuple which
/// represents a point at which a live range starts or ends. This is bit-packed
/// in 32 bits for memory efficiency and to allow for fast comparisons.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct LiveRangePoint {
    /// Bit-pack in 32 bits.
    ///
    /// inst:30 slot:2
    pub bits: u32,
}

impl LiveRangePoint {
    /// Returns a new `LiveRangePoint` at the given slot in an instruction.
    pub fn new(inst: Inst, slot: Slot) -> Self {
        Self {
            bits: (inst.index() as u32) << 2 | slot as u32,
        }
    }

    /// Returns the instruction that this `LiveRangePoint` is in.
    ///
    /// In the case of a [`LiveRangeSlot::Boundary`], this returns the following
    /// instruction.
    pub fn inst(self) -> Inst {
        Inst::new((self.bits >> 2) as usize)
    }

    /// Returns the slot inside an instruction that this `LiveRangePoint` points
    /// to.
    pub fn slot(self) -> Slot {
        match self.bits & 0b11 {
            0 => Slot::Boundary,
            1 => Slot::Early,
            2 => Slot::Normal,
            _ => unreachable!(),
        }
    }

    /// Rounds this point to the next instruction boundary.
    pub fn round_to_next_inst(self) -> Self {
        Self {
            bits: (self.bits + 3) & !3,
        }
    }

    /// Rounds this point to the previous instruction boundary.
    pub fn round_to_prev_inst(self) -> Self {
        Self {
            bits: self.bits & !3,
        }
    }

    /// Fixed reservations only span a single Use/Def in one instruction, so we
    /// can save space by only storing the start point and deriving the end
    /// point from it.
    pub fn end_for_fixed_reservation(self) -> LiveRangePoint {
        let expected = match self.slot() {
            // Use
            Slot::Boundary => self.inst().slot(Slot::Normal),
            // EarlyDef/Def
            Slot::Early | Slot::Normal => self.inst().next().slot(Slot::Boundary),
        };

        // Optimized implementation since this is in the hot path.
        let optimized = Self {
            // 0 => 3, then masked to 2
            // 1 => 4
            // 2 => 5, then masked to 4
            bits: (self.bits + 3) & !1,
        };
        debug_assert_eq!(optimized, expected);
        optimized
    }
}

impl fmt::Display for LiveRangePoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.inst(), self.slot())
    }
}

impl fmt::Debug for LiveRangePoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl Inst {
    /// Helper function on `Inst` to create a `LiveRangePoint`.
    pub(crate) fn slot(self, slot: Slot) -> LiveRangePoint {
        LiveRangePoint::new(self, slot)
    }
}

/// A range between two [`LiveRangePoint`] in the linear instruction space.
///
/// This may cross block boundaries, which indicates that the live range extends
/// across multiple basic blocks.
///
/// The live range of a `Value` or `VirtReg` is represented as a set of
/// non-overlapping `LiveRangeSegment`s.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct LiveRangeSegment {
    pub from: LiveRangePoint,
    pub to: LiveRangePoint,
}

impl LiveRangeSegment {
    /// Creates a new `LiveRangeSegment`.
    pub fn new(from: LiveRangePoint, to: LiveRangePoint) -> Self {
        Self { from, to }
    }

    /// Checks whether the `LiveRangeSegment` holds an empty range.
    ///
    /// Empty ranges are valid but have special semantics: no register needs to
    /// be allocated for them, but they may still hold useful information in
    /// the associated [`UseList`].
    pub fn is_empty(self) -> bool {
        self.from >= self.to
    }

    /// Returns the intersection of two ranges, or `None` if they don't overlap.
    pub fn intersection(self, other: Self) -> Option<Self> {
        if other.to <= self.from || other.from >= self.to {
            None
        } else {
            Some(Self {
                from: self.from.max(other.from),
                to: self.to.min(other.to),
            })
        }
    }

    /// Number of instructions covered by this live range segment.
    pub fn num_insts(self) -> u32 {
        (self.to.round_to_next_inst().bits - self.from.round_to_prev_inst().bits) >> 2
    }

    /// Splits the live range in the middle at the given point.
    pub fn split_at(self, mid: LiveRangePoint) -> (Self, Self) {
        debug_assert!(self.from < mid);
        debug_assert!(self.to > mid);
        (
            Self {
                from: self.from,
                to: mid,
            },
            Self {
                from: mid,
                to: self.to,
            },
        )
    }
}

impl fmt::Display for LiveRangeSegment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}", self.from, self.to)
    }
}

impl fmt::Debug for LiveRangeSegment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// A continuous segment of a value's live range.
#[derive(Debug, Clone, Copy)]
pub struct ValueSegment {
    /// Live range covered by this segment.
    pub live_range: LiveRangeSegment,

    /// Set of `Use`s associated with this live range segment.
    pub use_list: UseList,

    /// SSA value associated with this live range segment.
    ///
    /// Coalescing may produce virtual registers which cover multiple SSA values
    /// but each segment will only come from a single SSA value.
    pub value: Value,
}

impl ValueSegment {
    /// Utility function for splitting a [`ValueSegment`] into 2 halves at
    /// the given split point.
    pub fn split_at<'a>(
        segments: &'a mut [ValueSegment],
        uses: &Uses,
        before_inst: Inst,
    ) -> SplitResult<'a> {
        // The split point must be inside the live range of the vreg.
        let split_at = before_inst.slot(Slot::Boundary);
        debug_assert!(split_at > segments[0].live_range.from);
        debug_assert!(split_at < segments.last().unwrap().live_range.to);

        // Separate the list of segments into the set before the split and the set
        // after the split. These may overlap by exactly 1 segment if that segment
        // contains the split point.
        let first_segment_in_second_half =
            segments.partition_point(|seg| seg.live_range.to <= split_at);
        let last_segment_in_first_half =
            if segments[first_segment_in_second_half].live_range.from < split_at {
                first_segment_in_second_half
            } else {
                first_segment_in_second_half - 1
            };

        // If both vregs share a segment that is split down the middle, split it
        // into 2 separate segments. The shared segment is already present in both
        // vectors, it just needs to be truncated accordingly for each half.
        let second_live_range_and_use_list =
            if first_segment_in_second_half == last_segment_in_first_half {
                debug_assert!(!segments[first_segment_in_second_half].live_range.is_empty());
                let (first_uses, second_uses) = segments[first_segment_in_second_half]
                    .use_list
                    .split_at_inst(split_at.inst(), uses);
                let (first_range, second_range) = segments[first_segment_in_second_half]
                    .live_range
                    .split_at(split_at);
                segments[last_segment_in_first_half].live_range = first_range;
                segments[last_segment_in_first_half].use_list = first_uses;

                Some((second_range, second_uses))
            } else {
                None
            };

        SplitResult {
            segments,
            last_segment_in_first_half,
            first_segment_in_second_half,
            second_live_range_and_use_list,
        }
    }
}

/// Result of spliting a live range with [`ValueSegment::split_at`].
pub struct SplitResult<'a> {
    segments: &'a mut [ValueSegment],
    last_segment_in_first_half: usize,
    first_segment_in_second_half: usize,
    second_live_range_and_use_list: Option<(LiveRangeSegment, UseList)>,
}

impl<'a> SplitResult<'a> {
    /// First half of the split.
    pub fn first_half(&mut self) -> &mut [ValueSegment] {
        &mut self.segments[..=self.last_segment_in_first_half]
    }

    /// Second half of the split.
    pub fn into_second_half(self) -> &'a mut [ValueSegment] {
        // Mutate the range in-place to avoid temporary allocations.
        if let Some((second_range, second_uses)) = self.second_live_range_and_use_list {
            self.segments[self.first_segment_in_second_half].live_range = second_range;
            self.segments[self.first_segment_in_second_half].use_list = second_uses;
        }

        &mut self.segments[self.first_segment_in_second_half..]
    }
}
