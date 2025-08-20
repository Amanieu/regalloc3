use alloc::vec::Vec;
use core::fmt::Debug;
use core::ops::{Bound, ControlFlow};

use brie_tree::nonmax::NonMaxU32;
use brie_tree::{BTree, BTreeKey, Cursor, Iter};

use super::live_range::{LiveRangePoint, LiveRangeSegment, Slot, ValueSegment};
use super::virt_regs::{VirtReg, VirtRegs};
use crate::Stats;
use crate::entity::SecondaryMap;
use crate::entity::packed_option::PackedOption;
use crate::function::Value;
use crate::reginfo::{MAX_REG_UNITS, MAX_UNITS_PER_REG, PhysReg, RegInfo, RegUnit};

/// The kind of interference detected.
#[derive(Debug, Clone, Copy)]
pub enum InterferenceKind {
    /// Fixed interference which cannot be evicted.
    Fixed,

    /// Virtual register interference which can be evicted.
    VirtReg(VirtReg),
}

/// Interference detected between a virtual register and the existing
/// reservations on a physical register.
#[derive(Debug, Clone, Copy)]
pub struct Interference<S> {
    /// Subset of the live range at which the interference occurs.
    pub range: LiveRangeSegment,

    /// Whether this is fixed interference or virtual register inteference.
    pub kind: InterferenceKind,

    /// The register unit in which the interference occurs.
    pub unit: RegUnit,

    /// The segment at which the interference occurred.
    pub segment: S,
}

/// A live range segment that interference should be checked against.
pub trait InterferenceSegment: Copy + Debug {
    fn live_range(&self) -> LiveRangeSegment;
    fn value(&self) -> Value;
}

impl InterferenceSegment for ValueSegment {
    fn live_range(&self) -> LiveRangeSegment {
        self.live_range
    }

    fn value(&self) -> Value {
        self.value
    }
}

impl BTreeKey for LiveRangePoint {
    type Int = NonMaxU32;

    fn to_int(self) -> Self::Int {
        NonMaxU32::new(self.bits).unwrap()
    }

    fn from_int(int: Self::Int) -> Self {
        LiveRangePoint { bits: int.get() }
    }
}

/// An entry in the `BTree` for each register unit.
///
/// The key is the end point of the live range.
#[derive(Debug, Clone, Copy)]
struct Entry {
    /// Start point of the live range.
    from: LiveRangePoint,

    /// Virtual register assigned to this unit for the live range segment.
    ///
    /// This is `None` if this live range segment is for a fixed reservation.
    vreg: PackedOption<VirtReg>,

    /// This field has a different interpretation depending on `vreg`:
    ///
    /// - If `vreg` is `None` then this is a fixed reservation which cannot be
    ///   evicted. For fixed def reservations, `fixed_use == !0`. For fixed use
    ///   reservations, `fixed_use` holds the value index of that fixed use.
    ///   This is necessary because vreg segments with the same value can
    ///   overlap with it.
    ///
    /// - If `vreg` is `Some` then this is a virtual register reservation. It
    ///   may overlap with one or more fixed uses (which are removed from the
    ///   B-Tree when the segment is added). `fixed_uses` records the index of
    ///   the first overlapping value in `fixed_uses`, or `!0` if there are no
    ///   overlapping uses.
    fixed_use: u32,
}

/// Per-unit live range reservations.
#[derive(Default)]
struct UnitReservations {
    /// B-Tree of allocations on the register unit.
    btree: BTree<LiveRangePoint, Entry>,

    /// Reservations for fixed-register uses, sorted by start point.
    ///
    /// These are tracked separately because they may overlap with virtual
    /// register reservations that have the same value.
    fixed_uses: Vec<(LiveRangePoint, Value)>,
}

impl UnitReservations {
    /// Iterates over all the interference in the current unit.
    fn check_interference<B, S: InterferenceSegment>(
        &self,
        segments: &[S],
        unit: RegUnit,
        stats: &mut Stats,
        full_results: bool,
        mut f: impl FnMut(Interference<S>) -> ControlFlow<B>,
    ) -> ControlFlow<B> {
        stat!(stats, interference_checks);
        stat!(stats, interference_check_segments, segments.len());

        // Find the first reservation that ends after the start of the first
        // segment. This is the starting point for our scan.
        let mut iter = self
            .btree
            .iter_from(Bound::Excluded(segments[0].live_range().from));
        let Some(mut current_entry) = iter.next() else {
            return ControlFlow::Continue(());
        };

        'outer: for segment in segments {
            // Skip live range reservations that end before the start of the
            // current segment.
            while current_entry.0 <= segment.live_range().from {
                // TODO(perf): Integrate with btree to seek more efficiently.
                match iter.next() {
                    Some(entry) => current_entry = entry,
                    None => return ControlFlow::Continue(()),
                }
            }

            // Loop over all entries that overlap the segment.
            loop {
                let (to, entry) = current_entry;

                // If the live range reservation starts after the end of the current
                // segment then there is no overlap. Continue to the next segment.
                if entry.from >= segment.live_range().to {
                    continue 'outer;
                }

                // Calculate the overlap range. This is currently only used for
                // debug logging and is optimized out otherwise.
                let range = LiveRangeSegment::new(
                    entry.from.max(segment.live_range().from),
                    to.min(segment.live_range().to),
                );
                debug_assert!(range.from < range.to);

                // Invoke the callback to report any interference.
                if let Some(vreg) = entry.vreg.expand() {
                    stat!(stats, vreg_interference);
                    f(Interference {
                        range,
                        kind: InterferenceKind::VirtReg(vreg),
                        unit,
                        segment: *segment,
                    })?;

                    // If this vreg overlaps with one or more fixed use reservations
                    // then also check if the current segment overlaps any of those
                    // reservations. This is important for calculating eviction
                    // costs since fixed uses cannot be evicted.
                    if entry.fixed_use != !0 {
                        for &(from, value) in &self.fixed_uses[entry.fixed_use as usize..] {
                            if from >= segment.live_range().to {
                                break;
                            }
                            let to = from.end_for_fixed_use_reservation();
                            if to <= segment.live_range().from {
                                continue;
                            }
                            if value != segment.value() {
                                stat!(stats, inlined_fixed_use_interference);
                                f(Interference {
                                    range: LiveRangeSegment::new(from, to),
                                    kind: InterferenceKind::Fixed,
                                    unit,
                                    segment: *segment,
                                })?;
                            }
                        }

                        // If this entry ends after the end of the segment then
                        // it may still have fixed conflicts that interfere with
                        // the next segment. Continue to the next segment
                        // without advancing the cursor in this case.
                        if to > segment.live_range().to {
                            continue 'outer;
                        }
                    }
                } else {
                    // If fixed_use_idx is !0 then this indicates a fixed def
                    // which can never overlap a vreg segment. Otherwise this is
                    // a fixed use reservation which can overlap vreg segments
                    // that have the same value as the reservation.
                    if segment.value().index() != entry.fixed_use as usize {
                        if entry.fixed_use == !0 {
                            stat!(stats, fixed_def_interference);
                        } else {
                            stat!(stats, fixed_use_interference);
                        }
                        f(Interference {
                            range,
                            kind: InterferenceKind::Fixed,
                            unit,
                            segment: *segment,
                        })?;
                    }
                }

                // If this entry ends after the end of the segment then it may
                // also interfere with the next segment. Continue to the next
                // segment without advancing the cursor in this case.
                //
                // We only do this if full interference results are needed.
                if full_results && to > segment.live_range().to {
                    continue 'outer;
                }

                // Advance the cursor to the next entry.
                match iter.next() {
                    Some(entry) => current_entry = entry,
                    None => return ControlFlow::Continue(()),
                }
            }
        }

        ControlFlow::Continue(())
    }
}

/// Matrix which tracks, for each `RegUnit`, the portion of its live range which
/// has been allocated to a virtual register or is reserved for a fixed operand
/// or clobber.
pub struct RegMatrix {
    reservations: SecondaryMap<RegUnit, UnitReservations>,
}

impl RegMatrix {
    pub fn new() -> Self {
        Self {
            reservations: SecondaryMap::with_max_index(MAX_REG_UNITS),
        }
    }

    pub fn clear(&mut self) {
        // Preserve allocations when clearing.
        for reservations in self.reservations.values_mut() {
            reservations.btree.clear();
            reservations.fixed_uses.clear();
        }
    }

    /// Indicates that a portion of the given register is reserved for a fixed
    /// use constraint for the given live range and the given `Value`.
    ///
    /// This portion of the live range *can* be re-used by a virtual register,
    /// but only for a `ValueSegment` with a matching `Value`.
    ///
    /// This is important since it avoids the need to move a virtual register
    /// out of the way when it is used by a fixed operand while also being live
    /// through to later uses.
    ///
    /// This must be called in increasing instruction order.
    pub fn reserve_fixed_use(&mut self, unit: RegUnit, range: LiveRangeSegment, value: Value) {
        trace!("Reserving {range} in {unit} for fixed use of {value:?}");
        debug_assert_eq!(range.from.slot(), Slot::Boundary);
        debug_assert_eq!(range.to.slot(), Slot::Normal);
        debug_assert_eq!(range.to, range.from.end_for_fixed_use_reservation());

        self.reservations[unit].fixed_uses.push((range.from, value));
        let prev = self.reservations[unit].btree.insert(
            range.to,
            Entry {
                from: range.from,
                vreg: None.into(),
                fixed_use: value.index() as u32,
            },
        );

        // Check that all elements are still sorted.
        debug_assert!(
            self.reservations[unit]
                .fixed_uses
                .is_sorted_by(|a, b| a.0.end_for_fixed_use_reservation() <= b.0)
        );

        // Ensure there are no overlapping reservations.
        debug_assert!(prev.is_none());
    }

    /// Indicates that a portion of the given register is reserved for a fixed
    /// def or clobber constraint for the given live range.
    ///
    /// Unlike fixed uses, these can never overlap with the live range of a vreg
    /// and therefore don't need special handling.
    pub fn reserve_fixed_def(&mut self, unit: RegUnit, range: LiveRangeSegment) {
        trace!("Reserving {range} in {unit} for fixed def");
        debug_assert_eq!(range.to.slot(), Slot::Boundary);

        let prev = self.reservations[unit].btree.insert(
            range.to,
            Entry {
                from: range.from,
                vreg: None.into(),
                fixed_use: !0,
            },
        );

        // Ensure there are no overlapping reservations.
        debug_assert!(prev.is_none());
    }

    /// Checks whether the given register unit is free for the given live range
    /// segment.
    pub fn is_unit_free(&self, unit: RegUnit, check_range: LiveRangeSegment) -> bool {
        if let Some((_to, entry)) = self.reservations[unit]
            .btree
            .iter_from(Bound::Excluded(check_range.from))
            .next()
        {
            if entry.from <= check_range.to {
                return false;
            }
        }

        true
    }

    /// Iterates over all the interference between `segments` and existing
    /// assignments to `reg`.
    ///
    /// If `full_results` is true then all interferences are reported, otherwise
    /// each interfering vreg or fixed reservation is only reported once.
    ///
    /// The interference is returned in an arbitrary order.
    pub fn check_interference<B, S: InterferenceSegment>(
        &self,
        segments: &[S],
        reg: PhysReg,
        reginfo: &impl RegInfo,
        stats: &mut Stats,
        full_results: bool,
        mut f: impl FnMut(Interference<S>) -> ControlFlow<B>,
    ) -> ControlFlow<B> {
        for unit in reginfo.reg_units(reg) {
            self.reservations[unit].check_interference(
                segments,
                unit,
                stats,
                full_results,
                &mut f,
            )?;
        }
        ControlFlow::Continue(())
    }

    /// Assigns all segments of the given virtual register to a physical
    /// register.
    pub fn assign(
        &mut self,
        vreg: VirtReg,
        reg: PhysReg,
        virt_regs: &VirtRegs,
        reginfo: &impl RegInfo,
    ) {
        trace!("Assigning {vreg} to {reg}");

        // We need to track reservations in each register unit separately.
        for unit in reginfo.reg_units(reg) {
            let reservations = &mut self.reservations[unit];
            for segment in virt_regs.segments(vreg) {
                debug_assert!(!segment.live_range.is_empty());

                // Check if this segment overlaps any fixed reservations and
                // remove them from the btree. A reference to these is then
                // saved in the entry for the segment so that interference
                // checks can still take the fixed reservation into account.
                let mut cursor = reservations
                    .btree
                    .cursor_mut_at(Bound::Excluded(segment.live_range().from));
                let mut overlaps_fixed = false;
                while let Some((_to, &entry)) = cursor.entry() {
                    if entry.from >= segment.live_range.to {
                        break;
                    }
                    debug_assert_eq!(entry.vreg.expand(), None);
                    debug_assert_ne!(entry.fixed_use, !0);
                    overlaps_fixed = true;
                    cursor.remove();
                }
                let fixed_use = if overlaps_fixed {
                    reservations
                        .fixed_uses
                        .partition_point(|&(from, _value)| from < segment.live_range().from)
                        as u32
                } else {
                    !0
                };

                cursor.insert_before(
                    segment.live_range.to,
                    Entry {
                        from: segment.live_range.from,
                        vreg: Some(vreg).into(),
                        fixed_use,
                    },
                );
            }
        }
    }

    /// Evicts all segments of the given virtual register from a physical
    /// register.
    pub fn evict(
        &mut self,
        vreg: VirtReg,
        reg: PhysReg,
        virt_regs: &VirtRegs,
        reginfo: &impl RegInfo,
    ) {
        trace!("Evicting {vreg} from {reg}");

        // We need to track reservations in each register unit separately.
        for unit in reginfo.reg_units(reg) {
            let reservations = &mut self.reservations[unit];
            for segment in virt_regs.segments(vreg) {
                debug_assert!(!segment.live_range.is_empty());

                let entry = reservations.btree.remove(segment.live_range.to).unwrap();

                // Ensure the entry contained the expected virtual register.
                debug_assert_eq!(entry.from, segment.live_range.from);
                debug_assert_eq!(entry.vreg.expand(), Some(vreg));

                // If this segment overlapped with any fixed uses then we need
                // to restore them as individual fixed use entries.
                if entry.fixed_use != !0 {
                    for &(from, value) in &reservations.fixed_uses[entry.fixed_use as usize..] {
                        if from >= segment.live_range().to {
                            break;
                        }
                        let to = from.end_for_fixed_use_reservation();
                        reservations.btree.insert(
                            to,
                            Entry {
                                from,
                                vreg: None.into(),
                                fixed_use: value.index() as u32,
                            },
                        );
                    }
                }
            }
        }
    }

    /// Dumps the entire register matrix to the log.
    #[allow(dead_code)]
    pub fn dump(&self) {
        trace!("Register matrix:");
        for (unit, reservations) in &self.reservations {
            if reservations.btree.is_empty() {
                continue;
            }
            trace!("{unit}:");
            for (to, entry) in &reservations.btree {
                let segment = LiveRangeSegment::new(entry.from, to);
                if let Some(vreg) = entry.vreg.expand() {
                    trace!("- {segment}: {vreg}");
                    if entry.fixed_use != !0 {
                        for &(from, value) in &reservations.fixed_uses[entry.fixed_use as usize..] {
                            if from >= to {
                                break;
                            }
                            let to = from.end_for_fixed_use_reservation();
                            let segment = LiveRangeSegment::new(from, to);
                            trace!("  - {segment}: internal fixed use {value}");
                        }
                    }
                } else if entry.fixed_use == !0 {
                    trace!("- {segment}: fixed def");
                } else {
                    let value = Value::new(entry.fixed_use as usize);
                    trace!("- {segment}: fixed use {value}");
                }
            }
        }
    }
}

/// Cursor which scans the reservations in a unit for interference in the
/// forwards direction.
struct UnitInterferenceForwardCursor<'a> {
    /// The next entry to check for interfereence.
    current_entry: (LiveRangePoint, Entry),

    /// B-Tree iterator pointing after `current_entry`.
    iter: Iter<'a, LiveRangePoint, Entry>,
}

impl<'a> UnitInterferenceForwardCursor<'a> {
    fn advance_right<B, S: InterferenceSegment>(
        cursor: &mut Option<Self>,
        segment: &S,
        unit: RegUnit,
        full_results: bool,
        reservations: &UnitReservations,
        stats: &mut Stats,
        mut f: impl FnMut(Interference<S>) -> ControlFlow<B>,
    ) -> ControlFlow<B> {
        stat!(stats, interference_check_segments);

        // If the cursor is None then we have reached the end of the tree and
        // there is no more interference.
        let Some(inner) = cursor.as_mut() else {
            return ControlFlow::Continue(());
        };

        // Skip live range reservations that end before the start of the
        // current segment.
        while inner.current_entry.0 <= segment.live_range().from {
            // TODO(perf): Integrate with btree to seek more efficiently.
            match inner.iter.next() {
                Some((key, &value)) => inner.current_entry = (key, value),
                None => {
                    *cursor = None;
                    return ControlFlow::Continue(());
                }
            }
        }

        // Loop over all entries that overlap the segment.
        loop {
            let (to, entry) = inner.current_entry;

            // If the live range reservation starts after the end of the current
            // segment then there is no overlap. Continue to the next segment.
            if entry.from >= segment.live_range().to {
                return ControlFlow::Continue(());
            }

            // Calculate the overlap range. This is currently only used for
            // debug logging and is optimized out otherwise.
            let range = LiveRangeSegment::new(
                entry.from.max(segment.live_range().from),
                to.min(segment.live_range().to),
            );
            debug_assert!(range.from < range.to);

            // Invoke the callback to report any interference.
            if let Some(vreg) = entry.vreg.expand() {
                stat!(stats, vreg_interference);
                f(Interference {
                    range,
                    kind: InterferenceKind::VirtReg(vreg),
                    unit,
                    segment: *segment,
                })?;

                // If this vreg overlaps with one or more fixed use reservations
                // then also check if the current segment overlaps any of those
                // reservations. This is important for calculating eviction
                // costs since fixed uses cannot be evicted.
                if entry.fixed_use != !0 {
                    for &(from, value) in &reservations.fixed_uses[entry.fixed_use as usize..] {
                        if from >= segment.live_range().to {
                            break;
                        }
                        let to = from.end_for_fixed_use_reservation();
                        if to <= segment.live_range().from {
                            continue;
                        }
                        if value != segment.value() {
                            stat!(stats, inlined_fixed_use_interference);
                            f(Interference {
                                range: LiveRangeSegment::new(from, to),
                                kind: InterferenceKind::Fixed,
                                unit,
                                segment: *segment,
                            })?;
                        }
                    }

                    // If this entry ends after the end of the segment then
                    // it may still have fixed conflicts that interfere with
                    // the next segment. Continue to the next segment
                    // without advancing the cursor in this case.
                    if to > segment.live_range().to {
                        return ControlFlow::Continue(());
                    }
                }
            } else {
                // If fixed_use_idx is !0 then this indicates a fixed def
                // which can never overlap a vreg segment. Otherwise this is
                // a fixed use reservation which can overlap vreg segments
                // that have the same value as the reservation.
                if segment.value().index() != entry.fixed_use as usize {
                    if entry.fixed_use == !0 {
                        stat!(stats, fixed_def_interference);
                    } else {
                        stat!(stats, fixed_use_interference);
                    }
                    f(Interference {
                        range,
                        kind: InterferenceKind::Fixed,
                        unit,
                        segment: *segment,
                    })?;
                }
            }

            // If this entry ends after the end of the segment then it may
            // also interfere with the next segment. Continue to the next
            // segment without advancing the cursor in this case.
            //
            // We only do this if full interference results are needed.
            if full_results && to > segment.live_range().to {
                return ControlFlow::Continue(());
            }

            // Advance the cursor to the next entry.
            match inner.iter.next() {
                Some((key, &value)) => inner.current_entry = (key, value),
                None => {
                    *cursor = None;
                    return ControlFlow::Continue(());
                }
            }
        }
    }
}

/// Cursor which scans the reservations in a unit for interference in the
/// backwards direction.
struct UnitInterferenceBackwardCursor<'a> {
    /// The next entry to check for interfereence.
    current_entry: (LiveRangePoint, Entry),

    /// B-Tree cursor pointing after `current_entry`.
    cursor: Cursor<'a, LiveRangePoint, Entry>,
}

impl<'a> UnitInterferenceBackwardCursor<'a> {
    fn advance_left<B, S: InterferenceSegment>(
        cursor: &mut Option<Self>,
        segment: &S,
        unit: RegUnit,
        full_results: bool,
        reservations: &UnitReservations,
        stats: &mut Stats,
        mut f: impl FnMut(Interference<S>) -> ControlFlow<B>,
    ) -> ControlFlow<B> {
        stat!(stats, interference_check_segments);

        // If the cursor is None then we have reached the end of the tree and
        // there is no more interference.
        let Some(inner) = cursor.as_mut() else {
            return ControlFlow::Continue(());
        };

        // Skip live range reservations that start after the end of the
        // current segment.
        while inner.current_entry.1.from >= segment.live_range().to {
            // TODO(perf): Integrate with btree to seek more efficiently.
            if inner.cursor.prev() {
                let (key, &value) = inner.cursor.entry().unwrap();
                inner.current_entry = (key, value);
            } else {
                *cursor = None;
                return ControlFlow::Continue(());
            }
        }

        // Loop over all entries that overlap the segment.
        loop {
            let (to, entry) = inner.current_entry;

            // If the live range reservation ends before the start of the current
            // segment then there is no overlap. Continue to the next segment.
            if to <= segment.live_range().from {
                return ControlFlow::Continue(());
            }

            // Calculate the overlap range. This is currently only used for
            // debug logging and is optimized out otherwise.
            let range = LiveRangeSegment::new(
                entry.from.max(segment.live_range().from),
                to.min(segment.live_range().to),
            );
            debug_assert!(range.from < range.to);

            // Invoke the callback to report any interference.
            if let Some(vreg) = entry.vreg.expand() {
                stat!(stats, vreg_interference);
                f(Interference {
                    range,
                    kind: InterferenceKind::VirtReg(vreg),
                    unit,
                    segment: *segment,
                })?;

                // If this vreg overlaps with one or more fixed use reservations
                // then also check if the current segment overlaps any of those
                // reservations. This is important for calculating eviction
                // costs since fixed uses cannot be evicted.
                if entry.fixed_use != !0 {
                    for &(from, value) in &reservations.fixed_uses[entry.fixed_use as usize..] {
                        if from >= segment.live_range().to {
                            break;
                        }
                        let to = from.end_for_fixed_use_reservation();
                        if to <= segment.live_range().from {
                            continue;
                        }
                        if value != segment.value() {
                            stat!(stats, inlined_fixed_use_interference);
                            f(Interference {
                                range: LiveRangeSegment::new(from, to),
                                kind: InterferenceKind::Fixed,
                                unit,
                                segment: *segment,
                            })?;
                        }
                    }

                    // If this entry starts before the start of the segment then
                    // it may still have fixed conflicts that interfere with
                    // the next segment. Continue to the next segment
                    // without advancing the cursor in this case.
                    if entry.from < segment.live_range().from {
                        return ControlFlow::Continue(());
                    }
                }
            } else {
                // If fixed_use_idx is !0 then this indicates a fixed def
                // which can never overlap a vreg segment. Otherwise this is
                // a fixed use reservation which can overlap vreg segments
                // that have the same value as the reservation.
                if segment.value().index() != entry.fixed_use as usize {
                    if entry.fixed_use == !0 {
                        stat!(stats, fixed_def_interference);
                    } else {
                        stat!(stats, fixed_use_interference);
                    }
                    f(Interference {
                        range,
                        kind: InterferenceKind::Fixed,
                        unit,
                        segment: *segment,
                    })?;
                }
            }

            // If this entry starts before the start of the segment then it may
            // also interfere with the next segment. Continue to the next
            // segment without advancing the cursor in this case.
            //
            // We only do this if full interference results are needed.
            if full_results && entry.from < segment.live_range().from {
                return ControlFlow::Continue(());
            }

            // Advance the cursor to the next entry.
            if inner.cursor.prev() {
                let (key, &value) = inner.cursor.entry().unwrap();
                inner.current_entry = (key, value);
            } else {
                *cursor = None;
                return ControlFlow::Continue(());
            }
        }
    }
}

/// Cursor for incremental interference checking.
///
/// This works by starting from the initial segment and then expanding backwards
/// and forwards from that point.
pub struct InterferenceCursor<'a, R: RegInfo> {
    reg: PhysReg,
    reginfo: &'a R,
    reg_matrix: &'a RegMatrix,
    cursors: [(
        Option<UnitInterferenceBackwardCursor<'a>>,
        Option<UnitInterferenceForwardCursor<'a>>,
    ); MAX_UNITS_PER_REG],
}

impl<'a, R: RegInfo> InterferenceCursor<'a, R> {
    pub fn new(
        initial_segment: &impl InterferenceSegment,
        reg_matrix: &'a RegMatrix,
        reg: PhysReg,
        reginfo: &'a R,
        stats: &mut Stats,
    ) -> Self {
        let mut cursors = [const { (None, None) }; MAX_UNITS_PER_REG];
        for (i, unit) in reginfo.reg_units(reg).enumerate() {
            stat!(stats, interference_checks);

            // Find the first reservation that ends after the start of the first
            // segment. This is the starting point for our scan.
            let mut cursor = reg_matrix.reservations[unit]
                .btree
                .cursor_at(Bound::Excluded(initial_segment.live_range().from));
            let mut right_iter = cursor.iter();
            let right_entry = right_iter.next();

            let left_entry = if right_entry
                .is_none_or(|(_to, entry)| entry.from >= initial_segment.live_range().from)
            {
                if cursor.prev() { cursor.entry() } else { None }
            } else {
                right_entry
            };

            let left = left_entry.map(|(key, &value)| UnitInterferenceBackwardCursor {
                cursor,
                current_entry: (key, value),
            });
            let right = right_entry.map(|(key, &value)| UnitInterferenceForwardCursor {
                iter: right_iter,
                current_entry: (key, value),
            });

            cursors[i] = (left, right);
        }

        Self {
            reg,
            reginfo,
            reg_matrix,
            cursors,
        }
    }

    pub fn advance_left<B, S: InterferenceSegment>(
        &mut self,
        segment: &S,
        stats: &mut Stats,
        mut f: impl FnMut(Interference<S>) -> ControlFlow<B>,
    ) -> ControlFlow<B> {
        for (i, unit) in self.reginfo.reg_units(self.reg).enumerate() {
            UnitInterferenceBackwardCursor::advance_left(
                &mut self.cursors[i].0,
                segment,
                unit,
                true,
                &self.reg_matrix.reservations[unit],
                stats,
                &mut f,
            )?;
        }
        ControlFlow::Continue(())
    }

    pub fn advance_right<B, S: InterferenceSegment>(
        &mut self,
        segment: &S,
        stats: &mut Stats,
        mut f: impl FnMut(Interference<S>) -> ControlFlow<B>,
    ) -> ControlFlow<B> {
        for (i, unit) in self.reginfo.reg_units(self.reg).enumerate() {
            UnitInterferenceForwardCursor::advance_right(
                &mut self.cursors[i].1,
                segment,
                unit,
                true,
                &self.reg_matrix.reservations[unit],
                stats,
                &mut f,
            )?;
        }
        ControlFlow::Continue(())
    }
}
