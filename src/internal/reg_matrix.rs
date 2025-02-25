use alloc::collections::BTreeMap;
use alloc::collections::btree_map::Cursor;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::mem;
use core::ops::Bound;

use super::live_range::ValueSegment;
use super::live_range::{LiveRangePoint, LiveRangeSegment, Slot};
use super::virt_regs::{VirtReg, VirtRegs};
use crate::entity::SecondaryMap;
use crate::entity::packed_option::PackedOption;
use crate::function::Value;
use crate::reginfo::{MAX_REG_UNITS, PhysReg, RegInfo, RegUnit};

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
pub struct Interference {
    /// Subset of the live range at which the interference occurs.
    pub range: LiveRangeSegment,

    /// Whether this is fixed interference or virtual register inteference.
    pub kind: InterferenceKind,

    /// The register unit in which the interference occurs.
    pub unit: RegUnit,
}

/// Iterator over the interference between a virtual register and other
/// virtual registers already assigned to a `RegUnit`.
struct VirtRegInterferenceIter<'a> {
    unit: RegUnit,
    btree: &'a BTreeMap<LiveRangePoint, (LiveRangePoint, VirtReg)>,
    cursor: Cursor<'a, LiveRangePoint, (LiveRangePoint, VirtReg)>,
    segments: &'a [ValueSegment],
}

impl Iterator for VirtRegInterferenceIter<'_> {
    type Item = Interference;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Get the next segment to look at. If there are no more segments
            // then we are done.
            let segment = *self.segments.first()?;

            // Get the next live range reservation in the current register. If
            // we have reached the end of the register then there are no more
            // interferences.
            let (&to, &(from, vreg)) = self.cursor.peek_next()?;
            let range = LiveRangeSegment::new(from, to);

            // Advance the segment iterator or the reservation iterator
            // depending on which one has the earliest end point on their range.
            if range.to < segment.live_range.to {
                self.cursor.next();

                // If the next range is still below the target segment, do a
                // tree search to find the next point to start scanning.
                // TODO(perf): Double-check this and maybe probe more
                if let Some((&key, _entry)) = self.cursor.peek_next() {
                    if key <= segment.live_range.from {
                        self.cursor = self
                            .btree
                            .lower_bound(Bound::Excluded(&segment.live_range.from));
                    }
                }
            } else {
                self.segments = &self.segments[1..];
            }

            // If the ranges don't overlap then loop back to find the next
            // intersecting range.
            let Some(intersection) = range.intersection(segment.live_range) else {
                continue;
            };

            // Return the interference.
            return Some(Interference {
                range: intersection,
                kind: InterferenceKind::VirtReg(vreg),
                unit: self.unit,
            });
        }
    }
}

/// Iterator over the interference between a virtual register and fixed register
/// constraints and clobbers on a `RegUnit`.
struct FixedInterferenceIter<'a> {
    unit: RegUnit,
    fixed: &'a [(LiveRangePoint, PackedOption<Value>)],
    segments: &'a [ValueSegment],
}

impl Iterator for FixedInterferenceIter<'_> {
    type Item = Interference;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Get the next segment to look at. If there are no more segments
            // then we are done.
            let segment = *self.segments.first()?;

            // Get the next live range reservation in the current register. If
            // we have reached the end of the register then there are no more
            // interferences.
            let &(from, value) = self.fixed.first()?;
            let range = LiveRangeSegment::new(from, from.end_for_fixed_reservation());

            // Advance the segment iterator or the reservation iterator
            // depending on which one has the earliest end point on their range.
            if range.to < segment.live_range.to {
                self.fixed = &self.fixed[1..];

                // If the next range is still below the target segment, do a
                // tree search to find the next point to start scanning.
                // TODO(perf): Double-check this and maybe probe more
                if let Some(&(from, _value)) = self.fixed.first() {
                    if from.end_for_fixed_reservation() <= segment.live_range.from {
                        let mid = self.fixed.partition_point(|&(from, _value)| {
                            from.end_for_fixed_reservation() <= segment.live_range.from
                        });
                        self.fixed = &self.fixed[mid..];
                    }
                }
            } else {
                self.segments = &self.segments[1..];
            }

            // If the ranges don't overlap then loop back to find the next
            // intersecting range.
            let Some(intersection) = range.intersection(segment.live_range) else {
                continue;
            };

            // If the interferening reservation is a fixed-register constraint
            // then it can be coalesced with the segment if it has the same SSA
            // value. In that case there is no actual interference.
            if value.expand() == Some(segment.value) {
                continue;
            }

            // Return the interference.
            return Some(Interference {
                range: intersection,
                kind: InterferenceKind::Fixed,
                unit: self.unit,
            });
        }
    }
}

/// Per-unit live range reservations.
#[derive(Default, Clone)]
struct UnitReservations {
    /// Reservations for fixed-register constraints and clobbers.
    ///
    /// These are sorted by start point.
    fixed: Vec<(LiveRangePoint, PackedOption<Value>)>,

    /// Reservations for virtual register segments.
    ///
    /// These are held in a B-Tree with entries encoded in 12 bytes.
    vregs: BTreeMap<LiveRangePoint, (LiveRangePoint, VirtReg)>,
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
            reservations.fixed.clear();
            reservations.vregs.clear();
        }
    }

    /// Returns whether there are any active reservations on the given physical
    /// register.
    pub fn is_reg_used(&self, reg: PhysReg, reginfo: &impl RegInfo) -> bool {
        reginfo.reg_units(reg).all(|unit| {
            self.reservations[unit].fixed.is_empty() && self.reservations[unit].vregs.is_empty()
        })
    }

    /// Indicates that a portion of the given register is reserved for a fixed
    /// constraint for the given live range and the given `Value`.
    ///
    /// This portion of the live range *can* be re-used by a virtual register,
    /// but only for a `ValueSegment` with a matching `Value`.
    ///
    /// This is important since it avoids the need to move a virtual register
    /// out of the way when it is used by a fixed operand while also being live
    /// through to later uses.
    ///
    /// This must be called in increasing instruction order and before any
    /// clobbers are added for the current instruction.
    pub fn reserve_fixed(&mut self, unit: RegUnit, range: LiveRangeSegment, value: Value) {
        trace!("Reserving {range} in {unit} with value {value:?}");
        debug_assert_eq!(range.to, range.from.end_for_fixed_reservation());

        // We need to ensure that elements in the reservations list are always
        // properly sorted. If a fixed def is inserted before a fixed use for
        // the current instruction, then we need to swap their ordering.
        let mut to_insert = (range.from, Some(value).into());
        if let Some(last) = self.reservations[unit].fixed.last_mut() {
            debug_assert!(last.0.inst() <= range.from.inst());
            debug_assert_ne!(last.0, range.from);
            if last.0.inst() == range.from.inst() {
                debug_assert_ne!(last.1.expand(), None);
            }
            if last.0 > range.from {
                mem::swap(&mut to_insert, last);

                // Check that all elements are still sorted.
                debug_assert!(
                    self.reservations[unit]
                        .fixed
                        .windows(2)
                        .all(|window| window[0].0.end_for_fixed_reservation() <= window[1].0)
                );
            }
        }

        self.reservations[unit].fixed.push(to_insert);
    }

    /// Indicates that a portion of the given register is clobbered for the
    /// given live range.
    ///
    /// This must be called in increasing instruction order and after any
    /// fixed constraint reservations are added for the current instruction.
    pub fn reserve_clobber(&mut self, unit: RegUnit, range: LiveRangeSegment) {
        trace!("Reserving {range} in {unit} for clobber");
        debug_assert_eq!(range.to, range.from.end_for_fixed_reservation());
        debug_assert_eq!(range.from.slot(), Slot::Normal);

        self.reservations[unit]
            .fixed
            .push((range.from, None.into()));
    }

    /// Checks whether the given register unit is free for the given live range
    /// segment.
    pub fn is_unit_free(&self, unit: RegUnit, check_range: LiveRangeSegment) -> bool {
        // Check for a vreg conflict.
        if let Some((&to, &(from, _vreg))) = self.reservations[unit]
            .vregs
            .lower_bound(Bound::Excluded(&check_range.from))
            .peek_next()
        {
            let range = LiveRangeSegment::new(from, to);
            if range.intersection(check_range).is_some() {
                return false;
            }
        }

        // Check for a fixed conflict.
        self.reservations[unit]
            .fixed
            .binary_search_by(|&(from, _value)| {
                let to = from.end_for_fixed_reservation();
                if to <= check_range.from {
                    Ordering::Less
                } else if from >= check_range.to {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            })
            .is_err()
    }

    /// Returns an iterator over all the interference between `vreg` and the
    /// existing assignments to `reg`.
    ///
    /// The interference is returned in an arbitrary order.
    pub fn interference<'a>(
        &'a self,
        vreg: VirtReg,
        reg: PhysReg,
        virt_regs: &'a VirtRegs,
        reginfo: &'a impl RegInfo,
    ) -> impl Iterator<Item = Interference> + 'a {
        let segments = virt_regs.segments(vreg);
        reginfo.reg_units(reg).flat_map(move |unit| {
            let btree = &self.reservations[unit].vregs;
            let fixed = &self.reservations[unit].fixed;
            let cursor = btree.lower_bound(Bound::Excluded(&segments[0].live_range.from));
            VirtRegInterferenceIter {
                unit,
                btree,
                cursor,
                segments,
            }
            .chain(FixedInterferenceIter {
                unit,
                fixed,
                segments,
            })
        })
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
            for segment in virt_regs.segments(vreg) {
                debug_assert!(!segment.live_range.is_empty());

                let prev = self.reservations[unit]
                    .vregs
                    .insert(segment.live_range.to, (segment.live_range.from, vreg));

                // Ensure there are no overlapping reservations.
                debug_assert!(prev.is_none());
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
            for segment in virt_regs.segments(vreg) {
                debug_assert!(!segment.live_range.is_empty());

                let prev = self.reservations[unit].vregs.remove(&segment.live_range.to);

                // Ensure the entry contained the expected virtual register.
                debug_assert_eq!(prev, Some((segment.live_range.from, vreg)));
            }
        }
    }

    /// Dumps the entire register matrix to the log.
    #[allow(dead_code)]
    pub fn dump(&self) {
        trace!("Register matrix:");
        for (unit, reservations) in &self.reservations {
            if reservations.vregs.is_empty() && reservations.fixed.is_empty() {
                continue;
            }
            trace!("{unit}:");
            for (&to, &(from, vreg)) in &reservations.vregs {
                let segment = LiveRangeSegment::new(from, to);
                trace!("- {segment}: {vreg}");
            }
            for &(from, value) in &reservations.fixed {
                let segment = LiveRangeSegment::new(from, from.end_for_fixed_reservation());
                if let Some(value) = value.expand() {
                    trace!("- {segment}: fixed {value}");
                } else {
                    trace!("- {segment}: clobber");
                }
            }
        }
    }
}
