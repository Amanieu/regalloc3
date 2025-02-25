//! `AllocationUnit` helper type.

use core::{fmt, iter};

use crate::entity::{EntityRef, ReservedValue};
use crate::output::{Allocation, AllocationKind, SpillSlot};
use crate::reginfo::{RegInfo, RegUnit, MAX_REG_UNITS};

/// Entity type which can represent either a [`RegUnit`] or a [`SpillSlot`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct AllocationUnit {
    /// Indices below `MAX_REG_UNITS` indicate a `RegUnit`. Otherwise
    /// this indicates a `SpillSlot` with index `index - MAX_REG_UNITS`.
    index: u32,
}

impl EntityRef for AllocationUnit {
    #[inline]
    fn new(index: usize) -> Self {
        debug_assert!(index < (u32::MAX as usize));
        Self {
            index: index as u32,
        }
    }

    #[inline]
    fn index(self) -> usize {
        self.index as usize
    }
}

impl ReservedValue for AllocationUnit {
    #[inline]
    fn reserved_value() -> Self {
        Self { index: u32::MAX }
    }

    #[inline]
    fn is_reserved_value(&self) -> bool {
        self.index == u32::MAX
    }
}

impl AllocationUnit {
    /// Constructs a new `AllocationUnit`.
    #[inline]
    #[must_use]
    pub(crate) const fn new(kind: AllocationUnitKind) -> Self {
        let index = match kind {
            AllocationUnitKind::SpillSlot(spillslot) => spillslot.index() + MAX_REG_UNITS,
            AllocationUnitKind::Reg(unit) => unit.index(),
        };
        Self {
            index: index as u32,
        }
    }

    /// Creates an allocation into a `PhysReg`.
    #[inline]
    #[must_use]
    pub(crate) fn reg(unit: RegUnit) -> AllocationUnit {
        AllocationUnit::new(AllocationUnitKind::Reg(unit))
    }

    /// Creates an allocation into a `SpillSlot`.
    #[inline]
    #[must_use]
    pub(crate) fn spillslot(spillslot: SpillSlot) -> AllocationUnit {
        AllocationUnit::new(AllocationUnitKind::SpillSlot(spillslot))
    }

    /// Expands the `AllocationKind` into an enum that can be matched on.
    #[inline]
    #[must_use]
    pub fn kind(self) -> AllocationUnitKind {
        if self.index() < MAX_REG_UNITS {
            AllocationUnitKind::Reg(RegUnit::new(self.index()))
        } else {
            AllocationUnitKind::SpillSlot(SpillSlot::new(self.index() - MAX_REG_UNITS))
        }
    }
}

/// Expanded form of `AllocationUnit` as an enum.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum AllocationUnitKind {
    Reg(RegUnit),
    SpillSlot(SpillSlot),
}

impl fmt::Debug for AllocationUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for AllocationUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind() {
            AllocationUnitKind::Reg(unit) => unit.fmt(f),
            AllocationUnitKind::SpillSlot(slot) => slot.fmt(f),
        }
    }
}

impl Allocation {
    /// Returns an iterator over the units in an allocation.
    pub(crate) fn units(self, reginfo: &impl RegInfo) -> impl Iterator<Item = AllocationUnit> + '_ {
        enum EitherIter<A, B> {
            A(A),
            B(B),
        }
        impl<A, B> Iterator for EitherIter<A, B>
        where
            A: Iterator,
            B: Iterator<Item = A::Item>,
        {
            type Item = A::Item;

            fn next(&mut self) -> Option<Self::Item> {
                match self {
                    EitherIter::A(a) => a.next(),
                    EitherIter::B(b) => b.next(),
                }
            }
        }

        match self.kind() {
            AllocationKind::PhysReg(reg) => {
                EitherIter::A(reginfo.reg_units(reg).map(AllocationUnit::reg))
            }
            AllocationKind::SpillSlot(slot) => {
                EitherIter::B(iter::once(AllocationUnit::spillslot(slot)))
            }
        }
    }
}
