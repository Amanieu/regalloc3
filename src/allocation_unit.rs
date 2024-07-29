//! `AllocationUnit` and `AllocationUnitMap`/`AllocationUnitSet` helper types.

use core::ops::{Index, IndexMut};
use core::{fmt, iter};

use cranelift_entity::{EntitySet, SecondaryMap};

use crate::output::{Allocation, AllocationKind, SpillSlot};
use crate::reginfo::{RegInfo, RegUnit, RegUnitSet};

/// Helper type which abstracts over a register unit or spill slot.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum AllocationUnit {
    Reg(RegUnit),
    Slot(SpillSlot),
}

impl fmt::Display for AllocationUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            AllocationUnit::Reg(unit) => unit.fmt(f),
            AllocationUnit::Slot(slot) => slot.fmt(f),
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
            AllocationKind::PhysReg(reg) => EitherIter::A(
                reginfo
                    .reg_units(reg)
                    .iter()
                    .map(|&unit| AllocationUnit::Reg(unit)),
            ),
            AllocationKind::SpillSlot(slot) => {
                EitherIter::B(iter::once(AllocationUnit::Slot(slot)))
            }
        }
    }
}

/// A wrapper around [`SecondaryMap`] which takes an `AllocationUnit` as an
/// index.
#[derive(Debug, Clone, Default)]
pub(crate) struct AllocationUnitMap<T: Clone> {
    unit: SecondaryMap<RegUnit, T>,
    slot: SecondaryMap<SpillSlot, T>,
}

impl<T: Clone> AllocationUnitMap<T> {
    pub(crate) fn new() -> Self
    where
        T: Default,
    {
        Self {
            unit: SecondaryMap::new(),
            slot: SecondaryMap::new(),
        }
    }

    pub(crate) fn clear(&mut self) {
        self.unit.clear();
        self.slot.clear();
    }
}

impl<T: Clone> Index<AllocationUnit> for AllocationUnitMap<T> {
    type Output = T;

    fn index(&self, index: AllocationUnit) -> &Self::Output {
        match index {
            AllocationUnit::Reg(unit) => &self.unit[unit],
            AllocationUnit::Slot(slot) => &self.slot[slot],
        }
    }
}

impl<T: Clone> IndexMut<AllocationUnit> for AllocationUnitMap<T> {
    fn index_mut(&mut self, index: AllocationUnit) -> &mut Self::Output {
        match index {
            AllocationUnit::Reg(unit) => &mut self.unit[unit],
            AllocationUnit::Slot(slot) => &mut self.slot[slot],
        }
    }
}

/// A wrapper around [`EntitySet`] which takes an `AllocationUnit` as an
/// index.
#[derive(Debug, Clone, Default)]
pub(crate) struct AllocationUnitSet {
    unit: RegUnitSet,
    slot: EntitySet<SpillSlot>,
}

impl AllocationUnitSet {
    pub(crate) fn new() -> Self {
        Self {
            unit: RegUnitSet::new(),
            slot: EntitySet::new(),
        }
    }

    pub(crate) fn clear(&mut self) {
        self.unit.clear();
        self.slot.clear();
    }

    pub(crate) fn contains(&self, unit: AllocationUnit) -> bool {
        match unit {
            AllocationUnit::Reg(unit) => self.unit.contains(unit),
            AllocationUnit::Slot(slot) => self.slot.contains(slot),
        }
    }

    pub(crate) fn insert(&mut self, unit: AllocationUnit) {
        match unit {
            AllocationUnit::Reg(unit) => {
                self.unit.insert(unit);
            }
            AllocationUnit::Slot(slot) => {
                self.slot.insert(slot);
            }
        }
    }
}
