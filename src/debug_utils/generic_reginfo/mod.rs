//! Generic implementation of a [`RegInfo`] which can be used for testing the
//! register allocator.

use alloc::vec::Vec;
use core::fmt;

use cranelift_entity::PrimaryMap;

use crate::reginfo::{
    AllocationOrderSet, PhysReg, RegBank, RegClass, RegClassSet, RegGroup, RegInfo, RegOrRegGroup,
    RegOrRegGroupSet, RegUnit, SpillSlotSize,
};

#[cfg(feature = "arbitrary")]
mod arbitrary;
#[cfg(feature = "arbitrary")]
pub use arbitrary::ArbitraryRegInfoConfig;

use super::DisplayRegInfo;
#[cfg(feature = "parse")]
mod parse;

#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct RegBankData {
    top_level_class: RegClass,
    stack_to_stack_class: RegClass,
    reftype_class: Option<RegClass>,
    spillslot_size: SpillSlotSize,
}

#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct RegClassData {
    bank: RegBank,
    includes_spillslots: bool,
    spill_cost: f32,
    group_size: u8,
    members: RegOrRegGroupSet,
    sub_classes: RegClassSet,
    preferred_regs: Vec<RegOrRegGroup>,
    non_preferred_regs: Vec<RegOrRegGroup>,
    callee_saved_preferred_regs: Vec<RegOrRegGroup>,
    callee_saved_non_preferred_regs: Vec<RegOrRegGroup>,
}

#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct PhysRegData {
    bank: Option<RegBank>,
    is_fixed_stack: bool,
    units: Vec<RegUnit>,
}

#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct RegGroupData {
    regs: Vec<PhysReg>,
}

/// A generic implementation of [`RegInfo`] which can be constructed from an
/// existing `RegInfo` or parsed from a text representation.
///
/// This is primarily useful for development and debugging of the register
/// allocator since it enables working with user-readable and editable forms of
/// a machine register description.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GenericRegInfo {
    banks: PrimaryMap<RegBank, RegBankData>,
    classes: PrimaryMap<RegClass, RegClassData>,
    regs: PrimaryMap<PhysReg, PhysRegData>,
    groups: PrimaryMap<RegGroup, RegGroupData>,
}

impl fmt::Debug for GenericRegInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        DisplayRegInfo(self).fmt(f)
    }
}

impl fmt::Display for GenericRegInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        DisplayRegInfo(self).fmt(f)
    }
}

impl GenericRegInfo {
    /// Constructs a `GenericRegInfo` from an existing object which implements
    /// the [`RegInfo`] trait.
    pub fn from_reginfo(reginfo: &impl RegInfo) -> Self {
        let mut banks = PrimaryMap::new();
        let mut classes = PrimaryMap::new();
        let mut regs = PrimaryMap::new();
        let mut groups = PrimaryMap::new();
        for bank in reginfo.banks() {
            banks.push(RegBankData {
                top_level_class: reginfo.top_level_class(bank),
                stack_to_stack_class: reginfo.stack_to_stack_class(bank),
                reftype_class: reginfo.reftype_class(bank),
                spillslot_size: reginfo.spillslot_size(bank),
            });
        }
        for class in reginfo.classes() {
            classes.push(RegClassData {
                bank: reginfo.bank_for_class(class),
                includes_spillslots: reginfo.class_includes_spillslots(class),
                spill_cost: reginfo.class_spill_cost(class),
                group_size: reginfo.class_group_size(class) as u8,
                members: reginfo.class_members(class),
                sub_classes: reginfo.sub_classes(class),
                preferred_regs: reginfo
                    .allocation_order(class, AllocationOrderSet::Preferred)
                    .into(),
                non_preferred_regs: reginfo
                    .allocation_order(class, AllocationOrderSet::NonPreferred)
                    .into(),
                callee_saved_preferred_regs: reginfo
                    .allocation_order(class, AllocationOrderSet::CalleeSavedPreferred)
                    .into(),
                callee_saved_non_preferred_regs: reginfo
                    .allocation_order(class, AllocationOrderSet::CalleeSavedNonPreferred)
                    .into(),
            });
        }
        for reg in reginfo.regs() {
            let bank = reginfo.bank_for_reg(reg);
            let is_fixed_stack = reginfo.is_memory(reg);
            regs.push(PhysRegData {
                bank,
                is_fixed_stack,
                units: reginfo.reg_units(reg).into(),
            });
        }
        for group in reginfo.reg_groups() {
            groups.push(RegGroupData {
                regs: reginfo.reg_group_members(group).into(),
            });
        }
        Self {
            banks,
            classes,
            regs,
            groups,
        }
    }
}

impl RegInfo for GenericRegInfo {
    #[inline]
    fn num_banks(&self) -> usize {
        self.banks.len()
    }

    #[inline]
    fn top_level_class(&self, bank: RegBank) -> RegClass {
        self.banks[bank].top_level_class
    }

    #[inline]
    fn stack_to_stack_class(&self, bank: RegBank) -> RegClass {
        self.banks[bank].stack_to_stack_class
    }

    #[inline]
    fn bank_for_class(&self, class: RegClass) -> RegBank {
        self.classes[class].bank
    }

    #[inline]
    fn bank_for_reg(&self, reg: PhysReg) -> Option<RegBank> {
        self.regs[reg].bank
    }

    #[inline]
    fn spillslot_size(&self, bank: RegBank) -> SpillSlotSize {
        self.banks[bank].spillslot_size
    }

    #[inline]
    fn num_classes(&self) -> usize {
        self.classes.len()
    }

    #[inline]
    fn class_members(&self, class: RegClass) -> RegOrRegGroupSet {
        self.classes[class].members
    }

    #[inline]
    fn class_includes_spillslots(&self, class: RegClass) -> bool {
        self.classes[class].includes_spillslots
    }

    #[inline]
    fn class_spill_cost(&self, class: RegClass) -> f32 {
        self.classes[class].spill_cost
    }

    #[inline]
    fn allocation_order(&self, class: RegClass, set: AllocationOrderSet) -> &[RegOrRegGroup] {
        match set {
            AllocationOrderSet::Preferred => &self.classes[class].preferred_regs,
            AllocationOrderSet::NonPreferred => &self.classes[class].non_preferred_regs,
            AllocationOrderSet::CalleeSavedPreferred => {
                &self.classes[class].callee_saved_preferred_regs
            }
            AllocationOrderSet::CalleeSavedNonPreferred => {
                &self.classes[class].callee_saved_non_preferred_regs
            }
        }
    }

    #[inline]
    fn sub_classes(&self, class: RegClass) -> RegClassSet {
        self.classes[class].sub_classes
    }

    #[inline]
    fn class_group_size(&self, class: RegClass) -> usize {
        self.classes[class].group_size as usize
    }

    #[inline]
    fn num_regs(&self) -> usize {
        self.regs.len()
    }

    #[inline]
    fn reg_units(&self, reg: PhysReg) -> &[RegUnit] {
        &self.regs[reg].units
    }

    #[inline]
    fn is_memory(&self, reg: PhysReg) -> bool {
        self.regs[reg].is_fixed_stack
    }

    #[inline]
    fn num_reg_groups(&self) -> usize {
        self.groups.len()
    }

    #[inline]
    fn reg_group_members(&self, group: RegGroup) -> &[PhysReg] {
        &self.groups[group].regs
    }

    #[inline]
    fn group_for_reg(&self, reg: PhysReg, group_index: usize, class: RegClass) -> Option<RegGroup> {
        debug_assert_ne!(self.classes[class].group_size, 1);
        for group in &self.classes[class].members {
            let group = group.as_multi();
            if self.groups[group].regs[group_index] == reg {
                return Some(group);
            }
        }
        None
    }

    #[inline]
    fn reftype_class(&self, bank: RegBank) -> Option<RegClass> {
        self.banks[bank].reftype_class
    }
}
