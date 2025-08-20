use alloc::vec;
use alloc::vec::Vec;
use core::ops::RangeInclusive;

use arbitrary::{Arbitrary, Result, Unstructured};

use super::{GenericRegInfo, PhysRegData, RegBankData, RegClassData, RegGroupData};
use crate::entity::PrimaryMap;
use crate::reginfo::{
    MAX_GROUP_SIZE, MAX_REG_UNITS, PhysReg, RegBank, RegClass, RegClassSet, RegGroup, RegGroupSet,
    RegInfo, RegUnit, SpillSlotSize,
};

/// Configuration options for [`GenericRegInfo::arbitrary_with_config`].
///
/// These are ranges from which a value is arbitrarily chosen when generating a
/// function.
///
/// It's generally fine to just use `Default::default` for this.
#[derive(Debug, Clone)]
pub struct ArbitraryRegInfoConfig {
    /// Number of register banks.
    pub num_banks: RangeInclusive<usize>,

    /// Number of registers in each register bank.
    pub regs_per_bank: RangeInclusive<usize>,

    /// Number of register classes in each register bank in addition to the
    /// top-level class and stack-to-stack class.
    pub extra_classes_per_bank: RangeInclusive<usize>,

    /// Number of register units per register.
    pub units_per_reg: RangeInclusive<usize>,
}

impl Default for ArbitraryRegInfoConfig {
    fn default() -> Self {
        Self {
            num_banks: 1..=5,
            regs_per_bank: 1..=20,
            extra_classes_per_bank: 1..=5,
            units_per_reg: 1..=8,
        }
    }
}

impl<'a> Arbitrary<'a> for GenericRegInfo {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        GenericRegInfo::arbitrary_with_config(u, Default::default())
    }
}

impl GenericRegInfo {
    /// Constructs a randomly-generated `GenericRegInfo`.
    ///
    /// The returned `GenericRegInfo` is guaranteed to pass validation.
    pub fn arbitrary_with_config(
        u: &mut Unstructured<'_>,
        config: ArbitraryRegInfoConfig,
    ) -> Result<Self> {
        let mut builder = RegInfoBuilder::new(u, config);

        for _ in 0..builder.u.int_in_range(builder.config.num_banks.clone())? {
            builder.gen_bank()?;
        }

        Ok(builder.reginfo)
    }
}

struct RegInfoBuilder<'a, 'b> {
    /// Source of randomness.
    u: &'a mut Unstructured<'b>,

    /// RegInfo that is being built.
    reginfo: GenericRegInfo,

    /// Configuration options
    config: ArbitraryRegInfoConfig,
}

impl<'a, 'b> RegInfoBuilder<'a, 'b> {
    /// Initializes the `RegInfoBuilder`.
    fn new(u: &'a mut Unstructured<'b>, config: ArbitraryRegInfoConfig) -> Self {
        let reginfo = GenericRegInfo {
            banks: PrimaryMap::new(),
            classes: PrimaryMap::new(),
            regs: PrimaryMap::new(),
            groups: PrimaryMap::new(),
        };

        Self { u, reginfo, config }
    }

    /// Generates a reasonable spill cost for a register class.
    fn spill_cost(&mut self) -> Result<f32> {
        Ok(*self.u.choose(&[0.0, 0.1, 0.5, 0.75, 1.0, 2.0])?)
    }

    /// Generates a reasonable spillslot size for a register bank.
    fn spillslot_size(&mut self) -> Result<SpillSlotSize> {
        Ok(SpillSlotSize::from_log2_bytes(self.u.int_in_range(0..=15)?))
    }

    /// Creates a new register bank.
    fn gen_bank(&mut self) -> Result<RegBank> {
        let bank = self.reginfo.banks.next_key();

        // Set of register units available for use in the bank.
        let mut units: Vec<_> = (0..MAX_REG_UNITS).map(RegUnit::new).collect();

        // Create registers in the bank.
        let mut regs = vec![];
        for _ in 0..self.u.int_in_range(self.config.regs_per_bank.clone())? {
            if units.is_empty() {
                break;
            }
            regs.push(self.gen_reg(bank, &mut units)?);
        }

        // Ensure at least 1 register is not in memory.
        self.reginfo.regs[regs[0]].is_fixed_stack = false;

        // Generate a top-level class.
        let allocation_order = self.gen_allocation_order(&regs, true)?;
        let top_level_class = RegClassData {
            bank,
            includes_spillslots: true,
            spill_cost: self.spill_cost()?,
            group_size: 1,
            members: regs.iter().copied().collect(),
            group_members: RegGroupSet::new(),
            sub_classes: RegClassSet::from_iter([self.reginfo.classes.next_key()]),
            allocation_order,
            group_allocation_order: vec![],
        };
        let top_level_class = self.reginfo.classes.push(top_level_class);

        // Generate a stack-to-stack class.
        let stack_to_stack_class = self.gen_subclass(top_level_class, true)?;

        // Generate additional classes.
        let mut classes = vec![top_level_class, stack_to_stack_class];
        for _ in 0..self
            .u
            .int_in_range(self.config.extra_classes_per_bank.clone())?
        {
            let superclass = *self.u.choose(&classes)?;
            let class = self.gen_subclass(superclass, false)?;
            classes.push(class);
        }

        let bank_data = RegBankData {
            top_level_class,
            stack_to_stack_class,
            spillslot_size: self.spillslot_size()?,
        };
        Ok(self.reginfo.banks.push(bank_data))
    }

    /// Generates an allocation order for a register class.
    fn gen_allocation_order<T: Copy>(
        &mut self,
        members: &[T],
        allows_spillslots: bool,
    ) -> Result<Vec<T>> {
        let mut out = vec![];
        for &member in members {
            if self.u.arbitrary()? {
                out.push(member);
            }
        }

        // Can't have an empty allocation order if spillslots aren't allowed.
        if !allows_spillslots && out.is_empty() {
            let member = *self.u.choose(members)?;
            out.push(member);
        }

        Ok(out)
    }

    /// Generates a sub-class of an existing class.
    fn gen_subclass(
        &mut self,
        superclass: RegClass,
        is_stack_to_stack_class: bool,
    ) -> Result<RegClass> {
        // Gather the set of possible members from the superclass.
        //
        // For stack_to_stack_class we additional restrict members so they are
        // real registers.
        let mut members: Vec<_> = self.reginfo.classes[superclass]
            .members
            .into_iter()
            .filter(|&reg| !is_stack_to_stack_class || !self.reginfo.is_memory(reg))
            .collect();
        let mut group_members: Vec<_> = self.reginfo.classes[superclass]
            .group_members
            .into_iter()
            .collect();

        // Check if we should generate a group subclass from a non-group class.
        let mut group_size = self.reginfo.class_group_size(superclass);
        if group_size == 1 && !is_stack_to_stack_class && self.u.arbitrary()? {
            let new_group_size = self.u.int_in_range(2..=MAX_GROUP_SIZE)?;
            if members.len() >= new_group_size {
                group_members = self.gen_groups(new_group_size, &members)?;
                members = vec![];
                group_size = new_group_size;
            }
        }

        // Remove some members at random to make it a proper subclass.
        if group_size == 1 {
            while members.len() > 1 && self.u.arbitrary()? {
                members.swap_remove(self.u.choose_index(members.len())?);
            }
        } else {
            while group_members.len() > 1 && self.u.arbitrary()? {
                group_members.swap_remove(self.u.choose_index(group_members.len())?);
            }
        }

        let bank = self.reginfo.bank_for_class(superclass);
        let class = self.reginfo.classes.next_key();
        let includes_spillslots = !is_stack_to_stack_class
            && group_size == 1
            && self.reginfo.class_includes_spillslots(superclass)
            && self.u.arbitrary()?;
        let mut class_data = RegClassData {
            bank,
            includes_spillslots,
            spill_cost: self.spill_cost()?,
            group_size: group_size as u8,
            members: members.iter().copied().collect(),
            group_members: group_members.iter().copied().collect(),
            sub_classes: RegClassSet::from_iter([class]),
            allocation_order: vec![],
            group_allocation_order: vec![],
        };

        if group_size == 1 {
            class_data.allocation_order =
                self.gen_allocation_order(&members, includes_spillslots)?;
        } else {
            class_data.group_allocation_order =
                self.gen_allocation_order(&group_members, includes_spillslots)?;
        }

        // Insert as sub-class of all our superclasses.
        for c in self.reginfo.classes() {
            if self.reginfo.classes[c].sub_classes.contains(superclass) {
                self.reginfo.classes[c].sub_classes.insert(class);
            }
        }

        Ok(self.reginfo.classes.push(class_data))
    }

    /// Creates register groups from the given registers.
    fn gen_groups(&mut self, group_size: usize, members: &[PhysReg]) -> Result<Vec<RegGroup>> {
        // Start with all members at every group index.
        let mut all = vec![members.to_vec(); group_size];

        // Fix up lists to ensure that the same register is not used multiple
        // times in a group.
        let mut temp = vec![];
        for group_idx in 1..group_size {
            for idx in 0..members.len() {
                while (0..group_idx)
                    .map(|i| all[i][idx])
                    .any(|r| r == all[group_idx][idx])
                {
                    temp.push(all[group_idx].swap_remove(idx));
                }
                all[group_idx].append(&mut temp);
            }
        }

        // Create the register groups
        let mut out = vec![];
        for i in 0..all.len() {
            let regs = (0..group_size).map(|group_idx| all[group_idx][i]).collect();
            let group = self.reginfo.groups.push(RegGroupData { regs });
            out.push(group);
        }

        Ok(out)
    }

    /// Creates a new register in a bank, making sure it doesn't share a unit
    /// with an existing register in the bank.
    fn gen_reg(&mut self, bank: RegBank, units_available: &mut Vec<RegUnit>) -> Result<PhysReg> {
        let mut units = vec![];
        for _ in 0..self.u.int_in_range(self.config.units_per_reg.clone())? {
            let idx = self.u.choose_index(units_available.len())?;
            units.push(units_available.swap_remove(idx));
        }
        Ok(self.reginfo.regs.push(PhysRegData {
            bank: Some(bank),
            is_fixed_stack: self.u.arbitrary()?,
            units,
        }))
    }
}
