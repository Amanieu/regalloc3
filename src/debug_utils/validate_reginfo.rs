//! Register information validation.

use core::fmt;

use anyhow::{bail, ensure, Result};

use crate::entity::SecondaryMap;
use crate::reginfo::{
    AllocationOrderSet, PhysReg, PhysRegSet, RegBank, RegClass, RegGroup, RegInfo, RegUnit,
    RegUnitSet, MAX_GROUP_SIZE, MAX_PHYSREGS, MAX_REG_BANKS, MAX_REG_CLASSES, MAX_REG_GROUPS,
    MAX_REG_UNITS, MAX_UNITS_PER_REG,
};

/// Checks `reginfo` to ensure it satisfies all of the pre-conditions required
/// by the register allocator.
pub fn validate_reginfo(reginfo: &impl RegInfo) -> Result<()> {
    let mut ctx = Context {
        reginfo,
        bank_units: SecondaryMap::with_max_index(reginfo.num_banks()),
    };
    ctx.check_reginfo()?;
    Ok(())
}

/// An entity reference.
///
/// This is used by [`ValidationError::InvalidEntity`] to report entity
/// references with an invalid index.
#[derive(Debug, Clone, Copy)]
enum Entity {
    RegUnit(RegUnit),
    PhysReg(PhysReg),
    RegGroup(RegGroup),
    RegClass(RegClass),
    RegBank(RegBank),
}

impl fmt::Display for Entity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Entity::RegUnit(x) => x.fmt(f),
            Entity::PhysReg(x) => x.fmt(f),
            Entity::RegGroup(x) => x.fmt(f),
            Entity::RegClass(x) => x.fmt(f),
            Entity::RegBank(x) => x.fmt(f),
        }
    }
}

/// State used for validation.
struct Context<'a, R> {
    reginfo: &'a R,
    bank_units: SecondaryMap<RegBank, RegUnitSet>,
}

impl<R: RegInfo> Context<'_, R> {
    /// Check that an entity refers to a valid object.
    fn check_entity(&self, entity: Entity) -> Result<()> {
        let (index, len) = match entity {
            Entity::RegUnit(x) => (x.index(), MAX_REG_UNITS),
            Entity::PhysReg(x) => (x.index(), self.reginfo.num_regs()),
            Entity::RegGroup(x) => (x.index(), self.reginfo.num_reg_groups()),
            Entity::RegClass(x) => (x.index(), self.reginfo.num_classes()),
            Entity::RegBank(x) => (x.index(), self.reginfo.num_banks()),
        };
        ensure!(index < len, "{entity}: Invalid entity reference");
        Ok(())
    }

    /// Check the limits on the number of entities.
    fn check_limits(&self) -> Result<()> {
        ensure!(
            self.reginfo.num_regs() <= MAX_PHYSREGS,
            "Too many registers: {} (max: {MAX_PHYSREGS})",
            self.reginfo.num_regs(),
        );
        ensure!(
            self.reginfo.num_reg_groups() <= MAX_REG_GROUPS,
            "Too many register groups: {} (max: {MAX_REG_GROUPS})",
            self.reginfo.num_reg_groups(),
        );
        ensure!(
            self.reginfo.num_classes() <= MAX_REG_CLASSES,
            "Too many register classes: {} (max: {MAX_REG_CLASSES})",
            self.reginfo.num_classes(),
        );
        ensure!(
            self.reginfo.num_banks() <= MAX_REG_BANKS,
            "Too many register banks: {} (max: {MAX_REG_BANKS})",
            self.reginfo.num_banks(),
        );
        Ok(())
    }

    /// Check a register bank.
    fn check_bank(&mut self, bank: RegBank) -> Result<()> {
        // Check that the top-level class is part of this bank and allows
        // spillslots
        let top_level_class = self.reginfo.top_level_class(bank);
        self.check_entity(Entity::RegClass(top_level_class))?;
        ensure!(
            self.reginfo.bank_for_class(top_level_class) == bank,
            "{bank}: Top-level class {top_level_class} is not in bank"
        );
        ensure!(
            self.reginfo.class_includes_spillslots(top_level_class),
            "{bank}: Top-level class {top_level_class} must include spillslots"
        );
        ensure!(
            self.reginfo.class_group_size(top_level_class) == 1,
            "{bank}: Top-level class {top_level_class} must have a group size of 1"
        );

        // Check stack_to_stack_class
        let stack_to_stack_class = self.reginfo.stack_to_stack_class(bank);
        self.check_entity(Entity::RegClass(stack_to_stack_class))?;
        ensure!(
            self.reginfo.bank_for_class(stack_to_stack_class) == bank,
            "{bank}: Stack-to-stack class {stack_to_stack_class} is not in bank"
        );
        ensure!(
            self.reginfo.class_group_size(stack_to_stack_class) == 1,
            "{stack_to_stack_class}: Stack-to-stack class must have a group size of 1"
        );

        // Check registers in the bank
        let mut empty = true;
        for reg in self.reginfo.regs() {
            if self.reginfo.bank_for_reg(reg) == Some(bank) {
                empty = false;
                ensure!(
                    self.reginfo.class_members(top_level_class).contains(reg),
                    "{bank}: {reg} not in top-level class {top_level_class}"
                );
            }
        }
        ensure!(!empty, "{bank} has no registers");

        // Check stack_to_stack_class
        for reg in &self.reginfo.class_members(stack_to_stack_class) {
            ensure!(
                !self.reginfo.is_memory(reg),
                "{bank}: {reg} in stack-to-stack {stack_to_stack_class} cannot be in memory"
            );
        }

        Ok(())
    }

    /// Check a register class.
    fn check_class(&mut self, class: RegClass) -> Result<()> {
        let bank = self.reginfo.bank_for_class(class);
        self.check_entity(Entity::RegBank(bank))?;

        let group_size = self.reginfo.class_group_size(class);
        ensure!(
            group_size <= MAX_GROUP_SIZE,
            "{class}: Group size {group_size} too large (max: {MAX_GROUP_SIZE})"
        );
        ensure!(group_size != 0, "{class}: Invalid group size of 0");
        if group_size != 1 {
            ensure!(
                !self.reginfo.class_includes_spillslots(class),
                "{class}: Group class cannot include spillslots"
            );
        }

        if group_size != 1 {
            let mut regs_per_index = [PhysRegSet::new(); MAX_GROUP_SIZE];
            for group in &self.reginfo.class_group_members(class) {
                // Check that class members have the same group size as the class.
                let members = self.reginfo.reg_group_members(group);
                ensure!(
                    members.len() == group_size,
                    "{group} group size ({}) doesn't match {class} group size ({group_size})",
                    members.len()
                );

                for (group_index, &reg) in members.iter().enumerate() {
                    // Check that class members are in the same bank as the class.
                    ensure!(
                        self.reginfo.bank_for_reg(reg) == Some(bank),
                        "{class} contains {reg} which is outside {bank}"
                    );

                    // Check that, at each index, each register is unique within the class.
                    ensure!(
                        !regs_per_index[group_index].contains(reg),
                        "{class} has duplicate register {reg} at group index {group_index}"
                    );
                    regs_per_index[group_index].insert(reg);

                    // Check the reverse mapping in group_for_reg.
                    ensure!(
                        self.reginfo.group_for_reg(reg, group_index, class) == Some(group),
                        "Inconsistent group_for_reg({reg}, {group_index}, {class}): got {:?} \
                         expected {:?}",
                        self.reginfo.group_for_reg(reg, group_index, class),
                        Some(group)
                    );
                }
            }
        } else {
            for reg in &self.reginfo.class_members(class) {
                // Check that class members are in the same bank as the class.
                ensure!(
                    self.reginfo.bank_for_reg(reg) == Some(bank),
                    "{class} contains {reg} which is outside {bank}"
                );
            }
        }

        // Check that the allocation order isn't empty if spillslot are not
        // allowed or if this is a group register class.
        if group_size == 1 {
            if !self.reginfo.class_includes_spillslots(class) {
                ensure!(
                    AllocationOrderSet::each()
                        .any(|set| !self.reginfo.allocation_order(class, set).is_empty()),
                    "{class} cannot have an empty allocation order unless it allows spillslots"
                );
            }
            ensure!(
                AllocationOrderSet::each()
                    .all(|set| self.reginfo.group_allocation_order(class, set).is_empty()),
                "{class}: Non-group class cannot have a group allocation order"
            );
        } else {
            ensure!(
                AllocationOrderSet::each()
                    .any(|set| !self.reginfo.group_allocation_order(class, set).is_empty()),
                "{class}: Group class cannot have an empty allocation order"
            );
            ensure!(
                AllocationOrderSet::each()
                    .all(|set| self.reginfo.allocation_order(class, set).is_empty()),
                "{class}: Non-group class cannot have a non-group allocation order"
            );
        }

        // Check that the allocation order only contains class members.
        if group_size == 1 {
            for &reg in
                AllocationOrderSet::each().flat_map(|set| self.reginfo.allocation_order(class, set))
            {
                self.check_entity(Entity::PhysReg(reg))?;
                ensure!(
                    self.reginfo.class_members(class).contains(reg),
                    "{class}: Allocation order contains {reg} which is outside class"
                );
            }
        } else {
            for &group in AllocationOrderSet::each()
                .flat_map(|set| self.reginfo.group_allocation_order(class, set))
            {
                self.check_entity(Entity::RegGroup(group))?;
                ensure!(
                    self.reginfo.class_group_members(class).contains(group),
                    "{class}: Allocation order contains {group} which is outside class"
                );
            }
        }

        // Check subclasses
        let top_level_class = self.reginfo.top_level_class(bank);
        ensure!(
            self.reginfo.sub_classes(top_level_class).contains(class),
            "{class} must be a subclass of the top-level class for {bank} ({top_level_class})"
        );
        ensure!(
            self.reginfo.sub_classes(class).contains(class),
            "{class} must be a subclass of itself"
        );
        for subclass in &self.reginfo.sub_classes(class) {
            ensure!(
                subclass.index() >= class.index(),
                "{class} must not have any subclasses with a lower index than itself"
            );
            ensure!(
                self.reginfo.bank_for_class(subclass) == bank,
                "Subclass {subclass} of {class} is not in {bank}"
            );
            if !self.reginfo.class_includes_spillslots(class) {
                ensure!(
                    !self.reginfo.class_includes_spillslots(subclass),
                    "{subclass} allows spillslots but is subclass of {class} which doesn't"
                );
            }

            if group_size == 1 && self.reginfo.class_group_size(subclass) > 1 {
                for group in &self.reginfo.class_group_members(subclass) {
                    for &member in self.reginfo.reg_group_members(group) {
                        ensure!(
                            self.reginfo.class_members(class).contains(member),
                            "Superclass {class} of {subclass} doesn't contain {member} (member of \
                             {group})"
                        );
                    }
                }
            } else {
                ensure!(
                    self.reginfo.class_group_size(subclass) == group_size,
                    "Subclass {subclass} must have same group size as {class}"
                );
                for reg in &self.reginfo.class_members(subclass) {
                    ensure!(
                        self.reginfo.class_members(class).contains(reg),
                        "Subclass {subclass} of {class} doesn't contain {reg}"
                    );
                }
            }
        }

        Ok(())
    }

    /// Check a register.
    fn check_reg(&mut self, reg: PhysReg) -> Result<()> {
        // Non-allocatable registers have no constraints.
        if let Some(bank) = self.reginfo.bank_for_reg(reg) {
            self.check_entity(Entity::RegBank(bank))?;
            ensure!(
                !self.reginfo.reg_units(reg).is_empty(),
                "{reg}: Allocatable register must have at least 1 register unit"
            );

            ensure!(
                self.reginfo.reg_units(reg).len() <= MAX_UNITS_PER_REG,
                "{reg} has too many register units (max is {MAX_UNITS_PER_REG})"
            );

            for &unit in self.reginfo.reg_units(reg) {
                self.check_entity(Entity::RegUnit(unit))?;
                ensure!(
                    !self.bank_units[bank].contains(unit),
                    "{unit} in {reg} overlaps with other registers in {bank}"
                );
                self.bank_units[bank].insert(unit);
            }
        }

        Ok(())
    }

    /// Check a register group.
    fn check_reg_group(&mut self, group: RegGroup) -> Result<()> {
        let members = self.reginfo.reg_group_members(group);
        ensure!(
            members.len() >= 2 || members.len() > MAX_GROUP_SIZE,
            "{group}: Invalid group size {}",
            members.len()
        );
        let Some(bank) = self.reginfo.bank_for_reg(members[0]) else {
            bail!(
                "{group}: Register member {} must be in a register bank",
                members[0]
            );
        };

        // Check for overlaps within a group.
        let mut set = PhysRegSet::new();
        for &member in members {
            self.check_entity(Entity::PhysReg(member))?;
            ensure!(
                self.reginfo.bank_for_reg(member) == Some(bank),
                "{group}: Group member {member} expected to be from {bank}"
            );
            ensure!(
                !set.contains(member),
                "{group} contains duplicate member {member}"
            );
            set.insert(member);
        }
        Ok(())
    }

    /// Main entry point for `RegInfo` validation.
    fn check_reginfo(&mut self) -> Result<()> {
        self.check_limits()?;

        for bank in self.reginfo.banks() {
            self.check_bank(bank)?;
        }

        for class in self.reginfo.classes() {
            self.check_class(class)?;
        }

        for reg in self.reginfo.regs() {
            self.check_reg(reg)?;
        }

        for group in self.reginfo.reg_groups() {
            self.check_reg_group(group)?;
        }

        Ok(())
    }
}
