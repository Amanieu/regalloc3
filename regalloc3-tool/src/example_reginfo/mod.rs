use core::fmt;

use anyhow::Result;
use clap::ValueEnum;
use regalloc3::entity::PrimaryMap;
use regalloc3::reginfo::{PhysReg, RegBank, RegClass, RegGroup, RegUnit};

mod aarch64;
mod riscv;

/// Architecture to generate the register definitions for.
#[derive(Debug, Copy, Clone, ValueEnum)]
pub enum Arch {
    /// ARM AArch64
    Aarch64,

    /// RISC-V
    Riscv,
}

impl Arch {
    pub fn gen(self, num_fixed_stack: usize) -> RegInfo {
        match self {
            Arch::Aarch64 => aarch64::make_aarch64_reginfo(num_fixed_stack),
            Arch::Riscv => riscv::make_riscv_reginfo(num_fixed_stack),
        }
    }
}

impl fmt::Display for Arch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Arch::Aarch64 => f.write_str("aarch64"),
            Arch::Riscv => f.write_str("riscv"),
        }
    }
}

/// Helper type to display a comma-separated list of displayable values.
struct DisplayIter<T>(T);
impl<T: Iterator + Clone> fmt::Display for DisplayIter<T>
where
    T::Item: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, val) in self.0.clone().enumerate() {
            if i == 0 {
                write!(f, "{}", val)?;
            } else {
                write!(f, " {}", val)?;
            }
        }
        Ok(())
    }
}

pub struct RegInfo {
    num_fixed_stack: usize,
    arch: Arch,
    units: PrimaryMap<RegUnit, ()>,
    regs: PrimaryMap<PhysReg, RegData>,
    groups: PrimaryMap<RegGroup, RegGroupData>,
    banks: PrimaryMap<RegBank, RegBankData>,
    classes: PrimaryMap<RegClass, RegClassData>,
}

impl RegInfo {
    /// Helper function to generate a sequence of register units.
    fn make_units(&mut self, count: usize) -> Vec<RegUnit> {
        (0..count).map(|_| self.units.push(())).collect()
    }

    /// Helper function to generate a sequence of registers.
    fn make_regs(
        &mut self,
        count: usize,
        mut make_reg: impl FnMut(usize) -> RegData,
    ) -> Vec<PhysReg> {
        let mut out = vec![];
        for i in 0..count {
            out.push(self.regs.push(make_reg(i)))
        }
        out
    }

    /// Helper function to generate a sequence of register groups.
    fn make_reg_group(
        &mut self,
        count: usize,
        mut make_group: impl FnMut(usize) -> RegGroupData,
    ) -> Vec<RegGroup> {
        let mut out = vec![];
        for i in 0..count {
            out.push(self.groups.push(make_group(i)))
        }
        out
    }

    /// Writes the register definition to the given sink.
    pub fn emit(&self, f: &mut impl fmt::Write) -> Result<()> {
        writeln!(f, "; Generated by regalloc3-tool, do not edit manually")?;
        writeln!(
            f,
            "; Re-generate with: cargo run -p regalloc3-tool -- example-reginfo -f {} {}",
            self.num_fixed_stack, self.arch,
        )?;

        writeln!(f)?;
        for (reg, regdata) in self.regs.iter() {
            let location = if regdata.is_stack { "stack" } else { "reg" };
            if regdata.units.is_empty() {
                writeln!(f, "{reg} = {location} nonallocatable ; {}", regdata.name,)?;
            } else {
                writeln!(
                    f,
                    "{reg} = {location} {} ; {}",
                    DisplayIter(regdata.units.iter()),
                    regdata.name,
                )?;
            }
        }

        writeln!(f)?;
        for (group, groupdata) in self.groups.iter() {
            writeln!(
                f,
                "{group} = {} ; ({})",
                DisplayIter(groupdata.regs.iter()),
                DisplayIter(groupdata.regs.iter().map(|&r| &self.regs[r].name)),
            )?;
        }

        for (bank, bankdata) in self.banks.iter() {
            writeln!(f)?;
            writeln!(f, "; {}", bankdata.desc)?;
            writeln!(f, "{bank} {{")?;
            writeln!(f, "    top_level_class = {}", bankdata.top_level_class)?;
            writeln!(
                f,
                "    stack_to_stack_class = {}",
                bankdata.stack_to_stack_class
            )?;
            writeln!(f, "    spillslot_size = {}", bankdata.spillslot_size)?;

            for &class in &bankdata.classes {
                let classdata = &self.classes[class];
                writeln!(f)?;
                writeln!(f, "    ; {}", classdata.desc)?;
                if let Some(superclass) = classdata.superclass {
                    writeln!(f, "    {class}: {superclass} {{")?;
                } else {
                    writeln!(f, "    {class} {{")?;
                }
                if classdata.group_size != 1 {
                    writeln!(f, "        group_size = {}", classdata.group_size)?;
                }
                if classdata.allows_spillslots {
                    writeln!(f, "        allows_spillslots")?;
                }
                writeln!(f, "        spill_cost = {}", classdata.spill_cost)?;
                writeln!(f, "        members = {}", classdata.members)?;
                if !classdata.preferred_regs.is_empty() {
                    writeln!(f, "        preferred_regs = {}", classdata.preferred_regs)?;
                }
                if !classdata.non_preferred_regs.is_empty() {
                    writeln!(
                        f,
                        "        non_preferred_regs = {}",
                        classdata.non_preferred_regs
                    )?;
                }
                if !classdata.callee_saved_preferred_regs.is_empty() {
                    writeln!(
                        f,
                        "        callee_saved_preferred_regs = {}",
                        classdata.callee_saved_preferred_regs
                    )?;
                }
                if !classdata.callee_saved_non_preferred_regs.is_empty() {
                    writeln!(
                        f,
                        "        callee_saved_non_preferred_regs = {}",
                        classdata.callee_saved_non_preferred_regs
                    )?;
                }
                writeln!(f, "    }}")?;
            }

            writeln!(f, "}}")?;
        }

        Ok(())
    }
}

/// Information about a register.
struct RegData {
    name: String,
    is_stack: bool,
    units: Vec<RegUnit>,
}

/// Information about a register group.
struct RegGroupData {
    regs: Vec<PhysReg>,
}

/// Information about a register bank.
struct RegBankData {
    desc: String,
    top_level_class: RegClass,
    stack_to_stack_class: RegClass,
    spillslot_size: usize,
    classes: Vec<RegClass>,
}

/// A list of registers or register groups
enum RegGroupList {
    Single(Vec<PhysReg>),
    Multi(Vec<RegGroup>),
}
impl RegGroupList {
    fn is_empty(&self) -> bool {
        match self {
            RegGroupList::Single(v) => v.is_empty(),
            RegGroupList::Multi(v) => v.is_empty(),
        }
    }
}

impl fmt::Display for RegGroupList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegGroupList::Single(regs) => write!(f, "{}", DisplayIter(regs.iter())),
            RegGroupList::Multi(groups) => write!(f, "{}", DisplayIter(groups.iter())),
        }
    }
}

/// Information about a register class.
struct RegClassData {
    desc: String,
    superclass: Option<RegClass>,
    group_size: usize,
    allows_spillslots: bool,
    spill_cost: f32,
    members: RegGroupList,
    preferred_regs: RegGroupList,
    non_preferred_regs: RegGroupList,
    callee_saved_preferred_regs: RegGroupList,
    callee_saved_non_preferred_regs: RegGroupList,
}
