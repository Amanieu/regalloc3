//! Support for displaying human-readable representations of types implementing
//! [`Function`] or [`RegInfo`].

use core::cell::Cell;
use core::fmt;

use crate::function::{Block, Function, OperandConstraint, OperandKind, RematCost, TerminatorKind};
use crate::output::{Allocation, AllocationKind, Output, OutputInst};
use crate::reginfo::{AllocationOrderSet, RegClass, RegClassSet, RegInfo};

/// Helper type to display a comma-separated list of displayable values.
pub(crate) struct DisplayIter<T> {
    iter: Cell<Option<T>>,
    separator: &'static str,
}
impl<T: IntoIterator> fmt::Display for DisplayIter<T>
where
    T::Item: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, val) in self.iter.take().unwrap().into_iter().enumerate() {
            if i == 0 {
                write!(f, "{val}")?;
            } else {
                write!(f, "{} {val}", self.separator)?;
            }
        }
        Ok(())
    }
}

pub(crate) fn display_iter<I: IntoIterator<Item = impl fmt::Display>>(
    iter: I,
    separator: &'static str,
) -> DisplayIter<I> {
    DisplayIter {
        iter: Cell::new(Some(iter)),
        separator,
    }
}

/// Wrapper around a type implementing [`Function`] that provides a [`Display`]
/// implementation which dumps the function in a format that is both
/// human-readable and machine-parseable.
///
/// The returned string can be re-parsed into a function by using
/// [`GenericFunction::parse`] (requires the `parse` cargo feature).
///
/// [`GenericFunction::parse`]: crate::debug_utils::GenericFunction::parse
/// [`Display`]: core::fmt::Display
pub struct DisplayFunction<'a, F: Function>(pub &'a F);

impl<F: Function> fmt::Debug for DisplayFunction<'_, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl<F: Function> fmt::Display for DisplayFunction<'_, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Value declarations
        for value in self.0.values() {
            let bank = self.0.value_bank(value);
            write!(f, "{value} = {bank}")?;
            if let Some((cost, class)) = self.0.can_rematerialize(value) {
                let cost = match cost {
                    RematCost::CheaperThanMove => "cheaper_than_move",
                    RematCost::CheaperThanLoad => "cheaper_than_load",
                };
                write!(f, " remat({cost}, {class})")?;
            }
            writeln!(f)?;
        }

        // Blocks
        for block in self.0.blocks() {
            writeln!(f)?;
            writeln!(
                f,
                "{block}({}) freq({}){}:",
                display_iter(self.0.block_params(block), ","),
                self.0.block_frequency(block),
                if self.0.block_is_critical_edge(block) {
                    " critical_edge"
                } else {
                    ""
                }
            )?;

            // Predecessors are emitted as a comment. They are recomputed after
            // parsing. Since the ordering can be arbitrary, sort them when dumping.
            let mut predecessors = self.0.block_preds(block).to_vec();
            predecessors.sort_unstable();
            writeln!(f, "; predecessors: {}", display_iter(predecessors, ","))?;

            for inst in self.0.block_insts(block).iter() {
                // Base opcode. For terminators this also encodes the list of block
                // successors.
                write!(f, "    {inst}: ")?;
                match self.0.terminator_kind(inst) {
                    None => write!(f, "inst")?,
                    Some(TerminatorKind::Ret) => write!(f, "ret")?,
                    Some(TerminatorKind::Jump) => write!(
                        f,
                        "jump {}({})",
                        self.0.block_succs(block)[0],
                        display_iter(self.0.jump_blockparams(block), ",")
                    )?,
                    Some(TerminatorKind::Branch) => write!(
                        f,
                        "branch({})",
                        display_iter(self.0.block_succs(block), ",")
                    )?,
                };

                // Attributes
                if self.0.can_eliminate_dead_inst(inst) {
                    write!(f, " pure")?;
                }

                // Operands and clobbers
                for operand in self.0.inst_operands(inst) {
                    f.write_str(" ")?;
                    let constraint = operand.constraint();
                    match operand.kind() {
                        OperandKind::Def(value) => write!(f, "Def({value}):{constraint}")?,
                        OperandKind::Use(value) => write!(f, "Use({value}):{constraint}")?,
                        OperandKind::EarlyDef(value) => {
                            write!(f, "EarlyDef({value}):{constraint}")?;
                        }
                        OperandKind::DefGroup(group) => write!(
                            f,
                            "Def({}):{constraint}",
                            display_iter(self.0.value_group_members(group), ",")
                        )?,
                        OperandKind::UseGroup(group) => write!(
                            f,
                            "Use({}):{constraint}",
                            display_iter(self.0.value_group_members(group), ",")
                        )?,
                        OperandKind::EarlyDefGroup(group) => write!(
                            f,
                            "EarlyDef({}):{constraint}",
                            display_iter(self.0.value_group_members(group), ",")
                        )?,
                        OperandKind::NonAllocatable => {
                            write!(f, "NonAllocatable:{constraint}")?;
                        }
                    }
                }
                for unit in self.0.inst_clobbers(inst) {
                    write!(f, " Clobber:{unit}")?;
                }

                writeln!(f)?;
            }
        }

        Ok(())
    }
}

/// Wrapper around a type implementing [`RegInfo`] that provides a [`Display`]
/// implementation which dumps the register description in a format that is both
/// human-readable and machine-parseable.
///
/// The returned string can be re-parsed into a register description by using
/// [`GenericRegInfo::parse`] (requires the `parse` cargo feature).
///
/// [`GenericRegInfo::parse`]: crate::debug_utils::GenericRegInfo::parse
/// [`Display`]: core::fmt::Display
pub struct DisplayRegInfo<'a, R: RegInfo>(pub &'a R);

impl<R: RegInfo> fmt::Debug for DisplayRegInfo<'_, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// Helper function to compute superclasses for a register class.
fn superclasses(class: RegClass, reginfo: &impl RegInfo) -> RegClassSet {
    // Collect all transitive superclass: all classes that have this class as
    // a subclass.
    let mut superclasses = RegClassSet::new();
    for superclass in reginfo.classes() {
        // Exclude the current class.
        if superclass == class {
            break;
        }

        if reginfo.sub_classes(superclass).contains(class) {
            superclasses.insert(superclass);
        }
    }

    // Only keep direct superclasses: if a superclass includes a different
    // superclass then it is redundant.
    let mut direct_superclasses = superclasses;
    for superclass in superclasses {
        if (reginfo.sub_classes(superclass) & superclasses).count() > 1 {
            direct_superclasses.remove(superclass);
        }
    }

    direct_superclasses
}

impl<R: RegInfo> fmt::Display for DisplayRegInfo<'_, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Register declarations
        for reg in self.0.regs() {
            let kind = if self.0.is_memory(reg) {
                "stack"
            } else {
                "reg"
            };
            write!(f, "{reg} = {kind}")?;
            if self.0.bank_for_reg(reg).is_none() {
                writeln!(f, " nonallocatable")?;
            } else {
                writeln!(f, " {}", display_iter(self.0.reg_units(reg), ""))?;
            }
        }

        // Register group declarations
        if self.0.num_reg_groups() != 0 {
            writeln!(f)?;
        }
        for group in self.0.reg_groups() {
            writeln!(
                f,
                "{group} = {}",
                display_iter(self.0.reg_group_members(group), "")
            )?;
        }

        // Register banks
        for bank in self.0.banks() {
            writeln!(f)?;
            writeln!(f, "{bank} {{")?;
            writeln!(f, "    top_level_class = {}", self.0.top_level_class(bank))?;
            writeln!(
                f,
                "    stack_to_stack_class = {}",
                self.0.stack_to_stack_class(bank)
            )?;
            writeln!(f, "    spillslot_size = {}", self.0.spillslot_size(bank))?;

            // Register classes in the bank
            for class in self.0.classes() {
                if self.0.bank_for_class(class) != bank {
                    continue;
                }

                writeln!(f)?;
                let superclasses = superclasses(class, self.0);
                if superclasses.is_empty() {
                    writeln!(f, "    {class} {{")?;
                } else {
                    writeln!(
                        f,
                        "    {class}: {} {{",
                        display_iter(superclasses.into_iter(), "")
                    )?;
                }

                let group_size = self.0.class_group_size(class);
                if self.0.class_includes_spillslots(class) {
                    writeln!(f, "        allows_spillslots")?;
                }
                writeln!(f, "        spill_cost = {}", self.0.class_spill_cost(class))?;
                if group_size != 1 {
                    writeln!(f, "        group_size = {group_size}")?;
                }

                // Write the class members.
                if group_size == 1 {
                    writeln!(
                        f,
                        "        members = {}",
                        display_iter(self.0.class_members(class).into_iter(), "")
                    )?;
                } else {
                    writeln!(
                        f,
                        "        members = {}",
                        display_iter(self.0.class_group_members(class).into_iter(), "")
                    )?;
                }

                // Write the allocation order.
                let mut write_order = |name, set| {
                    if group_size == 1 {
                        let order = self.0.allocation_order(class, set);
                        if order.is_empty() {
                            return Ok(());
                        }
                        writeln!(f, "        {name} = {}", display_iter(order.iter(), ""))
                    } else {
                        let order = self.0.group_allocation_order(class, set);
                        if order.is_empty() {
                            return Ok(());
                        }
                        writeln!(f, "        {name} = {}", display_iter(order.iter(), ""))
                    }
                };
                write_order("preferred_regs", AllocationOrderSet::Preferred)?;
                write_order("non_preferred_regs", AllocationOrderSet::NonPreferred)?;

                writeln!(f, "    }}")?;
            }

            writeln!(f, "}}")?;
        }

        Ok(())
    }
}

/// Internal helper type to format an output instruction.
pub(crate) struct DisplayOutputInst<'a, F, R> {
    pub(crate) inst: OutputInst<'a>,
    pub(crate) block: Block,
    pub(crate) output: &'a Output<'a, F, R>,
}

impl<F: Function, R: RegInfo> fmt::Display for DisplayOutputInst<'_, F, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let func = self.output.function();
        let reginfo = self.output.reginfo();
        match self.inst {
            OutputInst::Inst {
                inst,
                operand_allocs,
            } => {
                // Base opcode. For terminators this also encodes the list of block
                // successors.
                write!(f, "{inst}: ")?;
                match func.terminator_kind(inst) {
                    None => write!(f, "inst")?,
                    Some(TerminatorKind::Ret) => write!(f, "ret")?,
                    Some(TerminatorKind::Jump) => write!(
                        f,
                        "jump {}({})",
                        func.block_succs(self.block)[0],
                        display_iter(func.jump_blockparams(self.block), ",")
                    )?,
                    Some(TerminatorKind::Branch) => write!(
                        f,
                        "branch({})",
                        display_iter(func.block_succs(self.block), ",")
                    )?,
                };

                // Attributes
                if func.can_eliminate_dead_inst(inst) {
                    write!(f, " pure")?;
                }

                // Operands and clobbers
                for (&operand, &alloc) in func.inst_operands(inst).iter().zip(operand_allocs) {
                    f.write_str(" ")?;
                    let dump_group = |f: &mut fmt::Formatter<'_>,
                                      kind,
                                      value_group,
                                      constraint,
                                      alloc: Allocation| {
                        let AllocationKind::PhysReg(reg) = alloc.kind() else {
                            unreachable!()
                        };
                        let class = match constraint {
                            OperandConstraint::Class(class) => class,
                            OperandConstraint::Fixed(_) => unreachable!(),
                            OperandConstraint::Reuse(target) => {
                                let OperandConstraint::Class(class) =
                                    func.inst_operands(inst)[target].constraint()
                                else {
                                    unreachable!()
                                };
                                class
                            }
                        };
                        if let Some(reg_group) = reginfo.group_for_reg(reg, 0, class) {
                            write!(
                                f,
                                "{kind}({}):({})",
                                display_iter(func.value_group_members(value_group), ","),
                                display_iter(reginfo.reg_group_members(reg_group), ","),
                            )?;
                        } else {
                            write!(
                                f,
                                "{kind}({}):<invalid reg group>",
                                display_iter(func.value_group_members(value_group), ",")
                            )?;
                        };
                        Ok(())
                    };
                    match operand.kind() {
                        OperandKind::Def(value) => write!(f, "Def({value}):{alloc}")?,
                        OperandKind::Use(value) => write!(f, "Use({value}):{alloc}")?,
                        OperandKind::EarlyDef(value) => write!(f, "EarlyDef({value}):{alloc}")?,
                        OperandKind::DefGroup(group) => {
                            dump_group(f, "Def", group, operand.constraint(), alloc)?;
                        }
                        OperandKind::UseGroup(group) => {
                            dump_group(f, "Use", group, operand.constraint(), alloc)?;
                        }
                        OperandKind::EarlyDefGroup(group) => {
                            dump_group(f, "EarlyDef", group, operand.constraint(), alloc)?;
                        }
                        OperandKind::NonAllocatable => write!(f, "NonAllocatable:{alloc}")?,
                    }
                }
                for unit in func.inst_clobbers(inst) {
                    write!(f, " Clobber:{unit}")?;
                }
            }
            OutputInst::Rematerialize { value, to } => {
                write!(f, "remat {to} <- {value}")?;
            }
            OutputInst::Move { from, to, value } => {
                if let Some(value) = value {
                    write!(f, "move {to} <- {from} ({value})")?;
                } else {
                    let name = match to.kind() {
                        AllocationKind::PhysReg(_) => "restore",
                        AllocationKind::SpillSlot(_) => "evict",
                    };
                    write!(f, "{name} {to} <- {from} ")?;
                }
            }
        }
        Ok(())
    }
}

impl<F: Function, R: RegInfo> fmt::Display for Output<'_, F, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let func = self.function();

        // Spill slots
        let spillslot_area_size = self.stack_layout().spillslot_area_size();
        writeln!(f, "spillslot_area_size = {spillslot_area_size}")?;
        for slot in self.stack_layout().spillslots() {
            let offset = self.stack_layout().spillslot_offset(slot);
            let size = self.stack_layout().spillslot_size(slot);
            writeln!(f, "{slot}: offset={offset} size={size}")?;
        }

        // Blocks
        for block in func.blocks() {
            writeln!(f)?;
            writeln!(
                f,
                "{block}({}):",
                display_iter(func.block_params(block), ","),
            )?;

            // Predecessors are emitted as a comment. Since the ordering can be
            // arbitrary, sort them when dumping.
            let mut predecessors = func.block_preds(block).to_vec();
            predecessors.sort_unstable();
            writeln!(f, "; predecessors: {}", display_iter(predecessors, ","))?;

            for inst in self.output_insts(block) {
                writeln!(
                    f,
                    "    {}",
                    DisplayOutputInst {
                        inst,
                        block,
                        output: self
                    }
                )?;
            }
        }

        // Value locations
        writeln!(f)?;
        for (value, range, alloc) in self.value_locations() {
            writeln!(f, "{value} in {range} => {alloc}")?;
        }

        Ok(())
    }
}
