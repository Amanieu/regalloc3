//! Input function validation.

use alloc::vec;
use alloc::vec::Vec;
use core::{fmt, slice};

use anyhow::{bail, ensure, Result};
use cranelift_entity::{EntityRef as _, EntitySet, SecondaryMap};

use crate::debug_utils::dominator_tree::DominatorTree;
use crate::debug_utils::postorder::PostOrder;
use crate::function::{
    Block, Function, Inst, InstRange, Operand, OperandConstraint, OperandKind, Value, ValueGroup,
    MAX_BLOCKS, MAX_BLOCK_PARAMS, MAX_INSTS, MAX_INST_OPERANDS, MAX_VALUES,
};
use crate::reginfo::{AllocationOrderSet, PhysReg, RegInfo, RegUnitSet};

/// Checks `func` to ensure it satisfies all of the pre-conditions required by
/// the register allocator.
///
/// As long as this validation succeeds, the register allocator is guaranteed to
/// produce a valid allocation. The only exception is if operands are
/// overconstrained: the register allocator may still fail in that case even if
/// validation passes.
///
/// This assumes that `reginfo` has already been validated by
/// [`validate_reginfo`].
///
/// [`validate_reginfo`]: super::validate_reginfo()
pub fn validate_function(func: &impl Function, reginfo: &impl RegInfo) -> Result<()> {
    let mut ctx = Context {
        func,
        reginfo,
        value_defs: SecondaryMap::new(),
        early_fixed: RegUnitSet::new(),
        late_fixed: RegUnitSet::new(),
        used_value_groups: EntitySet::new(),
        reuse_targets: vec![],
        domtree: DominatorTree::new(),
    };
    ctx.check_function()?;
    Ok(())
}

/// An entity reference.
///
/// This is used by [`ValidationError::InvalidEntity`] to report entity
/// references with an invalid index.
#[derive(Debug, Clone, Copy)]
enum Entity {
    Value(Value),
    ValueGroup(ValueGroup),
    Inst(Inst),
}

impl fmt::Display for Entity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Entity::Value(x) => x.fmt(f),
            Entity::ValueGroup(x) => x.fmt(f),
            Entity::Inst(x) => x.fmt(f),
        }
    }
}

/// Point at which a `Value` is defined.
#[derive(Copy, Clone)]
enum ValueDef {
    Blockparam(Block),
    Inst(Block, Inst),
}

/// Wrapper around either a `Value` or a `ValueGroup`.
#[derive(Copy, Clone)]
enum ValueOrGroup {
    Value(Value),
    Group(ValueGroup),
}

impl ValueOrGroup {
    /// Returns the members of the `ValueGroup`, or the single `Value`.
    fn members<'a>(&'a self, func: &'a impl Function) -> &'a [Value] {
        match *self {
            ValueOrGroup::Value(ref value) => slice::from_ref(value),
            ValueOrGroup::Group(group) => func.value_group_members(group),
        }
    }
}

/// State used for validation.
struct Context<'a, F, R> {
    func: &'a F,
    reginfo: &'a R,
    value_defs: SecondaryMap<Value, Option<ValueDef>>,
    early_fixed: RegUnitSet,
    late_fixed: RegUnitSet,
    used_value_groups: EntitySet<ValueGroup>,
    reuse_targets: Vec<usize>,
    domtree: DominatorTree,
}

impl<F: Function, R: RegInfo> Context<'_, F, R> {
    /// Check that an entity refers to a valid object.
    fn check_entity(&mut self, entity: Entity) -> Result<()> {
        let (index, len) = match entity {
            Entity::Value(x) => (x.index(), self.func.num_values()),
            Entity::ValueGroup(x) => {
                ensure!(
                    self.used_value_groups.insert(x),
                    "{x} cannot be used multiple times in a function"
                );
                (x.index(), self.func.num_value_groups())
            }
            Entity::Inst(x) => (x.index(), self.func.num_insts()),
        };
        ensure!(index < len, "{entity}: Invalid entity reference");
        Ok(())
    }

    /// Checks a range of instructions.
    fn check_inst_range(&mut self, range: InstRange) -> Result<()> {
        self.check_entity(Entity::Inst(range.from))?;
        self.check_entity(Entity::Inst(range.last()))?;
        ensure!(range.to >= range.from, "{range}: Invalid instruction range");
        Ok(())
    }

    /// Check the limits on the number of entities.
    fn check_limits(&self) -> Result<()> {
        ensure!(
            self.func.num_values() <= MAX_VALUES,
            "Too many values: {} (max: {MAX_VALUES})",
            self.func.num_values(),
        );
        ensure!(
            self.func.num_blocks() <= MAX_BLOCKS,
            "Too many blocks: {} (max: {MAX_BLOCKS})",
            self.func.num_blocks(),
        );
        ensure!(
            self.func.num_insts() <= MAX_INSTS,
            "Too many instructions: {} (max: {MAX_INSTS})",
            self.func.num_insts(),
        );
        Ok(())
    }

    /// Record the definition of a value and check for duplicate definitions.
    fn check_value_def(&mut self, value: Value, def: ValueDef) -> Result<()> {
        match self.value_defs[value] {
            Some(_) => bail!("{value} defined multiple times"),
            ref mut opt @ None => *opt = Some(def),
        }
        Ok(())
    }

    /// Check for multiple conflicting uses of a fixed register in a single
    /// instruction.
    fn check_fixed(&mut self, inst: Inst, reg: PhysReg, early: bool) -> Result<()> {
        let set = if early {
            &mut self.early_fixed
        } else {
            &mut self.late_fixed
        };
        for &unit in self.reginfo.reg_units(reg) {
            ensure!(!set.contains(unit), "{inst}: Conflicting uses of {reg}");
            set.insert(unit);
        }
        Ok(())
    }

    /// Check an operand with a constraint (not `NonAllocatable`).
    fn check_constraint(
        &mut self,
        inst: Inst,
        operands: &[Operand],
        operand: Operand,
        value_or_group: ValueOrGroup,
    ) -> Result<()> {
        match operand.constraint() {
            OperandConstraint::Class(class) => {
                let bank = self.reginfo.bank_for_class(class);
                let group_size = self.reginfo.class_group_size(class);
                match value_or_group {
                    ValueOrGroup::Value(value) => {
                        ensure!(
                            group_size == 1,
                            "{inst} {operand}: Value used with group register class"
                        );
                        let value_bank = self.func.value_bank(value);
                        ensure!(
                            bank == value_bank,
                            "{inst} {operand}: {value} used with different register banks: {bank} \
                             vs {value_bank}"
                        );
                    }
                    ValueOrGroup::Group(group) => {
                        let members = self.func.value_group_members(group);
                        ensure!(
                            group_size == members.len(),
                            "{inst} {operand}: group size mismatch {group_size} vs {}",
                            members.len()
                        );
                        for &value in members {
                            let value_bank = self.func.value_bank(value);
                            ensure!(
                                bank == value_bank,
                                "{inst} {operand}: {value} used with different register banks: \
                                 {bank} vs {value_bank}"
                            );
                        }
                    }
                }
            }
            OperandConstraint::Fixed(reg) => {
                match value_or_group {
                    ValueOrGroup::Value(value) => {
                        let Some(bank) = self.reginfo.bank_for_reg(reg) else {
                            bail!("{inst} {operand}: Use of non-allocatable register");
                        };
                        let value_bank = self.func.value_bank(value);
                        ensure!(
                            bank == value_bank,
                            "{inst} {operand}: {value} used with different register banks: {bank} \
                             vs {value_bank}"
                        );
                    }
                    ValueOrGroup::Group(_) => {
                        bail!(
                            "{inst} {operand}: Fixed constraint cannot be used with register \
                             groups"
                        );
                    }
                }

                // Check for conflicting fixed-register constraints.
                match operand.kind() {
                    OperandKind::Def(_) => {
                        self.check_fixed(inst, reg, false)?;
                    }
                    OperandKind::Use(_) => {
                        self.check_fixed(inst, reg, true)?;
                    }
                    OperandKind::EarlyDef(_) => {
                        self.check_fixed(inst, reg, true)?;
                        self.check_fixed(inst, reg, false)?;
                    }
                    OperandKind::DefGroup(_)
                    | OperandKind::UseGroup(_)
                    | OperandKind::EarlyDefGroup(_)
                    | OperandKind::NonAllocatable => unreachable!(),
                }
            }
            OperandConstraint::Reuse(index) => {
                match operand.kind() {
                    OperandKind::Def(_)
                    | OperandKind::EarlyDef(_)
                    | OperandKind::DefGroup(_)
                    | OperandKind::EarlyDefGroup(_) => {}
                    OperandKind::Use(_)
                    | OperandKind::UseGroup(_)
                    | OperandKind::NonAllocatable => {
                        bail!("{inst} {operand}: Reuse operand must be a Def or EarlyDef")
                    }
                }
                let Some(&target_operand) = operands.get(index) else {
                    bail!("{inst} {operand}: Invalid index for reuse operand");
                };
                ensure!(
                    !self.reuse_targets.contains(&index),
                    "{inst} {operand}: multiple reuse of same target operand {index}"
                );
                self.reuse_targets.push(index);
                let target_value_or_group = match target_operand.kind() {
                    OperandKind::Use(target_value) => ValueOrGroup::Value(target_value),
                    OperandKind::UseGroup(target_group) => ValueOrGroup::Group(target_group),
                    OperandKind::Def(_)
                    | OperandKind::EarlyDef(_)
                    | OperandKind::DefGroup(_)
                    | OperandKind::EarlyDefGroup(_)
                    | OperandKind::NonAllocatable => {
                        bail!(
                            "{inst} {operand} -> {target_operand}: Reuse operand target must be a \
                             Use"
                        );
                    }
                };
                ensure!(
                    matches!(target_operand.constraint(), OperandConstraint::Class(_)),
                    "{inst} {operand} -> {target_operand}: Reuse operand target must have a Class \
                     constraint"
                );

                // Ensure both source and target have the same group width.
                match (value_or_group, target_value_or_group) {
                    (ValueOrGroup::Value(_), ValueOrGroup::Value(_)) => {}
                    (ValueOrGroup::Group(group), ValueOrGroup::Group(target_group))
                        if self.func.value_group_members(group).len()
                            == self.func.value_group_members(target_group).len() => {}
                    _ => {
                        bail!(
                            "{inst} {operand} -> {target_operand}: Tied operands must either both \
                             be values or both be groups of the same size"
                        );
                    }
                }

                // Ensure both values are from the same register bank.
                for (&source_value, &target_value) in value_or_group
                    .members(self.func)
                    .iter()
                    .zip(target_value_or_group.members(self.func))
                {
                    let target_bank = self.func.value_bank(target_value);
                    let source_bank = self.func.value_bank(source_value);
                    ensure!(
                        source_bank == target_bank,
                        "{inst}: Tied operand with different register banks: {operand} \
                         ({source_bank}) vs {target_operand} ({target_bank})"
                    );
                }
            }
        }
        Ok(())
    }

    /// Check an instruction.
    fn check_inst(&mut self, block: Block, inst: Inst) -> Result<()> {
        // These are temporary for the scope of this instruction.
        self.early_fixed.clear();
        self.late_fixed.clear();
        self.reuse_targets.clear();

        let operands = self.func.inst_operands(inst);
        ensure!(
            operands.len() <= MAX_INST_OPERANDS,
            "{inst}: Too many operands: {} (max: {MAX_INST_OPERANDS})",
            operands.len(),
        );
        for &op in operands {
            match op.kind() {
                OperandKind::Def(value) | OperandKind::EarlyDef(value) => {
                    self.check_entity(Entity::Value(value))?;
                    self.check_value_def(value, ValueDef::Inst(block, inst))?;
                    self.check_constraint(inst, operands, op, ValueOrGroup::Value(value))?;
                }
                OperandKind::Use(value) => {
                    self.check_entity(Entity::Value(value))?;
                    self.check_constraint(inst, operands, op, ValueOrGroup::Value(value))?;
                }
                OperandKind::DefGroup(group) | OperandKind::EarlyDefGroup(group) => {
                    self.check_entity(Entity::ValueGroup(group))?;
                    for &value in self.func.value_group_members(group) {
                        self.check_entity(Entity::Value(value))?;
                        self.check_value_def(value, ValueDef::Inst(block, inst))?;
                    }
                    self.check_constraint(inst, operands, op, ValueOrGroup::Group(group))?;
                }
                OperandKind::UseGroup(group) => {
                    self.check_entity(Entity::ValueGroup(group))?;
                    for &value in self.func.value_group_members(group) {
                        self.check_entity(Entity::Value(value))?;
                    }
                    self.check_constraint(inst, operands, op, ValueOrGroup::Group(group))?;
                }
                OperandKind::NonAllocatable => match op.constraint() {
                    OperandConstraint::Fixed(reg) => {
                        ensure!(
                            self.reginfo.bank_for_reg(reg).is_none(),
                            "{inst} {op}: NonAllocatable register must be outside a bank"
                        );
                    }
                    OperandConstraint::Class(_) | OperandConstraint::Reuse(_) => {
                        bail!("{inst} {op}: NonAllocatable operand must have a Fixed constraint")
                    }
                },
            }
        }
        Ok(())
    }

    /// Check a basic block.
    fn check_block(&mut self, block: Block) -> Result<()> {
        let insts = self.func.block_insts(block);
        self.check_inst_range(insts)?;

        // Block frequency must be positive. This also excludes zero and NaN.
        ensure!(
            self.func.block_frequency(block) > 0.0,
            "{block}: Frequency must be positive and non-zero"
        );

        // Instruction indicies must be ordered by block and with no gaps.
        if block.index() != 0 {
            let prev_block = Block::new(block.index() - 1);
            let prev_insts = self.func.block_insts(prev_block);
            ensure!(
                insts.from == prev_insts.to,
                "{block}: Instructions are not ordered by block"
            );
        } else {
            ensure!(
                insts.from.index() == 0,
                "{block}: Instructions are not ordered by block"
            );
        }
        if block.index() != self.func.num_blocks() - 1 {
            let next_block = Block::new(block.index() + 1);
            let next_insts = self.func.block_insts(next_block);
            ensure!(
                next_insts.from == insts.to,
                "{block}: Instructions are not ordered by block"
            );
        } else {
            ensure!(
                insts.to.index() == self.func.num_insts(),
                "{block}: Instructions are not ordered by block"
            );
        }

        // Check consistency of successors & predecessors.
        for &pred in self.func.block_preds(block) {
            ensure!(
                self.func.block_succs(pred).contains(&block),
                "Inconsistent predecessors and successors between {pred} and {block}"
            );
        }
        for &succ in self.func.block_succs(block) {
            ensure!(
                self.func.block_preds(succ).contains(&block),
                "Inconsistent predecessors and successors between {block} and {succ}"
            );
        }

        // Check for crtical edges. If we have more than one predecessors, those
        // must only have one successor (this block).
        if self.func.block_preds(block).len() > 1 {
            for &pred in self.func.block_preds(block) {
                ensure!(
                    self.func.block_succs(pred).len() == 1,
                    "Critical edge beween {pred} and {block}"
                );
            }
        }

        // Check incoming block parameters.
        if !self.func.block_params(block).is_empty() {
            ensure!(
                self.func.block_preds(block).len() > 1,
                "{block}: Block parameters are only allowed on blocks with multiple predecessors"
            );
        }
        ensure!(
            self.func.block_params(block).len() < MAX_BLOCK_PARAMS,
            "{block} has too many block parameters (max {MAX_BLOCK_PARAMS})"
        );
        for &param in self.func.block_params(block) {
            self.check_entity(Entity::Value(param))?;
            self.check_value_def(param, ValueDef::Blockparam(block))?;
        }

        // Check outgoing block parameters.
        let mut allow_terminator_operands = true;
        if !self.func.jump_blockparams(block).is_empty() {
            let &[succ] = self.func.block_succs(block) else {
                bail!("{block}: Branch blockparams can only be used with a single successor");
            };
            ensure!(
                self.func.block_params(succ).len() == self.func.jump_blockparams(block).len(),
                "Blockparam count mismatch between {block} and {succ}"
            );
            for (&blockparam_out, &blockparam_in) in self
                .func
                .jump_blockparams(block)
                .iter()
                .zip(self.func.block_params(succ).iter())
            {
                self.check_entity(Entity::Value(blockparam_out))?;
                let bank_out = self.func.value_bank(blockparam_out);
                let bank_in = self.func.value_bank(blockparam_in);
                ensure!(
                    bank_out == bank_in,
                    "Blockparams with register banks: {blockparam_out} ({bank_out}) vs \
                     {blockparam_in} ({bank_in})"
                );
            }

            // If the target block has more than one predecessor, the terminator
            // instruction cannot have operands.
            if self.func.block_preds(succ).len() > 1 {
                allow_terminator_operands = false;
            }
        }

        // Check instructions.
        let mut terminator = None;
        for inst in insts.iter() {
            if let Some(inst) = terminator {
                bail!("{inst}: Terminator in middle of block");
            }
            if self.func.inst_is_terminator(inst) {
                ensure!(
                    !self.func.can_eliminate_dead_inst(inst),
                    "{inst}: Terminator cannot be marked as a pure instruction"
                );
                if !allow_terminator_operands {
                    ensure!(
                        self.func.inst_operands(inst).is_empty(),
                        "{inst}: Terminator cannot have operands when the successor block has \
                         multiple predecessors"
                    );
                    ensure!(
                        self.func.inst_clobbers(inst).is_empty(),
                        "{inst}: Terminator cannot have clobbers when the successor block has \
                         multiple predecessors"
                    );
                }
                if self.func.block_succs(block).is_empty() {
                    for op in self.func.inst_operands(inst) {
                        match op.kind() {
                            OperandKind::Def(_) | OperandKind::DefGroup(_) => bail!(
                                "{inst}: Terminator with no successors cannot have Def operands, \
                                 only Use/EarlyDef"
                            ),
                            OperandKind::Use(_)
                            | OperandKind::EarlyDef(_)
                            | OperandKind::UseGroup(_)
                            | OperandKind::EarlyDefGroup(_)
                            | OperandKind::NonAllocatable => {}
                        }
                    }
                    ensure!(
                        self.func.inst_clobbers(inst).is_empty(),
                        "{inst}: Terminator with no successors cannot have clobbers"
                    );
                }
                terminator = Some(inst);
            }
            self.check_inst(block, inst)?;
        }
        ensure!(terminator.is_some(), "{block}: Missing terminator");

        Ok(())
    }

    /// Check that defs dominate uses.
    ///
    /// At this point the dominator tree should be valid and all values should
    /// have a `ValueDef`.
    fn check_ssa_dominance(&self, block: Block) -> Result<()> {
        // Check that the block's immediate dominator is correct.
        ensure!(
            self.func.block_immediate_dominator(block) == self.domtree.immediate_dominator(block),
            "{block} has incorrect immediate dominator: got {:?}, expected {:?}",
            self.func.block_immediate_dominator(block),
            self.domtree.immediate_dominator(block)
        );

        if let Some(idom) = self.domtree.immediate_dominator(block) {
            // This also ensures that all defs come before uses in the linear
            // instruction ordering.
            ensure!(
                idom.index() < block.index(),
                "{idom} dominates {block} but has higher index"
            );
        }

        for inst in self.func.block_insts(block).iter() {
            for op in self.func.inst_operands(inst) {
                if let OperandKind::Use(value) = op.kind() {
                    let Some(def) = self.value_defs[value] else {
                        bail!("{value} used without being defined");
                    };
                    let dominates = match def {
                        ValueDef::Blockparam(def_block) => self.domtree.dominates(def_block, block),
                        ValueDef::Inst(def_block, def_inst) => {
                            if def_block == block {
                                // Can't use a value defined in the same instruction.
                                def_inst.index() < inst.index()
                            } else {
                                self.domtree.dominates(def_block, block)
                            }
                        }
                    };
                    ensure!(
                        dominates,
                        "{value} definition does not dominate use at {inst}"
                    );
                }
            }
        }
        for &value in self.func.jump_blockparams(block) {
            let Some(def) = self.value_defs[value] else {
                bail!("{value} used without being defined");
            };
            let dominates = match def {
                ValueDef::Blockparam(def_block) | ValueDef::Inst(def_block, _) => {
                    self.domtree.dominates(def_block, block)
                }
            };
            ensure!(
                dominates,
                "{value} definition does not dominate use as outgoing blockparam in {block}"
            );
        }
        Ok(())
    }

    /// Main entry point for `Function` validation.
    fn check_function(&mut self) -> Result<()> {
        self.check_limits()?;

        // Check blocks and instructions. This also records a `ValueDef` for
        // each defined value.
        for block in self.func.blocks() {
            self.check_block(block)?;
        }

        // Check the entry block.
        ensure!(
            self.func.block_preds(Block::ENTRY_BLOCK).is_empty(),
            "{}: Entry block cannot have predecessors",
            Block::ENTRY_BLOCK
        );
        ensure!(
            self.func.block_params(Block::ENTRY_BLOCK).is_empty(),
            "{}: Entry block cannot have block parameters",
            Block::ENTRY_BLOCK
        );

        // Check that all blocks are reachable.
        let postorder = PostOrder::for_function(self.func);
        if postorder.cfg_postorder().len() != self.func.num_blocks() {
            for block in self.func.blocks() {
                ensure!(
                    postorder.is_reachable(block),
                    "{block} is not reachable from the entry block"
                );
            }

            // There must be at least one unreachable block, so we can't get here.
            unreachable!();
        }

        // Check that defs dominate uses, as required by SSA.
        self.domtree.compute(self.func, &postorder);
        for block in self.func.blocks() {
            self.check_ssa_dominance(block)?;
        }

        // Check values.
        for value in self.func.values() {
            // Used values without a definition are caught by the prior
            // dominance check.
            ensure!(self.value_defs[value].is_some(), "{value} is unused");

            if let Some((_cost, class)) = self.func.can_rematerialize(value) {
                ensure!(
                    self.reginfo.class_group_size(class) == 1,
                    "{value} cannot be rematerialized with group register class {class}"
                );
                let bank = self.reginfo.bank_for_class(class);
                let value_bank = self.func.value_bank(value);
                ensure!(
                    bank == value_bank,
                    "{value} cannot be rematerialized with different register banks: {bank} vs \
                     {value_bank}"
                );

                ensure!(
                    AllocationOrderSet::each()
                        .any(|set| !self.reginfo.allocation_order(class, set).is_empty()),
                    "{value} cannot be rematerialized into {class} which has an empty allocation \
                     order"
                );
                for reg in &self.reginfo.class_members(class) {
                    ensure!(
                        self.reginfo.class_includes_spillslots(class)
                            || !self.reginfo.is_memory(reg.as_single()),
                        "{value} cannot be rematerialized into {class} which has in-memory \
                         members but doesn't include spill slots"
                    );
                }
            }
        }

        Ok(())
    }
}
