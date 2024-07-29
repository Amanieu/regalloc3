use alloc::vec;
use alloc::vec::Vec;
use core::ops::RangeInclusive;

use arbitrary::{Result, Unstructured};
use cranelift_entity::{EntityRef, PrimaryMap, SecondaryMap};

use super::{BlockData, GenericFunction, InstData};
use crate::debug_utils::generic_function::ValueData;
use crate::function::{
    Block, Inst, InstRange, Operand, OperandConstraint, OperandKind, RematCost, Value,
};
use crate::internal::dominator_tree::DominatorTree;
use crate::internal::postorder::PostOrder;
use crate::reginfo::{
    AllocationOrderSet, PhysReg, RegBank, RegClass, RegInfo, RegOrRegGroup, RegUnit, RegUnitSet,
    MAX_REG_UNITS,
};

/// Configuration options for [`GenericFunction::arbitrary`].
///
/// These are ranges from which a value is arbitrarily chosen when generating a
/// function.
///
/// It's generally fine to just use `Default::default` for this.
#[derive(Debug, Clone)]
pub struct ArbitraryFunctionConfig {
    /// Number of CFG edges. This also implicitly controls the number of blocks
    /// in a function since all blocks must be reachable from the entry block.
    pub cfg_edges: RangeInclusive<usize>,

    /// Number of block parameters for each block that is allowed to have them.
    pub blockparams_per_block: RangeInclusive<usize>,

    /// Number of instructions per block, excluding the terminator instruction.
    pub insts_per_block: RangeInclusive<usize>,

    /// Number of definition operands (`Def` and `EarlyDef`) per instruction.
    ///
    /// Some instructions may exceed this limit due to the way the algorithm
    /// works. This is because all used values need a definition, which may
    /// force extra definitions to be added.
    pub defs_per_inst: RangeInclusive<usize>,

    /// Number of non-definition operands (`Use` and `NonAllocatable`) per
    /// instruction.
    pub uses_per_inst: RangeInclusive<usize>,

    /// Number of clobbers per instruction.
    pub clobbers_per_inst: RangeInclusive<usize>,
}

impl Default for ArbitraryFunctionConfig {
    fn default() -> Self {
        Self {
            cfg_edges: 0..=20,
            blockparams_per_block: 0..=30,
            insts_per_block: 0..=15,
            defs_per_inst: 0..=20,
            uses_per_inst: 0..=20,
            clobbers_per_inst: 0..=20,
        }
    }
}

impl GenericFunction {
    /// Constructs a randomly-generated `GenericFunction`.
    ///
    /// This function is guaranteed to pass validation with the given (valid)
    /// [`RegInfo`] implementation.
    pub fn arbitrary_with_config(
        reginfo: &impl RegInfo,
        u: &mut Unstructured<'_>,
        config: ArbitraryFunctionConfig,
    ) -> Result<Self> {
        let mut builder = FunctionBuilder::new(u, reginfo, config);
        builder.gen_cfg_skeleton()?;

        let postorder = PostOrder::for_function(&builder.func);
        builder.domtree.compute(&builder.func, &postorder);

        builder.add_blockparams()?;

        for block in postorder.cfg_postorder() {
            builder.gen_block_insts(block)?;
        }

        builder.finalize()?;

        Ok(builder.func)
    }
}

struct FunctionBuilder<'a, 'b, R> {
    /// Source of randomness.
    u: &'a mut Unstructured<'b>,

    /// Register description.
    reginfo: &'a R,

    /// Function that is being built.
    func: GenericFunction,

    /// Configuration options
    config: ArbitraryFunctionConfig,

    /// Dominator tree of the function.
    domtree: DominatorTree,

    /// Instructions for each basic block, in reverse order.
    ///
    /// These are later copied to the `GenericFunction` in proper block order
    /// once instructions for all blocks have been generated.
    block_insts: SecondaryMap<Block, Vec<InstData>>,

    /// List of register classes per register bank.
    class_per_bank: SecondaryMap<RegBank, Vec<RegClass>>,

    /// List of register classes that are suitable for rematerialization.
    remat_class_per_bank: SecondaryMap<RegBank, Vec<RegClass>>,

    /// List of registers per register bank.
    reg_per_bank: SecondaryMap<RegBank, Vec<PhysReg>>,

    /// List of non-allocatable registers.
    non_allocatable_regs: Vec<PhysReg>,

    /// Values that have been used but not yet defined, for each block they
    /// should be defined in. This does not include incoming blockparams for
    /// this block.
    ///
    /// As values are defined, they are removed from this list.
    defs_by_blocks: SecondaryMap<Block, Vec<Value>>,

    /// Scratch space used to collect candidates for using an existing value.
    use_candidates: Vec<Value>,

    /// Registers currently allocated with a fixed constraint in the current
    /// instruction.
    early_fixed: RegUnitSet,
    late_fixed: RegUnitSet,

    /// Def operands which are suitable for `OperandConstraint::Reuse`.
    reuse_operands: Vec<usize>,
}

impl<'a, 'b, R: RegInfo> FunctionBuilder<'a, 'b, R> {
    /// Initializes the `FunctionBuilder`.
    fn new(u: &'a mut Unstructured<'b>, reginfo: &'a R, config: ArbitraryFunctionConfig) -> Self {
        let func = GenericFunction {
            blocks: PrimaryMap::new(),
            insts: PrimaryMap::new(),
            values: PrimaryMap::new(),
            value_groups: PrimaryMap::new(),
            safepoints: vec![],
            reftype_values: vec![],
        };

        let mut class_per_bank: SecondaryMap<RegBank, Vec<RegClass>> = SecondaryMap::new();
        let mut remat_class_per_bank: SecondaryMap<RegBank, Vec<RegClass>> = SecondaryMap::new();
        let mut reg_per_bank: SecondaryMap<RegBank, Vec<PhysReg>> = SecondaryMap::new();
        let mut non_allocatable_regs = vec![];
        for class in reginfo.classes() {
            class_per_bank[reginfo.bank_for_class(class)].push(class);
            if reginfo.class_group_size(class) == 1
                && AllocationOrderSet::each()
                    .any(|set| !reginfo.allocation_order(class, set).is_empty())
                && (reginfo.class_includes_spillslots(class)
                    || reginfo
                        .regs()
                        .filter(|&reg| reginfo.class_contains(class, RegOrRegGroup::single(reg)))
                        .all(|reg| !reginfo.is_memory(reg)))
            {
                remat_class_per_bank[reginfo.bank_for_class(class)].push(class);
            }
        }
        for reg in reginfo.regs() {
            if let Some(bank) = reginfo.bank_for_reg(reg) {
                reg_per_bank[bank].push(reg);
            } else {
                non_allocatable_regs.push(reg);
            }
        }

        Self {
            u,
            reginfo,
            func,
            config,
            domtree: DominatorTree::new(),
            block_insts: SecondaryMap::new(),
            class_per_bank,
            remat_class_per_bank,
            reg_per_bank,
            non_allocatable_regs,
            defs_by_blocks: SecondaryMap::new(),
            use_candidates: vec![],
            early_fixed: RegUnitSet::new(),
            late_fixed: RegUnitSet::new(),
            reuse_operands: vec![],
        }
    }

    /// Generates a reasonable block frequency.
    fn block_frequency(&mut self) -> Result<f32> {
        Ok(2.0f32.powi(self.u.int_in_range(-10..=10)?))
    }

    /// Generates a function skeleton with a randomly generated CFG.
    ///
    /// These blocks do not contain any instructions yet.
    fn gen_cfg_skeleton(&mut self) -> Result<()> {
        // Create the entry block.
        let entry_frequency = self.block_frequency()?;
        self.func.blocks.push(BlockData {
            insts: InstRange::new(Inst::new(0), Inst::new(0)),
            preds: vec![],
            succs: vec![],
            block_params_in: vec![],
            block_params_out: vec![],
            frequency: entry_frequency,
        });

        // To avoid critical edges, we need to ensure that blocks with multiple
        // successors only jump to blocks with a single predecessors, and that
        // blocks with multiple predecessors are only jumped to from blocks with
        // a single successor.
        let mut can_add_succ = vec![Block::ENTRY_BLOCK];
        let mut can_add_pred = vec![];

        // Repeatedly add edges to the CFG, either to a new block, or to an
        // existing block.
        for _ in 0..self.u.int_in_range(self.config.cfg_edges.clone())? {
            if can_add_succ.is_empty() {
                break;
            }

            let from = *self.u.choose(&can_add_succ)?;
            let mut to = None;

            // If the chosen block has no successors, try linking it to an
            // existing block that accepts predecessors.
            if !can_add_pred.is_empty()
                && self.func.blocks[from].succs.is_empty()
                && self.u.arbitrary()?
            {
                to = Some(*self.u.choose(&can_add_pred)?);
            }

            if let Some(to) = to {
                // Create an edge to an existing block.
                self.func.blocks[from].succs.push(to);
                self.func.blocks[to].preds.push(from);

                // If the `to` block now has multiple predecessors, prevent
                // adding new successors to them.
                if self.func.blocks[to].preds.len() > 1 {
                    can_add_succ.retain(|b| !self.func.blocks[to].preds.contains(b));
                }
            } else {
                // Create an edge to a new block.
                let frequency = self.block_frequency()?;
                let to = self.func.blocks.push(BlockData {
                    insts: InstRange::new(Inst::new(0), Inst::new(0)),
                    preds: vec![from],
                    succs: vec![],
                    block_params_in: vec![],
                    block_params_out: vec![],
                    frequency,
                });
                self.func.blocks[from].succs.push(to);

                // We can add more predecessors to the new block if it is the
                // only successor of `from`.
                if self.func.blocks[from].succs.len() == 1 {
                    can_add_pred.push(to);
                }
                can_add_succ.push(to);
            }

            // If the `from` block now has multiple successors, prevent
            // adding new predecessors to them.
            if self.func.blocks[from].succs.len() > 1 {
                can_add_pred.retain(|b| !self.func.blocks[from].succs.contains(b));
            }
        }

        Ok(())
    }

    /// Defines a new value with randomized properties.
    fn new_value(&mut self, bank: RegBank) -> Result<Value> {
        let remat = if !self.remat_class_per_bank[bank].is_empty() && self.u.arbitrary()? {
            let cost = if self.u.arbitrary()? {
                RematCost::CheaperThanMove
            } else {
                RematCost::CheaperThanLoad
            };
            let class = *self.u.choose(&self.remat_class_per_bank[bank])?;
            Some((cost, class))
        } else {
            None
        };
        let value = self.func.values.push(ValueData { bank, remat });
        if self.reginfo.reftype_class(bank).is_some() && self.u.arbitrary()? {
            self.func.reftype_values.push(value);
        }
        Ok(value)
    }

    /// Adds incoming block parameters to blocks that allow them.
    fn add_blockparams(&mut self) -> Result<()> {
        // The entry block cannot have blockparams.
        for block in self.func.blocks.keys().skip(1) {
            // Blockparams are only allowed if we have multiple predecessors.
            if self.func.blocks[block].preds.len() <= 1 {
                continue;
            }

            // Define some values for incoming blockparams.
            for _ in 0..self
                .u
                .int_in_range(self.config.blockparams_per_block.clone())?
            {
                let bank = RegBank::new(self.u.choose_index(self.reginfo.num_banks())?);
                let value = self.new_value(bank)?;
                self.func.blocks[block].block_params_in.push(value);
            }
        }

        Ok(())
    }

    /// Generate the contents of a basic block.
    ///
    /// This should be called in CFG post-order so that uses come before
    /// definitions.
    fn gen_block_insts(&mut self, block: Block) -> Result<()> {
        // We generate instructions in reverse, starting with the terminator.
        // This allows us to know exactly what values need to be defined later.
        let num_insts = self.u.int_in_range(self.config.insts_per_block.clone())?;

        // Set up outgoing blockparams if we have a single successor.
        if let [succ] = self.func.blocks[block].succs[..] {
            for idx in 0..self.func.blocks[succ].block_params_in.len() {
                let bank = self.func.values[self.func.blocks[succ].block_params_in[idx]].bank;
                let first_block_for_def = if num_insts == 0 {
                    // If this is the entry block then we don't have an
                    // immediate dominator. We will later be force to emit an
                    // instruction to define values for the block.
                    self.domtree.immediate_dominator(block).unwrap_or(block)
                } else {
                    block
                };
                let value = self.get_value_for_use(bank, first_block_for_def)?;
                self.func.blocks[block].block_params_out.push(value);
            }

            // If the successor has more than one predecessor, our terminator
            // cannot have operands. Create an empty terminator now before
            // adding any instructions.
            if self.func.blocks[succ].preds.len() > 1 {
                self.block_insts[block].push(InstData {
                    operands: vec![],
                    clobbers: vec![],
                    block,
                    is_terminator: true,
                    is_pure: false,
                });
            }
        }

        // We need a terminator instruction if an empty one wasn't created
        // above. This one is allowed to have operands like a normal
        // instruction.
        if self.block_insts[block].is_empty() {
            let mut terminator = self.gen_inst(
                block,
                num_insts == 0,
                self.func.blocks[block].succs.is_empty(),
            )?;
            terminator.is_pure = false;
            terminator.is_terminator = true;
            self.block_insts[block].push(terminator);
        }

        // Generate a sequence of instructions in the block.
        for idx in 0..num_insts {
            // If this is the last instruction we generate (i.e. the first one
            // in the block) then we need to ensure all definitions assigned to
            // this block are actually processed.
            let inst = self.gen_inst(block, idx == num_insts - 1, false)?;
            self.block_insts[block].push(inst);
        }

        // If this is the entry block and no instructions were generated, we
        // may need to force an instruction to exist to define any values that
        // still need to be defined.
        if num_insts == 0 && !self.defs_by_blocks[block].is_empty() {
            let inst = self.gen_inst(block, true, false)?;
            self.block_insts[block].push(inst);
        }

        Ok(())
    }

    /// Checks if the given register can be used as an early/late fixed-register
    /// constraint, and if successful, marks all of the register's sub-units as
    /// in-use for the current instruction.
    fn check_fixed_conflict(&mut self, reg: PhysReg, early: bool, late: bool) -> bool {
        let units = self.reginfo.reg_units(reg);
        if units.iter().any(|&unit| {
            (early && self.early_fixed.contains(unit)) || (late && self.late_fixed.contains(unit))
        }) {
            return false;
        }
        if early {
            self.early_fixed.extend(units.iter().copied());
        }
        if late {
            self.late_fixed.extend(units.iter().copied());
        }
        true
    }

    /// Returns a value for use as an `OperandKind::Use` in the given block.
    ///
    /// Because we process instructions in post-order, uses are processed before
    /// definitions. If necessary, this function will create a new value to be
    /// defined as a later point in the current block or in a dominating block.
    fn get_value_for_use(&mut self, bank: RegBank, first_block_for_def: Block) -> Result<Value> {
        // Try to reuse an existing value.
        if self.u.arbitrary()? {
            self.use_candidates.clear();

            // Walk up the dominator tree to collect all values that we can use.
            // Specifically: any value whose definition dominates us and which
            // comes from a compatible register bank.
            let mut reuse_block = first_block_for_def;
            loop {
                self.use_candidates.extend(
                    self.defs_by_blocks[reuse_block]
                        .iter()
                        .chain(&self.func.blocks[reuse_block].block_params_in)
                        .copied()
                        .filter(|&value| self.func.values[value].bank == bank),
                );

                let Some(idom) = self.domtree.immediate_dominator(reuse_block) else {
                    break;
                };
                reuse_block = idom;
            }

            // If no suitable values exist, just define a new one.
            if !self.use_candidates.is_empty() {
                return self.u.choose(&self.use_candidates).copied();
            }
        }

        // Walk up the dominator tree to find a block in which to define this
        // value.
        let mut def_block = first_block_for_def;
        while let Some(idom) = self.domtree.immediate_dominator(def_block) {
            // This halves the probability every time we go up the tree.
            if self.u.arbitrary()? {
                break;
            }

            def_block = idom;
        }

        // Define a new value and record it to be defined in the chosen block.
        let value = self.new_value(bank)?;
        self.defs_by_blocks[def_block].push(value);
        Ok(value)
    }

    /// Generates an operand which uses a value.
    fn gen_use(&mut self, first_block_for_def: Block, inst: &mut InstData) -> Result<Operand> {
        // Generate a NonAllocatable operand if we have non-allocatable registers.
        if !self.non_allocatable_regs.is_empty() && self.u.arbitrary()? {
            let reg = *self.u.choose(&self.non_allocatable_regs)?;
            return Ok(Operand::fixed_nonallocatable(reg));
        }

        // Try to use a fixed register. We only do this if it doesn't introduce
        // a conflict with an existing fixed-register constraint.
        if self.u.arbitrary()? {
            let bank = RegBank::new(self.u.choose_index(self.reginfo.num_banks())?);
            let reg = *self.u.choose(&self.reg_per_bank[bank])?;
            if self.check_fixed_conflict(reg, true, false) {
                let value = self.get_value_for_use(bank, first_block_for_def)?;
                return Ok(Operand::new(
                    OperandKind::Use(value),
                    OperandConstraint::Fixed(reg),
                ));
            }
        }

        let class = if !self.reuse_operands.is_empty() && self.u.arbitrary()? {
            // Reuse the class of a Def and turn that Def into a Reuse.
            let idx = self.u.choose_index(self.reuse_operands.len())?;
            let def_idx = self.reuse_operands.swap_remove(idx);
            let OperandConstraint::Class(class) = inst.operands[def_idx].constraint() else {
                unreachable!();
            };
            inst.operands[def_idx] = Operand::new(
                inst.operands[def_idx].kind(),
                OperandConstraint::Reuse(inst.operands.len()),
            );
            class
        } else {
            // Pick a register class to use.
            RegClass::new(self.u.choose_index(self.reginfo.num_classes())?)
        };

        let bank = self.reginfo.bank_for_class(class);
        let group_size = self.reginfo.class_group_size(class);
        let kind = if group_size > 1 {
            let mut values = vec![];
            for _ in 0..group_size {
                values.push(self.get_value_for_use(bank, first_block_for_def)?);
            }
            let value_group = self.func.value_groups.push(values);
            OperandKind::UseGroup(value_group)
        } else {
            let value = self.get_value_for_use(bank, first_block_for_def)?;
            OperandKind::Use(value)
        };
        Ok(Operand::new(kind, OperandConstraint::Class(class)))
    }

    /// Generates an operand which defines a value.
    fn gen_def(
        &mut self,
        block: Block,
        value: Value,
        is_ret: bool,
        op_idx: usize,
    ) -> Result<Operand> {
        let bank = self.func.values[value].bank;

        // Try to use a fixed register.
        if self.u.arbitrary()? {
            let reg = *self.u.choose(&self.reg_per_bank[bank])?;
            let kind = if !is_ret && self.u.arbitrary()? {
                OperandKind::Def(value)
            } else {
                OperandKind::EarlyDef(value)
            };
            if self.check_fixed_conflict(reg, matches!(kind, OperandKind::EarlyDef(_)), true) {
                return Ok(Operand::new(kind, OperandConstraint::Fixed(reg)));
            }
        }

        // Pick a register class to use.
        let class = *self.u.choose(&self.class_per_bank[bank])?;
        let group_size = self.reginfo.class_group_size(class);
        let kind = if group_size > 1 {
            let mut values = vec![value];
            for _ in 1..group_size {
                // Try to find another definition from the definition list in
                // the current block.
                let value = if !self.defs_by_blocks[block].is_empty() && self.u.arbitrary()? {
                    let value_idx = self.u.choose_index(self.defs_by_blocks[block].len())?;
                    if self.func.values[self.defs_by_blocks[block][value_idx]].bank == bank {
                        self.defs_by_blocks[block].remove(value_idx)
                    } else {
                        self.new_value(bank)?
                    }
                } else {
                    self.new_value(bank)?
                };
                values.push(value);
            }
            if self.u.arbitrary()? {
                // Later change this into a Reuse.
                self.reuse_operands.push(op_idx);
            }
            let value_group = self.func.value_groups.push(values);
            if !is_ret && self.u.arbitrary()? {
                OperandKind::DefGroup(value_group)
            } else {
                OperandKind::EarlyDefGroup(value_group)
            }
        } else {
            if self.u.arbitrary()? {
                // Later change this into a Reuse.
                self.reuse_operands.push(op_idx);
            }
            if !is_ret && self.u.arbitrary()? {
                OperandKind::Def(value)
            } else {
                OperandKind::EarlyDef(value)
            }
        };
        Ok(Operand::new(kind, OperandConstraint::Class(class)))
    }

    /// Generates a single instruction in the given block.
    fn gen_inst(&mut self, block: Block, is_first_inst: bool, is_ret: bool) -> Result<InstData> {
        // These are temporary for the scope of this instruction.
        self.early_fixed.clear();
        self.late_fixed.clear();
        self.reuse_operands.clear();

        let mut inst = InstData {
            operands: vec![],
            clobbers: vec![],
            block,
            is_terminator: false,
            is_pure: self.u.arbitrary()?,
        };

        // Add operands which define values.
        for _ in 0..self.u.int_in_range(self.config.defs_per_inst.clone())? {
            // Generate a value that was previously used, otherwise generate a
            // new dead value.
            let value = if self.u.arbitrary()? && !self.defs_by_blocks[block].is_empty() {
                self.defs_by_blocks[block].pop().unwrap()
            } else {
                let bank = RegBank::new(self.u.choose_index(self.reginfo.num_banks())?);
                self.new_value(bank)?
            };
            inst.operands
                .push(self.gen_def(block, value, is_ret, inst.operands.len())?);
        }

        // If this is the first instruction in a block then we need to define
        // any remaining values to be defined in this block.
        if is_first_inst {
            while let Some(value) = self.defs_by_blocks[block].pop() {
                inst.operands
                    .push(self.gen_def(block, value, is_ret, inst.operands.len())?);
            }
        }

        // Lowest block in the dominator tree in which values that we use in the
        // current instruction are going to be defined. If we are emitting the
        // first instruction of the block that values must come from the
        // immediate dominator. In the case of the first instruction of the
        // entry block we cannot add any use operands at all.
        let first_block_for_def = if is_first_inst {
            self.domtree.immediate_dominator(block)
        } else {
            Some(block)
        };

        // Add operands which use values.
        if let Some(first_block_for_def) = first_block_for_def {
            for _ in 0..self.u.int_in_range(self.config.uses_per_inst.clone())? {
                let op = self.gen_use(first_block_for_def, &mut inst)?;
                inst.operands.push(op);
            }
        }

        // Add clobbers. This can be anything, including overlaps with fixed
        // constraints on the same instruction.
        if !is_ret {
            for _ in 0..self.u.int_in_range(self.config.clobbers_per_inst.clone())? {
                let unit = RegUnit::new(self.u.int_in_range(0..=MAX_REG_UNITS - 1)?);
                inst.clobbers.push(unit);
            }
        }

        Ok(inst)
    }

    /// Finalizes the function by assigning instruction numbers to each
    /// instruction, in block order.
    fn finalize(&mut self) -> Result<()> {
        for (block, blockdata) in &mut self.func.blocks {
            // Instructions were generated in reverse, un-reverse them and add them
            // to the block.
            let from = self.func.insts.next_key();
            for inst in self.block_insts[block].drain(..).rev() {
                let inst = self.func.insts.push(inst);
                if (!self.func.insts[inst].is_terminator || blockdata.succs.len() != 1)
                    && self.u.arbitrary()?
                {
                    self.func.safepoints.push(inst);
                }
            }
            let to = self.func.insts.next_key();
            blockdata.insts = InstRange::new(from, to);
        }
        Ok(())
    }
}
