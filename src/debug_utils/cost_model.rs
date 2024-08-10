use crate::{
    function::{Function, OperandConstraint, OperandKind, RematCost},
    output::{Output, OutputInst},
    reginfo::RegInfo,
};

/// Cost model to evaluate and compare the quality of different register
/// allocation algorithms.
///
/// A default set of costs (based on LLVM's) are provided by the [`Default`]
/// implementation.
#[derive(Debug, Clone)]
pub struct CostModel {
    /// Cost of a load from memory into a register.
    pub load_cost: f32,

    /// Cost of a store from a register into memory.
    pub store_cost: f32,

    /// Cost of a move from one register to another.
    pub move_cost: f32,

    /// Cost of rematerializing a value with [`RematCost::CheaperThanLoad`].
    ///
    /// The cost for [`RematCost::CheaperThanMove`] is assumed to be zero.
    pub remat_cost: f32,
}

impl Default for CostModel {
    fn default() -> Self {
        // These numbers come from LLVM's RegAllocScore.cpp.
        Self {
            load_cost: 4.0,
            store_cost: 1.0,
            move_cost: 0.2,
            remat_cost: 1.0,
        }
    }
}

impl CostModel {
    /// Evaluates the quality of the register allocator's output using the cost
    /// model.
    ///
    /// The returned value is approximately indicative of the execution time of
    /// the function based on the block frequencies provided.
    #[must_use]
    pub fn evaluate(&self, output: &Output<'_, impl Function, impl RegInfo>) -> f32 {
        let mut score = 0.0;
        let reginfo = output.reginfo();
        let func = output.function();
        for block in func.blocks() {
            let freq = func.block_frequency(block);
            for inst in output.output_insts(block) {
                match inst {
                    OutputInst::Inst {
                        inst,
                        operand_allocs,
                        stack_map: _,
                    } => {
                        // Penalize instruction operands that are assigned to
                        // memory instead of registers.
                        for (&op, &alloc) in func.inst_operands(inst).iter().zip(operand_allocs) {
                            if !alloc.is_memory(reginfo) {
                                continue;
                            }
                            let class = match op.constraint() {
                                OperandConstraint::Class(class) => class,
                                OperandConstraint::Fixed(_) => continue,
                                OperandConstraint::Reuse(idx) => {
                                    let OperandConstraint::Class(class) =
                                        func.inst_operands(inst)[idx].constraint()
                                    else {
                                        unreachable!()
                                    };
                                    class
                                }
                            };
                            let cost = match op.kind() {
                                OperandKind::Def(_) | OperandKind::EarlyDef(_) => {
                                    self.store_cost - self.move_cost
                                }
                                OperandKind::Use(_) => self.load_cost - self.move_cost,
                                OperandKind::DefGroup(_)
                                | OperandKind::UseGroup(_)
                                | OperandKind::EarlyDefGroup(_)
                                | OperandKind::NonAllocatable => continue,
                            };

                            // Scale this with the class spill cost: a class
                            // spill cost of 0 means that there is no cost to
                            // choosing a spill slot instead of a register.
                            score += cost * freq * reginfo.class_spill_cost(class);
                        }
                    }
                    OutputInst::Rematerialize { value, to } => {
                        // Treat a rematerialization as having 2 parts: actually
                        // computing the value and then moving it to its
                        // destination.
                        let remat_cost = match func.can_rematerialize(value).unwrap().0 {
                            RematCost::CheaperThanMove => 0.0,
                            RematCost::CheaperThanLoad => self.remat_cost,
                        };
                        let move_cost = match to.is_memory(reginfo) {
                            false => self.move_cost,
                            true => self.store_cost,
                        };
                        score += (remat_cost + move_cost) * freq;
                    }
                    OutputInst::Move { from, to, value: _ } => {
                        let cost = match (from.is_memory(reginfo), to.is_memory(reginfo)) {
                            (false, false) => self.move_cost,
                            (false, true) => self.store_cost,
                            (true, false) => self.load_cost,
                            (true, true) => {
                                unreachable!()
                            }
                        };
                        score += cost * freq;
                    }
                }
            }
        }
        score
    }
}
