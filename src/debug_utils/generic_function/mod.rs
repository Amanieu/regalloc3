//! Generic implementation of a [`Function`] which can be used for testing the
//! register allocator.

use alloc::vec::Vec;
use core::fmt;

use crate::entity::PrimaryMap;
use crate::entity::packed_option::PackedOption;
use crate::function::{
    Block, Function, Inst, InstRange, Operand, RematCost, TerminatorKind, Value, ValueGroup,
};
use crate::reginfo::{RegBank, RegClass, RegUnit};

#[cfg(feature = "arbitrary")]
mod arbitrary;
#[cfg(feature = "arbitrary")]
pub use arbitrary::ArbitraryFunctionConfig;

use super::DisplayFunction;
#[cfg(feature = "parse")]
mod parse;

#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct BlockData {
    insts: InstRange,
    preds: Vec<Block>,
    succs: Vec<Block>,
    block_params_in: Vec<Value>,
    block_params_out: Vec<Value>,
    immediate_dominator: PackedOption<Block>,
    frequency: f32,
    is_critical_edge: bool,
}

#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct InstData {
    operands: Vec<Operand>,
    clobbers: Vec<RegUnit>,
    block: Block,
    terminator_kind: Option<TerminatorKind>,
    is_pure: bool,
}

#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct ValueData {
    bank: RegBank,
    remat: Option<(RematCost, RegClass)>,
}

/// A generic implementation of [`Function`] which can be constructed from an
/// existing `Function` or parsed from a text representation.
///
/// This is primarily useful for development and debugging of the register
/// allocator since it enables working with user-readable and editable  forms of
/// the register allocator input.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GenericFunction {
    blocks: PrimaryMap<Block, BlockData>,
    insts: PrimaryMap<Inst, InstData>,
    values: PrimaryMap<Value, ValueData>,
    value_groups: PrimaryMap<ValueGroup, Vec<Value>>,
}

impl fmt::Debug for GenericFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        DisplayFunction(self).fmt(f)
    }
}

impl fmt::Display for GenericFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        DisplayFunction(self).fmt(f)
    }
}

impl GenericFunction {
    /// Constructs a `GenericFunction` from an existing object which implements
    /// the [`Function`] trait.
    pub fn from_function(func: &impl Function) -> Self {
        let mut blocks = PrimaryMap::new();
        let mut insts = PrimaryMap::new();
        let mut values = PrimaryMap::new();
        let mut value_groups = PrimaryMap::new();
        for block in func.blocks() {
            blocks.push(BlockData {
                insts: func.block_insts(block),
                preds: func.block_preds(block).into(),
                succs: func.block_succs(block).into(),
                block_params_in: func.block_params(block).into(),
                block_params_out: func.jump_blockparams(block).into(),
                frequency: func.block_frequency(block),
                immediate_dominator: func.block_immediate_dominator(block).into(),
                is_critical_edge: func.block_is_critical_edge(block),
            });
        }
        for inst in func.insts() {
            insts.push(InstData {
                operands: func.inst_operands(inst).into(),
                clobbers: func.inst_clobbers(inst).collect(),
                block: func.inst_block(inst),
                terminator_kind: func.terminator_kind(inst),
                is_pure: func.can_eliminate_dead_inst(inst),
            });
        }
        for value in func.values() {
            values.push(ValueData {
                bank: func.value_bank(value),
                remat: func.can_rematerialize(value),
            });
        }
        for group in func.value_groups() {
            value_groups.push(func.value_group_members(group).into());
        }
        Self {
            blocks,
            insts,
            values,
            value_groups,
        }
    }
}

impl Function for GenericFunction {
    #[inline]
    fn num_insts(&self) -> usize {
        self.insts.len()
    }

    #[inline]
    fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    #[inline]
    fn block_insts(&self, block: Block) -> InstRange {
        self.blocks[block].insts
    }

    #[inline]
    fn block_succs(&self, block: Block) -> &[Block] {
        &self.blocks[block].succs
    }

    #[inline]
    fn block_preds(&self, block: Block) -> &[Block] {
        &self.blocks[block].preds
    }

    #[inline]
    fn block_immediate_dominator(&self, block: Block) -> Option<Block> {
        self.blocks[block].immediate_dominator.expand()
    }

    #[inline]
    fn block_params(&self, block: Block) -> &[Value] {
        &self.blocks[block].block_params_in
    }

    #[inline]
    fn terminator_kind(&self, inst: Inst) -> Option<TerminatorKind> {
        self.insts[inst].terminator_kind
    }

    #[inline]
    fn jump_blockparams(&self, block: Block) -> &[Value] {
        &self.blocks[block].block_params_out
    }

    #[inline]
    fn inst_block(&self, inst: Inst) -> Block {
        self.insts[inst].block
    }

    #[inline]
    fn block_frequency(&self, block: Block) -> f32 {
        self.blocks[block].frequency
    }

    #[inline]
    fn block_is_critical_edge(&self, block: Block) -> bool {
        self.blocks[block].is_critical_edge
    }

    #[inline]
    fn inst_operands(&self, inst: Inst) -> &[Operand] {
        &self.insts[inst].operands
    }

    #[inline]
    fn inst_clobbers(&self, inst: Inst) -> impl Iterator<Item = RegUnit> {
        self.insts[inst].clobbers.iter().copied()
    }

    #[inline]
    fn num_values(&self) -> usize {
        self.values.len()
    }

    #[inline]
    fn value_bank(&self, value: Value) -> RegBank {
        self.values[value].bank
    }

    #[inline]
    fn num_value_groups(&self) -> usize {
        self.value_groups.len()
    }

    #[inline]
    fn value_group_members(&self, group: ValueGroup) -> &[Value] {
        &self.value_groups[group]
    }

    #[inline]
    fn can_rematerialize(&self, value: Value) -> Option<(RematCost, RegClass)> {
        self.values[value].remat
    }

    #[inline]
    fn can_eliminate_dead_inst(&self, inst: Inst) -> bool {
        self.insts[inst].is_pure
    }
}
