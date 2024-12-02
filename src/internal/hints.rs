//! Hints for live ranges that are linked to a fixed-def or fixed-use.
//!
//! Fixed-def and fixed-use constraints are represented internally as a fixed
//! reservation which is connected to the rest of a value's live range with an
//! implicit copy. To minimize copies, we prefer allocating live ranges linked
//! to a fixed def/use to that fixed register.

use core::fmt;

use alloc::vec;
use alloc::vec::Vec;

use crate::{
    entity::EntitySet,
    function::{Function, Inst, Value},
    reginfo::PhysReg,
};

use super::live_range::LiveRangeSegment;

/// Key used for sorting hints so that we can easily retrieve a range of keys
/// for a value.
///
/// This encodes whether the hint is from an incoming value or an outgoing one,
/// which is useful because at the start of a segment we only want to consider
/// incoming value and at the end only outgoing values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct HintKey {
    /// Bit-pack in 64 bits.
    ///
    /// value:32 inst:31 pos:1
    bits: u64,
}

impl fmt::Display for HintKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = Value::new((self.bits >> 32) as usize);
        let inst = Inst::new((self.bits >> 1) as usize & 0x7ffffff);
        let pos = if self.bits & 1 == 0 { "out" } else { "in" };
        write!(f, "{value} at {inst}-{pos}")
    }
}

impl HintKey {
    /// A hint about an incoming value at the given instruction boundary.
    pub fn incoming(value: Value, inst: Inst) -> Self {
        Self {
            bits: ((value.index() as u64) << 32) | ((inst.index() as u64) << 1) | 1,
        }
    }

    /// A hint about an outgoing value at the given instruction boundary.
    pub fn outgoing(value: Value, inst: Inst) -> Self {
        Self {
            bits: ((value.index() as u64) << 32) | ((inst.index() as u64) << 1),
        }
    }
}

#[derive(Debug)]
pub struct Hint {
    key: HintKey,
    pub reg: PhysReg,
    pub weight: f32,
}

impl fmt::Display for Hint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({}) for {}", self.reg, self.weight, self.key)
    }
}

/// Fixed register hints.
pub struct Hints {
    hints: Vec<Hint>,
    has_hint: EntitySet<Value>,
}

impl Hints {
    pub fn new() -> Self {
        Self {
            hints: vec![],
            has_hint: EntitySet::new(),
        }
    }

    pub fn clear(&mut self, func: &impl Function) {
        self.hints.clear();
        self.has_hint.clear_and_resize(func.num_values());
    }

    /// Records a fixed definition at the given instruction.
    ///
    /// This adds an incoming fixed register hint at all successor instructions.
    pub fn add_fixed_def(&mut self, value: Value, inst: Inst, reg: PhysReg, func: &impl Function) {
        let block = func.inst_block(inst);
        if func.inst_is_terminator(inst) {
            let succs = func.block_succs(block);
            let is_jump = succs.len() == 1 && func.block_preds(succs[0]).len() > 1;
            for &succ in succs {
                self.hints.push(Hint {
                    key: HintKey::incoming(value, func.block_insts(succ).from),
                    reg,
                    weight: func.block_frequency(if is_jump { block } else { succ }),
                });
            }
        } else {
            self.hints.push(Hint {
                key: HintKey::incoming(value, inst.next()),
                reg,
                weight: func.block_frequency(block),
            });
        }
        self.has_hint.insert(value);
    }

    /// Records a fixed use at the given instruction.
    ///
    /// This adds an outgoing fixed register hint at all predecessor instructions.
    ///
    /// If this use is at the first instruction of a block then `blockparam_idx`
    /// should indicate whether this is a block parameter.
    pub fn add_fixed_use(
        &mut self,
        value: Value,
        inst: Inst,
        reg: PhysReg,
        blockparam_idx: Option<u32>,
        func: &impl Function,
    ) {
        let block = func.inst_block(inst);
        if inst == func.block_insts(block).from {
            let preds = func.block_preds(block);
            let is_jump = preds.len() > 1;
            for &pred in preds {
                let value = if let Some(idx) = blockparam_idx {
                    func.jump_blockparams(pred)[idx as usize]
                } else {
                    value
                };
                if blockparam_idx.is_some() {
                    self.has_hint.insert(value);
                }
                self.hints.push(Hint {
                    key: HintKey::outgoing(value, func.block_insts(pred).to),
                    reg,
                    weight: func.block_frequency(if is_jump { pred } else { block }),
                });
            }
            if blockparam_idx.is_none() {
                self.has_hint.insert(value);
            }
        } else {
            self.hints.push(Hint {
                key: HintKey::outgoing(value, inst),
                reg,
                weight: func.block_frequency(block),
            });
            self.has_hint.insert(value);
        }
    }

    /// Sorts hints so that they can be efficiently queried by
    /// `hints_for_segment`.
    pub fn sort_hints(&mut self) {
        self.hints.sort_unstable_by_key(|hint| hint.key);

        trace!("Fixed register hints:");
        for hint in &self.hints {
            trace!("- {hint}");
        }
    }

    /// Quickly checks whether the given value has a fixed-register hint.
    pub fn has_fixed_hint(&self, value: Value) -> bool {
        self.has_hint.contains(value)
    }

    /// Returns an iterator over all the fixed-register hints covered by the
    /// given live range segment.
    pub fn hints_for_segment(
        &self,
        value: Value,
        seg: LiveRangeSegment,
    ) -> impl Iterator<Item = &Hint> + '_ {
        let start = HintKey::incoming(value, seg.from.inst());
        let end = HintKey::outgoing(value, seg.to.round_to_next_inst().inst());
        let from = self.hints.partition_point(|hint| hint.key < start);
        self.hints[from..]
            .iter()
            .take_while(move |hint| hint.key <= end)
    }

    /// Given a live range split, returns whether the left and right sides of
    /// the split still have a fixed register hint.
    pub fn hints_for_split(
        &self,
        value: Value,
        seg: LiveRangeSegment,
        split_at: Inst,
    ) -> (bool, bool) {
        let start = HintKey::incoming(value, seg.from.inst());
        let end = HintKey::outgoing(value, seg.to.round_to_next_inst().inst());
        let mid_key = HintKey::incoming(value, split_at);
        let mid = self.hints.partition_point(|hint| hint.key < mid_key);
        let first = mid != 0 && self.hints[mid - 1].key >= start;
        let second = self.hints.get(mid).is_some_and(|hint| hint.key <= end);
        (first, second)
    }
}
