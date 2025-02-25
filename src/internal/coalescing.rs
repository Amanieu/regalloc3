//! Live range coalescing pass.
//!
//! This pass groups multiple [`Value`]s into a [`ValueSet`] which allows them
//! to be allocated as a single virtual register. The live ranges of `Value`s in
//! a `ValueSet` must not overlap.
//!
//! This allows eliminating moves which would otherwise be needed if two values
//! are allocated to different registers. Specifically we try to merge:
//! - Incoming and outgoing block parameters.
//! - Definitions which reuse an input register.
//! - Values which should be placed in the same register group.

use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::mem;

use smallvec::SmallVec;

use super::live_range::ValueSegment;
use super::uses::Uses;
use super::value_live_ranges::ValueLiveRanges;
use crate::Stats;
use crate::entity::SecondaryMap;
use crate::entity::packed_option::PackedOption;
use crate::function::{Block, Function, OperandConstraint, OperandKind, Value, ValueGroup};
use crate::internal::value_live_ranges::ValueSet;
use crate::union_find::UnionFind;

pub struct Coalescing {
    /// Mapping of `Value` to `ValueSet`.
    ///
    /// Initially we start with a 1-to-1 mapping. As we unify sets, the
    /// index of the set leader in the union-find data structure represents the
    /// `ValueSet` containing all `Value`s in the set.
    set_for_value: UnionFind<Value>,

    /// Scratch space for computing the block ordering by priority.
    blocks_by_priority: Vec<Block>,

    /// For each value, the last `ValueGroup` that it was seen to be part of.
    ///
    /// This is used to merge values that are used together in groups.
    ///
    /// Consider this example:
    /// - inst Def(%0, %1, %2)
    /// - inst Use(%2) Def(%3)
    /// - inst Use(%0, %1, %3)
    ///
    /// We want to merge %2 and %3 so that the same virtual register group can
    /// be used for the 1st and 3rd instructions.
    last_group_for_value: SecondaryMap<Value, PackedOption<ValueGroup>>,
}

impl Coalescing {
    pub fn new() -> Self {
        Self {
            set_for_value: UnionFind::new(),
            blocks_by_priority: vec![],
            last_group_for_value: SecondaryMap::new(),
        }
    }

    /// Returns the `ValueSet` containing the given `Value`.
    pub fn set_for_value(&mut self, value: Value) -> ValueSet {
        let leader = self.set_for_value.find(value);
        ValueSet::new(leader.index())
    }

    /// Runs the coalescing pass to group values into `ValueSet`s.
    pub fn run(
        &mut self,
        func: &impl Function,
        uses: &Uses,
        value_live_ranges: &mut ValueLiveRanges,
        stats: &mut Stats,
    ) {
        self.set_for_value.reset(func.num_values());
        self.last_group_for_value
            .clear_and_resize(func.num_values());

        self.compute_block_order(func);

        for i in 0..self.blocks_by_priority.len() {
            self.coalesce_in_block(self.blocks_by_priority[i], func, value_live_ranges, stats);
        }

        stat!(
            stats,
            value_sets,
            value_live_ranges.all_value_sets().count()
        );

        value_live_ranges.dump(uses);
    }

    /// Computes an ordering of blocks sorted by priority.
    ///
    /// Each merge eliminates the need for one move instruction in the final
    /// program. However a successful merge can cause a future merge to fail
    /// due to interference. Therefore we want to prioritize merges that have
    /// the highest impact:
    ///
    /// * We want to eliminate moves in loops that are executed many times.
    /// * We want to eliminate moves in split critical edge blocks, since it
    ///   allows the entire block to be eliminated with jump threading.
    ///
    /// The heuristic is based on `compareMBBPriority` from LLVM.
    fn compute_block_order(&mut self, func: &impl Function) {
        self.blocks_by_priority.clear();
        self.blocks_by_priority.extend(func.blocks());

        let cmp_freq = |a: Block, b: Block| {
            if func.block_frequency(a) < func.block_frequency(b) {
                Ordering::Less
            } else if func.block_frequency(a) > func.block_frequency(b) {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        };
        let is_split_edge = |block: Block| {
            // A split critical edge is an artifical block which only exists for
            // the sake of passing blockparams to another block.
            func.block_insts(block).len() == 1 && func.block_succs(block).len() == 1
        };
        let num_connections =
            |block: Block| func.block_succs(block).len() + func.block_preds(block).len();

        self.blocks_by_priority.sort_by(|&a, &b| {
            // Prioritize hot blocks first.
            cmp_freq(a, b)
                .reverse()
                // Prioritize critical edges to enable jump threading.
                .then_with(|| is_split_edge(a).cmp(&is_split_edge(b)).reverse())
                // Prioritize blocks which are more connected in the CFG.
                .then_with(|| num_connections(a).cmp(&num_connections(b)).reverse())
            // Otherwise follow the original block order.
        });
    }

    /// Find pairs of values to merge in the given block.
    ///
    /// We merge pairs where allocating both values to the same register would
    /// eliminate the need for a move instruction.
    fn coalesce_in_block(
        &mut self,
        block: Block,
        func: &impl Function,
        value_live_ranges: &mut ValueLiveRanges,
        stats: &mut Stats,
    ) {
        trace!("Coalescing values in {block}...");

        // Find merge opportunities in instruction operands.
        for inst in func.block_insts(block).iter() {
            let operands = func.inst_operands(inst);
            for &op in operands {
                // Merge the values of output operands that reuse the allocation
                // of an input with that input value.
                if let OperandConstraint::Reuse(idx) = op.constraint() {
                    let use_op = operands[idx];
                    match (op.kind(), use_op.kind()) {
                        (
                            OperandKind::Def(def_value) | OperandKind::EarlyDef(def_value),
                            OperandKind::Use(use_value),
                        ) => {
                            trace!("Reused operand: {use_value} -> {def_value}");
                            if self.coalesce_values(use_value, def_value, value_live_ranges, stats)
                            {
                                stat!(stats, coalesced_tied);
                            } else {
                                stat!(stats, coalesced_failed_tied);
                            }
                        }
                        (
                            OperandKind::DefGroup(def_value_group)
                            | OperandKind::EarlyDefGroup(def_value_group),
                            OperandKind::UseGroup(use_value_group),
                        ) => {
                            for (&def_value, &use_value) in func
                                .value_group_members(def_value_group)
                                .iter()
                                .zip(func.value_group_members(use_value_group))
                            {
                                trace!("Reused group operand: {use_value} -> {def_value}");
                                if self.coalesce_values(
                                    use_value,
                                    def_value,
                                    value_live_ranges,
                                    stats,
                                ) {
                                    stat!(stats, coalesced_tied_group);
                                } else {
                                    stat!(stats, coalesced_failed_tied_group);
                                }
                            }
                        }
                        _ => unreachable!(),
                    }
                } else if let OperandKind::DefGroup(value_group)
                | OperandKind::EarlyDefGroup(value_group)
                | OperandKind::UseGroup(value_group) = op.kind()
                {
                    // If a value is used in 2 separate group operands, try to
                    // merge all the other group members with each other.
                    let members = func.value_group_members(value_group);
                    for &value in members {
                        if let Some(prev_group) = self.last_group_for_value[value].expand() {
                            let prev_members = func.value_group_members(prev_group);
                            if members.len() != prev_members.len() {
                                continue;
                            }
                            trace!(
                                "Merging groups which both contain {value}: {prev_group} and \
                                 {value_group}"
                            );
                            for (idx, (&value, &prev_value)) in
                                members.iter().zip(prev_members).enumerate()
                            {
                                if value != prev_value {
                                    trace!(
                                        "Merging {prev_group}[{idx}]:{prev_value} and \
                                         {value_group}[{idx}]:{value}"
                                    );
                                    if self.coalesce_values(
                                        value,
                                        prev_value,
                                        value_live_ranges,
                                        stats,
                                    ) {
                                        stat!(stats, coalesced_group);
                                    } else {
                                        stat!(stats, coalesced_failed_group);
                                    }
                                }
                            }
                        }
                        self.last_group_for_value[value] = Some(value_group).into();
                    }
                }
            }
        }

        // Merge matching block parameters.
        if let &[succ] = func.block_succs(block) {
            for (&blockparam_out, &blockparam_in) in func
                .jump_blockparams(block)
                .iter()
                .zip(func.block_params(succ))
            {
                trace!("Block parameter: {blockparam_out} -> {blockparam_in}");
                if self.coalesce_values(blockparam_out, blockparam_in, value_live_ranges, stats) {
                    stat!(stats, coalesced_blockparam);
                } else {
                    stat!(stats, coalesced_failed_blockparam);
                }
            }
        }
    }

    /// Attempts to merge the two given values into the same `ValueSet`.
    ///
    /// Returns whether a merge was performed.
    fn coalesce_values(
        &mut self,
        a: Value,
        b: Value,
        value_live_ranges: &mut ValueLiveRanges,
        stats: &mut Stats,
    ) -> bool {
        trace!("Trying to merge {a} and {b} into the same value set...");

        let mut merged = false;
        self.set_for_value.try_union(a, b, |set_a, set_b| {
            let set_a = ValueSet::new(set_a.index());
            let set_b = ValueSet::new(set_b.index());

            // Check if any of the segments in both sets overlap. This can be
            // done efficiently since the segments are always sorted.
            let mut segments_a = &value_live_ranges[set_a][..];
            let mut segments_b = &value_live_ranges[set_b][..];

            // If either set has an empty live range then there is no point in
            // merging them.
            if segments_a.is_empty() || segments_b.is_empty() {
                return false;
            }

            // Fast path if one set of segments are all before/after the other.
            if segments_a[0].live_range.from >= segments_b.last().unwrap().live_range.to {
                trace!("-> fast-path merging {set_b} into {set_a}");
                stat!(stats, coalesce_fast_path);
                let mut segments = mem::take(&mut value_live_ranges[set_b]);
                segments.extend_from_slice(&value_live_ranges[set_a]);
                value_live_ranges[set_a] = segments;
                merged = true;
                return true;
            } else if segments_b[0].live_range.from >= segments_a.last().unwrap().live_range.to {
                trace!("-> fast-path merging {set_b} into {set_a}");
                stat!(stats, coalesce_fast_path);
                let segments = mem::take(&mut value_live_ranges[set_b]);
                value_live_ranges[set_a].extend_from_slice(&segments[..]);
                merged = true;
                return true;
            }

            // Otherwise iterate through the segments in lockstep to figure out
            // if any segments overlap.
            while let (Some(seg_a), Some(seg_b)) = (segments_a.first(), segments_b.first()) {
                if seg_a.live_range.from >= seg_b.live_range.to {
                    segments_b = &segments_b[1..];
                } else if seg_a.live_range.to <= seg_b.live_range.from {
                    segments_a = &segments_a[1..];
                } else {
                    trace!(
                        "-> overlap between {set_a} ({}) and {set_b} ({})",
                        seg_a.live_range, seg_b.live_range
                    );
                    return false;
                }
            }

            // Move all segments to the unified ValueSet.
            trace!("-> merging {set_b} into {set_a}");
            stat!(stats, coalesce_slow_path);
            let segments = merge(&value_live_ranges[set_a], &value_live_ranges[set_b]);
            value_live_ranges[set_a] = segments;
            value_live_ranges[set_b].clear();
            merged = true;
            true
        });
        merged
    }
}

/// Merges 2 sorted value segment lists into a new list.
fn merge(mut a: &[ValueSegment], mut b: &[ValueSegment]) -> SmallVec<[ValueSegment; 4]> {
    let mut out = SmallVec::with_capacity(a.len() + b.len());
    loop {
        match (a, b) {
            (&[seg_a, ref rest_a @ ..], &[seg_b, ref rest_b @ ..]) => {
                debug_assert_ne!(seg_a.live_range.from, seg_b.live_range.from);
                let seg = if seg_a.live_range.from < seg_b.live_range.from {
                    a = rest_a;
                    seg_a
                } else {
                    b = rest_b;
                    seg_b
                };
                out.push(seg);
            }
            (&[], &[_, ..]) => {
                out.extend_from_slice(b);
                break;
            }
            (&[_, ..], &[]) => {
                out.extend_from_slice(a);
                break;
            }
            (&[], &[]) => break,
        }
    }
    debug_assert!(out.is_sorted_by_key(|seg: &ValueSegment| seg.live_range.from));
    debug_assert!(out.is_sorted_by_key(|seg: &ValueSegment| seg.live_range.to));
    out
}
