//! Live range splitting and spilling.

use alloc::vec;
use alloc::vec::Vec;

use super::queue::VirtRegOrGroup;
use super::{AbstractVirtRegGroup, Assignment, Context, Stage};
use crate::SplitStrategy;
use crate::function::{Function, Inst, InstRange, OperandKind};
use crate::internal::live_range::{Slot, ValueSegment};
use crate::internal::reg_matrix::{InterferenceKind, RegMatrix};
use crate::internal::uses::{UseKind, Uses};
use crate::internal::virt_regs::{VirtReg, VirtRegGroup, VirtRegs};
use crate::reginfo::{PhysReg, RegInfo};

/// Information about a use that we may want to include or exclude from a split.
///
/// Multiple uses at the same instruction are merged together.
#[derive(Debug)]
struct SplitUse {
    /// Instruction at which this use occurs.
    inst: Inst,

    /// Split cost of all uses at this instruction, weighed by block frequency.
    weight: f32,

    /// Weight which is only taken into account if the split extends to the left
    /// pf this use.
    ///
    /// This is needed for uses which occur after the end of the live range of a
    /// segment such as a `FixedUse` or `TiedUse`.
    left_weight: f32,

    /// Weight which is only taken into account if the split extends to the
    /// right pf this use.
    ///
    /// This is needed for uses which occur before the start of the live range
    /// of a segment such as a `FixedDef`.
    right_weight: f32,
}

/// Information about a gap between 2 `SplitUse`.
///
/// For a virtual register to be successfully allocated to a gap, its spill
/// weight must be greater than the spill weight of any interfering virtual
/// register in this gap.
#[derive(Debug)]
struct SplitGap {
    range: InstRange,

    weight: f32,

    /// Number live instructions in this gap.
    live_insts: u32,

    /// Lowest block frequency of all blocks in this gap.
    min_freq: f32,
}

/// A proposal for splitting based on the interference patterns of the current
/// allocations in a physical register.
#[derive(Debug)]
struct SplitProposal {
    /// Register whose interference was split around.
    ///
    /// This is used as a hint for the allocation order of the split-off part.
    reg: PhysReg,

    /// Left boundary to split at.
    left: Option<Inst>,

    /// Right boundary to split at.
    right: Option<Inst>,

    /// Sum of weighed spill costs covered by the split.
    use_weight: f32,

    /// Estimated number of live instructions in the split.
    ///
    /// We may over-estimate this in some cases since we assume instructions
    /// with uses are live, but we will never under-estimate this. This is
    /// important since we really need to be sure that the produced vreg will
    /// actually be able to evict its interference.
    live_insts: u32,

    /// Maximum inteference weight that will need to be evicted in the split
    /// region.
    interference_weight: f32,

    /// Estimated cost of performing the split.
    ///
    /// This is used to determine whether spilling is more profitable when
    /// allowed by the register class.
    split_cost: f32,
}

impl SplitProposal {
    /// Score used to select the best split proposal.
    fn score(&self) -> (f32, u32, f32) {
        // - maximize weight covered by the split.
        // - if equal weight, maximize instructions,
        // - if also equal instructions, minimize interference weight
        (self.use_weight, self.live_insts, -self.interference_weight)
    }
}

/// Temporary state used when splitting.
pub struct Splitter {
    /// Scratch space for collecting segments when building a virtual register.
    segments: Vec<ValueSegment>,

    /// Scratch space for collecting minimal segments when spilling or isolating
    /// group uses.
    minimal_segments: Vec<ValueSegment>,

    /// Newly created virtual register from the minimal live ranges.
    new_vregs: Vec<VirtReg>,

    /// Instructions at which a virtual register is used, with weighed spill
    /// costs.
    uses: Vec<SplitUse>,

    /// Information about gaps between uses.
    gaps: Vec<SplitGap>,

    /// Maximum spill weights of inteferences in gaps.
    gap_interference_weights: Vec<f32>,
}

impl Splitter {
    pub fn new() -> Self {
        Self {
            segments: vec![],
            minimal_segments: vec![],
            new_vregs: vec![],
            gaps: vec![],
            uses: vec![],
            gap_interference_weights: vec![],
        }
    }
}

impl<F: Function, R: RegInfo> Context<'_, F, R> {
    /// Collects all instructions where the given virtual register is used.
    fn collect_uses(&mut self, vreg: VirtReg) {
        let segments = self.virt_regs.segments(vreg);
        let splitter = &mut self.allocator.splitter;
        splitter.uses.clear();

        // Add a use at the start of the first segment.
        splitter.uses.push(SplitUse {
            inst: segments[0].first_inst(),
            weight: 0.0,
            left_weight: 0.0,
            right_weight: 0.0,
        });

        for segment in segments {
            for u in &self.uses[segment.use_list] {
                // Ignore uses with no spill weight.
                let spill_cost = u.spill_cost(self.reginfo);
                if spill_cost == 0.0 {
                    continue;
                }
                let freq = self.func.block_frequency(self.func.inst_block(u.pos));
                let weight = spill_cost * freq;

                // Add a new use if this use is at a different position than the
                // last one.
                let mut split_use = splitter.uses.last_mut().unwrap();
                if u.pos < split_use.inst {
                    // This can only happen for a FixedDef which happens
                    // "before" the live range starts.
                    debug_assert!(matches!(u.kind, UseKind::FixedDef { .. }));
                    if splitter.uses.len() < 2
                        || splitter.uses[splitter.uses.len() - 2].inst != u.pos
                    {
                        splitter.uses.insert(
                            splitter.uses.len() - 1,
                            SplitUse {
                                inst: u.pos,
                                weight: 0.0,
                                left_weight: 0.0,
                                right_weight: 0.0,
                            },
                        );
                    }
                    let idx = splitter.uses.len() - 2;
                    split_use = &mut splitter.uses[idx];
                } else if u.pos > split_use.inst {
                    splitter.uses.push(SplitUse {
                        inst: u.pos,
                        weight: 0.0,
                        left_weight: 0.0,
                        right_weight: 0.0,
                    });
                    split_use = splitter.uses.last_mut().unwrap();
                }

                // Merge this with the last use. Special handling is needed if
                // the use is entirely outside the live range of the segment.
                if u.pos.next().slot(Slot::Boundary) == segment.live_range.from {
                    split_use.right_weight += weight;
                } else if u.pos.slot(Slot::Boundary) == segment.live_range.to {
                    split_use.left_weight += weight;
                } else {
                    split_use.weight += weight;
                }
            }
        }

        // Add a use at the end of the last segment.
        let last_segment = segments.last().unwrap();
        let last_inst = if last_segment.use_list.has_liveout() {
            segments.last().unwrap().live_range.to.inst().prev()
        } else {
            self.uses[last_segment.use_list].last().unwrap().pos
        };
        let last_use = splitter.uses.last_mut().unwrap();
        debug_assert!(last_inst >= last_use.inst);
        if last_use.inst != last_inst {
            splitter.uses.push(SplitUse {
                inst: last_inst,
                weight: 0.0,
                left_weight: 0.0,
                right_weight: 0.0,
            });
        }

        trace!("Uses which can be included in a split:");
        for (idx, u) in splitter.uses.iter().enumerate() {
            trace!(
                "  {idx}: {} weight={} left_weight={} right_weight={}",
                u.inst, u.weight, u.left_weight, u.right_weight,
            );
        }

        debug_assert!(splitter.uses.is_sorted_by(|a, b| a.inst < b.inst));
    }

    /// Finds the "best" use with the highest weight that we should build a
    /// split region around.
    ///
    /// Even if we can't build a region for it, we can isolate it and split it
    /// off a live range so that the other parts may be successfully allocated.
    fn find_best_use(&mut self) -> Option<usize> {
        let splitter = &self.allocator.splitter;
        if splitter.uses.len() <= 1 {
            trace!("No uses to build a split around");
            stat!(self.stats, no_split_uses);
            return None;
        }

        // Exclude uses with a weight of 0 from being selected. Since 0-weight
        // uses were already excluded in collect_uses, the only remaining ones
        // are implicit live-in/live-out uses.
        //
        // We also only consider uses which are actually live in their
        // instruction. Otherwise if we end up with a unit split then we would
        // not be making any progress.
        let mut best_idx = None;
        let mut best_weight = 0.0;
        for (idx, u) in splitter.uses.iter().enumerate() {
            if u.weight > best_weight {
                best_idx = Some(idx);
                best_weight = u.weight;
            }
        }

        if let Some(best_idx) = best_idx {
            trace!("Best use at index {best_idx} with weight {best_weight}");
        } else {
            trace!("No best use found");
            stat!(self.stats, no_best_split_use);
        }

        best_idx
    }

    /// Buids gaps between `SplitUse`s, also splitting at block frequency
    /// transitions so that we can extend splits to include an entire loop.
    ///
    /// Returns the index of the initial gap to start building a split from.
    fn collect_gaps(&mut self, best_use: usize) -> usize {
        let splitter = &mut self.allocator.splitter;
        splitter.gaps.clear();

        let mut initial_gap = None;
        if best_use == 0 {
            initial_gap = Some(0);
            splitter.gaps.push(SplitGap {
                range: InstRange::new(splitter.uses[0].inst, splitter.uses[0].inst.next()),
                weight: splitter.uses[0].weight,
                live_insts: 0,
                min_freq: self
                    .func
                    .block_frequency(self.func.inst_block(splitter.uses[0].inst)),
            });
        }

        let mut prev_use = &splitter.uses[0];
        for (idx, u) in splitter.uses.iter().enumerate().skip(1) {
            // Build a gap between this use an the previous one.
            if idx <= best_use {
                let reverse_from = splitter.gaps.len();
                let mut to = u.inst;
                let mut weight = u.left_weight;
                let start_block = self.func.inst_block(prev_use.inst);
                let mut end_block = self.func.inst_block(u.inst.prev());
                if start_block != end_block {
                    while let Some(lower_freq_block) =
                        self.split_placement.prev_lower_freq(end_block)
                    {
                        if lower_freq_block < start_block {
                            break;
                        }

                        let from = self.func.block_insts(lower_freq_block).to;
                        splitter.gaps.push(SplitGap {
                            range: InstRange::new(from, to),
                            weight,
                            live_insts: 0,
                            min_freq: self.func.block_frequency(end_block),
                        });
                        to = from;
                        weight = 0.0;
                        end_block = lower_freq_block;
                    }
                }
                splitter.gaps.push(SplitGap {
                    range: InstRange::new(prev_use.inst, to),
                    weight: weight + prev_use.right_weight + prev_use.weight,
                    live_insts: 0,
                    min_freq: self.func.block_frequency(end_block),
                });
                splitter.gaps[reverse_from..].reverse();
            } else {
                let mut from = prev_use.inst.next();
                let mut weight = prev_use.right_weight;
                let mut start_block = self.func.inst_block(prev_use.inst.next());
                let end_block = self.func.inst_block(u.inst);
                if start_block != end_block {
                    while let Some(lower_freq_block) =
                        self.split_placement.next_lower_freq(start_block)
                    {
                        if lower_freq_block > end_block {
                            break;
                        }

                        let to = self.func.block_insts(lower_freq_block).from;
                        splitter.gaps.push(SplitGap {
                            range: InstRange::new(from, to),
                            weight,
                            live_insts: 0,
                            min_freq: self.func.block_frequency(start_block),
                        });
                        from = to;
                        weight = 0.0;
                        start_block = lower_freq_block;
                    }
                }
                splitter.gaps.push(SplitGap {
                    range: InstRange::new(from, u.inst.next()),
                    weight: weight + u.left_weight + u.weight,
                    live_insts: 0,
                    min_freq: self.func.block_frequency(start_block),
                });
            }

            if idx == best_use {
                initial_gap = Some(splitter.gaps.len());
                splitter.gaps.push(SplitGap {
                    range: InstRange::new(u.inst, u.inst.next()),
                    weight: u.weight,
                    live_insts: 0,
                    min_freq: self.func.block_frequency(self.func.inst_block(u.inst)),
                });
            }
            prev_use = u;
        }

        trace!("Gaps between uses:");
        for (idx, gap) in splitter.gaps.iter().enumerate() {
            trace!(
                "  {idx}: {} weight={} min_freq={}{}",
                gap.range,
                gap.weight,
                gap.min_freq,
                if Some(idx) == initial_gap { " <==" } else { "" },
            );
        }

        debug_assert!(
            splitter
                .gaps
                .is_sorted_by(|a, b| a.range.to == b.range.from && a.range.from < b.range.to)
        );
        for gap in &splitter.gaps {
            debug_assert!(gap.range.from < gap.range.to);
        }

        initial_gap.unwrap()
    }

    /// Counts the number of live instruction in each gap.
    fn count_gap_live_insts(&mut self, vreg: VirtReg, initial_gap: usize) {
        let splitter = &mut self.allocator.splitter;
        let mut gaps = &mut splitter.gaps[..];
        let mut prev_segment_end = Inst::new(0);
        for segment in self.virt_regs.segments(vreg) {
            let seg_start = prev_segment_end.max(segment.live_range.from.inst());
            let seg_end = segment.live_range.to.round_to_next_inst().inst();

            while !gaps.is_empty() {
                if gaps[0].range.to <= seg_start {
                    gaps = &mut gaps[1..];
                    continue;
                }
                if gaps[0].range.from >= seg_end {
                    break;
                }
                let start = seg_start.max(gaps[0].range.from);
                let end = seg_end.min(gaps[0].range.to);
                gaps[0].live_insts += end.index() as u32 - start.index() as u32;
                if gaps[0].range.to > seg_end {
                    break;
                }
                gaps = &mut gaps[1..];
            }

            prev_segment_end = seg_end;
        }

        trace!("Gaps after counting live instructions:");
        for (idx, gap) in splitter.gaps.iter().enumerate() {
            trace!(
                "  {idx}: {} weight={} min_freq={} live_insts={}{}",
                gap.range,
                gap.weight,
                gap.min_freq,
                gap.live_insts,
                if idx == initial_gap { " <==" } else { "" },
            );
        }

        // Ensure the total live instruction count matches the one used by the
        // spill cost calculation function.
        debug_assert_eq!(
            splitter.gaps.iter().map(|gap| gap.live_insts).sum::<u32>(),
            ValueSegment::live_insts(self.virt_regs.segments(vreg))
        );
    }

    /// Collects the maximum spill weight of interference that needs to be
    /// evicted from a virtual register in order to allocate each gap.
    fn collect_gap_interference(
        reg: PhysReg,
        vreg: VirtReg,
        splitter: &mut Splitter,
        reg_matrix: &RegMatrix,
        virt_regs: &VirtRegs,
        reginfo: &impl RegInfo,
    ) {
        splitter.gap_interference_weights.clear();
        splitter
            .gap_interference_weights
            .resize(splitter.gaps.len(), 0.0);

        for interference in reg_matrix.interference(vreg, reg, virt_regs, reginfo) {
            // Fixed interference has a infinite spill weight since it cannot be
            // evicted.
            let weight = match interference.kind {
                InterferenceKind::Fixed => f32::INFINITY,
                InterferenceKind::VirtReg(virt_reg) => virt_regs[virt_reg].spill_weight,
            };

            let from = splitter
                .gaps
                .partition_point(|gap| gap.range.to <= interference.range.from.inst());
            for (idx, gap) in splitter.gaps.iter().enumerate().skip(from) {
                if gap.range.from >= interference.range.to.round_to_next_inst().inst() {
                    break;
                }
                splitter.gap_interference_weights[idx] =
                    splitter.gap_interference_weights[idx].max(weight);
            }
        }

        trace!("Gap interference for {reg}:");
        for (idx, weight) in splitter.gap_interference_weights.iter().enumerate() {
            trace!("  {idx}: {weight}");
        }
    }

    /// Builds a proposed split region starting from the best use that is small
    /// enough to evict any interference in the given physical register.
    fn find_split_region(&self, reg: PhysReg, initial_gap: usize) -> Option<SplitProposal> {
        let splitter = &self.allocator.splitter;

        let mut left = initial_gap;
        let mut right = initial_gap;
        let mut can_grow_left = left != 0;
        let mut can_grow_right = right != splitter.gaps.len() - 1;
        let mut insts = 1;
        let mut weight = splitter.gaps[initial_gap].weight;
        debug_assert_ne!(weight, 0.0);
        let mut interference_weight = splitter.gap_interference_weights[initial_gap];

        // Adjustment to apply to our estimated spill weight to avoid issues
        // with float precision. It's fine to under-estimate our spill
        // weight but we mustn't over-estimate it otherwise the new split
        // won't actually be able to evict interference.
        const FLOAT_PRECISION_ADJUST: f32 = 2007.0 / 2048.0; // 0.97998046875

        // If the initial gap has too much inteference to allocate then we can't
        // split around it with this register.
        if weight * FLOAT_PRECISION_ADJUST <= interference_weight {
            trace!(
                "Can't split around initial gap: weight={weight} interference_weight={interference_weight}"
            );
            return None;
        }

        loop {
            // Try to grow the region left or right preferring the gap with the
            // highest minimum frequency. Stop once we can't grow any more.
            let grow_left = match (can_grow_left, can_grow_right) {
                (true, true) => {
                    if splitter.gaps[left - 1].min_freq == splitter.gaps[right + 1].min_freq {
                        // In case of a tie, prefer growing to the gap with
                        // more instructions first. This empirically produces
                        // better allocation results.
                        splitter.gaps[left - 1].live_insts >= splitter.gaps[right + 1].live_insts
                    } else {
                        splitter.gaps[left - 1].min_freq >= splitter.gaps[right + 1].min_freq
                    }
                }
                (true, false) => true,
                (false, true) => false,
                (false, false) => break,
            };
            let gap_idx = if grow_left { left - 1 } else { right + 1 };

            // Compute the weights for the extended split.
            let new_insts = insts + splitter.gaps[gap_idx].live_insts;
            let new_weight = weight + splitter.gaps[gap_idx].weight;
            let estimated_spill_weight =
                (new_weight / new_insts as f32).min(f32::MAX) * FLOAT_PRECISION_ADJUST;
            let new_interference_weight =
                interference_weight.max(splitter.gap_interference_weights[gap_idx]);

            // Check if we can grow the region: we must be able to evict any
            // interfering virtual registers in the selected region.
            if estimated_spill_weight > new_interference_weight {
                if grow_left {
                    left -= 1;
                    can_grow_left = left != 0;
                } else {
                    right += 1;
                    can_grow_right = right != splitter.gaps.len() - 1;
                }
                weight = new_weight;
                interference_weight = new_interference_weight;
                insts = new_insts;
            } else {
                if grow_left {
                    can_grow_left = false;
                } else {
                    can_grow_right = false;
                }
            }
        }

        // Estimate the split cost based on the block frequencies just before
        // and after the split.
        let (left, left_split_cost) = if left != 0 {
            (
                Some(splitter.gaps[left].range.from),
                splitter.gaps[left - 1].min_freq,
            )
        } else {
            (None, 0.0)
        };
        let (right, right_split_cost) = if right != splitter.gaps.len() - 1 {
            (
                Some(splitter.gaps[right].range.to),
                splitter.gaps[right + 1].min_freq,
            )
        } else {
            (None, 0.0)
        };

        Some(SplitProposal {
            reg,
            left,
            right,
            use_weight: weight,
            live_insts: insts,
            interference_weight,
            split_cost: left_split_cost + right_split_cost,
        })
    }

    /// Actually perform the split at the given split points.
    ///
    /// `hint` is applied to the middle portion between `left` and `right`.
    fn do_split(
        &mut self,
        vreg: VirtReg,
        left: Option<Inst>,
        right: Option<Inst>,
        hint: PhysReg,
        interference_weight: f32,
    ) {
        trace!("Splitting {vreg}");
        stat!(self.stats, split_vregs);

        let splitter = &mut self.allocator.splitter;
        splitter.new_vregs.clear();
        splitter.segments.clear();
        splitter
            .segments
            .extend_from_slice(self.virt_regs.segments(vreg));

        // Helper function to create new virtual regsiters and initialize them
        // to unassigned with the given hint.
        let mut create_vregs =
            |segments: &mut [ValueSegment], uses: &mut Uses, hint: Option<PhysReg>| {
                self.virt_regs.create_vreg_from_segments(
                    segments,
                    self.func,
                    self.reginfo,
                    uses,
                    self.hints,
                    self.virt_reg_builder,
                    self.coalescing,
                    self.stats,
                    &mut splitter.new_vregs,
                );
                self.allocator
                    .assignments
                    .grow_to_with(self.virt_regs.num_virt_regs(), || Assignment::Unassigned {
                        evicted_for_preference: false,
                        hint: hint.into(),
                    });
            };

        let mut segments = &mut splitter.segments[..];
        if let Some(left) = left {
            let mut split = ValueSegment::split_segments_at(segments, self.uses, self.hints, left);
            create_vregs(split.first_half(), self.uses, None);
            segments = split.into_second_half();
        }

        if let Some(right) = right {
            let mut split2 =
                ValueSegment::split_segments_at(segments, self.uses, self.hints, right);
            create_vregs(split2.first_half(), self.uses, Some(hint));
            create_vregs(split2.into_second_half(), self.uses, None);
        } else {
            create_vregs(segments, self.uses, Some(hint));
        }

        // At least one of the new virtual registers must be able to evict the
        // interference, otherwise we aren't making progress.
        debug_assert!(splitter.new_vregs.iter().any(|&vreg| {
            self.virt_regs[vreg].spill_weight > interference_weight
                || ValueSegment::live_insts(self.virt_regs.segments(vreg)) <= 1
        }));

        // The original vreg is no longer used after this point.
        self.allocator.assignments[vreg] = Assignment::Dead;

        self.queue_new_vregs();
    }

    /// Splits the given virtual register into smaller pieces.
    ///
    /// Given a virtual register which cannot be allocated due to interference
    /// and which has too low of a spill weight to evict other interfering
    /// virtual registers, attempt to split it into smaller pieces that are
    /// alloctable.
    pub(super) fn split_or_spill(&mut self, vreg: impl AbstractVirtRegGroup) {
        if self.split_strategy == SplitStrategy::Spill {
            self.spill(vreg);
            return;
        }

        // We can't split register groups so instead isolate all group uses into
        // minimal segments and then try again with the remaining segments.
        if let VirtRegOrGroup::Group(group) = vreg.into() {
            self.isolate_group_uses(group);
            return;
        }
        let vreg = vreg.first_vreg(self.virt_regs);

        // If this virtual register has a spill weight of 0 then it
        // can't evict any interference and must be spilled. Such virtual
        // registers have no uses or only uses that don't care about being on
        // the stack.
        if self.virt_regs[vreg].spill_weight == 0.0 {
            stat!(self.stats, spill_weight_zero);
            self.spill(vreg);
            return;
        }

        // Collect a linear list of all the places where the virtual register is
        // used, with associated total spill weights.
        self.collect_uses(vreg);
        stat!(
            self.stats,
            num_split_uses,
            self.allocator.splitter.uses.len()
        );

        // Find the "best" use (highest weight) to build a split region around.
        let Some(best_use) = self.find_best_use() else {
            self.spill(vreg);
            return;
        };

        // Collect information about gaps between uses.
        let initial_gap = self.collect_gaps(best_use);
        stat!(
            self.stats,
            num_split_gaps,
            self.allocator.splitter.gaps.len()
        );

        // Count the number of live instructions in each gap.
        self.count_gap_live_insts(vreg, initial_gap);

        let mut best_split = None;
        for candidate in self.allocator.allocation_order.order() {
            Self::collect_gap_interference(
                candidate.reg.as_single(),
                vreg,
                &mut self.allocator.splitter,
                self.reg_matrix,
                self.virt_regs,
                self.reginfo,
            );
            if let Some(new_split) = self.find_split_region(candidate.reg.as_single(), initial_gap)
            {
                trace!("Proposed split: {new_split:?}");
                if best_split
                    .as_ref()
                    .is_none_or(|best_split: &SplitProposal| new_split.score() > best_split.score())
                {
                    best_split = Some(new_split);
                }
            } else {
                stat!(self.stats, unevictable_initial_gap);
            }
        }
        let Some(best_split) = best_split else {
            trace!("Best use couldn't be evicted on any register, forcing a spill");
            stat!(self.stats, no_best_split);
            self.spill(vreg);
            return;
        };
        trace!("Best split: {best_split:?}");

        // If we returned the whole range then it means that we can actually
        // evict the interference immediately. Attempt to do so.
        //
        // This can happen due to evictions between now and when this register
        // was previously processed for the evict stage.
        if best_split.left.is_none() && best_split.right.is_none() {
            trace!("Null split proposed, try evicting inference instead of splitting");
            stat!(self.stats, evict_for_null_split);
            if !self.try_evict(vreg) {
                panic!("Failed to evict {vreg} when interference has lower weight");
            }
            return;
        }

        // If the register class for this virtual register allows allocation to
        // a spill slot then only split if this is more profitable than
        // spilling.
        if self
            .reginfo
            .class_includes_spillslots(self.virt_regs[vreg].class)
        {
            let mut spill_cost = 0.0;
            for segment in self.virt_regs.segments(vreg) {
                for &u in &self.uses[segment.use_list] {
                    let block_freq = self.func.block_frequency(self.func.inst_block(u.pos));
                    spill_cost += u.spill_cost(self.reginfo) * block_freq;
                }
            }
            trace!("{vreg} is directly spillable with a spill cost of {spill_cost}");
            if spill_cost < best_split.split_cost {
                stat!(self.stats, spill_cheaper_than_split);
                self.spill(vreg);
                return;
            }
        } else {
            trace!("{vreg} is not directly spillable");
        }

        // Finally perform the actual split.
        self.do_split(
            vreg,
            best_split.left,
            best_split.right,
            best_split.reg,
            best_split.interference_weight,
        );
    }

    // Invalidate any existing ValueGroup mappings for a vreg: the group they
    // point to will no longer be valid after it is spilled. New mappings will
    // be created when building the new virtual registers for the split product.
    fn invalidate_value_group_mapping(&mut self, vreg: impl AbstractVirtRegGroup) {
        if vreg.is_group() {
            let vreg = vreg.first_vreg(self.virt_regs);
            for segment in self.virt_regs.segments(vreg) {
                for &u in &self.uses[segment.use_list] {
                    if let UseKind::GroupClassUse {
                        slot,
                        class: _,
                        group_index: _,
                    }
                    | UseKind::GroupClassDef {
                        slot,
                        class: _,
                        group_index: _,
                    } = u.kind
                    {
                        let value_group = match self.func.inst_operands(u.pos)[slot as usize].kind()
                        {
                            OperandKind::DefGroup(group)
                            | OperandKind::UseGroup(group)
                            | OperandKind::EarlyDefGroup(group) => group,
                            OperandKind::Def(_)
                            | OperandKind::Use(_)
                            | OperandKind::EarlyDef(_)
                            | OperandKind::NonAllocatable => unreachable!(),
                        };
                        self.virt_reg_builder
                            .invalidate_value_group_mapping(value_group);
                    }
                }
            }
        }
    }

    /// Isolates all group uses in the given register group by splitting them
    /// off into minimal segments that only cover a single instruction (or part
    /// of one) and are therefore always allocatable.
    ///
    /// The remaining live range pieces are collected into a separate non-group
    /// virtual register for each original group member, which can then be split
    /// as normal.
    fn isolate_group_uses(&mut self, group: VirtRegGroup) {
        self.invalidate_value_group_mapping(group);

        // Find any group uses and split them into minimal segments
        let splitter = &mut self.allocator.splitter;
        splitter.minimal_segments.clear();
        splitter.new_vregs.clear();
        for group_index in 0..self.virt_regs.group_members(group).len() {
            splitter.segments.clear();
            let vreg = self.virt_regs.group_members(group)[group_index];
            trace!("Isolating group uses in {vreg}");
            stat!(self.stats, isolated_group_vregs);
            for &segment in self.virt_regs.segments(vreg) {
                let mut segment = segment;
                'outer: loop {
                    for &u in &self.uses[segment.use_list] {
                        match u.kind {
                            UseKind::GroupClassUse {
                                slot: _,
                                class: _,
                                group_index: use_group_index,
                            }
                            | UseKind::GroupClassDef {
                                slot: _,
                                class: _,
                                group_index: use_group_index,
                            } => {
                                debug_assert_eq!(use_group_index as usize, group_index);
                            }
                            UseKind::ClassUse { .. }
                            | UseKind::ClassDef { .. }
                            | UseKind::FixedDef { .. }
                            | UseKind::FixedUse { .. }
                            | UseKind::TiedUse { .. }
                            | UseKind::ConstraintConflict { .. }
                            | UseKind::BlockparamIn { .. }
                            | UseKind::BlockparamOut { .. } => continue,
                        };

                        trace!("Splitting around group use {}: {}", u.pos, u.kind);

                        // If there is a live range segment with no uses before
                        // this use, collect that segment for the non-group
                        // portion.
                        if u.pos != segment.live_range.from.round_to_prev_inst().inst() {
                            let (before, after) = segment.split_at(u.pos, self.uses, self.hints);
                            splitter.segments.push(before);
                            segment = after;
                        }

                        if u.pos.next() == segment.live_range.to.round_to_next_inst().inst() {
                            // If this use is on the last instruction of the segment
                            // then split off the rest of the segment into a
                            // minimal segment.
                            trace!(
                                "Generating minimal segment for {} at {}",
                                segment.value, segment.live_range
                            );
                            splitter.minimal_segments.push(segment);
                            break 'outer;
                        } else {
                            // Otherwise split the segment at the next
                            // instruction boundary after the use. The first
                            // half is split as a minimal segment and continue
                            // processing the remaining half.
                            let (before, after) =
                                segment.split_at(u.pos.next(), self.uses, self.hints);
                            trace!(
                                "Generating minimal segment for {} at {}",
                                before.value, before.live_range
                            );
                            splitter.minimal_segments.push(before);
                            segment = after;
                            continue 'outer;
                        }
                    }

                    // If there are no uses left in the segment, add it to the
                    // non-group portion of the virtual register.
                    splitter.segments.push(segment);
                    break;
                }
            }

            // The original vreg is no longer used after this point.
            self.allocator.assignments[vreg] = Assignment::Dead;

            // Build a new virtual register with the non-group parts of the
            // live range.
            if !splitter.segments.is_empty() {
                self.virt_regs.create_vreg_from_segments(
                    &mut splitter.segments,
                    self.func,
                    self.reginfo,
                    self.uses,
                    self.hints,
                    self.virt_reg_builder,
                    self.coalescing,
                    self.stats,
                    &mut splitter.new_vregs,
                );
            }
        }

        // Create a new virtual register for each minimal segment that was created.
        splitter.minimal_segments.iter().for_each(|&segment| {
            stat!(self.stats, isolated_group_minimal_segments);
            self.virt_regs.create_vreg_from_segments(
                &mut [segment],
                self.func,
                self.reginfo,
                self.uses,
                self.hints,
                self.virt_reg_builder,
                self.coalescing,
                self.stats,
                &mut splitter.new_vregs,
            );
        });

        // Initialize assignments for the new virtual registers.
        self.allocator
            .assignments
            .grow_to(self.virt_regs.num_virt_regs());

        self.queue_new_vregs();
    }

    /// Spills the given virtual register to a spillslot.
    ///
    /// Any uses which must be in a register are split off into separate virtual
    /// registers which only cover a single instruction and therefore are
    /// unspillable.
    pub(super) fn spill(&mut self, vreg: impl AbstractVirtRegGroup) {
        // TODO(perf): Fast path if class allows spillslots?

        self.invalidate_value_group_mapping(vreg);

        // Find any uses that can't be spilled and split them into minimal
        // segments that only cover a single instruction (or part of one).
        //
        // Any remaining live range between these minimal segments is spilled to
        // the stack.
        let splitter = &mut self.allocator.splitter;
        splitter.minimal_segments.clear();
        for (group_index, vreg) in vreg.vregs(self.virt_regs).enumerate() {
            trace!("Spilling {vreg}");
            stat!(self.stats, spilled_vregs);
            for &segment in self.virt_regs.segments(vreg) {
                // If the value of that segment is rematerializable then we
                // don't need to spill it. This will result in it having no
                // allocation, which the move resolver will handle by
                // rematerializing the value.
                let can_remat = self.func.can_rematerialize(segment.value).is_some();

                let mut segment = segment;
                'outer: loop {
                    let mut must_spill = false;
                    for &u in &self.uses[segment.use_list] {
                        // Ignore uses that can be assigned to spill slots.
                        let can_spill = match u.kind {
                            UseKind::ClassUse { slot: _, class }
                            | UseKind::ClassDef { slot: _, class } => {
                                if self.reginfo.class_includes_spillslots(class) {
                                    // Even if the value is rematerializable, we
                                    // must *still* spill it because ClassUse
                                    // and ClassDef constraints need an
                                    // allocation.
                                    must_spill = true;
                                    true
                                } else {
                                    false
                                }
                            }
                            UseKind::GroupClassUse {
                                slot: _,
                                class,
                                group_index: use_group_index,
                            }
                            | UseKind::GroupClassDef {
                                slot: _,
                                class,
                                group_index: use_group_index,
                            } => {
                                debug_assert!(!self.reginfo.class_includes_spillslots(class));
                                debug_assert_eq!(use_group_index as usize, group_index);
                                false
                            }
                            UseKind::FixedDef { .. }
                            | UseKind::FixedUse { .. }
                            | UseKind::TiedUse { .. }
                            | UseKind::ConstraintConflict { .. }
                            | UseKind::BlockparamIn { .. }
                            | UseKind::BlockparamOut { .. } => true,
                        };
                        if can_spill {
                            trace!(
                                "Spillable use of {} at {}: {}",
                                segment.value, u.pos, u.kind
                            );
                            continue;
                        }

                        trace!("Splitting around unspillable use {}: {}", u.pos, u.kind);

                        // If there is a live range segment with no uses before
                        // this use, spill that segment.
                        if u.pos != segment.live_range.from.round_to_prev_inst().inst() {
                            let (before, after) = segment.split_at(u.pos, self.uses, self.hints);
                            if must_spill || !can_remat {
                                let set = self.coalescing.set_for_value(before.value);
                                self.spill_allocator.spill_segment(set, before);
                            } else {
                                trace!(
                                    "Rematerializing segment for {} at {}",
                                    before.value, before.live_range
                                );
                                self.allocator.remat_segments.push(before);
                            }
                            segment = after;
                        }

                        if u.pos.next() == segment.live_range.to.round_to_next_inst().inst() {
                            // If this use is on the last instruction of the segment
                            // then split off the rest of the segment into a
                            // minimal segment.
                            trace!(
                                "Generating minimal segment for {} at {}",
                                segment.value, segment.live_range
                            );
                            splitter.minimal_segments.push(segment);
                            break 'outer;
                        } else {
                            // Otherwise split the segment at the next
                            // instruction boundary after the use. The first
                            // half is split as a minimal segment and continue
                            // processing the remaining half.
                            let (before, after) =
                                segment.split_at(u.pos.next(), self.uses, self.hints);
                            trace!(
                                "Generating minimal segment for {} at {}",
                                before.value, before.live_range
                            );
                            splitter.minimal_segments.push(before);
                            segment = after;
                            continue 'outer;
                        }
                    }

                    // If there are no uses left in the segment, spill it. The
                    // live range must be non-empty at this point, this is
                    // checked in spill_segment.
                    if must_spill || !can_remat {
                        let set = self.coalescing.set_for_value(segment.value);
                        self.spill_allocator.spill_segment(set, segment);
                    } else {
                        trace!(
                            "Rematerializing segment for {} at {}",
                            segment.value, segment.live_range
                        );
                        self.allocator.remat_segments.push(segment);
                    }
                    break;
                }
            }

            // The original vreg is no longer used after this point.
            self.allocator.assignments[vreg] = Assignment::Dead;
        }

        // Create a new virtual register for each minimal segment that was
        // created.
        splitter.new_vregs.clear();
        splitter.minimal_segments.iter().for_each(|&segment| {
            stat!(self.stats, spill_minimal_segments);
            self.virt_regs.create_vreg_from_segments(
                &mut [segment],
                self.func,
                self.reginfo,
                self.uses,
                self.hints,
                self.virt_reg_builder,
                self.coalescing,
                self.stats,
                &mut splitter.new_vregs,
            );
        });

        // Initialize assignments for the new virtual registers.
        self.allocator
            .assignments
            .grow_to(self.virt_regs.num_virt_regs());

        self.queue_new_vregs();
    }

    fn queue_new_vregs(&mut self) {
        // Then queue the newly created virtual registers. This needs to be done
        // after all split products are created so that virtual register groups
        // are complete.
        for &vreg in &self.allocator.splitter.new_vregs {
            trace!("Created split product {vreg}");

            // If the virtual register is part of a group, enqueue the whole
            // group instead.
            let vreg_or_group = if let Some(group) = self.virt_regs[vreg].group.expand() {
                // Only do this once for the group leader.
                if self.virt_regs[vreg].group_index != 0 {
                    continue;
                }
                VirtRegOrGroup::Group(group)
            } else {
                VirtRegOrGroup::Reg(vreg)
            };

            trace!("Queuing split product {vreg_or_group}");
            self.allocator
                .queue
                .enqueue(vreg_or_group, Stage::Evict, self.virt_regs);
        }
    }
}
