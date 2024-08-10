//! This pass generates the move instructions that connect the virtual register
//! allocations.
//!
//! The algorithm here is based on the concept of half-moves from regalloc2.
//! The basic idea to emit the source of a move separately from the destination
//! of a move, and key both to a `(inst, value)` so they can find each other.
//!
//! All the information necessary to produce half-moves is in the use list
//! associated with every virtual register and spilled segments, so we just scan
//! through them in a single pass to generate half-moves and assign allocations
//! to instruction operands.
//!
//! Once all half-moves are produced, they are collected at each instruction
//! boundaries and passed to the parallel move resolver which will emit a
//! sequence of move instructions to be emitted between the two instructions.

use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::hash::{Hash, Hasher};

use cranelift_entity::packed_option::PackedOption;
use cranelift_entity::EntityRef;
use hashbrown::HashMap;
use rustc_hash::FxBuildHasher;

use super::allocations::Allocations;
use super::allocator::Allocator;
use super::live_range::Slot;
use super::parallel_moves::ParallelMoves;
use super::reg_matrix::RegMatrix;
use super::spill_allocator::SpillAllocator;
use super::uses::{Use, UseKind, Uses};
use super::value_live_ranges::ValueSegment;
use super::virt_regs::VirtRegs;
use crate::function::{Block, Function, Inst, Value};
use crate::internal::live_range::LiveRangeSegment;
use crate::output::{Allocation, AllocationKind};
use crate::reginfo::{RegClass, RegInfo};
use crate::{MoveOptimizationLevel, Stats};

/// Position in which to insert a move.
///
/// All moves occur at the boundary before an instruction. However some moves
/// must happen before other moves. This is specifically useful when moves need
/// to be inserted both at the start of a block for live-in values and before
/// the first instruction in the block to satisfy constraints of that instruction.
///
/// When connecting multiple predecessor blocks to a single successor, moves
/// must occur before the jump instruction in the predecessor blocks (this is
/// why such jump instructions are not allowed to have operands).
///
/// The problem is that this jump instruction may have other moves before it.
/// These prior moves *must* be processed before any moves used to resolve
/// block-to-block liveness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MovePosition {
    /// Bit-pack in 32 bits.
    ///
    /// inst:31 pos:1
    bits: u32,
}

impl fmt::Display for MovePosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.bits & 1 == 0 {
            write!(f, "{}-early", self.inst())
        } else {
            write!(f, "{}-late", self.inst())
        }
    }
}

impl MovePosition {
    /// An early move.
    ///
    /// This is the default used for almost all moves.
    fn early(inst: Inst) -> Self {
        Self {
            bits: (inst.index() as u32) << 1,
        }
    }

    /// A late move.
    ///
    /// This is only used for moves on a jump instruction where the successor
    /// block has multiple predecessors.
    ///
    /// Such moves still happen before the instruction itself, it's just that
    /// they are ordered to happen after early moves.
    fn late(inst: Inst) -> Self {
        Self {
            bits: ((inst.index() as u32) << 1) | 1,
        }
    }

    /// Whether this is a late move on a jump instruction.
    fn is_late(self) -> bool {
        self.bits & 1 != 0
    }

    /// Instruction before which the move must be placed.
    fn inst(self) -> Inst {
        Inst::new((self.bits >> 1) as usize)
    }
}

/// An edit represents either a move between 2 locations or a rematerialization
/// of a value into a location.
///
/// Valid combinations are:
/// - Move: value:Some from:Some to:Some
/// - Emergency spill: value:None from:Some(reg) to:None(spillslot)
/// - Emergency reload: value:None from:Some(spillslot) to:None(reg)
/// - Rematerialization: value:Some from:None to:Some
///
/// If `to` is `None` then it means the entire edit has be optimized away to a
/// nop. This is only done in the move optimization pass.
#[derive(Debug, Clone, Copy)]
pub struct Edit {
    pub value: PackedOption<Value>,
    pub from: PackedOption<Allocation>,
    pub to: PackedOption<Allocation>,
}

impl fmt::Display for Edit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(to) = self.to.expand() {
            if let Some(from) = self.from.expand() {
                write!(f, "move {:?} from {} to {to}", self.value, from,)
            } else {
                write!(f, "remat {} in {to}", self.value.unwrap())
            }
        } else {
            f.write_str("nop")
        }
    }
}

/// Information needed to emit a dest half-move for a pair of tied operands.
///
/// This needs to be deferred until all allocations have been assigned so that
/// we can find the allocation that was assigned to the def operand slot.
#[derive(Debug)]
struct TiedMove {
    move_pos: MovePosition,
    inst: Inst,
    value: Value,
    def_slot: u16,
    class: RegClass,
    group_index: u8,
    is_blockparam: bool,
}

#[derive(Debug)]
struct TiedOperands {
    inst: Inst,
    def_slot: u16,
    use_slot: u16,
}

// Key for identifying matching half-move pairs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HalfMoveKey {
    pos: MovePosition,
    value: Value,
}

// Custom hasher which hashes a single u64 instead of 2 u32
impl Hash for HalfMoveKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let key = self.pos.bits as u64 | ((self.value.index() as u64) << 32);
        key.hash(state);
    }
}

pub struct MoveResolver {
    source_half_moves: HashMap<HalfMoveKey, Allocation, FxBuildHasher>,
    dest_half_moves: Vec<(MovePosition, Value, Allocation)>,
    tied_moves: Vec<TiedMove>,
    tied_operands: Vec<TiedOperands>,
    edits: Vec<(Inst, Edit)>,
    blockparam_allocs: Vec<(Block, Value, Allocation)>,
    parallel_move_resolver: ParallelMoves,
}

impl MoveResolver {
    pub fn new() -> Self {
        Self {
            source_half_moves: HashMap::default(),
            dest_half_moves: vec![],
            tied_moves: vec![],
            tied_operands: vec![],
            edits: vec![],
            blockparam_allocs: vec![],
            parallel_move_resolver: ParallelMoves::new(),
        }
    }

    /// Processes all virtual register segments to assign allocations to the
    /// output vector and generates move instruction that connect live ranges
    /// together.
    pub fn generate_moves(
        &mut self,
        allocator: &Allocator,
        virt_regs: &VirtRegs,
        spill_allocator: &mut SpillAllocator,
        uses: &Uses,
        allocations: &mut Allocations,
        reg_matrix: &RegMatrix,
        stats: &mut Stats,
        func: &impl Function,
        reginfo: &impl RegInfo,
        move_optimization: MoveOptimizationLevel,
    ) {
        self.source_half_moves.clear();
        self.dest_half_moves.clear();
        self.tied_moves.clear();
        self.tied_operands.clear();
        self.blockparam_allocs.clear();

        let mut ctx = Context {
            func,
            move_resolver: self,
            uses,
            allocations,
            live_in: None,
            fixed_def: None,
        };

        trace!("Adding half-moves from vregs");
        for (vreg, reg) in allocator.assignments() {
            trace!("Adding half-moves from {vreg}");
            for segment in virt_regs[vreg].segments(virt_regs) {
                ctx.process_segment(segment, Some(Allocation::reg(reg)));
            }
        }

        trace!("Adding half-moves from spilled segments");
        for (spillslot, segment) in spill_allocator.spilled_segments() {
            ctx.process_segment(segment, Some(Allocation::spillslot(spillslot)));
        }

        trace!("Adding half-moves from empty segments");
        for segment in &allocator.empty_segments {
            ctx.process_segment(segment, None);
        }

        trace!("Adding half-moves from rematerialized segments");
        for segment in &allocator.remat_segments {
            self.process_remat_segment(segment, uses);
        }

        trace!("Adding half-moves from tied uses");
        for tied in &self.tied_moves {
            // This should be filled in by now since all segments have been
            // processed.
            let mut def_alloc = allocations.inst_allocations(tied.inst)[tied.def_slot as usize];

            // We need to do some extra work for tied groups to find the
            // corresponding allocation, but this is a rare case in practice
            // anyways.
            if tied.group_index != 0 {
                let AllocationKind::PhysReg(reg) = def_alloc.kind() else {
                    unreachable!("Group use should not be tied to a spill slot")
                };
                let group = reginfo
                    .group_for_reg(reg, 0, tied.class)
                    .expect("invalid reg group for group tied def");
                def_alloc =
                    Allocation::reg(reginfo.reg_group_members(group)[tied.group_index as usize]);
            }

            trace!(
                "Dest half-move at {}: {} to {def_alloc}",
                tied.move_pos,
                tied.value
            );
            self.dest_half_moves
                .push((tied.move_pos, tied.value, def_alloc));

            if tied.is_blockparam {
                // Record the allocation assigned to the block parameter
                // for the move optimizer.
                let block = func.inst_block(tied.inst);
                self.blockparam_allocs.push((block, tied.value, def_alloc));
            }
        }

        // Copy tied def allocations to the corresponding use slot.
        for tied in &self.tied_operands {
            let alloc = allocations.inst_allocations(tied.inst)[tied.def_slot as usize];
            allocations.set_allocation(tied.inst, tied.use_slot, alloc);
        }

        self.resolve_moves(
            spill_allocator,
            allocations,
            reg_matrix,
            stats,
            func,
            reginfo,
        );

        // The move optimizer needs per-block information on incoming
        // blockparams for each block, so ensure this is properly sorted.
        if move_optimization != MoveOptimizationLevel::Off {
            self.blockparam_allocs
                .sort_unstable_by_key(|&(block, _, _)| block);
        } else {
            self.blockparam_allocs.clear();
        }
    }

    /// After all half-moves have been generated, resolve half-move pairs into
    /// full moves at each instruction boundary.
    ///
    /// Because these moves logically happen in parallel, we need to resolve the
    /// set of parallel moves into a sequence of move operations, possibly using
    /// one or more scratch registers.
    fn resolve_moves(
        &mut self,
        spill_allocator: &mut SpillAllocator,
        allocations: &mut Allocations,
        reg_matrix: &RegMatrix,
        stats: &mut Stats,
        func: &impl Function,
        reginfo: &impl RegInfo,
    ) {
        self.edits.clear();

        // Ensure that all allocations are filled in at this point.
        allocations.assert_all_assigned();

        self.parallel_move_resolver.clear();
        self.dest_half_moves
            .sort_unstable_by_key(|&(pos, _, _)| pos);
        for half_moves in self.dest_half_moves.chunk_by(|a, b| a.0 == b.0) {
            let pos = half_moves[0].0;
            trace!("Processing parallel moves at {pos}:");

            self.parallel_move_resolver.new_parallel_move();
            for &(_, value, dest) in half_moves {
                let source = self
                    .source_half_moves
                    .get(&HalfMoveKey { pos, value })
                    .copied();
                if let Some(source) = source {
                    if source != dest {
                        trace!("- Move {value} from {source} to {dest}");
                        self.parallel_move_resolver
                            .add_move(source, dest, value, func, reginfo);
                    }
                } else {
                    trace!("- Remat {value} in {dest}");
                    self.parallel_move_resolver
                        .add_remat(dest, value, func, reginfo);
                }
            }

            self.parallel_move_resolver.resolve(
                reginfo,
                func,
                |unit| {
                    // If this is a late move then this is on a jump instruction
                    // and we need to check whether the unit is free in the
                    // successor block.
                    let inst = if pos.is_late() {
                        let block = func.inst_block(pos.inst());
                        let succ = func.block_succs(block)[0];
                        func.block_insts(succ).from
                    } else {
                        pos.inst()
                    };
                    reg_matrix.is_unit_free(
                        unit,
                        LiveRangeSegment::new(inst.slot(Slot::Boundary), inst.slot(Slot::Early)),
                    )
                },
                |size| spill_allocator.alloc_emergency_spillslot(size),
            );

            trace!("Resolved sequential moves at {pos}:");
            self.edits
                .extend(self.parallel_move_resolver.edits().map(|edit| {
                    trace!("- {edit}");
                    stat!(stats, edits);
                    if let Some(from) = edit.from.expand() {
                        if from.is_memory(reginfo) {
                            stat!(stats, reloads);
                        } else if edit.to.unwrap().is_memory(reginfo) {
                            stat!(stats, spills);
                        } else {
                            stat!(stats, moves);
                        }
                    } else {
                        stat!(stats, remats);
                    }

                    (pos.inst(), edit)
                }));
        }
    }

    fn emit_source_half_move(&mut self, pos: MovePosition, value: Value, alloc: Allocation) {
        trace!("Source half-move at {pos}: {value} in {alloc}");
        let prev = self
            .source_half_moves
            .insert(HalfMoveKey { pos, value }, alloc);
        debug_assert!(!prev.is_some_and(|prev| prev != alloc));
    }

    fn emit_dest_half_move(&mut self, pos: MovePosition, value: Value, alloc: Allocation) {
        trace!("Dest half-move at {pos}: {value} to {alloc}");
        self.dest_half_moves.push((pos, value, alloc));
    }

    /// Special handling for segments that are rematerialized: we only need to
    /// emit destination half-moves for fixed uses and tied uses.
    fn process_remat_segment(&mut self, segment: &ValueSegment, uses: &Uses) {
        trace!(
            "Processing rematerialized segment {} ({})",
            segment.live_range,
            segment.value
        );

        for &u in &uses[segment.use_list] {
            trace!("-> {} {}", u.pos(), u.kind);
            match u.kind {
                UseKind::FixedUse { reg } => {
                    self.emit_dest_half_move(
                        MovePosition::early(u.pos()),
                        u.value,
                        Allocation::reg(reg),
                    );
                }
                UseKind::TiedUse {
                    use_slot,
                    def_slot,
                    class,
                    group_index,
                } => {
                    // Copy the allocation of the def slot to the use slot. Only do
                    // this for the first member of a group.
                    if group_index == 0 {
                        self.tied_operands.push(TiedOperands {
                            inst: u.pos(),
                            def_slot,
                            use_slot,
                        });
                    }

                    // Rematerialize the value into the def slot.
                    self.tied_moves.push(TiedMove {
                        move_pos: MovePosition::early(u.pos()),
                        inst: u.pos(),
                        value: u.value,
                        def_slot,
                        class,
                        group_index,
                        is_blockparam: false,
                    });
                }

                // Class uses cannot be directly rematerialized: we need the
                // allocator to actually select an allocation for the slot.
                UseKind::ClassUse { slot: _, class: _ }
                | UseKind::ClassDef { slot: _, class: _ }
                | UseKind::GroupClassUse {
                    slot: _,
                    class: _,
                    group_index: _,
                }
                | UseKind::GroupClassDef {
                    slot: _,
                    class: _,
                    group_index: _,
                }
                | UseKind::StackMap { class: _ } => unreachable!("Cannot rematerialize class use"),

                // Ignore everything else.
                UseKind::FixedDef { reg: _ }
                | UseKind::ConstraintConflict {}
                | UseKind::BlockparamIn { blockparam_idx: _ }
                | UseKind::BlockparamOut {} => {}
            }
        }
    }

    /// Returns the list of edits starting from the given instruction.
    pub fn edits_from(&self, inst: Inst) -> &[(Inst, Edit)] {
        let idx = self.edits.partition_point(|&(pos, _)| pos < inst);
        &self.edits[idx..]
    }

    /// Returns the list of edits starting from the given instruction.
    pub fn edits_from_mut(&mut self, inst: Inst) -> &mut [(Inst, Edit)] {
        let idx = self.edits.partition_point(|&(pos, _)| pos < inst);
        &mut self.edits[idx..]
    }

    /// Returns the locations for block parameter values at the start of a
    /// block.
    pub fn blockparam_allocs(
        &self,
        block: Block,
    ) -> impl Iterator<Item = (Value, Allocation)> + '_ {
        let idx = self
            .blockparam_allocs
            .partition_point(|&(block2, _, _)| block2 < block);
        self.blockparam_allocs[idx..]
            .iter()
            .take_while(move |&&(block2, _, _)| block2 == block)
            .map(|&(_, value, alloc)| (value, alloc))
    }
}

// Track whether the current instruction has an external live-in value
// on entry. In such cases we don't need to emit source half-moves for
// fixed uses since the half-move is already provided elsewhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LiveInKind {
    /// Value comes from a source half-move at the start of the
    /// instruction.
    ///
    /// `from_same_segment` is true if the incoming value is from a predecessor
    /// block that is part of the the same segment, and which will therefore not
    /// have created a source half-move.
    Single { from_same_segment: bool },

    /// Value comes from late source half-moves on the terminator
    /// instructions of all predecessor blocks.
    Multi { blockparam_idx: Option<u32> },
}

struct Context<'a, F: Function> {
    func: &'a F,
    move_resolver: &'a mut MoveResolver,
    uses: &'a Uses,
    allocations: &'a mut Allocations,

    /// Indicates if the value is live-in from another segment at the start of
    /// an instruction.
    live_in: Option<(Inst, LiveInKind)>,

    /// Indicates that a fixed definition immediately precedes the given
    /// instruction. Any moves should use that fixed definition as their source.
    fixed_def: Option<(Inst, Allocation)>,
}

impl<F: Function> Context<'_, F> {
    /// Handle live-ins at the start of a block.
    fn handle_block_live_in(
        &mut self,
        inst: Inst,
        block: Block,
        segment: &ValueSegment,
        alloc: Option<Allocation>,
    ) {
        trace!("-> block livein {block}:{inst}");

        // Live-ins from multiple predecessors need special handling since the
        // moves happen in predecessor blocks.
        if let &[pred] = self.func.block_preds(block) {
            // If the segment's live range ends here then we don't need a
            // destination half move. Any uses with no live range (fixed/tied)
            // will use the incoming values directly from self.live_in.
            let mut from_same_segment = false;
            if segment.live_range.to != inst.slot(Slot::Boundary) {
                // If the predecessor block is in the same segment as us, it
                // is known to have the same value so we can avoid
                // generating a move.
                let block_end = self.func.block_insts(pred).to.slot(Slot::Boundary);
                if block_end > segment.live_range.from && block_end <= segment.live_range.to {
                    from_same_segment = true;
                } else {
                    debug_assert!(!segment.live_range.is_empty());
                    self.move_resolver.emit_dest_half_move(
                        MovePosition::early(inst),
                        segment.value,
                        alloc.expect("split live range must have an allocation"),
                    );
                }
            }
            self.live_in = Some((inst, LiveInKind::Single { from_same_segment }));
        } else {
            // If the segment's live range ends here then we don't need a
            // destination half move. Any uses with no live range (fixed/tied)
            // will use the incoming values directly from self.live_in.
            if segment.live_range.to != inst.slot(Slot::Boundary) {
                // We have an incoming value from multiple predecessor
                // blocks, emit a destination half-move before the jump
                // instruction in the predecessor blocks.
                for &pred in self.func.block_preds(block) {
                    // If the predecessor block is in the same segment as us, it
                    // is known to have the same value so we can avoid
                    // generating a move.
                    let block_end = self.func.block_insts(pred).to.slot(Slot::Boundary);
                    if !(block_end > segment.live_range.from && block_end <= segment.live_range.to)
                    {
                        debug_assert!(!segment.live_range.is_empty());
                        self.move_resolver.emit_dest_half_move(
                            MovePosition::late(self.func.block_insts(pred).last()),
                            segment.value,
                            alloc.expect("missing allocation for block live-in"),
                        );
                    }
                }
            }

            // Indicate that tied/fixed uses on the first instruction
            // should emit half-moves in predecessor blocks instead of
            // at the start of this block.
            self.live_in = Some((
                inst,
                LiveInKind::Multi {
                    blockparam_idx: None,
                },
            ));
        }
    }

    /// Handle live-outs at the end of a block.
    fn handle_block_live_out(
        &mut self,
        terminator: Inst,
        block: Block,
        segment: &ValueSegment,
        alloc: Option<Allocation>,
    ) {
        trace!("-> block liveout {block}:{terminator}");

        // If we only have a single successor and that successor has multiple
        // predecessors then this is a jump terminator.
        let succs = self.func.block_succs(block);
        if succs.len() == 1 && self.func.block_preds(succs[0]).len() > 1 {
            // If the successor block is in the same segment as us, then the
            // code in handle_block_live_in will not emit a move for it. We
            // can avoid emitting a source half-move here as well.
            let block_start = self.func.block_insts(succs[0]).from.slot(Slot::Boundary);
            if !(block_start >= segment.live_range.from && block_start < segment.live_range.to) {
                debug_assert!(!segment.live_range.is_empty());
                // For jump terminators that target a block with multiple
                // predecessors, the move must be in this block just before
                // the jump instruction.
                self.move_resolver.emit_source_half_move(
                    MovePosition::late(terminator),
                    segment.value,
                    // Values are live through the entire jump instruction,
                    // so this cannot be an empty segment.
                    alloc.expect("missing allocation for jump terminator"),
                );
            }
        } else {
            // Otherwise this is a branch terminator.

            // If this terminator produced a fixed definition, use that
            // as the move source.
            let alloc_out = match self.fixed_def {
                Some((pos, alloc)) if pos == terminator.next() => alloc,
                _ => alloc.expect("missing allocation for terminator"),
            };

            // For branch terminators that target blocks with a single
            // predecessor, the move must be at the start of each
            // successor block.
            for &succ in succs {
                // If the successor block is in the same segment as us, then the
                // code in handle_block_live_in will not emit a move for it. We
                // can avoid emitting a source half-move here as well.
                //
                // We can't do this if the block ends with a fixed definition:
                // the live range will look like it starts after the end of the
                // block, which won't trigger the check in handle_block_live_in.
                // Also it would have the wrong allocation anyways.
                let block_start = self.func.block_insts(succ).from.slot(Slot::Boundary);
                if !(block_start >= segment.live_range.from
                    && block_start < segment.live_range.to
                    && self.fixed_def.is_none())
                {
                    self.move_resolver.emit_source_half_move(
                        MovePosition::early(self.func.block_insts(succ).from),
                        segment.value,
                        alloc_out,
                    );
                }
            }
        }
    }

    /// Helper function to deal with uses that move an incoming value to another
    /// location. This figures out where the values comes from and which moves
    /// need to be inserted.
    ///
    /// If an `Allocation` is provided to the callback then the callback must
    /// emit a source half-move itself.
    fn for_each_move_source(
        &mut self,
        inst: Inst,
        segment: &ValueSegment,
        alloc: Option<Allocation>,
        mut f: impl FnMut(&mut Self, Value, MovePosition, Option<Allocation>, bool),
    ) {
        // If there is a live-in value at this point, move from the sources of
        // the live-in. These source will have already provided source
        // half-moves.
        //
        // This is necessary to handle segments with empty live ranges which
        // ontain just a tied/fixed use, and therefore have no assigned allocation.
        if let Some((_pos, kind)) = self.live_in.filter(|&(pos, _kind)| pos == inst) {
            match kind {
                LiveInKind::Single { from_same_segment } => {
                    let move_pos = MovePosition::early(inst);

                    // If the live-in is from the same segment then its source
                    // half-move will have been elided. We need to make sure to
                    // emit one here.
                    let alloc_out = if from_same_segment {
                        Some(alloc.expect("missing allocation"))
                    } else {
                        None
                    };

                    f(self, segment.value, move_pos, alloc_out, false);
                }
                LiveInKind::Multi { blockparam_idx } => {
                    for &pred in self.func.block_preds(self.func.inst_block(inst)) {
                        let mut alloc_out = None;
                        let value = match blockparam_idx {
                            Some(blockparam_idx) => {
                                self.func.jump_blockparams(pred)[blockparam_idx as usize]
                            }
                            None => {
                                // If the live-in is from the same segment then
                                // its source half-move will have been elided.
                                // We need to make sure to emit one here.
                                //
                                // This can't happen for blockparam live-ins
                                // since a segment can only hold a single value.
                                let block_end = self.func.block_insts(pred).to.slot(Slot::Boundary);
                                if block_end > segment.live_range.from
                                    && block_end <= segment.live_range.to
                                {
                                    alloc_out = Some(alloc.expect("missing allocation"));
                                }
                                segment.value
                            }
                        };
                        let move_pos = MovePosition::late(self.func.block_insts(pred).last());
                        f(self, value, move_pos, alloc_out, blockparam_idx.is_some());
                    }
                }
            }
        } else {
            // If the instruction immediately preceding the use was a
            // fixed definition then we should use that location as the
            // source of the move.
            let alloc_out = match self.fixed_def {
                Some((pos, alloc)) if pos == inst => alloc,
                _ => alloc.expect("missing allocation"),
            };

            let move_pos = MovePosition::early(inst);
            f(self, segment.value, move_pos, Some(alloc_out), false);
        }
    }

    fn handle_use(&mut self, u: &Use, segment: &ValueSegment, alloc: Option<Allocation>) {
        match u.kind {
            UseKind::FixedDef { reg } => {
                // Record that, at the next instruction boundary, the
                // incoming value is from a fixed definition rather than the
                // normal allocation for this segment.
                self.fixed_def = Some((u.pos().next(), Allocation::reg(reg)));

                // Nothing to do here if the segment ends right after
                // this definition or if this fixed def is on a
                // terminator instruction: the terminator will take care of
                // connecting the fixed register to successors.
                if !self.func.inst_is_terminator(u.pos())
                    && segment.live_range.to != u.pos().next().slot(Slot::Boundary)
                {
                    // If the segment isn't a single fixed def, then it must
                    // have a non-empty live range, which means it must have
                    // an allocation.
                    let pos = MovePosition::early(u.pos().next());
                    self.move_resolver
                        .emit_source_half_move(pos, u.value, Allocation::reg(reg));
                    if alloc != Some(Allocation::reg(reg)) {
                        // Nothing to do if this segment is already assigned
                        // to the desired register.
                        self.move_resolver.emit_dest_half_move(
                            pos,
                            u.value,
                            alloc.expect("missing allocation for fixed def"),
                        );
                    }

                    // The fixed def effectively acts as a live-in at
                    // the next instruction boundary since it provides
                    // a source half-move.
                    self.live_in = Some((
                        u.pos().next(),
                        LiveInKind::Single {
                            from_same_segment: false,
                        },
                    ));
                }
            }
            UseKind::FixedUse { reg } => {
                self.for_each_move_source(
                    u.pos(),
                    segment,
                    alloc,
                    |self_, src_value, move_pos, src_alloc, is_blockparam| {
                        if let Some(alloc) = src_alloc {
                            // Nothing to do if this segment is already assigned
                            // to the desired register.
                            if alloc == Allocation::reg(reg) {
                                return;
                            }
                            self_
                                .move_resolver
                                .emit_source_half_move(move_pos, src_value, alloc);
                        }
                        self_.move_resolver.emit_dest_half_move(
                            move_pos,
                            src_value,
                            Allocation::reg(reg),
                        );

                        if is_blockparam {
                            // Tell the move optimizer that the fixed register
                            // now holds the blockparam value.
                            let block = self.func.inst_block(u.pos());
                            self_.move_resolver.blockparam_allocs.push((
                                block,
                                u.value,
                                Allocation::reg(reg),
                            ));
                        }
                    },
                );
            }
            UseKind::TiedUse {
                use_slot,
                def_slot,
                class,
                group_index,
            } => {
                // Copy the allocation of the def slot to the use slot. Only do
                // this for the first member of a group.
                if group_index == 0 {
                    self.move_resolver.tied_operands.push(TiedOperands {
                        inst: u.pos(),
                        def_slot,
                        use_slot,
                    });
                }

                self.for_each_move_source(
                    u.pos(),
                    segment,
                    alloc,
                    |self_, src_value, move_pos, src_alloc, is_blockparam| {
                        if let Some(alloc) = src_alloc {
                            self_
                                .move_resolver
                                .emit_source_half_move(move_pos, src_value, alloc);
                        }

                        // We defer emitting the destination half-move until after
                        // all segments have been processed since at this point the
                        // allocation for def_slot may not have been filled in yet.
                        self_.move_resolver.tied_moves.push(TiedMove {
                            move_pos,
                            inst: u.pos(),
                            value: src_value,
                            def_slot,
                            class,
                            group_index,
                            is_blockparam,
                        });
                    },
                );
            }
            UseKind::ConstraintConflict {} => {
                // Ensure that a source half-move exists for this value.
                self.for_each_move_source(
                    u.pos(),
                    segment,
                    alloc,
                    |self_, src_value, move_pos, src_alloc, _is_blockparam| {
                        if let Some(alloc) = src_alloc {
                            self_
                                .move_resolver
                                .emit_source_half_move(move_pos, src_value, alloc);
                        }

                        // The conflicting vreg will have a live-in that will
                        // read from the appropriate source.
                    },
                );
            }
            UseKind::ClassUse { slot, class: _ } | UseKind::ClassDef { slot, class: _ } => {
                // Register class uses don't have any moves associated with
                // them. We just need to record the allocation assigned to
                // the operand slot.
                self.allocations.set_allocation(
                    u.pos(),
                    slot,
                    alloc.expect("missing allocation for class use"),
                );
            }
            UseKind::StackMap { class: _ } => {
                self.allocations.add_stack_map_alloc(
                    u.pos(),
                    alloc.expect("missing allocation for stack map use"),
                );
            }
            UseKind::GroupClassUse {
                slot,
                class: _,
                group_index,
            }
            | UseKind::GroupClassDef {
                slot,
                class: _,
                group_index,
            } => {
                // For register groups, only assign the allocation for the
                // first group member.
                if group_index == 0 {
                    self.allocations.set_allocation(
                        u.pos(),
                        slot,
                        alloc.expect("missing allocation for group class use"),
                    );
                }
            }
            UseKind::BlockparamOut {} => {
                // Treat this like a block live-out for a jump terminator.
                self.move_resolver.emit_source_half_move(
                    MovePosition::late(u.pos()),
                    u.value,
                    // Values are live through the entire jump instruction,
                    // so this cannot be an empty segment.
                    alloc.expect("missing allocation for jump terminator"),
                );
            }
            UseKind::BlockparamIn { blockparam_idx } => {
                // If the segment ends at the start of the block then this is a
                // dead blockparam or it is followed by just a single fixed/tied
                // use. That use will take care of inserting the corresponding
                // half-move.
                if segment.live_range.to != u.pos().slot(Slot::Boundary) {
                    // We have an incoming value from multiple predecessor
                    // blocks, emit a destination half-move before the jump
                    // instruction in the predecessor blocks.
                    //
                    // We use the value of the corresponding outgoing blockparam
                    // so that rematerializations are properly handled.
                    let alloc = alloc.expect("missing allocation for blockparam live-in");
                    for &pred in self.func.block_preds(self.func.inst_block(u.pos())) {
                        let value = self.func.jump_blockparams(pred)[blockparam_idx as usize];
                        self.move_resolver.emit_dest_half_move(
                            MovePosition::late(self.func.block_insts(pred).last()),
                            value,
                            alloc,
                        );
                    }

                    // Record the allocation assigned to the block parameter for
                    // the move optimizer.
                    let block = self.func.inst_block(u.pos());
                    self.move_resolver
                        .blockparam_allocs
                        .push((block, u.value, alloc));
                }

                // Indicate that fixed uses on the first instruction
                // should emit half-moves in predecessor blocks instead of
                // at the start of this block.
                self.live_in = Some((
                    u.pos(),
                    LiveInKind::Multi {
                        blockparam_idx: Some(blockparam_idx),
                    },
                ));
            }
        }
    }

    fn process_segment(&mut self, segment: &ValueSegment, alloc: Option<Allocation>) {
        trace!(
            "Processing segment {} ({}) with allocation {alloc:?}",
            segment.live_range,
            segment.value
        );

        // Emit a destination half-move when this value is live-in from another
        // segment.
        let mut current_block;
        if segment.use_list.has_livein() {
            debug_assert_eq!(segment.live_range.from.slot(), Slot::Boundary);

            let first_inst = segment.live_range.from.inst();
            let first_block = self.func.inst_block(first_inst);
            if first_inst == self.func.block_insts(first_block).from {
                self.handle_block_live_in(first_inst, first_block, segment, alloc);
            } else {
                trace!("-> split livein");

                // If the segment's live range ends here then we don't need a
                // destination half move. Any uses with no live range (fixed/tied)
                // will use the incoming values directly from self.live_in.
                if segment.live_range.to != first_inst.slot(Slot::Boundary) {
                    self.move_resolver.emit_dest_half_move(
                        MovePosition::early(first_inst),
                        segment.value,
                        alloc.expect("split live range must have an allocation"),
                    );
                }
                self.live_in = Some((
                    first_inst,
                    LiveInKind::Single {
                        from_same_segment: false,
                    },
                ));
            }

            current_block = first_block;
        } else {
            // A segment with no live-in must start with a definition.
            self.live_in = None;
            current_block = self.func.inst_block(self.uses[segment.use_list][0].pos());
        }

        self.fixed_def = None;
        for u in &self.uses[segment.use_list] {
            // Handle any block boundaries between the previous use and the new
            // one. We need to emit half-moves
            let block = self.func.inst_block(u.pos());
            while current_block != block {
                trace!(
                    "Segment crosses block bounary between {current_block} and {}",
                    current_block.next()
                );
                let terminator = self.func.block_insts(current_block).last();
                self.handle_block_live_out(terminator, current_block, segment, alloc);
                current_block = current_block.next();
                self.handle_block_live_in(terminator.next(), current_block, segment, alloc);
            }

            trace!("-> {} {}", u.pos(), u.kind);
            self.handle_use(u, segment, alloc);
        }

        // Emit a source half-move if the value is live-out to another segment.
        if segment.use_list.has_liveout() {
            debug_assert_eq!(segment.live_range.to.slot(), Slot::Boundary);
            let inst = segment.live_range.to.inst().prev();
            let block = self.func.inst_block(inst);

            // Handle any block boundaries between the previous use and the new
            // one. We need to emit half-moves
            while current_block != block {
                trace!(
                    "Segment crosses block bounary between {current_block} and {}",
                    current_block.next()
                );
                let terminator = self.func.block_insts(current_block).last();
                self.handle_block_live_out(terminator, current_block, segment, alloc);
                current_block = current_block.next();
                self.handle_block_live_in(terminator.next(), current_block, segment, alloc);
            }

            if inst == self.func.block_insts(block).last() {
                self.handle_block_live_out(inst, block, segment, alloc);
            } else {
                trace!("-> split liveout");

                // If the last instruction in the segment was a fixed definition
                // then we should use that allocation as the source of the move.
                let alloc_out = match self.fixed_def {
                    Some((pos, alloc)) if pos == segment.live_range.to.inst() => alloc,
                    _ => alloc.expect("missing allocation for segment with live-out"),
                };

                // Emit a move after the last instruction. This is guaranteed to be
                // in the same block: split live-in/live-out are never inserted at
                // block boundaries.
                self.move_resolver.emit_source_half_move(
                    MovePosition::early(segment.live_range.to.inst()),
                    segment.value,
                    alloc_out,
                );
            }
        }
    }
}
