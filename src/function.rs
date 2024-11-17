//! The [`Function`] trait which describes the input function.
//!
//! This includes all details of the function including basic blocks,
//! instructions and their operands.
//!
//! # Control flow graph and instructions
//!
//! The allocator operates on an input program that is in a standard CFG
//! representation: the function body is a sequence of basic blocks, and
//! each block has a sequence of instructions and zero or more
//! successors. The allocator also requires the client to provide
//! predecessors for each block, and these must be consistent with the
//! successors. The entry block may not have any predecessors.
//!
//! The CFG must have *no critical edges*. A critical edge is an edge from
//! block A to block B such that A has more than one successor *and* B has
//! more than one predecessor.
//!
//! Instructions are opaque to the allocator: their behavior is entirely
//! described using a vector of [`Operand`]s. Every block must end with
//! a terminator instruction. The kind of terminator is determined by the
//! number of successor blocks in a function:
//! * A `jump` terminator has a single successor block *and* that successor
//!   block has more than one predecessor blocks.
//! * A `branch` terminator has multiple successor blocks, each of which only
//!   has a single predecessor block.
//! * A `ret` has no successor blocks and return from the function.
//!
//! Both instructions and blocks are named by indices in contiguous index
//! spaces. A block's instructions must be a contiguous range of
//! instruction indices, and block i's first instruction must come
//! immediately after block i-1's last instruction.
//!
//! Basic blocks in the CFG must be topologically sorted with regards to the
//! dominator tree of the CFG. This means that if block A dominates block B then
//! A must have a lower block index than B. Additionally, the heuristics used
//! by the register allocator work best if loop bodies are continuous and
//! properly nested. The best way to achieve this is to order block by a
//! reverse-postorder (RPO) of the CFG, with loop back-edges prioritized to
//! come earlier than other edges.
//!
//! The first block is treated as the entry block of the CFG. It cannot have
//! any predecessors and all other blocks must be reachable from it.
//!
//! # Operands and values
//!
//! Every instruction operates on values by way of [`Operand`]s. An operand
//! consists of the 2 parts: the [`OperandKind`] and the [`OperandConstraint`].
//!
//! A [`Value`] represents an abstract value produced by an instruction and
//! consumed by other instructions.
//!
//! [`OperandKind`] describes what the operand is doing to a [`Value`]. This is
//! usually one of two things: it defines ([`OperandKind::Def`]) a value as an
//! output or it uses ([`OperandKind::Use`]) a value as an input.
//!
//! [`OperandConstraint`] describes the constraints on which [`PhysReg`] can be
//! assigned to an operand. This is usually [`OperandConstraint::Class`] which
//! means that any register in a given [`RegClass`] can be selected. Some
//! instructions (e.g. a function call) require values to be in a fixed
//! register. For those, [`OperandConstraint::Fixed`] can be used.
//!
//! # SSA and block parameters
//!
//! The function described by the [`Function`] trait must be in [Static Single-Assignment]
//! (SSA) form. This means that each [`Value`] can only be defined once, by a
//! single instruction. Additionally, because the CFG must be ordered by
//! dominance, this means that the definition of a value must always appear
//! before any uses of that value *in the linear instruction order*: the index
//! of the defining [`Inst`] must be less than that of any instruction using the
//! value.
//!
//! At control-flow join points, a [`Value`] may need to represent the incoming
//! value from multiple blocks, each of which has a separate definition. This is
//! achieved using *basic block parameters*, which are equivalent to phi-nodes.
//!
//! A basic block with multiple predecessor blocks may specify a set of block
//! parameters with [`Function::block_params`]. This defines new [`Value`]s at
//! the start of the block. All predecessor blocks, which must end with a `jump`
//! terminator, must provide a matching number of outgoing [`Value`]s with
//! [`Function::jump_blockparams`]. On entry to the block, the block parameter
//! values will be initialized with the value from the block that control flow
//! actually came from.
//!
//! # Reusing an input register for an output
//!
//! Some instructions, particularly on ISAs like x86, only allow a single
//! register to be specified as both an input and output, where the instruction
//! clobber the original input value with the output value.
//!
//! In the case of fixed registers, this can be represented by using the same
//! fixed register for both an input and an output operand. However for
//! class-constraint operands an explicit relationship needs to be created
//! between the input and output operand to ensure they are assigned to the same
//! register.
//!
//! This is achieved by using [`OperandConstraint::Reuse`] for the output
//! operand and specifying the index of the corresponding input operand, which
//! must have [`OperandConstraint::Class`]. This will force the register
//! allocator to assign to the output operand the same register as the
//! designated input operand.
//!
//! # Non-allocatable registers
//!
//! [`OperandKind::NonAllocatable`] can be used to specify a [`PhysReg`] that
//! should be ignored by the register allocator and passed on to the output
//! [`Allocation`] unmodified. This is intended to help with client that want to
//! make use of special registers like the stack pointer or a hardware zero
//! register without needing to change the number of operands in an instruction.
//!
//! Any registers used with `OperandKind::NonAllocatable` must not be part of
//! any register bank or register class.
//!
//! [Static Single-Assignment]: https://en.wikipedia.org/wiki/Static_single-assignment_form
//! [`Allocation`]: super::output::Allocation

use core::fmt;

use crate::entity::iter::Keys;
use crate::entity::packed_option::ReservedValue as _;
use crate::entity::EntityRange;
use crate::reginfo::{PhysReg, RegBank, RegClass, RegUnit, MAX_PHYSREGS};

/// Maximum number of basic blocks.
pub const MAX_BLOCKS: usize = 1 << 28;

/// Maximum number of instructions.
pub const MAX_INSTS: usize = 1 << 28;

/// Maximum number of SSA values.
pub const MAX_VALUES: usize = 1 << 28;

/// Maximum number of operands per instruction.
pub const MAX_INST_OPERANDS: usize = 1 << 10;

/// Maximum number of basic block parameters.
pub const MAX_BLOCK_PARAMS: usize = 1 << 28;

entity_def! {
    /// An opaque reference to a basic block in the input function.
    ///
    /// The register allocator will work correctly with arbitrary block orderings,
    /// however it performs best if blocks are arranged in reverse post-order, and
    /// with loop back-edges ordered before loop exits.
    pub entity Block(u32, "block");


    /// An instruction index in the input function.
    ///
    /// An instruction is opaque: it only interacts with the register allocator
    /// through the constraints defined on its operands.
    ///
    /// Instruction indices must be continuous and ordered according to the block
    /// they are in. This means that the last instruction for block N must be
    /// immediately followed by the first instruction in block N+1.
    ///
    /// Where an `Inst` represents a point between 2 instructions rather than an
    /// instruction, this always refers to the point *before* the given instruction.
    pub entity Inst(u32, "inst");

    /// An opaque reference to an SSA value in the input function.
    pub entity Value(u32, "%");

    /// A reference to a list of [`Value`]s.
    ///
    /// Each `ValueGroup` must only be used in a single [`Operand`] in a function,
    /// even if the same set of value is used multiple times.
    pub entity ValueGroup(u32, "group");
}

impl Block {
    /// The entry block is always block 0 since it dominates all other blocks.
    pub const ENTRY_BLOCK: Block = Block(0);

    /// Returns an index pointing to the next block.
    #[inline]
    #[must_use]
    pub fn next(self) -> Self {
        debug_assert!(!self.is_reserved_value());
        Self(self.0 + 1)
    }

    /// Returns an index pointing to the previous block.
    #[inline]
    #[must_use]
    pub fn prev(self) -> Self {
        debug_assert!(!self.is_reserved_value());
        Self(self.0 - 1)
    }
}

impl Inst {
    /// Returns an index pointing to the next instruction.
    #[inline]
    #[must_use]
    pub fn next(self) -> Self {
        debug_assert!(!self.is_reserved_value());
        Self(self.0 + 1)
    }

    /// Returns an index pointing to the previous instruction.
    #[inline]
    #[must_use]
    pub fn prev(self) -> Self {
        debug_assert!(!self.is_reserved_value());
        Self(self.0 - 1)
    }
}

/// A range of instructions in the input function.
pub type InstRange = EntityRange<Inst>;

/// The "kind" of an operand, which describes how an instruction makes use of
/// an operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OperandKind {
    /// A write of a `Value`.
    ///
    /// The SSA property requires that each `Value` only be defined in a single
    /// place in the entire function.
    Def(Value),

    /// A read of a `Value`.
    Use(Value),

    /// An "early" write of a `Value`.
    ///
    /// Normally, the same `PhysReg` may be assigned to a `Use` and a `Def`
    /// operand because these are assumed to have non-overlapping live ranges.
    /// To avoid this, an `EarlyDef` operand can be used. It is identical to a
    /// `Def` except that it is guaranteed not to conflict with any input
    /// operand.
    ///
    /// `EarlyDef` is primarily useful for pseudo-instructions that expand to
    /// multiple instruction after register allocation. In such sequences, an
    /// output or scratch register may need to be written to before all input
    /// registers have been read.
    EarlyDef(Value),

    /// A write of a group of `Value`s.
    ///
    /// This is analogous to [`OperandKind::Def`] but requires that the
    /// corresponding constraint use an [`OperandConstraint::Class`] with a
    /// register class that has same group size as the [`ValueGroup`].
    DefGroup(ValueGroup),

    /// A read of a group of `Value`s.
    ///
    /// This is analogous to [`OperandKind::Use`] but requires that the
    /// corresponding constraint use an [`OperandConstraint::Class`] with a
    /// register class that has same group size as the [`ValueGroup`].
    UseGroup(ValueGroup),

    /// An "early" write of a group of `Value`s.
    ///
    /// This is analogous to [`OperandKind::EarlyDef`] but requires that the
    /// corresponding constraint use an [`OperandConstraint::Class`] with a
    /// register class that has same group size as the [`ValueGroup`].
    EarlyDefGroup(ValueGroup),

    /// Use of a fixed non-allocatable register.
    ///
    /// This must be used with `OperandConstraint::Fixed`. The given `PhysReg`
    /// is directly assigned to the operand without tracking its live range for
    /// checking interferences.
    ///
    /// This is intended for use with reserved registers that are manually
    /// managed by the client such as the stack pointer or a hardware zero
    /// register. It must not be used with a `PhysReg` that is part of an
    /// allocatable register class.
    NonAllocatable,
}

/// An `OperandConstraint` specifies which value should be assigned to this
/// operand and any constraints on the location where this value should be
/// placed.
///
/// The allocator's result will always satisfy all given constraints; however,
/// if the input has a combination of constraints that are impossible to
/// satisfy, then allocation may fail. Providing impossible constraints
/// is usually a programming error in the client, rather than a function of bad
/// input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OperandConstraint {
    /// Operand must be in a register of the given class.
    Class(RegClass),

    /// Operand must be in the given fixed register.
    Fixed(PhysReg),

    /// Ties an [`OperandKind::Def`] or [`OperandKind::EarlyDef`] to the same
    /// [`PhysReg`] as an input operand.
    ///
    /// The target operand must be an [`OperandKind::Use`] with an
    /// [`OperandConstraint::Class`]. It is also valid for register groups with
    /// the corresponding [`OperandKind::DefGroup`]/[`OperandKind::EarlyDefGroup`]
    /// and [`OperandKind::UseGroup`].
    Reuse(usize),
}

impl fmt::Display for OperandConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Class(rc) => write!(f, "{rc}"),
            Self::Fixed(reg) => write!(f, "{reg}"),
            Self::Reuse(idx) => write!(f, "reuse({idx})"),
        }
    }
}

/// Information about an operand in an instruction.
///
/// `Operand` encodes everything about a mention of a register in an
/// instruction: the associated SSA value, how it is used by the instruction
/// (read, write), and any constraints on the `Allocation` that will be
/// selected for the operand.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
// Packing so that the overall size is 6 bytes instead of 8.
#[repr(packed(2))]
pub struct Operand {
    /// `OperandKind` encoded in 32 bits:
    ///
    /// kind:3 unused:1 value:28
    kind: u32,

    /// `OperandConstraint` encoded in 16 bits:
    ///
    /// type:2 unused:2 index:12
    constraint: u16,
}

impl Operand {
    /// Constructs a new operand.
    #[inline]
    #[must_use]
    pub fn new(kind: OperandKind, constraint: OperandConstraint) -> Self {
        let kind_field = match kind {
            #[allow(clippy::identity_op)]
            OperandKind::Def(value) => 0 << 29 | value.index() as u32,
            OperandKind::Use(value) => 1 << 29 | value.index() as u32,
            OperandKind::EarlyDef(value) => 2 << 29 | value.index() as u32,
            OperandKind::DefGroup(group) => 3 << 29 | group.index() as u32,
            OperandKind::UseGroup(group) => 4 << 29 | group.index() as u32,
            OperandKind::EarlyDefGroup(group) => 5 << 29 | group.index() as u32,
            OperandKind::NonAllocatable => 6 << 29,
        };
        let constraint = match constraint {
            #[allow(clippy::identity_op)]
            OperandConstraint::Class(class) => 0 << 14 | class.index() as u16,
            OperandConstraint::Fixed(reg) => 1 << 14 | reg.index() as u16,
            OperandConstraint::Reuse(index) => 2 << 14 | index as u16,
        };
        Self {
            kind: kind_field,
            constraint,
        }
    }

    /// Create an `Operand` that designates a use of a `Value` that must be in
    /// a register from the given register class. This value is read by the
    /// instruction and may be overwritten by an output of the instruction.
    #[inline]
    #[must_use]
    pub fn regclass_use(value: Value, class: RegClass) -> Self {
        Self::new(OperandKind::Use(value), OperandConstraint::Class(class))
    }

    /// Create an `Operand` that designates a definition of a `Value` that must
    /// be in a register from the given register class. This value is written to
    /// by the instruction and may reuse an input register for this output.
    #[inline]
    #[must_use]
    pub fn regclass_def(value: Value, class: RegClass) -> Self {
        Self::new(OperandKind::Def(value), OperandConstraint::Class(class))
    }

    /// Create an `Operand` that designates a definition of a `Value` that must
    /// be in a register from the given register class. This value is written to
    /// by the instruction but may not reuse an input register for this output.
    ///
    /// This kind of constraint is useful to reserve a "scratch" register for a
    /// pseudo-instruction which expands to a sequence of instructions after
    /// register allocation.
    #[inline]
    #[must_use]
    pub fn regclass_early_def(value: Value, class: RegClass) -> Self {
        Self::new(
            OperandKind::EarlyDef(value),
            OperandConstraint::Class(class),
        )
    }

    /// Create an `Operand` that designates a use of a `Value` that must be in a
    /// specific register. This value is read by the instruction and may be
    /// overwritten by an output of the instruction.
    #[inline]
    #[must_use]
    pub fn fixed_use(value: Value, reg: PhysReg) -> Self {
        Self::new(OperandKind::Use(value), OperandConstraint::Fixed(reg))
    }

    /// Create an `Operand` that designates a definition of a `Value` that must
    /// be in a specific register. This value is written to by the instruction
    /// and may reuse an input register for this output.
    #[inline]
    #[must_use]
    pub fn fixed_def(value: Value, reg: PhysReg) -> Self {
        Self::new(OperandKind::Def(value), OperandConstraint::Fixed(reg))
    }

    /// Create an `Operand` that designates a definition of a `Value` that must
    /// be in a specific register. This value is written to
    /// by the instruction but may not reuse an input register for this output.
    #[inline]
    #[must_use]
    pub fn fixed_early_def(value: Value, reg: PhysReg) -> Self {
        Self::new(OperandKind::EarlyDef(value), OperandConstraint::Fixed(reg))
    }

    /// Create an `Operand` that designates a definition of a `Value` that must
    /// be in the same register as an input to the instruction. The input is
    /// identified by `input_idx` (is the `idx`th `Operand` for the instruction)
    /// and must have a `regclass_use` constraint.
    #[inline]
    #[must_use]
    pub fn reuse_def(value: Value, input_idx: usize) -> Self {
        Self::new(OperandKind::Def(value), OperandConstraint::Reuse(input_idx))
    }

    /// Create an `Operand` that designates a definition of a `Value` that must
    /// be in the same register as an input to the instruction. The input is
    /// identified by `input_idx` (is the `idx`th `Operand` for the instruction)
    /// and must have a `regclass_use` constraint.
    ///
    /// Additionally, this constrains the operand to not reuse the register of
    /// any *other* input operand.
    #[inline]
    #[must_use]
    pub fn reuse_early_def(value: Value, input_idx: usize) -> Self {
        Self::new(
            OperandKind::EarlyDef(value),
            OperandConstraint::Reuse(input_idx),
        )
    }

    /// Create an `Operand` that always results in an assignment to the
    /// given fixed `reg`, *without* tracking liveranges in that register.
    ///
    /// This is intended for use with reserved registers that are manually
    /// managed by the client such as the stack pointer or a hardware zero
    /// register. It must not be used with a `PhysReg` that is part of an
    /// allocatable register class.
    #[inline]
    #[must_use]
    pub fn fixed_nonallocatable(reg: PhysReg) -> Self {
        Self::new(OperandKind::NonAllocatable, OperandConstraint::Fixed(reg))
    }

    /// Returns the "kind" of this operand which describes how the operand is
    /// used by the instruction.
    #[inline]
    #[must_use]
    pub fn kind(self) -> OperandKind {
        let value = Value::new(self.kind as usize & (MAX_VALUES - 1));
        let group = ValueGroup::new(self.kind as usize & (MAX_VALUES - 1));
        match (self.kind >> 29) & 7 {
            0 => OperandKind::Def(value),
            1 => OperandKind::Use(value),
            2 => OperandKind::EarlyDef(value),
            3 => OperandKind::DefGroup(group),
            4 => OperandKind::UseGroup(group),
            5 => OperandKind::EarlyDefGroup(group),
            6 => OperandKind::NonAllocatable,
            _ => unreachable!(),
        }
    }

    /// Returns the constraint on this operand which describes the requirements
    /// for the `Allocation` it will be assigned to.
    #[inline]
    #[must_use]
    pub fn constraint(self) -> OperandConstraint {
        let index = self.constraint as usize & (MAX_PHYSREGS - 1);
        match self.constraint >> 14 {
            0 => OperandConstraint::Class(RegClass::new(index)),
            1 => OperandConstraint::Fixed(PhysReg::new(index)),
            2 => OperandConstraint::Reuse(index),
            _ => unreachable!(),
        }
    }
}

impl fmt::Debug for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = self.kind();
        let (kind, value) = match kind {
            OperandKind::Def(ref value) => ("Def", value as &dyn fmt::Display),
            OperandKind::Use(ref value) => ("Use", value as &dyn fmt::Display),
            OperandKind::EarlyDef(ref value) => ("EarlyDef", value as &dyn fmt::Display),
            OperandKind::DefGroup(ref group) => ("Def", group as &dyn fmt::Display),
            OperandKind::UseGroup(ref group) => ("Use", group as &dyn fmt::Display),
            OperandKind::EarlyDefGroup(ref group) => ("EarlyDef", group as &dyn fmt::Display),
            OperandKind::NonAllocatable => {
                return write!(f, "NonAllocatable:{}", self.constraint());
            }
        };
        write!(f, "{kind}({value}):{}", self.constraint())
    }
}

/// Information about the cost of rematerializing a value into a register.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RematCost {
    /// Rematerialization is cheaper than a register-register move and should
    /// always been preferred instead of a move.
    CheaperThanMove,

    /// Rematerialization is more expensive than a move but cheaper than a
    /// memory load. It should only be performed if doing so avoids a memory
    /// access instrution.
    CheaperThanLoad,
}

/// A trait defined by the register allocator client to provide access to its
/// machine-instruction / CFG representation.
///
/// See the [module-level documentation] for more details.
///
/// [module-level documentation]: self
//
// (This trait's design is inspired by, and derives heavily from, the
// trait of the same name in regalloc2.)
pub trait Function {
    // -------------
    // CFG traversal
    // -------------

    /// How many instructions are there?
    fn num_insts(&self) -> usize;

    /// Iterator over all the [`Inst`]s in this function.
    #[inline]
    fn insts(&self) -> Keys<Inst> {
        Keys::with_len(self.num_insts())
    }

    /// How many blocks are there?
    ///
    /// All blocks must be reachable from the entry block.
    fn num_blocks(&self) -> usize;

    /// Iterator over all the [`Block`]s in this function.
    #[inline]
    fn blocks(&self) -> Keys<Block> {
        Keys::with_len(self.num_blocks())
    }

    /// Provide the range of instruction indices contained in each block.
    fn block_insts(&self, block: Block) -> InstRange;

    /// Returns the block containing the given instruction.
    fn inst_block(&self, inst: Inst) -> Block;

    /// Get CFG successors for a given block.
    fn block_succs(&self, block: Block) -> &[Block];

    /// Get the CFG predecessors for a given block.
    fn block_preds(&self, block: Block) -> &[Block];

    /// Returns the immediate dominator of the given block, or `None` if it is
    /// the entry block.
    fn block_immediate_dominator(&self, block: Block) -> Option<Block>;

    /// Returns whether block `a` dominates block `b`.
    ///
    /// This should return true if `a == b`.
    fn block_dominates(&self, a: Block, mut b: Block) -> bool {
        // Walk b up the dominator tree until a is reached or we go past it.
        // This works because the block ordering is required to be topologically
        // ordered with regards to the dominator tree.
        while a < b {
            match self.block_immediate_dominator(b) {
                Some(idom) => b = idom,
                None => return false,
            }
        }
        a == b
    }

    /// Get the block parameters for a given block.
    ///
    /// Block parameters are only allowed on blocks with more than one
    /// predecessors.
    fn block_params(&self, block: Block) -> &[Value];

    /// Determine whether an instruction is an end-of-block branch or return.
    ///
    /// All blocks must end with a terminator instruction, and terminator
    /// instructions are not allowed in the body of a block.
    fn inst_is_terminator(&self, inst: Inst) -> bool;

    /// If `block` ends with a jump terminator, returns the outgoing
    /// block arguments.
    ///
    /// * Jump arguments are only allowed on blocks with a single successor.
    /// * The number of arguments must match the number incoming blockparams in
    ///   the successor.
    /// * If a block has outgoing branch arguments then the terminator
    ///   instruction cannot have any operands.
    fn jump_blockparams(&self, block: Block) -> &[Value];

    /// Returns the estimated execution frequency of this block.
    ///
    /// The allocator uses this to prefer placing moves in lower-frequency
    /// blocks and to prioritize registers for values that are used in higher-
    /// frequency blocks.
    ///
    /// This number be be non-zero and positive. In general, good numbers to use
    /// are in the range of 10e-9 to 10e9 since this avoids issues floats
    /// overflowing into infinity and precision loss.
    fn block_frequency(&self, block: Block) -> f32;

    // --------------------------
    // Instruction register slots
    // --------------------------

    /// Get the `Operand`s for an instruction.
    fn inst_operands(&self, inst: Inst) -> &[Operand];

    /// Get the clobbers for an instruction; these are the registers units
    /// that, after the instruction has executed, hold values that are
    /// arbitrary, separately from the usual outputs to the
    /// instruction. It is invalid to read a register that has been
    /// clobbered; the register allocator is free to assume that
    /// clobbered registers are filled with garbage and available for
    /// reuse. It will avoid storing any value in a clobbered register
    /// that must be live across the instruction.
    ///
    /// Another way of seeing this is that a clobber is equivalent to
    /// a `Def` of a fresh [`Value`] that is not used anywhere else
    /// in the program, with a fixed-register constraint that places
    /// it in a given `PhysReg` chosen by the client prior to regalloc.
    ///
    /// Every register written by an instruction must either
    /// correspond to (be assigned to) an Operand of kind `Def`, or
    /// else must be a "clobber".
    ///
    /// This can be used to, for example, describe ABI-specified
    /// registers that are not preserved by a call instruction, or
    /// fixed physical registers written by an instruction but not
    /// used as an output, or fixed physical registers used as
    /// temps within an instruction out of necessity.
    ///
    /// Duplicate clobbers are not allowed and clobbers may not overlap with any
    /// fixed-def operands on the same instruction.
    fn inst_clobbers(&self, inst: Inst) -> &[RegUnit];

    // -----------------------
    // Values and value groups
    // -----------------------

    /// Get the number of SSA values in use in this function.
    fn num_values(&self) -> usize;

    /// Iterator over all the [`Value`]s in this function.
    #[inline]
    fn values(&self) -> Keys<Value> {
        Keys::with_len(self.num_values())
    }

    /// The register bank associated with the given value.
    ///
    /// A given value can only be associated with a single register bank,
    /// typically based on its type in the source IR. All operands using this
    /// value must have constraints from the same bank and it may only be
    /// linked to block parameters from the same bank.
    fn value_bank(&self, value: Value) -> RegBank;

    /// Get the number of value groups in use in this function.
    fn num_value_groups(&self) -> usize;

    /// Iterator over all the [`ValueGroup`]s in this function.
    #[inline]
    fn value_groups(&self) -> Keys<ValueGroup> {
        Keys::with_len(self.num_value_groups())
    }

    /// Get the members of a value group.
    fn value_group_members(&self, group: ValueGroup) -> &[Value];

    // -----------------
    // Rematerialization
    // -----------------

    /// Whether a [`Value`] can be re-materialized "cheaply" (specifically,
    /// cheaper than a stack spill and reload).
    ///
    /// There are strict restrictions on rematerializable values: they cannot
    /// depend on any other allocatable register and they are only allowed to
    /// use a single destination register as scratch space.
    ///
    /// Additionally, the register class for the rematerialization target must:
    /// - have an non-empty allocation order.
    /// - either allow spillsots or only have registers for which
    ///   [`RegInfo::is_memory`] is false.
    ///
    /// [`RegInfo::is_memory`]: super::reginfo::RegInfo::is_memory
    fn can_rematerialize(&self, value: Value) -> Option<(RematCost, RegClass)>;

    /// If all the outputs of an instruction are dead (never used), can the
    /// instruction be removed (i.e. it has no side effects apart from its
    /// outputs).
    ///
    /// Instruction outputs can become dead due to rematerialization when all
    /// existing users are using the rematerialized value and the original
    /// instruction is no longer needed.
    // TODO: We don't do this yet.
    fn can_eliminate_dead_inst(&self, inst: Inst) -> bool;
}
