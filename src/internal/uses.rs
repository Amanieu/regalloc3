use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::ops::{Index, IndexMut, Range};

use super::live_range::{LiveRangePoint, Slot};
use crate::function::Inst;
use crate::reginfo::{PhysReg, RegClass, RegInfo};

/// A `Use` describes the way a value is used in a live range.
///
/// Once each [`VirtReg`] is mapped to an [`Allocation`] this information is
/// used to map allocations back to the original instruction operands and to
/// generate the necessary move instructions to connect the dataflow across
/// live ranges.
///
/// [`VirtReg`]: super::virt_regs::VirtReg
/// [`Allocation`]: super::output::Allocation
#[derive(Debug, Clone, Copy)]
pub struct Use {
    /// The position of the use in the function.
    pub pos: Inst,

    /// How the value is used in the instruction.
    pub kind: UseKind,
}

/// Cost of a spill/reload of a register from memory.
pub const SPILL_RELOAD_COST: f32 = 1.0;

/// Cost of a move between 2 registers.
pub const MOVE_COST: f32 = 0.5;

impl Use {
    /// Spill cost for this use.
    ///
    /// This is calculated as the cost to be paid if the virtual register
    /// containing this use is spilled to the stack instead of allocated to a
    /// register.
    pub fn spill_cost(self, reginfo: &impl RegInfo) -> f32 {
        match self.kind {
            // Fixed uses/defs are simple: just pay the cost of the
            // spill/reload, except if reg represents a memory location.
            UseKind::FixedDef { reg } | UseKind::FixedUse { reg } => {
                if reginfo.is_memory(reg) {
                    0.0
                } else {
                    SPILL_RELOAD_COST
                }
            }

            // Some instructions can directly accept stack operands, use the
            // class spill cost to determine how much this costs.
            UseKind::ClassUse { slot: _, class }
            | UseKind::ClassDef { slot: _, class }
            | UseKind::GroupClassUse {
                slot: _,
                class,
                group_index: _,
            }
            | UseKind::GroupClassDef {
                slot: _,
                class,
                group_index: _,
            } => reginfo.class_spill_cost(class),

            // Tied uses and constraint conflicts involve a move from one vreg
            // to another.
            //
            // By allocating this to a register instead of a spill slot we can
            // use a register-register move which is cheaper than memory access.
            UseKind::TiedUse {
                use_slot: _,
                def_slot: _,
                class: _,
                group_index: _,
            }
            | UseKind::ConstraintConflict {} => SPILL_RELOAD_COST - MOVE_COST,

            // Blockparams don't care about being in a spill slot or in a
            // register. It may matter a little if this introduces a copy, but
            // that is mostly covered by the live-in/live-out cost penalty.
            UseKind::BlockparamIn { blockparam_idx: _ } | UseKind::BlockparamOut {} => 0.0,
        }
    }

    /// Returns the live range point at which this use ends.
    ///
    /// For a use, this is the point at which the value is used.
    ///
    /// For a definition, this is the next instruction boundary.
    pub fn end_point(self) -> LiveRangePoint {
        let inst = self.pos;
        match self.kind {
            UseKind::FixedDef { .. } => inst.next().slot(Slot::Boundary),
            UseKind::FixedUse { .. } => inst.slot(Slot::Boundary),
            UseKind::TiedUse { .. } => inst.slot(Slot::Boundary),
            UseKind::ConstraintConflict {} => inst.slot(Slot::Boundary),
            UseKind::ClassUse { .. } => inst.slot(Slot::Normal),
            UseKind::ClassDef { .. } => inst.next().slot(Slot::Boundary),
            UseKind::GroupClassUse { .. } => inst.slot(Slot::Normal),
            UseKind::GroupClassDef { .. } => inst.next().slot(Slot::Boundary),
            UseKind::BlockparamIn { .. } => inst.slot(Slot::Boundary),
            UseKind::BlockparamOut {} => inst.next().slot(Slot::Boundary),
        }
    }
}

/// Description of what operation is done on a value in a live range.
#[derive(Debug, Clone, Copy)]
pub enum UseKind {
    /// A definition of the value in a fixed register.
    ///
    /// The portion of the live range corresponding to the definition
    /// (`Normal`/`Early` to the following `Boundary`) is actually excluded from
    /// the live range. We only record the live range starting from the next
    /// `Boundary` onwards. The fixed register is kept as a hint for the main
    /// allocation loop.
    ///
    /// This is done so that a move is automatically generated from the fixed
    /// register to the allocation that is selected for this live range.
    ///
    /// If the `FixedDef` is at the end of this range then a source half-move
    /// will be generated for this value to propagate the value to subsequent
    /// live ranges.
    FixedDef {
        /// Fixed register.
        reg: PhysReg,
    },

    /// A use of the value in a fixed register.
    ///
    /// The portion of the live range corresponding to the use (`Boundary` to
    /// `Normal`) is actually excluded from the live range. We only record the
    /// live range ending at the preceding `Boundary`.
    ///
    /// This is done so that a move is automatically generated from the
    /// allocation that is selected for this live range to the fixed register.
    ///
    /// If the `FixedUse` live range is at the start of this range then a
    /// destination half-move will be generated for this value get the value
    /// from preceding live ranges.
    FixedUse {
        /// Fixed register.
        reg: PhysReg,
    },

    /// A use of the value that is tied to the same allocation as another
    /// definition in the same instruction.
    ///
    /// A move is emitted from the allocation for this live range to the
    /// allocation for the live range of the output operand. Then the allocation
    /// for the def slot is copied to the use slot.
    TiedUse {
        /// Input operand slot in the instruction.
        use_slot: u16,

        /// Output operand slot in the instruction.
        def_slot: u16,

        /// Register class of the tied operand.
        ///
        /// This is only used to resolve the allocation for tied groups.
        class: RegClass,

        /// Index of this value in its `ValueGroup`, or 0 if it is not in a
        /// group.
        group_index: u8,
    },

    /// If there are 2 conflicting register classes uses *on the same
    /// instruction* (meaning the conflict can't be resolved by splitting) then
    /// one of the uses is turned into a `ConstraintConflict` and the original
    /// use is moved to a separate virtual register.
    ///
    /// A move is emitted from the allocation for this live range to the
    /// allocation for the target virtual register.
    ConstraintConflict {},

    /// Use of the value in the given register class.
    ///
    /// This is used to calculate the register class requirements of a virtual
    /// register, and to map the selected `Allocation` to the corresponding
    /// operand slot.
    ClassUse {
        /// Operand slot in the instruction.
        slot: u16,

        /// Register class that the allocation for this operand must come from.
        class: RegClass,
    },

    /// Definition of the value in the given register class.
    ///
    /// This is used to calculate the register class requirements of a virtual
    /// register, and to map the selected `Allocation` to the corresponding
    /// operand slot.
    ClassDef {
        /// Operand slot in the instruction.
        slot: u16,

        /// Register class that the allocation for this operand must come from.
        class: RegClass,
    },

    /// Use value in the given register class as part of a group.
    ///
    /// This is used to calculate the register class requirements of a virtual
    /// register, and to map the selected `Allocation` to the corresponding
    /// operand slot.
    GroupClassUse {
        /// Operand slot in the instruction.
        slot: u16,

        /// Register class that the allocation for this operand must come from.
        class: RegClass,

        /// Index of this value in its `ValueGroup`.
        group_index: u8,
    },

    /// Definition of the value in the given register class as part of a group.
    ///
    /// This is used to calculate the register class requirements of a virtual
    /// register, and to map the selected `Allocation` to the corresponding
    /// operand slot.
    GroupClassDef {
        /// Operand slot in the instruction.
        slot: u16,

        /// Register class that the allocation for this operand must come from.
        class: RegClass,

        /// Index of this value in its `ValueGroup`.
        group_index: u8,
    },

    /// Indicates that the value is a block parameter live-in from multiple
    /// predecessor blocks.
    ///
    /// A destination half-move is emitted just before the corresponding
    /// `BlockparamOut` instruction in each of the block's predecessors.
    ///
    /// This is always guaranteed to be the first use when there are multiple
    /// uses within a single instruction.
    BlockparamIn { blockparam_idx: u32 },

    /// Indicates that the value is block parameter live-out on a terminator
    /// with a single successor block where the successor block has multiple
    /// predecessors.
    ///
    /// A source half-move is emitted before the jump instruction at the "late"
    /// position.
    ///
    /// This implies that the terminator cannot have any operands itself: a
    /// source half-move is emitted at the point before the terminator
    /// instruction.
    BlockparamOut {},
}
impl UseKind {
    /// Whether this `UseKind` represents the definition of a `Value`.
    pub fn is_def(self) -> bool {
        match self {
            UseKind::FixedDef { .. }
            | UseKind::ClassDef { .. }
            | UseKind::GroupClassDef { .. }
            | UseKind::BlockparamIn { .. } => true,
            UseKind::FixedUse { .. }
            | UseKind::TiedUse { .. }
            | UseKind::ConstraintConflict { .. }
            | UseKind::ClassUse { .. }
            | UseKind::GroupClassUse { .. }
            | UseKind::BlockparamOut { .. } => false,
        }
    }
}

impl fmt::Display for UseKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            UseKind::FixedDef { reg } => write!(f, "fixed_def: {reg}"),
            UseKind::FixedUse { reg } => write!(f, "fixed_use: {reg}"),
            UseKind::TiedUse {
                use_slot,
                def_slot,
                class,
                group_index,
            } => {
                write!(
                    f,
                    "tied_use: use_slot={use_slot} def_slot={def_slot} class={class} \
                     group_index={group_index}"
                )
            }
            UseKind::ConstraintConflict {} => write!(f, "constraint_conflict"),
            UseKind::ClassUse { slot, class } => {
                write!(f, "class_use: {class} slot={slot}")
            }
            UseKind::ClassDef { slot, class } => write!(f, "class_def: {class} slot={slot}"),
            UseKind::GroupClassUse {
                slot,
                class,
                group_index,
            } => write!(
                f,
                "group_class_use: {class} slot={slot} group_index={group_index}"
            ),
            UseKind::GroupClassDef {
                slot,
                class,
                group_index,
            } => write!(
                f,
                "group_class_def: {class} slot={slot} group_index={group_index}"
            ),
            UseKind::BlockparamIn { blockparam_idx } => {
                write!(f, "blockparam_in: idx={blockparam_idx}")
            }
            UseKind::BlockparamOut {} => {
                write!(f, "blockparam_out")
            }
        }
    }
}

/// Reference to a single [`Use`] in [`Uses`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UseIndex(u32);

/// List of [`Use`]s for the value covered by the live range, represented as a
/// range of entries in the global `Use` vector.
///
/// Uses are fixed once live ranges are constructed, and are not affected by
/// splits.
///
/// `UseList` also encodes 2 flags to indicate an implicit live-in from a
/// preceding segment and an implicit live-out to a later segment.
#[derive(Debug, Clone, Copy)]
pub struct UseList {
    from: u32,
    to: u32,
}

// Flags packed into the high bits of the use index.
const USE_INDEX_LIVE_IN: u32 = 1 << 31;
const USE_INDEX_LIVE_OUT: u32 = 1 << 31;
const USE_INDEX_FIXED_HINT: u32 = 1 << 30;
const USE_INDEX_FIXED_DEF: u32 = 1 << 30;
const USE_INDEX_MASK: u32 = !0 >> 2;

impl UseList {
    /// Returns an empty use list which doesn't contain any uses.
    pub fn empty() -> Self {
        Self { from: 0, to: 0 }
    }

    /// Indicates that the live range that contains this use list is live-in
    /// from a preceding segment at the start of the live range.
    pub fn has_livein(self) -> bool {
        self.from & USE_INDEX_LIVE_IN != 0
    }

    /// Indicates that the live range that contains this use list is live-out
    /// to another segment at the end of the live range.
    pub fn has_liveout(self) -> bool {
        self.to & USE_INDEX_LIVE_OUT != 0
    }

    /// Indicates that the live range that contains this use list has a fixed
    /// register hint.
    pub fn has_fixedhint(self) -> bool {
        self.to & USE_INDEX_FIXED_HINT != 0
    }

    /// Indicates that the first use in this list is a `FixedDef`.
    pub fn has_fixeddef(self) -> bool {
        self.from & USE_INDEX_FIXED_DEF != 0
    }

    /// Sets the live-in bit to the given value.
    pub fn set_livein(&mut self, val: bool) {
        if val {
            self.from |= USE_INDEX_LIVE_IN;
        } else {
            self.from &= !USE_INDEX_LIVE_IN;
        }
    }

    /// Sets the live-out bit to the given value.
    pub fn set_liveout(&mut self, val: bool) {
        if val {
            self.to |= USE_INDEX_LIVE_OUT;
        } else {
            self.to &= !USE_INDEX_LIVE_OUT;
        }
    }

    /// Sets the fixed-hint bit to the given value.
    pub fn set_fixedhint(&mut self, val: bool) {
        if val {
            self.to |= USE_INDEX_FIXED_HINT;
        } else {
            self.to &= !USE_INDEX_FIXED_HINT;
        }
    }

    /// Sets the fixed-def bit to the given value.
    pub fn set_fixeddef(&mut self, val: bool) {
        if val {
            self.from |= USE_INDEX_FIXED_DEF;
        } else {
            self.from &= !USE_INDEX_FIXED_DEF;
        }
    }

    /// Returns the number of [`Use`]s in this list.
    pub fn len(self) -> usize {
        self.indices().len()
    }

    /// Returns a [`UseIndex`] pointing to a single [`Use`] in this list.
    pub fn index(self, index: usize) -> UseIndex {
        debug_assert!(index < self.len());
        UseIndex((self.from & USE_INDEX_MASK) + index as u32)
    }

    /// Returns an iterator over all the [`UseIndex`] in this list.
    pub fn iter(self) -> impl DoubleEndedIterator<Item = UseIndex> + ExactSizeIterator {
        (0..self.len()).map(move |i| self.index(i))
    }

    /// Returns the range of indices encoded in the `UseList`.
    fn indices(self) -> Range<usize> {
        let from = (self.from & USE_INDEX_MASK) as usize;
        let to = (self.to & USE_INDEX_MASK) as usize;
        from..to
    }

    /// Splits the given use list at the given instruction.
    ///
    /// All uses before the given instruction are returned in the first list,
    /// and all uses at or after the given instruction are returned in the
    /// second list.
    pub fn split_at_inst(self, split_at: Inst, uses: &Uses) -> (Self, Self) {
        let split_index = uses[self].partition_point(|u| u.pos < split_at);
        let mid = UseIndex((self.from & USE_INDEX_MASK) + split_index as u32);
        let (mut first, mut second) = self.split_at_index(mid);

        // Mark the use lists as having a matching live-in/live-out.
        first.set_liveout(true);
        second.set_livein(true);

        (first, second)
    }

    /// Splits the given use list at the given `UseIndex`.
    ///
    /// The fixed-hint flag is cleared by this operation.
    pub fn split_at_index(self, mid: UseIndex) -> (Self, Self) {
        debug_assert!(mid.0 as usize >= self.indices().start);
        debug_assert!((mid.0 as usize) <= self.indices().end);
        let mut first = UseList {
            from: self.from,
            to: mid.0,
        };
        let second = UseList {
            from: mid.0,
            to: self.to & !USE_INDEX_FIXED_HINT,
        };
        if self.has_fixeddef() {
            first.set_fixeddef(true);
        }
        (first, second)
    }
}

pub struct Uses {
    /// Backing array of `Use`s for `UseList`.
    uses: Vec<Use>,
}

impl Uses {
    pub fn new() -> Uses {
        Self { uses: vec![] }
    }

    pub fn clear(&mut self) {
        self.uses.clear();
    }

    /// Adds a new `UseList` from an iterator.
    pub fn add_use_list(&mut self, iter: impl IntoIterator<Item = Use>) -> UseList {
        let from = self.uses.len() as u32;
        self.uses.extend(iter);
        let to = self.uses.len() as u32;
        UseList { from, to }
    }
}

impl Index<UseIndex> for Uses {
    type Output = Use;

    fn index(&self, index: UseIndex) -> &Self::Output {
        &self.uses[index.0 as usize]
    }
}

impl IndexMut<UseIndex> for Uses {
    fn index_mut(&mut self, index: UseIndex) -> &mut Self::Output {
        &mut self.uses[index.0 as usize]
    }
}

impl Index<UseList> for Uses {
    type Output = [Use];

    fn index(&self, index: UseList) -> &Self::Output {
        &self.uses[index.indices()]
    }
}

impl IndexMut<UseList> for Uses {
    fn index_mut(&mut self, index: UseList) -> &mut Self::Output {
        &mut self.uses[index.indices()]
    }
}
