use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::ops::{Index, IndexMut, Range};

use cranelift_entity::EntityRef;

use super::live_range::{LiveRangePoint, Slot};
use crate::function::{Inst, Value};
use crate::reginfo::{PhysReg, RegClass, RegInfo};

/// Position of a use in the function.
///
/// This is more fine-grained than just the instruction at which the use occurs
/// because we want to sort uses so that blockparam defintions always appear
/// first within each instruction. Move resolution relies on this property.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UsePosition {
    /// Bit-pack in 32 bits.
    ///
    /// inst:31 pos:1
    bits: u32,
}

impl fmt::Display for UsePosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inst().fmt(f)
    }
}

impl UsePosition {
    /// A use that represents the definition of a value.
    pub fn at_def(inst: Inst) -> Self {
        Self {
            bits: (inst.index() as u32) << 1,
        }
    }

    /// A use that represents the use of a value
    pub fn at_use(inst: Inst) -> Self {
        Self {
            bits: ((inst.index() as u32) << 1) | 1,
        }
    }

    /// Instruction at which the use occurs.
    pub fn inst(self) -> Inst {
        Inst::new((self.bits >> 1) as usize)
    }
}

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
    pub use_pos: UsePosition,

    /// The value that is being manipulated.
    ///
    /// This is only needed during live range calculation and while building
    /// virtual registers, and is otherwise redundant since this value is
    /// already available in [`ValueSegment::value`].
    ///
    /// [`ValueSegment::value`]: super::virt_regs::ValueSegment::value
    pub value: Value,

    /// How the value is used in the instruction.
    pub kind: UseKind,
}

/// Cost of a spill/reload of a register from memory.
pub const SPILL_RELOAD_COST: f32 = 1.0;

/// Cost of a move between 2 registers.
pub const MOVE_COST: f32 = 0.5;

impl Use {
    /// Returns the instruction at which the use occurs.
    pub fn pos(self) -> Inst {
        self.use_pos.inst()
    }

    /// Whether this `Use` represents the definition of a `Value`.
    pub fn is_def(self) -> bool {
        self.use_pos.bits & 1 == 0
    }

    /// Whether this `Use` represents the use of a `Value`.
    pub fn is_use(self) -> bool {
        self.use_pos.bits & 1 == 1
    }

    /// Spill cost for this use.
    ///
    /// This is calculated as the cost to be paid if the virtual register
    /// containing this use is spilled to the stack instead of allocated to a
    /// register.
    pub fn spill_cost(self, reginfo: &impl RegInfo) -> f32 {
        match self.kind {
            // Fixed uses/defs are simple: just pay the cost of the spill/reload.
            UseKind::FixedDef { reg: _ } | UseKind::FixedUse { reg: _ } => SPILL_RELOAD_COST,

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
        let inst = self.pos();
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
    /// live range ending at the preceding `Boundary`. The fixed register is
    /// kept as a hint for the main allocation loop.
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

/// Extra bit to indicate that the value is live-in/live-out.
const USE_LIST_EXTRA_BIT: u32 = 1 << 31;

impl UseList {
    /// Constructs a new `UseList` holding the given index range.
    ///
    /// The live-in/live-out flags are not set.
    pub fn new(from: UseIndex, to: UseIndex) -> Self {
        debug_assert!(from.0 < USE_LIST_EXTRA_BIT);
        debug_assert!(to.0 < USE_LIST_EXTRA_BIT);
        Self {
            from: from.0,
            to: to.0,
        }
    }

    /// Returns an empty use list which doesn't contain any uses.
    pub fn empty() -> Self {
        Self { from: 0, to: 0 }
    }

    /// Indicates that the live range that contains this use list is live-in
    /// from a preceding segment at the start of the live range.
    pub fn has_livein(self) -> bool {
        self.from & USE_LIST_EXTRA_BIT != 0
    }

    /// Indicates that the live range that contains this use list is live-out
    /// to another segment at the end of the live range.
    pub fn has_liveout(self) -> bool {
        self.to & USE_LIST_EXTRA_BIT != 0
    }

    /// Sets the live-in bit to the given value.
    pub fn set_livein(&mut self, val: bool) {
        if val {
            self.from |= USE_LIST_EXTRA_BIT;
        } else {
            self.from &= !USE_LIST_EXTRA_BIT;
        }
    }

    /// Sets the live-out bit to the given value.
    pub fn set_liveout(&mut self, val: bool) {
        if val {
            self.to |= USE_LIST_EXTRA_BIT;
        } else {
            self.to &= !USE_LIST_EXTRA_BIT;
        }
    }

    /// Returns the number of [`Use`]s in this list.
    pub fn len(self) -> usize {
        self.indices().len()
    }

    /// Returns a [`UseIndex`] pointing to a single [`Use`] in this list.
    pub fn index(self, index: usize) -> UseIndex {
        debug_assert!(index < self.len());
        UseIndex((self.from & !USE_LIST_EXTRA_BIT) + index as u32)
    }

    /// Returns an iterator over all the [`UseIndex`] in this list.
    pub fn iter(self) -> impl DoubleEndedIterator<Item = UseIndex> + ExactSizeIterator {
        (0..self.len()).map(move |i| self.index(i))
    }

    /// Returns the range of indices encoded in the `UseList`.
    fn indices(self) -> Range<usize> {
        let from = (self.from & !USE_LIST_EXTRA_BIT) as usize;
        let to = (self.to & !USE_LIST_EXTRA_BIT) as usize;
        from..to
    }

    /// Splits the given use list at the given instruction.
    ///
    /// All uses before the given instruction are returned in the first list,
    /// and all uses at or after the given instruction are returned in the
    /// second list.
    pub fn split_at_inst(self, split_at: Inst, uses: &Uses) -> (Self, Self) {
        let split_index = uses[self].partition_point(|u| u.pos() < split_at);
        let mid = UseIndex((self.from & !USE_LIST_EXTRA_BIT) + split_index as u32);
        let (mut first, mut second) = self.split_at_index(mid);

        // Mark the use lists as having a matching live-in/live-out.
        first.set_liveout(true);
        second.set_livein(true);
        (first, second)
    }

    /// Splits the given use list at the given `UseIndex`.
    pub fn split_at_index(self, mid: UseIndex) -> (Self, Self) {
        debug_assert!(mid.0 as usize >= self.indices().start);
        debug_assert!((mid.0 as usize) <= self.indices().end);
        let first = UseList {
            from: self.from,
            to: mid.0,
        };
        let second = UseList {
            from: mid.0,
            to: self.to,
        };
        (first, second)
    }
}

pub struct Uses {
    /// List of `Use`s ordered by value and position.
    uses: Vec<Use>,

    /// Portion of the `uses` vector that has been sorted with `sort_uses`.
    ///
    /// This portion has been sorted by `value` and then `pos`, which allows
    /// using `resolve_use_list` to retrieve a continuous slice of `Use`s for a
    /// `UseList`.
    sorted_until: usize,
}

impl Uses {
    pub fn new() -> Uses {
        Self {
            uses: vec![],
            sorted_until: 0,
        }
    }

    pub fn clear(&mut self) {
        self.uses.clear();
        self.sorted_until = 0;
    }

    /// Returns the `UseIndex` for the next use that will be inserted with
    /// `add_unsorted_use`.
    pub fn next_unsorted_use(&self) -> UseIndex {
        UseIndex(self.uses.len() as u32)
    }

    /// Adds an unsorted `Use` to the vector.
    pub fn add_unsorted_use(
        &mut self,
        use_pos: UsePosition,
        value: Value,
        kind: UseKind,
    ) -> UseIndex {
        trace!("Adding use of {value} at {}: {kind}", use_pos.inst());
        let idx = self.next_unsorted_use();
        self.uses.push(Use {
            use_pos,
            value,
            kind,
        });
        idx
    }

    /// Adds a new `UseList` after the uses have been sorted.
    ///
    /// `resolve_use_list` can no longer be called after this is used.
    pub fn add_use_list(&mut self, u: impl IntoIterator<Item = Use>) -> UseList {
        let from = self.uses.len() as u32;
        self.uses.extend(u);
        let to = self.uses.len() as u32;
        UseList { from, to }
    }

    /// Sorts the list of uses by `value` and then `pos`, which allows using
    /// `resolve_use_list` to retrieve a continuous slice of `Use`s for a
    /// `UseList`.
    ///
    /// This invalidates all current `UseList`s and `UseIndex`es.
    ///
    /// Returns a `UseIndex` pointing to the start of the use list for the
    /// first value (%0).
    pub fn sort_uses(&mut self) -> UseIndex {
        self.uses.sort_unstable_by_key(|u| {
            // Sort by value first, then by use position.
            (u.value.index() as u64) << 32 | u.use_pos.bits as u64
        });
        self.sorted_until = self.uses.len();

        UseIndex(0)
    }

    /// Returns a `UseList` which covers all the uses of `value`.
    ///
    /// This can only be called after `sort_uses` has been called.
    pub fn resolve_use_list(&self, value: Value, start: UseIndex) -> (UseList, UseIndex) {
        let end =
            UseIndex(self.uses[..self.sorted_until].partition_point(|u| u.value <= value) as u32);
        (UseList::new(start, end), end)
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
