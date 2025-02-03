//! The [`RegInfo`] trait which describes the registers and register classes in
//! the target ISA.
//!
//! The registers are described using a hierarchy of types:
//! * A [`RegBank`] is a top-level grouping of registers, and contains one or
//!   more [`RegClass`].
//! * A [`RegClass`] represents a selection of [`PhysReg`] or [`RegGroup`] to
//!   allocate for a instruction operand.
//! * A [`RegGroup`] represents a sequence of [`PhysReg`] which are allocated
//!   together for a single operand.
//! * A [`PhysReg`] consists of one or more [`RegUnit`] which are the smallest
//!   unit of allocation.
//!
//! # Physical registers
//!
//! A [`PhysReg`] is the unit of allocation that the register allocator works
//! with. Fundamentally, the job of the register allocator is to assign a
//! [`PhysReg`] (or a spill slot) to every use of a [`Value`] in a function.
//!
//! # Register banks
//!
//! A [`RegBank`] represents a set of [`PhysReg`] between which values can be
//! moved with `move` instructions. Register banks must not overlap: each
//! [`PhysReg`] may only be part of a single register bank.
//!
//! Each [`Value`] must be located in one register bank: all operand constraints
//! related to this value must use registers and register classes from that
//! register bank. Similarly, any pair of values linked together through tied
//! operands or  through block parameters must share the same register bank.
//!
//! # Register classes
//!
//! A [`RegClass`] represents a subset of the registers of a [`RegBank`] which
//! can be used by an instruction. Specifically, it describes which registers
//! are allowed to satisfy a given constraint, and whether the constraint can
//! be satisfied by stack spill slot (for instruction which can accept either a
//! register or a memory operand).
//!
//! Each register bank has an associated *top-level register class*. This is a
//! register class which covers the entire set of registers in the bank.
//! Top-level register classes are required to allow moves to/from spillslots.
//!
//! Each register bank also has an associated *stack-to-stack register class*.
//! This is used when a scratch register is required to resolve a move between
//! 2 memory locations.
//!
//! Register classes may overlap either completely or partially with other
//! register classes in the same bank.
//!
//! # Allocation order
//!
//! Each register class has an *allocation order* which is the order in which
//! the register allocator will attempt to select a suitable register for a
//! constraint of this class.
//!
//! The [`RegInfo`] trait specifies the allocation order as 4 sets of registers:
//! - [`AllocationOrderSet::Preferred`]
//! - [`AllocationOrderSet::NonPreferred`]
//! - [`AllocationOrderSet::CalleeSavedPreferred`]
//! - [`AllocationOrderSet::CalleeSavedNonPreferred`]
//!
//! Preferred registers generally have a more compact or efficient encoding than
//! non-preferred registers. Callee-saved registers are treated as normal
//! registers if they have already been used at least once in the function, and
//! are otherwise given the lowest priority. This minimizes the need for saving
//! and restoring these registers on function entry and exit.
//!
//! For a minimal implementation, it is possible to only specify registers as
//! [`AllocationOrderSet::Preferred`] and leave the other sets empty.
//!
//! Any registers that are members of a class but not in the allocation order
//! are only selected by the register allocator if it helps to satisfy a `Fixed`
//! register constraint and are otherwise never used.
//!
//! # Fixed stack slots
//!
//! A [`PhysReg`] can also be used to represent a memory location instead of a
//! register. This is useful to model stack slots for function arguments and
//! return values which
//!
//! Since it is undesirable to use these stack slots unless specifically needed,
//! these can be made part of a register class without being part of its
//! allocation order. This means that these [`PhysReg`] will only be selected if
//! necessary to satisfy a `Fixed` constraint, but not a `Class` constraint.
//!
//! # Compound registers
//!
//! Some architectures may have registers that overlap: for example on AArch32
//! the `D0` 64-bit register overlaps with the `S0` and `S1` 32-bit registers.
//! This is modeled in the register allocator by using "register units": `S0`
//! and `S1` can be defined as covering single register units `unit0` and
//! `unit1`, while `D0` can be defined as a compound register covering both
//! `unit0` and `unit1`.
//!
//! Liveness is only tracked at the register unit level, so the register units
//! must completely cover all the bits of the compound register.
//!
//! Overlaps are only allowed with registers in other banks (i.e. `D0` and `S0`
//! must be in different register banks). Within a bank no registers may share
//! a unit.
//!
//! Compound registers are still limited to holding a single [`Value`]. They
//! only interact with register units to avoid conflicts with values held in
//! other registers than they share a register unit with.
//!
//! # Register groups
//!
//! Some instructions take a sequence of registers as operands, where only the
//! first register is encoded in the instruction (e.g. `LD4` on AArch64). This
//! cannot be modeled using compound registers because each register needs to
//! be assigned a different [`Value`].
//!
//! The register allocator can model this using register groups, where a
//! [`RegGroup`] represents a sequence of [`PhysReg`]. Register classes are
//! generalized to contain either a set of [`PhysReg`] or a set of [`RegGroup`],
//! with the restriction that all register groups in a class must have the same
//! number of registers.
//!
//! Instruction operands can then be specified to require a group of values
//! using [`OperandKind::UseGroup`], [`OperandKind::DefGroup`] or
//! [`OperandKind::EarlyDefGroup`]. These operands take a [`ValueGroup`] which
//! represents a list of [`Value`]s.
//!
//! Group operands can only be used with [`OperandConstraint::Class`] and the
//! class must be a class of register groups where groups have the same number
//! of registers as the number of [`Value`]s in the [`ValueGroup`].
//!
//! The [`Allocation`] in [`OutputInst::Inst::operand_allocs`] will only contain a
//! single entry for the operand which holds the first register of the register
//! group that was allocated for the operand.
//!
//! [`Value`]: super::function::Value
//! [`ValueGroup`]: super::function::ValueGroup
//! [`OperandKind::UseGroup`]: super::function::OperandKind::UseGroup
//! [`OperandKind::DefGroup`]: super::function::OperandKind::DefGroup
//! [`OperandKind::EarlyDefGroup`]: super::function::OperandKind::EarlyDefGroup
//! [`OperandConstraint::Class`]: super::function::OperandConstraint::Class
//! [`OutputInst::Inst::operand_allocs`]: super::output::OutputInst::Inst::operand_allocs
//! [`Operand`]: super::function::Operand
//! [`Allocation`]: super::output::Allocation

use core::fmt;

use crate::entity::iter::Keys;
use crate::entity::SmallEntitySet;

/// Maximum number of register units.
pub const MAX_REG_UNITS: usize = 512;

/// Maximum number of registers.
pub const MAX_PHYSREGS: usize = 512;

/// Maximum number of register groups.
pub const MAX_REG_GROUPS: usize = 512;

/// Maximum number of register classes.
pub const MAX_REG_CLASSES: usize = 64;

/// Maximum number of register banks.
pub const MAX_REG_BANKS: usize = 64;

/// Maximum number of register units in a compound register.
pub const MAX_UNITS_PER_REG: usize = 8;

/// Maximum size of a register group.
pub const MAX_GROUP_SIZE: usize = 8;

entity_def! {
    /// A subset of a register which cannot be further split.
    ///
    /// For simple cases, a [`PhysReg`] will directly map to a single register unit.
    /// However some ISAs have registers which overlap with others: this can be
    /// modeled by having that register cover more than one register unit.
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub entity RegUnit(u16, "unit");

    /// A location to which a value can be mapped to.
    ///
    /// Note that despite the name, this doesn't have to be a machine register. A
    /// fixed stack slot (typically used for function arguments and return values)
    /// can also be represented as a `PhysReg`.
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub entity PhysReg(u16, "r");

    /// A sequence of multiple registers which are assigned together for a single
    /// operand.
    ///
    /// This type is only used for groups with more than one register. When only a
    /// single register is used, [`PhysReg`] is used directly.
    ///
    /// In most cases these only consist of a single [`PhysReg`], but some
    /// instruction operands require a sequence of registers from a specific set.
    /// An example of this is an AArch64 SIMD structured load which only encodes the
    /// first register of the sequence in the instruction.
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub entity RegGroup(u16, "rg");

    /// A set of [`PhysReg`] or [`RegGroup`] which can be allocated for an operand.
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub entity RegClass(u8, "class");

    /// A set of registers between which values can be copied with move
    /// instructions.
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub entity RegBank(u8, "bank");
}

/// A set of [`RegUnit`] encoded as a fixed-size bit set.
pub type RegUnitSet = SmallEntitySet<RegUnit, u64, { MAX_REG_UNITS / 64 }>;

/// A set of [`PhysReg`] encoded as a fixed-size bit set.
pub type PhysRegSet = SmallEntitySet<PhysReg, u64, { MAX_PHYSREGS / 64 }>;

/// A set of [`RegGroup`] encoded as a fixed-size bit set.
pub type RegGroupSet = SmallEntitySet<RegGroup, u64, { MAX_REG_GROUPS / 64 }>;

/// A set of [`RegClass`] encoded as a fixed-size bit set.
pub type RegClassSet = SmallEntitySet<RegClass, u64, { MAX_REG_CLASSES / 64 }>;

/// The size of a spillslot, which must be a power of two.
///
/// This is represented compactly as a `u8` holding the log2 of the size.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpillSlotSize(u8);

impl fmt::Display for SpillSlotSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.bytes())
    }
}

impl SpillSlotSize {
    /// Returns a `SpillSlotSize` of `bytes` bytes.
    ///
    /// `bytes` must be a power of two.
    #[inline]
    #[must_use]
    pub const fn new(bytes: u32) -> Self {
        debug_assert!(bytes.is_power_of_two());
        Self::from_log2_bytes(bytes.trailing_zeros())
    }

    /// Returns a `SpillSlotSize` of `1 << log2_bytes` bytes.
    ///
    /// `log2_bytes` must be less than 32.
    #[inline]
    #[must_use]
    pub const fn from_log2_bytes(log2_bytes: u32) -> Self {
        debug_assert!(log2_bytes < 32);
        Self(log2_bytes as u8)
    }

    /// Returns the size of the spill slot in bytes.
    #[inline]
    #[must_use]
    pub const fn bytes(self) -> u32 {
        1 << self.log2_bytes()
    }

    /// Returns the log2 of the size of the spill slot in bytes.
    #[inline]
    #[must_use]
    pub const fn log2_bytes(self) -> u32 {
        self.0 as u32
    }
}

/// Sets of registers in a [`RegClass`] grouped by their priority for register
/// allocation.
///
/// See [`RegInfo::allocation_order`] for more details.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AllocationOrderSet {
    /// Registers that are free to use and have a shorter or more efficient
    /// instruction encoding.
    Preferred,

    /// Registers that are free to use and have a longer or less efficient
    /// instruction encoding.
    NonPreferred,

    /// Registers that have a one-time cost on their first use in a function,
    /// but should otherwise be treated as `Preferred`.
    CalleeSavedPreferred,

    /// Registers that have a one-time cost on their first use in a function,
    /// but should otherwise be treated as `NonPreferred`.
    CalleeSavedNonPreferred,
}

impl AllocationOrderSet {
    /// Iterator for each set.
    pub(crate) fn each() -> impl DoubleEndedIterator<Item = Self> {
        [
            Self::Preferred,
            Self::NonPreferred,
            Self::CalleeSavedPreferred,
            Self::CalleeSavedNonPreferred,
        ]
        .into_iter()
    }
}

/// Trait which describes the physical registers and register classes that may
/// be allocated from.
///
/// This is separate from the `Function` trait since this data is usually
/// independent of the function currently being compiled.
///
/// See the [module-level documentation] for more details.
///
/// [module-level documentation]: self
pub trait RegInfo {
    // --------------
    // Register banks
    // --------------

    /// Get the number of [`RegBank`] in use.
    fn num_banks(&self) -> usize;

    /// Iterator over all the [`RegBank`]s.
    #[inline]
    fn banks(&self) -> Keys<RegBank> {
        Keys::with_len(self.num_banks())
    }

    /// Returns the top-level register class of the given register bank.
    ///
    /// This class encompasses all of the register in the bank, and is required
    /// to accept spillslot allocations. It is a superclass of all other classes
    /// in this bank.
    fn top_level_class(&self, bank: RegBank) -> RegClass;

    /// Returns the class from which to select scratch registers for splitting
    /// stack-to-stack moves.
    ///
    /// Move instructions on most ISAs cannot perform a memory-to-memory copy,
    /// so such moves must be split into 2 halves using an intermediate scratch
    /// register from this class.
    ///
    /// The returned class must not be a group class and cannot contain any
    /// register for which [`RegInfo::is_memory`] is true.
    fn stack_to_stack_class(&self, bank: RegBank) -> RegClass;

    /// Returns the register bank that the given [`RegClass`] is a member of.
    fn bank_for_class(&self, class: RegClass) -> RegBank;

    /// Returns the register bank that the given [`PhysReg`] is a member of.
    ///
    /// Returns `None` for non-allocatable registers.
    fn bank_for_reg(&self, reg: PhysReg) -> Option<RegBank>;

    /// Spillslot size needed for a value in this register bank.
    ///
    /// The spillslot is guaranteed to be aligned to this size.
    fn spillslot_size(&self, bank: RegBank) -> SpillSlotSize;

    // ----------------
    // Register classes
    // ----------------

    /// Get the number of [`RegClass`] in use.
    fn num_classes(&self) -> usize;

    /// Iterator over all [`RegClass`]es.
    #[inline]
    fn classes(&self) -> Keys<RegClass> {
        Keys::with_len(self.num_classes())
    }

    /// Returns the set of [`PhysReg`] contained within a register class.
    ///
    /// This indicates that one of these registers may be assigned to an operand
    /// constrainted to `class`. This is the case even if the register is not a
    /// part of the allocation order.
    ///
    /// This must be empty when `class_group_size > 1` for this class.
    fn class_members(&self, class: RegClass) -> PhysRegSet;

    /// Returns the set of [`RegGroup`] contained within a register class.
    ///
    /// This indicates that one of these register groups may be assigned to an
    /// operand constrainted to `class`. This is the case even if the register
    /// is not a part of the allocation order.
    ///
    /// This must be empty when `class_group_size == 1` for this class.
    fn class_group_members(&self, class: RegClass) -> RegGroupSet;

    /// Whether an operand constrainted to this register class can be allocated
    /// to a [`SpillSlot`] instead of a [`PhysReg`].
    ///
    /// This must return `true` for top-level register classes, and `false` for
    /// register group classes.
    ///
    /// [`SpillSlot`]: super::output::SpillSlot
    fn class_includes_spillslots(&self, class: RegClass) -> bool;

    /// The spill cost of a class is defined as the cost that needs to be paid
    /// if a value constrained to this class is spilled to the stack instead of
    /// allocated to a register.
    ///
    /// For most register classes, a default value of 1.0 can be used which
    /// indicates a cost of a single spill/reload instruction. In fact, there is
    /// no reason to use any other value if the class does not allow direct use
    /// of spillslots.
    ///
    /// For register classes that can be allocated to spillslots, the value
    /// should represent the cost of accessing a stack slot compared to a real
    /// register. A good value is aroud 0.5, which indicates a cost that is not
    /// as big as a full spill/reload, but still
    ///
    /// A spill cost of 0.0 indicates that there is no cost to spilling the
    /// value to the stack. This is useful for stack constraints (e.g. used for
    /// GC roots) or for values that are only read by trap handlers.
    fn class_spill_cost(&self, class: RegClass) -> f32;

    /// Returns a set of [`PhysReg`] to try allocating for an operand
    /// constrained to the given register class.
    ///
    /// Several sets of registers can be provided with different priority
    /// levels as determined by the [`AllocationOrderSet`] enum. The register
    /// allocator will attempt to select an available registers by probing the
    /// sets in this order:
    /// * Preferred registers.
    /// * Preferred callee-saved registers that have already been used in the
    ///   function.
    /// * Non-preferred registers.
    /// * Non-preferred callee-saved registers that have already been used in
    ///   the function.
    /// * Preferred callee-saved registers that have not yet been used in the
    ///   function.
    /// * Non-preferred callee-saved registers that have not yet been used in
    ///   the function.
    ///
    /// This arrangement prioritizes registers with smaller encodings and
    /// penalizes registers that need to be saved on function entry and
    /// restored on function exit (but only if that cost hasn't already been
    /// paid).
    ///
    /// All registers returned by this function must be a member of the register
    /// class. However not all members need to be in the allocation order.
    /// Registers outside the allocation order will only be selected to satisfy
    /// a fixed-register operand constraint. This is useful for "fake" registers
    /// such as fixed stack slots which are slower to access than a register.
    ///
    /// This must be empty when `class_group_size > 1` for this class.
    fn allocation_order(&self, class: RegClass, set: AllocationOrderSet) -> &[PhysReg];

    /// Returns a set of [`RegGroup`] to try allocating for an operand
    /// constrained to the given register class.
    ///
    /// This is similar to [`RegInfo::allocation_order`] except that it returns
    /// a list of [`RegGroup`] instead of [`PhysReg`].
    ///
    /// This must be empty when `class_group_size == 1` for this class.
    fn group_allocation_order(&self, class: RegClass, set: AllocationOrderSet) -> &[RegGroup];

    /// Returns the set of sub-classes of `class`, including itself.
    ///
    /// A sub-class must be from the same register bank as its superclass, and
    /// must contain a strict subset of the registers in the superclass.
    ///
    /// A class of register groups can only have other group classes of the same
    /// group size as sub-classes. However it may have a non-group class as a
    /// superclass as long as
    ///
    /// Classes must be topologically ordered: sub-classes must have a higher
    /// index than their superclass. Additionally, all register classes in a
    /// register bank must be a subclass of the top-level class of that bank.
    fn sub_classes(&self, class: RegClass) -> RegClassSet;

    /// Computes the largest common sub-class of the two register classes, or
    /// `None` if no common sub-class exists.
    #[inline]
    fn common_subclass(&self, a: RegClass, b: RegClass) -> Option<RegClass> {
        // Since classes are ordered topologically, this is the lowest indexed
        // class which is a sub-class of both original classes.
        (self.sub_classes(a) & self.sub_classes(b))
            .into_iter()
            .next()
    }

    /// The group size of registers in this class.
    ///
    /// If this returns a value greater than 1 then all [`PhysReg`] in this
    /// register class must be register groups with the given number of members,
    /// with each member coming from the register bank of this class.
    fn class_group_size(&self, class: RegClass) -> usize;

    // ---------
    // Registers
    // ---------

    /// Get the number of [`PhysReg`] in use.
    fn num_regs(&self) -> usize;

    /// Iterator over all [`PhysReg`]s.
    #[inline]
    fn regs(&self) -> Keys<PhysReg> {
        Keys::with_len(self.num_regs())
    }

    /// Returns the set of register units covered by a register.
    ///
    /// See the module-level documentation for more details.
    ///
    /// This is ignored for non-allocatable registers used with
    /// [`OperandKind::NonAllocatable`]. It can be an empty list in such cases.
    ///
    /// [`OperandKind::NonAllocatable`]: super::function::OperandKind::NonAllocatable
    fn reg_units(&self, reg: PhysReg) -> impl ExactSizeIterator<Item = RegUnit>;

    /// Returns whether this [`PhysReg`] represents a memory location (e.g. a
    /// fixed stack slot) instead of a register.
    ///
    /// [`Allocation::is_memory`] will forward to this function when the
    /// [`Allocation`] represents a [`PhysReg`].
    ///
    /// This has 2 effects on register allocation:
    /// - Memory to memory moves are resolved by using a temporary scratch
    ///   register since such more are not supported by hardware.
    /// - Rematerialization can be more aggressive when it can avoid a load from
    ///   memory, depending on [`RematCost`].
    ///
    /// [`RematCost`]: super::function::RematCost
    /// [`Allocation`]: super::output::Allocation
    /// [`Allocation::is_memory`]: super::output::Allocation::is_memory
    fn is_memory(&self, reg: PhysReg) -> bool;

    // ---------------
    // Register groups
    // ---------------

    /// Get the number of [`RegGroup`] in use.
    fn num_reg_groups(&self) -> usize;

    /// Iterator over all [`RegGroup`]s.
    #[inline]
    fn reg_groups(&self) -> Keys<RegGroup> {
        Keys::with_len(self.num_reg_groups())
    }

    /// Returns the sequence of registers for a [`RegGroup`].
    fn reg_group_members(&self, group: RegGroup) -> &[PhysReg];

    /// Searches members of `class` for a [`RegGroup`] which contains
    /// `reg` at position `group_index` within the group.
    ///
    /// This can only be called for register classes with a group size greater
    /// than 1.
    fn group_for_reg(&self, reg: PhysReg, group_index: usize, class: RegClass) -> Option<RegGroup>;
}
