//! Table of [`Allocation`]s for instruction operands and stack maps.

use alloc::vec;
use alloc::vec::Vec;

use cranelift_entity::packed_option::ReservedValue;
use cranelift_entity::EntityRef;

use crate::function::{Function, Inst};
use crate::output::Allocation;
use crate::{RegAllocError, Stats};

/// Information about the stack map for an instruction.
#[derive(Debug, Clone)]
struct Offsets {
    /// Offset of the operand allocations in the `allocations` vector.
    operands_offset: u32,

    /// Number of entries to allocate for the instruction's stack map.
    ///
    /// In the move resolution phase after offsets are calculated this is used
    /// to count the number of allocations that have been initialized so far in
    /// the stack map.
    num_stack_map: u32,

    /// Offset of the stack map allocations in the `allocations` vector.
    stack_map_offset: u32,
}

/// Mapping of [`Allocation`]s back to original instruction operands.
pub struct Allocations {
    /// Allocations for the operands of all instructions in the function.
    allocations: Vec<Allocation>,

    /// Offset of the allocations for a particular instruction in the
    /// `allocations` vector.
    offsets: Vec<Offsets>,
}

impl Allocations {
    pub fn new() -> Self {
        Self {
            allocations: vec![],
            offsets: vec![],
        }
    }

    /// Computes the offsets for each instruction's operands in the allocation
    /// vector.
    pub fn compute_alloc_offsets(
        &mut self,
        func: &impl Function,
        stats: &mut Stats,
    ) -> Result<(), RegAllocError> {
        self.allocations.clear();
        self.offsets.clear();
        self.offsets.resize(
            func.num_insts() + 1,
            Offsets {
                operands_offset: 0,
                num_stack_map: 0,
                stack_map_offset: 0,
            },
        );

        let mut offset = 0;
        for inst in func.insts() {
            self.offsets[inst.index()].operands_offset = offset;
            offset = offset
                .checked_add(func.inst_operands(inst).len() as u32)
                .ok_or(RegAllocError::FunctionTooBig)?;
        }

        // Insert a placeholder at the end to help calculate the offset for the
        // last instruction.
        self.offsets[func.num_insts()].operands_offset = offset;

        // Fill the allocation map with invalid allocations.
        self.allocations
            .resize(offset as usize, Allocation::reserved_value());
        stat!(stats, operands, offset as usize);

        Ok(())
    }

    /// Assigns the given [`Allocation`] to an instruction operand.
    pub fn set_allocation(&mut self, inst: Inst, slot: u16, alloc: Allocation) {
        let idx = self.offsets[inst.index()].operands_offset as usize + slot as usize;
        self.allocations[idx] = alloc;
    }

    /// Returns the allocations for an instruction's operands.
    pub fn inst_allocations(&self, inst: Inst) -> &[Allocation] {
        let start = self.offsets[inst.index()].operands_offset as usize;
        let end = self.offsets[inst.index() + 1].operands_offset as usize;
        &self.allocations[start..end]
    }

    /// Asserts that all allocations have been assigned.
    pub fn assert_all_assigned(&self) {
        self.allocations
            .iter()
            .for_each(|alloc| debug_assert!(!alloc.is_reserved_value()));
    }

    /// Reserves an entry in the stack map for an instruction.
    ///
    /// Must be called before `compute_stack_map_offsets`.
    pub fn reserve_stack_map_entry(&mut self, inst: Inst) {
        self.offsets[inst.index()].num_stack_map += 1;
    }

    /// Computes the offsets of stack map allocations for each instruction.
    pub fn compute_stack_map_offsets(&mut self, func: &impl Function) -> Result<(), RegAllocError> {
        let mut offset = self.allocations.len() as u32;
        for inst in func.insts() {
            self.offsets[inst.index()].stack_map_offset = offset;
            offset = offset
                .checked_add(self.offsets[inst.index()].num_stack_map)
                .ok_or(RegAllocError::FunctionTooBig)?;
            self.offsets[inst.index()].num_stack_map = 0;
        }

        // Insert a placeholder at the end to help calculate the offset for the
        // last instruction.
        self.offsets[func.num_insts()].stack_map_offset = offset;

        // Fill the allocation map with invalid allocations.
        self.allocations
            .resize(offset as usize, Allocation::reserved_value());

        Ok(())
    }

    /// Adds the given allocation to the stack map of an instruction.
    pub fn add_stack_map_alloc(&mut self, inst: Inst, alloc: Allocation) {
        let offset =
            self.offsets[inst.index()].stack_map_offset + self.offsets[inst.index()].num_stack_map;
        debug_assert!(offset < self.offsets[inst.index() + 1].stack_map_offset);
        self.offsets[inst.index()].num_stack_map += 1;
        self.allocations[offset as usize] = alloc;
    }

    /// Returns the allocations for an instruction's stack map.
    pub fn inst_stack_map(&self, inst: Inst) -> &[Allocation] {
        let start = self.offsets[inst.index()].stack_map_offset as usize;
        let end = self.offsets[inst.index() + 1].stack_map_offset as usize;
        &self.allocations[start..end]
    }
}
