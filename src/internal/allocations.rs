//! Table of [`Allocation`]s for instruction operands.

use alloc::vec;
use alloc::vec::Vec;

use cranelift_entity::packed_option::ReservedValue;
use cranelift_entity::EntityRef;

use crate::function::{Function, Inst};
use crate::output::Allocation;
use crate::{RegAllocError, Stats};

/// Mapping of [`Allocation`]s back to original instruction operands.
pub struct Allocations {
    /// Allocations for the operands of all instructions in the function.
    allocations: Vec<Allocation>,

    /// Offset of the allocations for a particular instruction in the
    /// `allocations` vector.
    operands_offset: Vec<u32>,
}

impl Allocations {
    pub fn new() -> Self {
        Self {
            allocations: vec![],
            operands_offset: vec![],
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
        self.operands_offset.clear();

        let mut offset = 0;
        for inst in func.insts() {
            self.operands_offset.push(offset);
            offset = offset
                .checked_add(func.inst_operands(inst).len() as u32)
                .ok_or(RegAllocError::FunctionTooBig)?;
        }

        // Insert a placeholder at the end to help calculate the offset for the
        // last instruction.
        self.operands_offset.push(offset);

        // Fill the allocation map with invalid allocations.
        self.allocations
            .resize(offset as usize, Allocation::reserved_value());
        stat!(stats, operands, offset as usize);

        Ok(())
    }

    /// Assigns the given [`Allocation`] to an instruction operand.
    pub fn set_allocation(&mut self, inst: Inst, slot: u16, alloc: Allocation) {
        let idx = self.operands_offset[inst.index()] as usize + slot as usize;
        self.allocations[idx] = alloc;
    }

    /// Returns the allocations for an instruction's operands.
    pub fn inst_allocations(&self, inst: Inst) -> &[Allocation] {
        let start = self.operands_offset[inst.index()] as usize;
        let end = self.operands_offset[inst.index() + 1] as usize;
        &self.allocations[start..end]
    }

    /// Returns a mutable slice of allocations for an instruction's operands.
    pub fn inst_allocations_mut(&mut self, inst: Inst) -> &mut [Allocation] {
        let start = self.operands_offset[inst.index()] as usize;
        let end = self.operands_offset[inst.index() + 1] as usize;
        &mut self.allocations[start..end]
    }

    /// Asserts that all allocations have been assigned.
    pub fn assert_all_assigned(&self) {
        self.allocations
            .iter()
            .for_each(|alloc| debug_assert!(!alloc.is_reserved_value()));
    }
}
