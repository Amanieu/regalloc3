//! Instruction-local constraint satisfiability checker.
//!
//! This checks that the allocator is guaranteed to find a solution for the
//! given constraints on an instruction.
//!
//! This analysis can be local to a single instruction because the allocator
//! can keep splitting live ranges until they are minimal and only cover a
//! single instruction.
//!
//! It is not sufficient to guarantee that a solution is possible however: the
//! allocator does not perform any backtracking when live ranges are
//! unsplittable. Instead we need to prove that regardless of operand allocation
//! order and which currently valid register is selected, allocation can always
//! continue until every operand is assigned.
//!
//! Consider this example:
//!
//! ```text
//! class0 = {r0, r1}
//! class1 = {r0}
//! Use(%0):class0 Use(%1):class1
//! ```
//!
//! If `%0` is assigned to `r0` then `%1` has no solution, but this isn't the
//! case if `%1` is assigned first.

use crate::{
    entity::{PrimaryMap, SecondaryMap, SmallEntitySet},
    function::{Function, Operand, OperandConstraint, OperandKind, Value, ValueGroup},
    reginfo::{
        MAX_PHYSREGS, MAX_REG_UNITS, PhysReg, RegClass, RegClassSet, RegGroup, RegInfo, RegUnit,
        RegUnitSet,
    },
};
use alloc::vec;
use alloc::vec::Vec;
use core::{error::Error, fmt};

/// Error indicating that the checker has found an operand which it cannot
/// guarantee that the allocator will be able to successfully allocate.
#[derive(Debug)]
pub struct ConstraintConflict {
    /// Index in the original instruction operand list.
    operand_idx: usize,

    /// Number of allocation candidates for the class.
    num_candidates: usize,

    /// Upper bound on candidates blocked by reservations and other operands.
    blocked: usize,
}

impl ConstraintConflict {
    /// Returns the index of the conflicting operand in the original instruction
    /// operand list.
    #[inline]
    pub fn conflicting_operand(&self) -> usize {
        self.operand_idx
    }
}

impl fmt::Display for ConstraintConflict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Satisfiability of operand {} cannot be guaranteed: {} candidates, \
             up to {} blocked by fixed reservations and other operands",
            self.operand_idx, self.num_candidates, self.blocked
        )
    }
}

impl Error for ConstraintConflict {}

entity_def! {
    /// An index into the allocation order of a register class.
    entity Candidate(u16, "candidate");
}

// A set of candidates.
type CandidateSet = SmallEntitySet<Candidate, u64, { MAX_PHYSREGS / 64 }>;

/// Pre-computed cached data for each class.
#[derive(Default)]
struct ClassData {
    /// Number of entries in the class's allocation order.
    num_candidates: usize,

    /// For each register unit, the set of candidates that become unavailable
    /// if that unit is reserved for a fixed constraint or clobber.
    blocked_by_unit: SecondaryMap<RegUnit, CandidateSet>,

    /// For each other register class, the number of candidates that become
    /// unavailable if a member of that class is already allocated. This is
    /// computed as a worst case estimate.
    ///
    /// Note that while candidates only refer to the allocation order of a
    /// class, we check against any member of other classes, not just the ones
    /// in the allocation order. The rationale here is that members outside the
    /// allocation order are only selected through hints, and we need our class
    /// to be allocatable without hints while considering the worst case of
    /// other register classes using their hints.
    max_blocked_by_class: SecondaryMap<RegClass, usize>,
}

/// The instruction phases occupied by an operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phases {
    Input = 0b01,
    Output = 0b10,
    Both = 0b11,
}

impl Phases {
    fn overlaps(self, other: Self) -> bool {
        self as u32 & other as u32 != 0
    }
}

/// Wrapper around either a `Value` or a `ValueGroup`.
#[derive(Copy, Clone)]
enum ValueOrGroup {
    Value(Value),
    Group(ValueGroup),
}

/// Information about an operand for the checker.
struct OperandData {
    /// Index in the original instruction operand list.
    idx: usize,

    /// Original register class of this operand.
    class: RegClass,

    /// Phases that this operand covers.
    phases: Phases,

    /// Value of this operand.
    value: ValueOrGroup,

    /// Classes represented by this operand after merging with other operands
    /// that it shares a value with. Each is a subclass of the original class
    /// and has the same group width.
    ///
    /// Across all operands sharing a value, these sets must represent every
    /// possible class resulting from those merges. Each possibility need only
    /// be represented on one operand.
    ///
    /// It is important to consider all possibilities here because the order in
    /// which the allocator merges operands with the same value can affect the
    /// feasibility of later merges.
    possible_classes: RegClassSet,
}

/// Instruction-local constraint satisfiability checker.
///
/// This checks that the allocator is guaranteed to find a solution for the
/// given constraints on an instruction.
///
/// This analysis can be local to a single instruction because the allocator
/// can keep splitting live ranges until they are minimal and only cover a
/// single instruction.
///
/// If all instructions pass this check then the allocator is guaranteed to find
/// an allocation solution.
pub struct ConstraintChecker<'a, R> {
    reginfo: &'a R,

    /// Cached per-class data
    class_data: SecondaryMap<RegClass, ClassData>,

    /// Cached set of register classes which don't contribute to register
    /// constraints and can be ignored.
    unconstrained_classes: RegClassSet,

    /// Classes that do not allow spill slots.
    register_classes: RegClassSet,

    /// Set of input operands that are reused by an output.
    reused_inputs: Vec<usize>,

    /// Operands that are relevant to the constraints.
    operands: Vec<OperandData>,

    /// Register units reserved in the input phase by fixed constraints.
    fixed_input: RegUnitSet,

    /// Register units reserved at the output phase by fixed constraints and clobbers.
    fixed_output: RegUnitSet,
}

impl<'a, R: RegInfo> ConstraintChecker<'a, R> {
    /// Creates a new `ConstraintChecker`.
    ///
    /// This pre-computes some per-class metadata needed by the checker so it
    /// doesn't have to be done for every instruction.
    pub fn new(reginfo: &'a R) -> Self {
        // Collect the set of register classes that don't accept spillslots.
        let mut register_classes = RegClassSet::new();
        for class in reginfo.classes() {
            if !reginfo.class_includes_spillslots(class) {
                register_classes.insert(class);
            }
        }

        // Collect register unit masks for each register-only class candidate.
        let mut candidate_units =
            SecondaryMap::<RegClass, PrimaryMap<Candidate, RegUnitSet>>::with_max_index(
                reginfo.num_classes(),
            );
        for class in register_classes {
            if reginfo.class_group_size(class) == 1 {
                for &reg in reginfo.allocation_order(class) {
                    let units = reginfo.reg_units(reg).collect();
                    candidate_units[class].push(units);
                }
            } else {
                for &group in reginfo.group_allocation_order(class) {
                    let units = reginfo
                        .reg_group_members(group)
                        .iter()
                        .flat_map(|&reg| reginfo.reg_units(reg))
                        .collect();
                    candidate_units[class].push(units);
                }
            }
        }

        let mut class_data =
            SecondaryMap::<RegClass, ClassData>::with_max_index(reginfo.num_classes());
        let mut reg_counts = SecondaryMap::<PhysReg, usize>::with_max_index(reginfo.num_regs());
        let mut group_counts =
            SecondaryMap::<RegGroup, usize>::with_max_index(reginfo.num_reg_groups());
        for class in register_classes {
            let data = &mut class_data[class];
            data.num_candidates = candidate_units[class].len();

            // Compute the candidates that are blocked by each register unit.
            data.blocked_by_unit.grow_to(MAX_REG_UNITS);
            for (candidate, units) in &candidate_units[class] {
                for unit in *units {
                    data.blocked_by_unit[unit].insert(candidate);
                }
            }

            // For each register and register group, count the number of
            // candidates that it overlaps with.
            for reg in reginfo.regs() {
                let mut blocked = CandidateSet::new();
                for unit in reginfo.reg_units(reg) {
                    blocked |= data.blocked_by_unit[unit];
                }
                reg_counts[reg] = blocked.count();
            }
            for group in reginfo.reg_groups() {
                let mut blocked = CandidateSet::new();
                for &reg in reginfo.reg_group_members(group) {
                    for unit in reginfo.reg_units(reg) {
                        blocked |= data.blocked_by_unit[unit];
                    }
                }
                group_counts[group] = blocked.count();
            }

            // For each other class, compute the maximum number of target
            // candidates blocked by one member of that class.
            data.max_blocked_by_class.grow_to(reginfo.num_classes());
            for other in register_classes {
                data.max_blocked_by_class[other] = if reginfo.class_group_size(other) == 1 {
                    reginfo
                        .class_members(other)
                        .into_iter()
                        .map(|reg| reg_counts[reg])
                        .max()
                        .unwrap_or(0)
                } else {
                    reginfo
                        .class_group_members(other)
                        .into_iter()
                        .map(|group| group_counts[group])
                        .max()
                        .unwrap_or(0)
                };
            }
        }

        // A spillable class can be ignored if merging it with another class
        // never produces a stricter register-only constraint.
        let mut unconstrained_classes = RegClassSet::new();
        for class in reginfo.classes() {
            if reginfo.class_includes_spillslots(class)
                && reginfo
                    .classes()
                    .all(|other| match reginfo.common_subclass(class, other) {
                        None => true,
                        Some(common) => {
                            common == other || reginfo.class_includes_spillslots(common)
                        }
                    })
            {
                unconstrained_classes.insert(class);
            }
        }

        Self {
            reginfo,
            class_data,
            unconstrained_classes,
            register_classes,
            reused_inputs: vec![],
            operands: vec![],
            fixed_input: RegUnitSet::new(),
            fixed_output: RegUnitSet::new(),
        }
    }

    /// Collect constraints from the operands and clobbers.
    fn collect_operands(&mut self, operands: &[Operand], clobbers: RegUnitSet) {
        self.reused_inputs.clear();
        self.operands.clear();
        self.fixed_input.clear();
        self.fixed_output = clobbers;

        for op in operands {
            if let OperandConstraint::Reuse(idx) = op.constraint() {
                self.reused_inputs.push(idx);
            }
        }
        for (idx, op) in operands.iter().enumerate() {
            // Reused inputs impose no separate register requirement: a move
            // supplies the output, which covers both phases with the input's
            // class.
            //
            // Known limitation: we do not model another input narrowing the
            // output's class through the reused input. This matches regalloc3:
            // value_live_ranges::try_reuse_value only folds another input into
            // the tie if the inherited output class already satisfies its
            // constraint, so folding never narrows that class. An allocator
            // that propagates narrower input constraints through ties would
            // need those possibilities represented across both phases here,
            // including when a tied scalar joins an input group.
            if self.reused_inputs.contains(&idx) {
                continue;
            }

            let (mut phases, value) = match op.kind() {
                OperandKind::Use(value) => (Phases::Input, ValueOrGroup::Value(value)),
                OperandKind::Def(value) => (Phases::Output, ValueOrGroup::Value(value)),
                OperandKind::EarlyDef(value) => (Phases::Both, ValueOrGroup::Value(value)),
                OperandKind::UseGroup(group) => (Phases::Input, ValueOrGroup::Group(group)),
                OperandKind::DefGroup(group) => (Phases::Output, ValueOrGroup::Group(group)),
                OperandKind::EarlyDefGroup(group) => (Phases::Both, ValueOrGroup::Group(group)),
                OperandKind::NonAllocatable => continue,
            };
            let class = match op.constraint() {
                OperandConstraint::Fixed(reg) => {
                    match phases {
                        Phases::Input => self.fixed_input.extend(self.reginfo.reg_units(reg)),
                        Phases::Output => self.fixed_output.extend(self.reginfo.reg_units(reg)),
                        Phases::Both => {
                            self.fixed_input.extend(self.reginfo.reg_units(reg));
                            self.fixed_output.extend(self.reginfo.reg_units(reg))
                        }
                    }
                    continue;
                }
                OperandConstraint::Class(class) => class,
                OperandConstraint::Reuse(input) => {
                    let OperandConstraint::Class(input_class) = operands[input].constraint() else {
                        unreachable!();
                    };
                    phases = Phases::Both;
                    input_class
                }
            };

            // Outputs cannot share their newly defined values with another
            // operand, so their classes cannot narrow through merging with
            // other operands in this instruction. Reuse outputs follow the
            // model described above and count as outputs too. Inputs may still
            // narrow unless their class is unconstrained.
            if (phases != Phases::Input && self.reginfo.class_includes_spillslots(class))
                || self.unconstrained_classes.contains(class)
            {
                continue;
            }

            self.operands.push(OperandData {
                idx,
                class,
                phases,
                value,
                possible_classes: RegClassSet::from_iter([class]),
            });
        }
    }

    /// Compute class possibilities so that every possible combination of merged
    /// operand constraints is represented by an operand record.
    fn merge_classes(&mut self, func: &impl Function) {
        for idx in 0..self.operands.len() {
            let target = &self.operands[idx];

            // Only input values can appear multiple times in the operand list.
            if target.phases != Phases::Input {
                continue;
            }

            let mut contributors = RegClassSet::new();
            for (other_idx, other) in self.operands.iter().enumerate() {
                // We can only merge with other inputs.
                if other.phases != Phases::Input {
                    continue;
                }

                // Operands with the same class don't introduce additional
                // constraints.
                if target.class == other.class {
                    continue;
                }

                // Check if the two operands share a value. If they do then the
                // allocator might merge both into a single operand constraint
                // with a shared register class.
                let compatible = match (target.value, other.value) {
                    // Represent each possible merged subset on its earliest
                    // group operand, or its earliest scalar if there is no
                    // group. That representative sees every other member of
                    // the subset, so its closure includes their combined class.
                    (ValueOrGroup::Value(a), ValueOrGroup::Value(b)) => idx < other_idx && a == b,
                    (ValueOrGroup::Group(a), ValueOrGroup::Group(b)) => {
                        idx < other_idx
                            && func.value_group_members(a) == func.value_group_members(b)
                    }
                    (ValueOrGroup::Group(group), ValueOrGroup::Value(value)) => {
                        func.value_group_members(group).contains(&value)
                    }
                    (ValueOrGroup::Value(_), ValueOrGroup::Group(_)) => false,
                };
                if compatible {
                    contributors.insert(other.class);
                }
            }

            // Compute the common subclass of the target's original class with
            // every subset of contributor classes.
            let mut possible_classes = RegClassSet::from_iter([target.class]);
            for contributor in contributors {
                for class in possible_classes {
                    if let Some(common) = self.reginfo.common_subclass(class, contributor) {
                        possible_classes.insert(common);
                    }
                }
            }

            // We no longer need spillable classes after this: they are only
            // useful for computing the possible classes, but do not constrain
            // the allocation problem itself.
            self.operands[idx].possible_classes = possible_classes & self.register_classes;
        }
    }

    /// Check that every represented class has a candidate left under the
    /// worst-case interference from fixed reservations and other operands.
    fn check_pressure(&self) -> Result<(), ConstraintConflict> {
        for (idx, operand) in self.operands.iter().enumerate() {
            let reserved = match operand.phases {
                Phases::Input => self.fixed_input,
                Phases::Output => self.fixed_output,
                Phases::Both => self.fixed_input | self.fixed_output,
            };

            for class in operand.possible_classes {
                let data = &self.class_data[class];

                // Count the number of candidates that are blocked by fixed
                // register constraints and clobbers.
                let mut fixed = CandidateSet::new();
                for unit in reserved {
                    fixed |= data.blocked_by_unit[unit];
                }
                let mut blocked = fixed.count();

                // Then for each other operand that we share a phase with, add
                // the number of candidates that are blocked by an allocation
                // for that operand.
                //
                // Because exhaustively checking every possible allocation from
                // other operands is impractically expensive, we choose a
                // conservative worst case: we count the highest number of
                // candidates that can be blocked by any member of any of the
                // possible classes for that operand.
                for (other_idx, other) in self.operands.iter().enumerate() {
                    if idx == other_idx || !operand.phases.overlaps(other.phases) {
                        continue;
                    }
                    blocked += other
                        .possible_classes
                        .into_iter()
                        .map(|class| data.max_blocked_by_class[class])
                        .max()
                        .unwrap_or(0);
                }

                // If this upper bound leaves no candidates guaranteed to be
                // available, conservatively assume that the constraints are
                // unsatisfiable.
                if blocked >= data.num_candidates {
                    return Err(ConstraintConflict {
                        operand_idx: operand.idx,
                        num_candidates: data.num_candidates,
                        blocked,
                    });
                }
            }
        }

        Ok(())
    }

    /// Checks the operand constraints for a single instruction to ensure the
    /// allocator can always find a solution for them.
    pub fn check(
        &mut self,
        func: &impl Function,
        operands: &[Operand],
        clobbers: RegUnitSet,
    ) -> Result<(), ConstraintConflict> {
        self.collect_operands(operands, clobbers);
        self.merge_classes(func);
        self.check_pressure()
    }
}
