//! Checks that `ParallelMoves` works correctly.

#![no_main]

use core::fmt;
use std::sync::OnceLock;

use arbitrary::{Arbitrary, Result, Unstructured};
use libfuzzer_sys::fuzz_target;
use regalloc3::debug_utils::{self, GenericRegInfo};
use regalloc3::entity::{PrimaryMap, SecondaryMap};
use regalloc3::function::{
    Block, Function, Inst, InstRange, Operand, RematCost, Value, ValueGroup,
};
use regalloc3::output::{Allocation, AllocationKind, SpillSlot};
use regalloc3::parallel_moves::ParallelMoves;
use regalloc3::reginfo::{
    PhysReg, RegBank, RegClass, RegInfo, RegOrRegGroup, RegUnit, RegUnitSet, SpillSlotSize,
    MAX_REG_UNITS,
};

/// Alternate between a  simple register description with 2 banks that overlap,
/// and an arbitrary register description.
fn reginfo(u: &mut Unstructured) -> Result<GenericRegInfo> {
    static REGINFO: OnceLock<GenericRegInfo> = OnceLock::new();
    if u.arbitrary()? {
        Ok(REGINFO
            .get_or_init(|| {
                let reginfo = GenericRegInfo::parse(
                    "\
r0 = reg unit0
r1 = reg unit1
r2 = reg unit2
r3 = reg unit3
r4 = reg unit4
r5 = reg unit5
r6 = reg unit6
r7 = reg unit7
r8 = stack unit8
r9 = stack unit9
r10 = stack unit10
r11 = stack unit11

r12 = reg unit0 unit1
r13 = reg unit2 unit3
r14 = reg unit4 unit5
r15 = reg unit6 unit7
r16 = stack unit12
r17 = stack unit13
r18 = stack unit14
r19 = stack unit15

bank0 {
    top_level_class = class0
    stack_to_stack_class = class1
    spillslot_size = 1

    class0 {
        allows_spillslots
        spill_cost = 1
        registers = r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11
    }

    class1: class0 {
        spill_cost = 1
        registers = r0 r1 r2 r3 r4 r5 r6 r7
        preferred_regs = r0 r1 r2 r3 r4 r5 r6 r7
    }
}

bank1 {
    top_level_class = class2
    stack_to_stack_class = class3
    spillslot_size = 1

    class2 {
        allows_spillslots
        spill_cost = 1
        registers = r12 r13 r14 r15 r16 r17 r18 r19
    }

    class3: class2 {
        spill_cost = 1
        registers = r12 r13 r14 r15
        preferred_regs = r12 r13 r14 r15
    }
}
",
                )
                .unwrap();
                debug_utils::validate_reginfo(&reginfo).expect("reginfo validation failed");
                reginfo
            })
            .clone())
    } else {
        GenericRegInfo::arbitrary(u)
    }
}

struct ValueData {
    bank: RegBank,
    alloc: Option<Allocation>,
    remat: Option<(RematCost, RegClass)>,
}

struct TestCase {
    moves: Vec<(Option<Allocation>, Allocation, Value)>,
    values: PrimaryMap<Value, ValueData>,
    available_units: RegUnitSet,
    spillslots: PrimaryMap<SpillSlot, SpillSlotSize>,
    reginfo: GenericRegInfo,
}

impl Arbitrary<'_> for TestCase {
    fn arbitrary(u: &mut Unstructured) -> Result<Self> {
        // Ensure the logger is initialized.
        let _ = pretty_env_logger::try_init();

        let reginfo = reginfo(u)?;

        // Generate values for move sources.
        let mut src_used_mask = RegUnitSet::new();
        let mut spillslots = PrimaryMap::new();
        let mut values = PrimaryMap::new();
        let mut slots_per_bank: SecondaryMap<RegBank, Vec<SpillSlot>> =
            SecondaryMap::with_max_index(reginfo.num_banks());
        for _ in 0..u.int_in_range(1..=20)? {
            let mut gen_alloc = || {
                // Try to get a register if it doesn't overlap an existing one.
                if u.arbitrary()? {
                    let reg = PhysReg::new(u.choose_index(reginfo.num_regs())?);
                    if reginfo
                        .reg_units(reg)
                        .iter()
                        .all(|&unit| !src_used_mask.contains(unit))
                    {
                        reginfo
                            .reg_units(reg)
                            .iter()
                            .for_each(|&unit| src_used_mask.insert(unit));
                        let bank = reginfo.bank_for_reg(reg).unwrap();
                        return Ok((Some(Allocation::reg(reg)), bank));
                    }
                }

                // Otherwise get a spillslot or force a rematerialization.
                let bank = RegBank::new(u.choose_index(reginfo.num_banks())?);
                if u.arbitrary()? {
                    Ok((None, bank))
                } else {
                    let spillslot = spillslots.push(reginfo.spillslot_size(bank));
                    slots_per_bank[bank].push(spillslot);
                    Ok((Some(Allocation::spillslot(spillslot)), bank))
                }
            };

            let (alloc, bank) = gen_alloc()?;
            let remat = if alloc.is_none() || u.arbitrary()? {
                let cost = if u.arbitrary()? {
                    RematCost::CheaperThanMove
                } else {
                    RematCost::CheaperThanLoad
                };
                let remat_classes: Vec<_> = reginfo
                    .classes()
                    .filter(|&class| {
                        reginfo.bank_for_class(class) == bank
                            && reginfo.class_group_size(class) == 1
                            && (reginfo.class_includes_spillslots(class)
                                || reginfo
                                    .class_members(class)
                                    .iter()
                                    .all(|reg| !reginfo.is_memory(reg.as_single())))
                    })
                    .collect();
                if remat_classes.is_empty() {
                    None
                } else {
                    let class = *u.choose(&remat_classes)?;
                    Some((cost, class))
                }
            } else {
                None
            };
            values.push(ValueData { bank, alloc, remat });
        }

        // Generate moves from one of the source values to an allocation.
        let mut dest_used_mask = RegUnitSet::new();
        let mut moves = vec![];
        for _ in 0..u.int_in_range(1..=20)? {
            let value = Value::new(u.choose_index(values.len())?);
            let bank = values[value].bank;
            let src = values[value].alloc;
            let mut gen_alloc = || {
                // Try to get a register if it doesn't overlap an existing one.
                if u.arbitrary()? {
                    let regs_in_bank: Vec<_> = reginfo
                        .regs()
                        .filter(|&reg| reginfo.bank_for_reg(reg) == Some(bank))
                        .collect();
                    let reg = *u.choose(&regs_in_bank)?;
                    if reginfo
                        .reg_units(reg)
                        .iter()
                        .all(|&unit| !dest_used_mask.contains(unit))
                    {
                        reginfo
                            .reg_units(reg)
                            .iter()
                            .for_each(|&unit| dest_used_mask.insert(unit));
                        return Ok(Allocation::reg(reg));
                    }
                }

                // Otherwise get a spillslot or create a new one.
                if !slots_per_bank[bank].is_empty() && u.arbitrary()? {
                    let idx = u.choose_index(slots_per_bank[bank].len())?;
                    Ok(Allocation::spillslot(slots_per_bank[bank].swap_remove(idx)))
                } else {
                    Ok(Allocation::spillslot(
                        spillslots.push(reginfo.spillslot_size(bank)),
                    ))
                }
            };
            let dest = gen_alloc()?;
            moves.push((src, dest, value));
        }

        Ok(TestCase {
            moves,
            values,
            available_units: !dest_used_mask,
            spillslots,
            reginfo,
        })
    }
}

// Minimal implementation of Function for value_bank and can_rematerialize.
impl Function for TestCase {
    fn num_insts(&self) -> usize {
        unreachable!()
    }

    fn num_blocks(&self) -> usize {
        unreachable!()
    }

    fn block_insts(&self, _block: Block) -> InstRange {
        unreachable!()
    }

    fn inst_block(&self, _inst: Inst) -> Block {
        unreachable!()
    }

    fn block_succs(&self, _block: Block) -> &[Block] {
        unreachable!()
    }

    fn block_preds(&self, _block: Block) -> &[Block] {
        unreachable!()
    }

    fn block_immediate_dominator(&self, _block: Block) -> Option<Block> {
        unreachable!()
    }

    fn block_params(&self, _block: Block) -> &[Value] {
        unreachable!()
    }

    fn inst_is_terminator(&self, _inst: Inst) -> bool {
        unreachable!()
    }

    fn jump_blockparams(&self, _block: Block) -> &[Value] {
        unreachable!()
    }

    fn block_frequency(&self, _block: Block) -> f32 {
        unreachable!()
    }

    fn inst_operands(&self, _inst: Inst) -> &[Operand] {
        unreachable!()
    }

    fn inst_clobbers(&self, _inst: Inst) -> &[RegUnit] {
        unreachable!()
    }

    fn num_values(&self) -> usize {
        self.values.len()
    }

    fn value_bank(&self, value: Value) -> RegBank {
        self.values[value].bank
    }

    fn num_value_groups(&self) -> usize {
        unreachable!()
    }

    fn value_group_members(&self, _group: ValueGroup) -> &[Value] {
        unreachable!()
    }

    fn can_rematerialize(&self, value: Value) -> Option<(RematCost, RegClass)> {
        self.values[value].remat
    }

    fn can_eliminate_dead_inst(&self, _inst: Inst) -> bool {
        unreachable!()
    }
}

impl fmt::Debug for TestCase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Reginfo:\n{}", self.reginfo)?;

        writeln!(f, "Spill slots:")?;
        for (slot, size) in self.spillslots.iter() {
            writeln!(f, "{slot}: size={size}")?;
        }
        writeln!(f)?;

        writeln!(f, "Input values:")?;
        for (value, &ValueData { bank, alloc, remat }) in &self.values {
            if let Some(alloc) = alloc {
                write!(f, "{value} in {alloc} = {bank}").unwrap();
            } else {
                write!(f, "{value} = {bank}").unwrap();
            }
            if let Some((cost, class)) = remat {
                let cost = match cost {
                    RematCost::CheaperThanMove => "cheaper_than_move",
                    RematCost::CheaperThanLoad => "cheaper_than_load",
                };
                write!(f, " remat({cost}, {class})").unwrap();
            }
            writeln!(f)?;
        }
        writeln!(f)?;

        writeln!(f, "Parallel moves:")?;
        for &(src, dest, value) in &self.moves {
            if let Some(src) = src {
                writeln!(f, "move {dest} <- {src} ({value})")?;
            } else {
                writeln!(f, "remat {dest} <- {value}")?;
            }
        }

        Ok(())
    }
}

fuzz_target!(|t: TestCase| {
    log::trace!("Test case:\n{t:?}");

    // Resolve the parallel moves into sequential moves.
    let mut parallel_moves = ParallelMoves::new();
    parallel_moves.prepare(&t, t.spillslots.len());
    parallel_moves.new_parallel_move();
    for &(src, dest, value) in &t.moves {
        if let Some(src) = src {
            if src != dest {
                parallel_moves.add_move(src, dest, value, &t, &t.reginfo);
            }
        } else {
            parallel_moves.add_remat(dest, value, &t, &t.reginfo);
        }
    }
    let mut spillslots = t.spillslots.clone();
    parallel_moves.resolve(
        &t.reginfo,
        &t,
        |unit| t.available_units.contains(unit),
        |size| {
            let slot = spillslots.push(size);
            log::trace!("Allocating emergency {slot} with size {size}");
            slot
        },
    );

    // A spill slot can either contain a value or an emergency spill.
    #[derive(Default, Clone, PartialEq, Eq, Debug)]
    enum SpillSlotContents {
        #[default]
        None,
        Value(Value),
        Spill(Vec<Option<Value>>),
    }

    // Prepare the initial state of all register units and spill slots.
    let mut unit_values = SecondaryMap::with_max_index(MAX_REG_UNITS);
    let mut spillslot_values = SecondaryMap::with_max_index(t.spillslots.len());
    for (
        value,
        &ValueData {
            bank: _,
            alloc,
            remat: _,
        },
    ) in &t.values
    {
        if let Some(alloc) = alloc {
            match alloc.kind() {
                AllocationKind::PhysReg(reg) => {
                    for &unit in t.reginfo.reg_units(reg) {
                        unit_values[unit] = Some(value);
                    }
                }
                AllocationKind::SpillSlot(slot) => {
                    spillslot_values[slot] = SpillSlotContents::Value(value);
                }
            }
        }
    }

    log::trace!("Generated move sequence:");
    for edit in parallel_moves.edits() {
        if let Some(from) = edit.from.expand() {
            log::trace!(
                "move {} <- {from} ({:?})",
                edit.to.unwrap(),
                edit.value.expand()
            );
        } else {
            let value = edit.value.unwrap();
            log::trace!("remat {} <- {value}", edit.to.unwrap());
        }
    }

    // Simulate the execution of the sequential moves.
    log::trace!("Executing sequential moves:");
    for edit in parallel_moves.edits() {
        if let Some(from) = edit.from.expand() {
            log::trace!(
                "move {} <- {from} ({:?})",
                edit.to.unwrap(),
                edit.value.expand()
            );
            if from.is_memory(&t.reginfo) && edit.to.unwrap().is_memory(&t.reginfo) {
                panic!("Stack-to-stack move from {from} to {}", edit.to.unwrap());
            }
            if let Some(value) = edit.value.expand() {
                match from.kind() {
                    AllocationKind::PhysReg(reg) => {
                        assert_eq!(t.reginfo.bank_for_reg(reg).unwrap(), t.value_bank(value));
                        t.reginfo
                            .reg_units(reg)
                            .iter()
                            .for_each(|&unit| assert_eq!(unit_values[unit], Some(value)));
                    }
                    AllocationKind::SpillSlot(slot) => {
                        assert_eq!(spillslot_values[slot], SpillSlotContents::Value(value));
                        assert_eq!(
                            spillslots[slot],
                            t.reginfo.spillslot_size(t.value_bank(value))
                        );
                    }
                }
                match edit.to.unwrap().kind() {
                    AllocationKind::PhysReg(reg) => {
                        assert_eq!(t.reginfo.bank_for_reg(reg).unwrap(), t.value_bank(value));
                        t.reginfo
                            .reg_units(reg)
                            .iter()
                            .for_each(|&unit| unit_values[unit] = Some(value));
                    }
                    AllocationKind::SpillSlot(slot) => {
                        spillslot_values[slot] = SpillSlotContents::Value(value);
                        assert_eq!(
                            spillslots[slot],
                            t.reginfo.spillslot_size(t.value_bank(value))
                        );
                    }
                }
            } else {
                match (from.kind(), edit.to.unwrap().kind()) {
                    (AllocationKind::PhysReg(_from), AllocationKind::PhysReg(_to)) => {
                        unreachable!()
                    }
                    (AllocationKind::SpillSlot(from), AllocationKind::PhysReg(to)) => {
                        match spillslot_values[from] {
                            SpillSlotContents::None => {
                                for &unit in t.reginfo.reg_units(to) {
                                    unit_values[unit] = None;
                                }
                            }
                            SpillSlotContents::Value(value) => {
                                for &unit in t.reginfo.reg_units(to) {
                                    unit_values[unit] = Some(value);
                                }
                            }
                            SpillSlotContents::Spill(ref values) => {
                                for (&value, &unit) in values.iter().zip(t.reginfo.reg_units(to)) {
                                    unit_values[unit] = value;
                                }
                            }
                        }
                        assert_eq!(
                            spillslots[from],
                            t.reginfo
                                .spillslot_size(t.reginfo.bank_for_reg(to).unwrap())
                        );
                    }
                    (AllocationKind::PhysReg(from), AllocationKind::SpillSlot(to)) => {
                        let values = t
                            .reginfo
                            .reg_units(from)
                            .iter()
                            .map(|&unit| unit_values[unit])
                            .collect();
                        spillslot_values[to] = SpillSlotContents::Spill(values);
                        assert_eq!(
                            spillslots[to],
                            t.reginfo
                                .spillslot_size(t.reginfo.bank_for_reg(from).unwrap())
                        );
                    }
                    (AllocationKind::SpillSlot(_from), AllocationKind::SpillSlot(_to)) => {
                        unreachable!()
                    }
                }
            }
        } else {
            let value = edit.value.unwrap();
            log::trace!("remat {} <- {value}", edit.to.unwrap());
            let Some((_cost, class)) = t.can_rematerialize(value) else {
                panic!("Can't rematerialize {value}");
            };
            match edit.to.unwrap().kind() {
                AllocationKind::PhysReg(reg) => {
                    assert!(t
                        .reginfo
                        .class_members(class)
                        .contains(RegOrRegGroup::single(reg)));
                    for &unit in t.reginfo.reg_units(reg) {
                        unit_values[unit] = Some(value);
                    }
                }
                AllocationKind::SpillSlot(slot) => {
                    assert!(t.reginfo.class_includes_spillslots(class));
                    assert_eq!(
                        spillslots[slot],
                        t.reginfo.spillslot_size(t.value_bank(value))
                    );
                    spillslot_values[slot] = SpillSlotContents::Value(value);
                }
            }
        }
    }

    // Check that the final states match move destinations.
    for &(_src, dest, value) in &t.moves {
        log::trace!("Checking final contents of {dest}, expecting {value}");
        match dest.kind() {
            AllocationKind::PhysReg(reg) => t
                .reginfo
                .reg_units(reg)
                .iter()
                .for_each(|&unit| assert_eq!(unit_values[unit], Some(value))),
            AllocationKind::SpillSlot(slot) => {
                assert_eq!(spillslot_values[slot], SpillSlotContents::Value(value))
            }
        }
    }
});
