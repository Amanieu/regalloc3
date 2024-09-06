use cranelift_entity::PrimaryMap;

use super::{Arch, RegBankData, RegClassData, RegData, RegGroupData, RegGroupList, RegInfo};

pub fn make_aarch64_reginfo(num_fixed_stack: usize) -> RegInfo {
    let mut reginfo = RegInfo {
        num_fixed_stack,
        arch: Arch::Aarch64,
        units: PrimaryMap::new(),
        regs: PrimaryMap::new(),
        groups: PrimaryMap::new(),
        banks: PrimaryMap::new(),
        classes: PrimaryMap::new(),
    };

    // X registers
    let x_units = reginfo.make_units(31);
    let x_stack_units = reginfo.make_units(num_fixed_stack);
    let x_regs = reginfo.make_regs(31, |i| RegData {
        is_stack: false,
        name: format!("x{i}"),
        units: vec![x_units[i]],
    });
    let _sp = reginfo.regs.push(RegData {
        is_stack: false,
        name: "sp".to_string(),
        units: vec![],
    });
    let _xzr = reginfo.regs.push(RegData {
        is_stack: false,
        name: "xzr".to_string(),
        units: vec![],
    });
    let x_fixed_stack = reginfo.make_regs(num_fixed_stack, |i| RegData {
        is_stack: true,
        name: format!("int_stack{i}"),
        units: vec![x_stack_units[i]],
    });
    // This excludes the (lr, xzr) pair.
    let casp_regs = reginfo.make_reg_group(15, |i| RegGroupData {
        regs: vec![x_regs[i * 2], x_regs[i * 2 + 1]],
    });
    let x_stack_class = reginfo.classes.push(RegClassData {
        desc: "General-purpose registers + stack".to_string(),
        superclass: None,
        group_size: 1,
        allows_spillslots: true,
        spill_cost: 0.5,
        members: RegGroupList::Single([&x_regs[..], &x_fixed_stack[..]].concat()),
        preferred_regs: RegGroupList::Single([&x_regs[0..=18], &x_regs[30..=30]].concat()),
        non_preferred_regs: RegGroupList::Single(vec![]),
        callee_saved_preferred_regs: RegGroupList::Single(x_regs[19..=29].into()),
        callee_saved_non_preferred_regs: RegGroupList::Single(vec![]),
    });
    let x_stack_only_class = reginfo.classes.push(RegClassData {
        desc: "General-purpose stack only".to_string(),
        superclass: Some(x_stack_class),
        group_size: 1,
        allows_spillslots: true,
        spill_cost: 0.0,
        members: RegGroupList::Single(x_fixed_stack.clone()),
        preferred_regs: RegGroupList::Single(vec![]),
        non_preferred_regs: RegGroupList::Single(vec![]),
        callee_saved_preferred_regs: RegGroupList::Single(vec![]),
        callee_saved_non_preferred_regs: RegGroupList::Single(vec![]),
    });
    let x_class = reginfo.classes.push(RegClassData {
        desc: "General-purpose registers".to_string(),
        superclass: Some(x_stack_class),
        group_size: 1,
        allows_spillslots: false,
        spill_cost: 1.0,
        members: RegGroupList::Single(x_regs.clone()),
        preferred_regs: RegGroupList::Single([&x_regs[0..=18], &x_regs[30..=30]].concat()),
        non_preferred_regs: RegGroupList::Single(vec![]),
        callee_saved_preferred_regs: RegGroupList::Single(x_regs[19..=29].into()),
        callee_saved_non_preferred_regs: RegGroupList::Single(vec![]),
    });
    let casp_class = reginfo.classes.push(RegClassData {
        desc: "Aligned register pairs for CASP".to_string(),
        superclass: Some(x_class),
        group_size: 2,
        allows_spillslots: false,
        spill_cost: 1.0,
        members: RegGroupList::Multi(casp_regs.clone()),
        preferred_regs: RegGroupList::Multi(casp_regs[0..=8].into()),
        non_preferred_regs: RegGroupList::Multi(vec![]),
        callee_saved_preferred_regs: RegGroupList::Multi(casp_regs[9..=14].into()),
        callee_saved_non_preferred_regs: RegGroupList::Multi(vec![]),
    });
    reginfo.banks.push(RegBankData {
        desc: "General-purpose registers".to_string(),
        top_level_class: x_stack_class,
        stack_to_stack_class: x_class,
        spillslot_size: 8,
        classes: vec![x_stack_class, x_stack_only_class, x_class, casp_class],
    });

    // D registers
    let d_units = reginfo.make_units(32);
    let d_stack_units = reginfo.make_units(num_fixed_stack);
    let d_regs = reginfo.make_regs(32, |i| RegData {
        is_stack: false,
        name: format!("d{i}"),
        units: vec![d_units[i]],
    });
    let d_fixed_stack = reginfo.make_regs(num_fixed_stack, |i| RegData {
        is_stack: true,
        name: format!("fp64_stack{i}"),
        units: vec![d_stack_units[i]],
    });
    let dd_regs = reginfo.make_reg_group(32, |i| RegGroupData {
        regs: vec![d_regs[i], d_regs[(i + 1) % 32]],
    });
    let ddd_regs = reginfo.make_reg_group(32, |i| RegGroupData {
        regs: vec![d_regs[i], d_regs[(i + 1) % 32], d_regs[(i + 2) % 32]],
    });
    let dddd_regs = reginfo.make_reg_group(32, |i| RegGroupData {
        regs: vec![
            d_regs[i],
            d_regs[(i + 1) % 32],
            d_regs[(i + 2) % 32],
            d_regs[(i + 3) % 32],
        ],
    });
    let d_stack_class = reginfo.classes.push(RegClassData {
        desc: "64-bit FP/SIMD registers + stack".to_string(),
        superclass: None,
        group_size: 1,
        allows_spillslots: true,
        spill_cost: 0.5,
        members: RegGroupList::Single([&d_regs[..], &d_fixed_stack[..]].concat()),
        preferred_regs: RegGroupList::Single([&d_regs[0..=7], &d_regs[16..=31]].concat()),
        non_preferred_regs: RegGroupList::Single(vec![]),
        callee_saved_preferred_regs: RegGroupList::Single(d_regs[8..=15].into()),
        callee_saved_non_preferred_regs: RegGroupList::Single(vec![]),
    });
    let d_stack_only_class = reginfo.classes.push(RegClassData {
        desc: "64-bit FP/SIMD stack only".to_string(),
        superclass: Some(d_stack_class),
        group_size: 1,
        allows_spillslots: true,
        spill_cost: 0.0,
        members: RegGroupList::Single(d_fixed_stack.clone()),
        preferred_regs: RegGroupList::Single(vec![]),
        non_preferred_regs: RegGroupList::Single(vec![]),
        callee_saved_preferred_regs: RegGroupList::Single(vec![]),
        callee_saved_non_preferred_regs: RegGroupList::Single(vec![]),
    });
    let d_class = reginfo.classes.push(RegClassData {
        desc: "64-bit FP/SIMD registers".to_string(),
        superclass: Some(d_stack_class),
        group_size: 1,
        allows_spillslots: false,
        spill_cost: 1.0,
        members: RegGroupList::Single(d_regs.clone()),
        preferred_regs: RegGroupList::Single([&d_regs[0..=7], &d_regs[16..=31]].concat()),
        non_preferred_regs: RegGroupList::Single(vec![]),
        callee_saved_preferred_regs: RegGroupList::Single(d_regs[8..=15].into()),
        callee_saved_non_preferred_regs: RegGroupList::Single(vec![]),
    });
    let dd_class = reginfo.classes.push(RegClassData {
        desc: "64-bit FP/SIMD register pairs".to_string(),
        superclass: Some(d_class),
        group_size: 2,
        allows_spillslots: false,
        spill_cost: 1.0,
        members: RegGroupList::Multi(dd_regs.clone()),
        preferred_regs: RegGroupList::Multi([&dd_regs[0..=6], &dd_regs[16..=31]].concat()),
        non_preferred_regs: RegGroupList::Multi(vec![]),
        callee_saved_preferred_regs: RegGroupList::Multi(dd_regs[7..=15].into()),
        callee_saved_non_preferred_regs: RegGroupList::Multi(vec![]),
    });
    let ddd_class = reginfo.classes.push(RegClassData {
        desc: "64-bit FP/SIMD register triples".to_string(),
        superclass: Some(d_class),
        group_size: 3,
        allows_spillslots: false,
        spill_cost: 1.0,
        members: RegGroupList::Multi(ddd_regs.clone()),
        preferred_regs: RegGroupList::Multi([&ddd_regs[0..=5], &ddd_regs[16..=31]].concat()),
        non_preferred_regs: RegGroupList::Multi(vec![]),
        callee_saved_preferred_regs: RegGroupList::Multi(ddd_regs[6..=15].into()),
        callee_saved_non_preferred_regs: RegGroupList::Multi(vec![]),
    });
    let dddd_class = reginfo.classes.push(RegClassData {
        desc: "64-bit FP/SIMD register quads".to_string(),
        superclass: Some(d_class),
        group_size: 4,
        allows_spillslots: false,
        spill_cost: 1.0,
        members: RegGroupList::Multi(dddd_regs.clone()),
        preferred_regs: RegGroupList::Multi([&dddd_regs[0..=4], &dddd_regs[16..=31]].concat()),
        non_preferred_regs: RegGroupList::Multi(vec![]),
        callee_saved_preferred_regs: RegGroupList::Multi(dddd_regs[5..=15].into()),
        callee_saved_non_preferred_regs: RegGroupList::Multi(vec![]),
    });
    reginfo.banks.push(RegBankData {
        desc: "64-bit FP/SIMD registers".to_string(),
        top_level_class: d_stack_class,
        stack_to_stack_class: d_class,
        spillslot_size: 8,
        classes: vec![
            d_stack_class,
            d_stack_only_class,
            d_class,
            dd_class,
            ddd_class,
            dddd_class,
        ],
    });

    // High half of D registers (for clobbers only).
    let dhigh_units = reginfo.make_units(32);

    // Q registers
    let q_stack_units = reginfo.make_units(num_fixed_stack);
    let q_regs = reginfo.make_regs(32, |i| RegData {
        is_stack: false,
        name: format!("q{i}"),
        units: vec![d_units[i], dhigh_units[i]],
    });
    let q_fixed_stack = reginfo.make_regs(num_fixed_stack, |i| RegData {
        is_stack: true,
        name: format!("fp128_stack{i}"),
        units: vec![q_stack_units[i]],
    });
    let qq_regs = reginfo.make_reg_group(32, |i| RegGroupData {
        regs: vec![q_regs[i], q_regs[(i + 1) % 32]],
    });
    let qqq_regs = reginfo.make_reg_group(32, |i| RegGroupData {
        regs: vec![q_regs[i], q_regs[(i + 1) % 32], q_regs[(i + 2) % 32]],
    });
    let qqqq_regs = reginfo.make_reg_group(32, |i| RegGroupData {
        regs: vec![
            q_regs[i],
            q_regs[(i + 1) % 32],
            q_regs[(i + 2) % 32],
            q_regs[(i + 3) % 32],
        ],
    });
    let q_stack_class = reginfo.classes.push(RegClassData {
        desc: "128-bit FP/SIMD registers + stack".to_string(),
        superclass: None,
        group_size: 1,
        allows_spillslots: true,
        spill_cost: 0.5,
        members: RegGroupList::Single([&q_regs[..], &q_fixed_stack[..]].concat()),
        preferred_regs: RegGroupList::Single(q_regs.clone()),
        non_preferred_regs: RegGroupList::Single(vec![]),
        callee_saved_preferred_regs: RegGroupList::Single(vec![]),
        callee_saved_non_preferred_regs: RegGroupList::Single(vec![]),
    });
    let q_stack_only_class = reginfo.classes.push(RegClassData {
        desc: "128-bit FP/SIMD stack only".to_string(),
        superclass: Some(q_stack_class),
        group_size: 1,
        allows_spillslots: true,
        spill_cost: 0.0,
        members: RegGroupList::Single(q_fixed_stack.clone()),
        preferred_regs: RegGroupList::Single(vec![]),
        non_preferred_regs: RegGroupList::Single(vec![]),
        callee_saved_preferred_regs: RegGroupList::Single(vec![]),
        callee_saved_non_preferred_regs: RegGroupList::Single(vec![]),
    });
    let q_class = reginfo.classes.push(RegClassData {
        desc: "128-bit FP/SIMD registers".to_string(),
        superclass: Some(q_stack_class),
        group_size: 1,
        allows_spillslots: false,
        spill_cost: 1.0,
        members: RegGroupList::Single(q_regs.clone()),
        preferred_regs: RegGroupList::Single(q_regs.clone()),
        non_preferred_regs: RegGroupList::Single(vec![]),
        callee_saved_preferred_regs: RegGroupList::Single(vec![]),
        callee_saved_non_preferred_regs: RegGroupList::Single(vec![]),
    });
    let qq_class = reginfo.classes.push(RegClassData {
        desc: "128-bit FP/SIMD register pairs".to_string(),
        superclass: Some(q_class),
        group_size: 2,
        allows_spillslots: false,
        spill_cost: 1.0,
        members: RegGroupList::Multi(qq_regs.clone()),
        preferred_regs: RegGroupList::Multi(qq_regs.clone()),
        non_preferred_regs: RegGroupList::Multi(vec![]),
        callee_saved_preferred_regs: RegGroupList::Multi(vec![]),
        callee_saved_non_preferred_regs: RegGroupList::Multi(vec![]),
    });
    let qqq_class = reginfo.classes.push(RegClassData {
        desc: "128-bit FP/SIMD register triples".to_string(),
        superclass: Some(q_class),
        group_size: 3,
        allows_spillslots: false,
        spill_cost: 1.0,
        members: RegGroupList::Multi(qqq_regs.clone()),
        preferred_regs: RegGroupList::Multi(qqq_regs.clone()),
        non_preferred_regs: RegGroupList::Multi(vec![]),
        callee_saved_preferred_regs: RegGroupList::Multi(vec![]),
        callee_saved_non_preferred_regs: RegGroupList::Multi(vec![]),
    });
    let qqqq_class = reginfo.classes.push(RegClassData {
        desc: "128-bit FP/SIMD register quads".to_string(),
        superclass: Some(q_class),
        group_size: 4,
        allows_spillslots: false,
        spill_cost: 1.0,
        members: RegGroupList::Multi(qqqq_regs.clone()),
        preferred_regs: RegGroupList::Multi(qqqq_regs.clone()),
        non_preferred_regs: RegGroupList::Multi(vec![]),
        callee_saved_preferred_regs: RegGroupList::Multi(vec![]),
        callee_saved_non_preferred_regs: RegGroupList::Multi(vec![]),
    });
    reginfo.banks.push(RegBankData {
        desc: "128-bit FP/SIMD registers".to_string(),
        top_level_class: q_stack_class,
        stack_to_stack_class: q_class,
        spillslot_size: 16,
        classes: vec![
            q_stack_class,
            q_stack_only_class,
            q_class,
            qq_class,
            qqq_class,
            qqqq_class,
        ],
    });

    reginfo
}
