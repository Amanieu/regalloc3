use regalloc3::entity::PrimaryMap;

use super::{Arch, RegBankData, RegClassData, RegData, RegGroupData, RegGroupList, RegInfo};

pub fn make_riscv_reginfo(num_fixed_stack: usize) -> RegInfo {
    let mut reginfo = RegInfo {
        num_fixed_stack,
        arch: Arch::Riscv,
        units: PrimaryMap::new(),
        regs: PrimaryMap::new(),
        groups: PrimaryMap::new(),
        banks: PrimaryMap::new(),
        classes: PrimaryMap::new(),
    };

    // General-purpose registers
    let x_units = reginfo.make_units(32 - 4);
    let x_stack_units = reginfo.make_units(num_fixed_stack);
    let all_x_regs = reginfo.make_regs(32, |i| match i {
        0 => RegData {
            is_stack: false,
            name: "zero".to_string(),
            units: vec![],
        },
        1 => RegData {
            is_stack: false,
            name: "ra".to_string(),
            units: vec![x_units[0]],
        },
        2 => RegData {
            is_stack: false,
            name: "sp".to_string(),
            units: vec![],
        },
        3 => RegData {
            is_stack: false,
            name: "gp".to_string(),
            units: vec![],
        },
        4 => RegData {
            is_stack: false,
            name: "tp".to_string(),
            units: vec![],
        },
        _ => RegData {
            is_stack: false,
            name: format!("x{i}"),
            units: vec![x_units[i - 5 + 1]],
        },
    });
    let x_regs: Vec<_> = all_x_regs
        .iter()
        .cloned()
        .filter(|i| !matches!(i.index(), 0 | 2 | 3 | 4))
        .collect();
    let x_fixed_stack = reginfo.make_regs(num_fixed_stack, |i| RegData {
        is_stack: true,
        name: format!("int_stack{i}"),
        units: vec![x_stack_units[i]],
    });
    // This excludes pairs with non-allocatable registers: [0, 1] [2, 3] [4, 5]
    let zacas_regs = reginfo.make_reg_group(13, |i| RegGroupData {
        regs: vec![all_x_regs[(i + 3) * 2], all_x_regs[(i + 3) * 2 + 1]],
    });
    let x_stack_class = reginfo.classes.push(RegClassData {
        desc: "General-purpose registers + stack".to_string(),
        superclass: None,
        group_size: 1,
        allows_spillslots: true,
        spill_cost: 0.5,
        members: RegGroupList::Single([&x_regs[..], &x_fixed_stack[..]].concat()),
        preferred_regs: RegGroupList::Single(all_x_regs[10..=15].into()),
        non_preferred_regs: RegGroupList::Single(
            [
                &all_x_regs[1..=1],
                &all_x_regs[5..=7],
                &all_x_regs[16..=17],
                &all_x_regs[28..=31],
            ]
            .concat(),
        ),
        callee_saved_preferred_regs: RegGroupList::Single(all_x_regs[8..=9].into()),
        callee_saved_non_preferred_regs: RegGroupList::Single(all_x_regs[18..=27].into()),
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
        preferred_regs: RegGroupList::Single(all_x_regs[10..=15].into()),
        non_preferred_regs: RegGroupList::Single(
            [
                &all_x_regs[1..=1],
                &all_x_regs[5..=7],
                &all_x_regs[16..=17],
                &all_x_regs[28..=31],
            ]
            .concat(),
        ),
        callee_saved_preferred_regs: RegGroupList::Single(all_x_regs[8..=9].into()),
        callee_saved_non_preferred_regs: RegGroupList::Single(all_x_regs[18..=27].into()),
    });
    let casp_class = reginfo.classes.push(RegClassData {
        desc: "Aligned register pairs for Zacas".to_string(),
        superclass: Some(x_class),
        group_size: 2,
        allows_spillslots: false,
        spill_cost: 1.0,
        members: RegGroupList::Multi(zacas_regs.clone()),
        preferred_regs: RegGroupList::Multi(
            [&zacas_regs[0..=0], &zacas_regs[2..=5], &zacas_regs[11..=12]].concat(),
        ),
        non_preferred_regs: RegGroupList::Multi(vec![]),
        callee_saved_preferred_regs: RegGroupList::Multi(
            [&zacas_regs[1..=1], &zacas_regs[6..=10]].concat(),
        ),
        callee_saved_non_preferred_regs: RegGroupList::Multi(vec![]),
    });
    reginfo.banks.push(RegBankData {
        desc: "General-purpose registers".to_string(),
        top_level_class: x_stack_class,
        stack_to_stack_class: x_class,
        spillslot_size: 8,
        classes: vec![x_stack_class, x_stack_only_class, x_class, casp_class],
    });

    // Float registers
    let f_units = reginfo.make_units(32);
    let f_stack_units = reginfo.make_units(num_fixed_stack);
    let f_regs = reginfo.make_regs(32, |i| RegData {
        is_stack: false,
        name: format!("f{i}"),
        units: vec![f_units[i]],
    });
    let f_fixed_stack = reginfo.make_regs(num_fixed_stack, |i| RegData {
        is_stack: true,
        name: format!("f_stack{i}"),
        units: vec![f_stack_units[i]],
    });
    let f_stack_class = reginfo.classes.push(RegClassData {
        desc: "Float registers + stack".to_string(),
        superclass: None,
        group_size: 1,
        allows_spillslots: true,
        spill_cost: 0.5,
        members: RegGroupList::Single([&f_regs[..], &f_fixed_stack[..]].concat()),
        preferred_regs: RegGroupList::Single(f_regs[10..=15].into()),
        non_preferred_regs: RegGroupList::Single(
            [&f_regs[0..=7], &f_regs[16..=17], &f_regs[28..=31]].concat(),
        ),
        callee_saved_preferred_regs: RegGroupList::Single(f_regs[8..=9].into()),
        callee_saved_non_preferred_regs: RegGroupList::Single(f_regs[18..=27].into()),
    });
    let f_stack_only_class = reginfo.classes.push(RegClassData {
        desc: "Float stack only".to_string(),
        superclass: Some(f_stack_class),
        group_size: 1,
        allows_spillslots: true,
        spill_cost: 0.0,
        members: RegGroupList::Single(f_fixed_stack.clone()),
        preferred_regs: RegGroupList::Single(vec![]),
        non_preferred_regs: RegGroupList::Single(vec![]),
        callee_saved_preferred_regs: RegGroupList::Single(vec![]),
        callee_saved_non_preferred_regs: RegGroupList::Single(vec![]),
    });
    let f_class = reginfo.classes.push(RegClassData {
        desc: "Float registers".to_string(),
        superclass: Some(f_stack_class),
        group_size: 1,
        allows_spillslots: false,
        spill_cost: 1.0,
        members: RegGroupList::Single(f_regs.clone()),
        preferred_regs: RegGroupList::Single(f_regs[10..=15].into()),
        non_preferred_regs: RegGroupList::Single(
            [&f_regs[0..=7], &f_regs[16..=17], &f_regs[28..=31]].concat(),
        ),
        callee_saved_preferred_regs: RegGroupList::Single(f_regs[8..=9].into()),
        callee_saved_non_preferred_regs: RegGroupList::Single(f_regs[18..=27].into()),
    });
    reginfo.banks.push(RegBankData {
        desc: "Float registers".to_string(),
        top_level_class: f_stack_class,
        stack_to_stack_class: f_class,
        spillslot_size: 8,
        classes: vec![f_stack_class, f_stack_only_class, f_class],
    });

    // Vector registers for LMUL=1,2,4,8
    let v_units = reginfo.make_units(32);
    for lmul in [1, 2, 4, 8] {
        let v_regs = reginfo.make_regs(32 / lmul, |i| RegData {
            is_stack: false,
            name: if lmul == 1 {
                format!("v{i}")
            } else {
                format!("v{}_x{lmul}", i * lmul)
            },
            units: (0..lmul).map(|j| v_units[i * lmul + j]).collect(),
        });
        let v_stack_units = reginfo.make_units(num_fixed_stack);
        let v_fixed_stack = reginfo.make_regs(num_fixed_stack, |i| RegData {
            is_stack: true,
            name: if lmul != 1 {
                format!("v_stack{i}_x{lmul}")
            } else {
                format!("v_stack{i}")
            },
            units: vec![v_stack_units[i]],
        });

        let v_stack_class = reginfo.classes.push(RegClassData {
            desc: format!("Vector registers + stack LMUL={lmul}"),
            superclass: None,
            group_size: 1,
            allows_spillslots: true,
            spill_cost: 0.5,
            members: RegGroupList::Single([&v_regs[..], &v_fixed_stack[..]].concat()),
            preferred_regs: RegGroupList::Single(v_regs.clone()),
            non_preferred_regs: RegGroupList::Single(vec![]),
            callee_saved_preferred_regs: RegGroupList::Single(vec![]),
            callee_saved_non_preferred_regs: RegGroupList::Single(vec![]),
        });
        let v_stack_only_class = reginfo.classes.push(RegClassData {
            desc: format!("Vector stack only LMUL={lmul}"),
            superclass: Some(v_stack_class),
            group_size: 1,
            allows_spillslots: true,
            spill_cost: 0.0,
            members: RegGroupList::Single(v_fixed_stack.clone()),
            preferred_regs: RegGroupList::Single(vec![]),
            non_preferred_regs: RegGroupList::Single(vec![]),
            callee_saved_preferred_regs: RegGroupList::Single(vec![]),
            callee_saved_non_preferred_regs: RegGroupList::Single(vec![]),
        });
        let v_class = reginfo.classes.push(RegClassData {
            desc: format!("Vector registers LMUL={lmul}"),
            superclass: Some(v_stack_class),
            group_size: 1,
            allows_spillslots: false,
            spill_cost: 1.0,
            members: RegGroupList::Single(v_regs.clone()),
            preferred_regs: RegGroupList::Single(v_regs.clone()),
            non_preferred_regs: RegGroupList::Single(vec![]),
            callee_saved_preferred_regs: RegGroupList::Single(vec![]),
            callee_saved_non_preferred_regs: RegGroupList::Single(vec![]),
        });

        // Register groups for segment load/store.
        let mut classes = vec![v_stack_class, v_stack_only_class, v_class];
        for num_seg in 2..=8 {
            if num_seg * lmul > 8 {
                break;
            }

            let windows = v_regs.windows(num_seg);
            let segment_regs = reginfo.make_reg_group(windows.len(), |i| RegGroupData {
                regs: v_regs.windows(num_seg).nth(i).unwrap().into(),
            });
            let segment_class = reginfo.classes.push(RegClassData {
                desc: format!(
                    "Vector register groups for {num_seg} segment load/store LMUL={lmul}"
                ),
                superclass: Some(v_class),
                group_size: num_seg,
                allows_spillslots: false,
                spill_cost: 1.0,
                members: RegGroupList::Multi(segment_regs.clone()),
                preferred_regs: RegGroupList::Multi(segment_regs.clone()),
                non_preferred_regs: RegGroupList::Multi(vec![]),
                callee_saved_preferred_regs: RegGroupList::Multi(vec![]),
                callee_saved_non_preferred_regs: RegGroupList::Multi(vec![]),
            });
            classes.push(segment_class);
        }

        reginfo.banks.push(RegBankData {
            desc: format!("Vector registers LMUL={lmul}"),
            top_level_class: v_stack_class,
            stack_to_stack_class: v_class,
            // Technically this is a scalable vector, but for now we just
            // expose it as 128-bit * LMUL.
            spillslot_size: 16 * lmul,
            classes,
        });
    }

    reginfo
}
