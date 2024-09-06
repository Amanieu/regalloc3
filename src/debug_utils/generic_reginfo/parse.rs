use alloc::vec;
use alloc::vec::Vec;
use core::str::FromStr;

use anyhow::Result;
use cranelift_entity::{EntityRef, PrimaryMap, SecondaryMap};
use pest::error::{Error, ErrorVariant};
use pest::iterators::Pair;
use pest::{Parser, Span};
use pest_derive::Parser;

use super::{GenericRegInfo, PhysRegData, RegBankData, RegClassData, RegGroupData};
use crate::reginfo::{
    PhysReg, RegBank, RegClass, RegClassSet, RegGroup, RegOrRegGroup, RegOrRegGroupSet,
    SpillSlotSize,
};

#[derive(Parser)]
#[grammar = "debug_utils/generic_reginfo/grammar.pest"]
pub struct FunctionParser;

/// Helper function to extract N sub-pairs when the layout of a rule is fixed.
fn extract<const N: usize>(pair: Pair<'_, Rule>, expected_rules: [Rule; N]) -> [Pair<'_, Rule>; N] {
    let mut out = [(); N].map(|()| pair.clone());
    let mut i = 0;
    for pair in pair.into_inner() {
        assert_eq!(pair.as_rule(), expected_rules[i]);
        out[i] = pair;
        i += 1;
    }
    assert_eq!(i, N);
    out
}

/// Helper function to emit a custom error at the given span.
fn custom_error(span: Span<'_>, msg: &str) -> Error<Rule> {
    Error::new_from_span(
        ErrorVariant::<Rule>::CustomError {
            message: msg.into(),
        },
        span,
    )
}

fn parse_number<T: FromStr>(pair: Pair<'_, Rule>) -> Result<T> {
    Ok(pair.as_str().parse().map_err(|_| {
        // This can only fail due to integer overflow, the rule only allows
        // digits.
        custom_error(pair.as_span(), "integer overflow")
    })?)
}

fn parse_entity<T: EntityRef>(pair: Pair<'_, Rule>) -> Result<T> {
    let [number] = extract(pair, [Rule::number]);
    let index = parse_number(number)?;
    Ok(T::new(index))
}

fn parse_expected_entity<T: EntityRef>(pair: Pair<'_, Rule>, expected: T) -> Result<()> {
    let span = pair.as_span();
    let entity = parse_entity::<T>(pair)?;
    if entity != expected {
        Err(custom_error(
            span,
            "must be declared in order and with no gaps",
        ))?;
    }
    Ok(())
}

fn parse_entity_list<T: EntityRef>(pair: Pair<'_, Rule>) -> Result<Vec<T>> {
    let mut out = vec![];
    for pair in pair.into_inner() {
        assert!(matches!(
            pair.as_rule(),
            Rule::unit | Rule::reg_group | Rule::reg | Rule::class | Rule::bank
        ));
        out.push(parse_entity(pair)?);
    }
    Ok(out)
}

fn parse_reg_group_list(
    pair: Pair<'_, Rule>,
    is_group: &mut Option<bool>,
) -> Result<Vec<RegOrRegGroup>> {
    let Some(list) = pair.into_inner().next() else {
        return Ok(vec![]);
    };
    Ok(match list.as_rule() {
        Rule::reg_list => {
            if *is_group == Some(true) {
                Err(custom_error(
                    list.as_span(),
                    "cannot mix registers and register groups in the same class",
                ))?;
            }
            *is_group = Some(false);
            parse_entity_list(list)?
                .into_iter()
                .map(RegOrRegGroup::single)
                .collect()
        }
        Rule::reg_group_list => {
            if *is_group == Some(false) {
                Err(custom_error(
                    list.as_span(),
                    "cannot mix registers and register groups in the same class",
                ))?;
            }
            *is_group = Some(true);
            parse_entity_list(list)?
                .into_iter()
                .map(RegOrRegGroup::multi)
                .collect()
        }
        _ => unreachable!(),
    })
}

fn parse_reg_def(pair: Pair<'_, Rule>, regs: &mut PrimaryMap<PhysReg, PhysRegData>) -> Result<()> {
    let [reg, location, kind] = extract(pair, [Rule::reg, Rule::reg_location, Rule::reg_kind]);
    parse_expected_entity(reg, regs.next_key())?;
    let is_fixed_stack = location.as_str() == "stack";
    let kind = kind.into_inner().next().unwrap();
    let units = match kind.as_rule() {
        Rule::nonallocatable => vec![],
        Rule::unit_list => parse_entity_list(kind)?,
        _ => unreachable!(),
    };
    regs.push(PhysRegData {
        bank: None,
        is_fixed_stack,
        units,
    });
    Ok(())
}

fn parse_reg_group_def(
    pair: Pair<'_, Rule>,
    groups: &mut PrimaryMap<RegGroup, RegGroupData>,
) -> Result<()> {
    let [group, reg_list] = extract(pair, [Rule::reg_group, Rule::reg_list]);
    parse_expected_entity(group, groups.next_key())?;
    let regs = parse_entity_list(reg_list)?;
    groups.push(RegGroupData { regs });
    Ok(())
}

fn parse_class_def(
    pair: Pair<'_, Rule>,
    bank: RegBank,
    classes: &mut PrimaryMap<RegClass, RegClassData>,
    regs: &mut PrimaryMap<PhysReg, PhysRegData>,
    superclasses: &mut SecondaryMap<RegClass, RegClassSet>,
) -> Result<()> {
    let span = pair.as_span();
    let mut group_size = None;
    let mut includes_spillslots = false;
    let mut spill_cost = None;
    let mut registers = None;
    let mut preferred_regs = None;
    let mut non_preferred_regs = None;
    let mut callee_saved_preferred_regs = None;
    let mut callee_saved_non_preferred_regs = None;
    let mut is_group = None;
    for pair in pair.into_inner() {
        match pair.as_rule() {
            Rule::class => {
                parse_expected_entity(pair, classes.next_key())?;
            }
            Rule::superclasses => {
                let [class_list] = extract(pair, [Rule::class_list]);
                let list = parse_entity_list(class_list)?;
                superclasses[classes.next_key()].extend(list);
            }
            Rule::group_size => {
                if group_size.is_some() {
                    Err(custom_error(pair.as_span(), "duplicate attribute"))?;
                }
                let [number] = extract(pair, [Rule::number]);
                group_size = Some(parse_number(number)?);
            }
            Rule::allows_spillslots => {
                if includes_spillslots {
                    Err(custom_error(pair.as_span(), "duplicate attribute"))?;
                }
                includes_spillslots = true;
            }
            Rule::spill_cost => {
                if spill_cost.is_some() {
                    Err(custom_error(pair.as_span(), "duplicate attribute"))?;
                }
                let [float] = extract(pair, [Rule::float]);
                spill_cost = Some(parse_number(float)?);
            }
            Rule::class_registers => {
                if registers.is_some() {
                    Err(custom_error(pair.as_span(), "duplicate attribute"))?;
                }
                let [reg_or_reg_group_list] = extract(pair, [Rule::reg_or_reg_group_list]);
                registers = Some(parse_reg_group_list(reg_or_reg_group_list, &mut is_group)?);
            }
            Rule::preferred_regs => {
                if preferred_regs.is_some() {
                    Err(custom_error(pair.as_span(), "duplicate attribute"))?;
                }
                let [reg_or_reg_group_list] = extract(pair, [Rule::reg_or_reg_group_list]);
                preferred_regs = Some(parse_reg_group_list(reg_or_reg_group_list, &mut is_group)?);
            }
            Rule::non_preferred_regs => {
                if non_preferred_regs.is_some() {
                    Err(custom_error(pair.as_span(), "duplicate attribute"))?;
                }
                let [reg_or_reg_group_list] = extract(pair, [Rule::reg_or_reg_group_list]);
                non_preferred_regs =
                    Some(parse_reg_group_list(reg_or_reg_group_list, &mut is_group)?);
            }
            Rule::callee_saved_preferred_regs => {
                if callee_saved_preferred_regs.is_some() {
                    Err(custom_error(pair.as_span(), "duplicate attribute"))?;
                }
                let [reg_or_reg_group_list] = extract(pair, [Rule::reg_or_reg_group_list]);
                callee_saved_preferred_regs =
                    Some(parse_reg_group_list(reg_or_reg_group_list, &mut is_group)?);
            }
            Rule::callee_saved_non_preferred_regs => {
                if callee_saved_non_preferred_regs.is_some() {
                    Err(custom_error(pair.as_span(), "duplicate attribute"))?;
                }
                let [reg_or_reg_group_list] = extract(pair, [Rule::reg_or_reg_group_list]);
                callee_saved_non_preferred_regs =
                    Some(parse_reg_group_list(reg_or_reg_group_list, &mut is_group)?);
            }
            _ => unreachable!(),
        }
    }
    let Some(spill_cost) = spill_cost else {
        Err(custom_error(span, "missing spill_cost attribute"))?
    };
    let Some(registers) = registers else {
        Err(custom_error(span, "missing registers attribute"))?
    };
    let group_size = group_size.unwrap_or(1);
    if let Some(is_group) = is_group {
        if is_group != (group_size != 1) {
            Err(custom_error(span, "group_size doesn't match registers"))?;
        }
        if !is_group {
            for &reg in &registers {
                if regs[reg.as_single()]
                    .bank
                    .is_some_and(|reg_bank| reg_bank != bank)
                {
                    Err(custom_error(span, "register used with different banks"))?;
                }
                regs[reg.as_single()].bank = Some(bank);
            }
        }
    };
    let mut members = RegOrRegGroupSet::new();
    for &reg in &registers {
        members.insert(reg);
    }
    let preferred_regs = preferred_regs.unwrap_or(vec![]);
    let non_preferred_regs = non_preferred_regs.unwrap_or(vec![]);
    let callee_saved_preferred_regs = callee_saved_preferred_regs.unwrap_or(vec![]);
    let callee_saved_non_preferred_regs = callee_saved_non_preferred_regs.unwrap_or(vec![]);
    classes.push(RegClassData {
        bank,
        includes_spillslots,
        spill_cost,
        group_size,
        members,
        sub_classes: RegClassSet::new(),
        preferred_regs,
        non_preferred_regs,
        callee_saved_preferred_regs,
        callee_saved_non_preferred_regs,
    });
    Ok(())
}

fn parse_bank_def(
    pair: Pair<'_, Rule>,
    banks: &mut PrimaryMap<RegBank, RegBankData>,
    classes: &mut PrimaryMap<RegClass, RegClassData>,
    regs: &mut PrimaryMap<PhysReg, PhysRegData>,
    superclasses: &mut SecondaryMap<RegClass, RegClassSet>,
) -> Result<()> {
    let span = pair.as_span();
    let mut top_level_class = None;
    let mut stack_to_stack_class = None;
    let mut spillslot_size = None;
    for pair in pair.into_inner() {
        match pair.as_rule() {
            Rule::bank => {
                parse_expected_entity(pair, banks.next_key())?;
            }
            Rule::top_level_class => {
                if top_level_class.is_some() {
                    Err(custom_error(pair.as_span(), "duplicate attribute"))?;
                }
                let [class] = extract(pair, [Rule::class]);
                top_level_class = Some(parse_entity(class)?);
            }
            Rule::stack_to_stack_class => {
                if stack_to_stack_class.is_some() {
                    Err(custom_error(pair.as_span(), "duplicate attribute"))?;
                }
                let [class] = extract(pair, [Rule::class]);
                stack_to_stack_class = Some(parse_entity(class)?);
            }
            Rule::spillslot_size => {
                if spillslot_size.is_some() {
                    Err(custom_error(pair.as_span(), "duplicate attribute"))?;
                }
                let [number] = extract(pair, [Rule::number]);
                let span = number.as_span();
                let size: u32 = parse_number(number)?;
                if !size.is_power_of_two() {
                    Err(custom_error(span, "spillslot size must be a power of two"))?;
                }
                spillslot_size = Some(SpillSlotSize::new(size));
            }
            Rule::class_def => {
                parse_class_def(pair, banks.next_key(), classes, regs, superclasses)?;
            }
            _ => unreachable!(),
        }
    }
    let Some(top_level_class) = top_level_class else {
        Err(custom_error(span, "missing top_level_class attribute"))?
    };
    let Some(stack_to_stack_class) = stack_to_stack_class else {
        Err(custom_error(span, "missing stack_to_stack_class attribute"))?
    };
    let Some(spillslot_size) = spillslot_size else {
        Err(custom_error(span, "missing spillslot_size attribute"))?
    };
    banks.push(RegBankData {
        top_level_class,
        stack_to_stack_class,
        spillslot_size,
    });
    Ok(())
}

/// The text format has superclasses, conver those into subclasses which are
/// needed by `GenericRegInfo`.
fn resolve_subclasses(
    classes: &mut PrimaryMap<RegClass, RegClassData>,
    superclasses: &SecondaryMap<RegClass, RegClassSet>,
) {
    // Propagate sub-classes backwards. This works because superclasses always
    // have a lower index than subclasses.
    for class in classes.keys().rev() {
        classes[class].sub_classes.insert(class);
        let sub_classes = classes[class].sub_classes;
        for superclass in &superclasses[class] {
            classes[superclass].sub_classes |= sub_classes;
        }
    }
}

impl GenericRegInfo {
    /// Parses a textual register description into a [`GenericRegInfo`].
    pub fn parse(input: &str) -> Result<Self> {
        let parse_result = FunctionParser::parse(Rule::reginfo, input)?;

        let mut banks = PrimaryMap::new();
        let mut classes = PrimaryMap::new();
        let mut regs = PrimaryMap::new();
        let mut groups = PrimaryMap::new();
        let mut superclasses = SecondaryMap::new();

        for pair in parse_result {
            match pair.as_rule() {
                Rule::reg_def => parse_reg_def(pair, &mut regs)?,
                Rule::reg_group_def => parse_reg_group_def(pair, &mut groups)?,
                Rule::bank_def => {
                    parse_bank_def(pair, &mut banks, &mut classes, &mut regs, &mut superclasses)?;
                }
                Rule::EOI => {}
                _ => unreachable!("{:?}", pair.as_rule()),
            }
        }

        resolve_subclasses(&mut classes, &superclasses);

        Ok(Self {
            banks,
            classes,
            regs,
            groups,
        })
    }
}
