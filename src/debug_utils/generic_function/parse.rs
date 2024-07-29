use alloc::vec;
use alloc::vec::Vec;
use core::str::FromStr;

use anyhow::Result;
use cranelift_entity::{EntityRef, PrimaryMap, SecondaryMap};
use pest::error::{Error, ErrorVariant};
use pest::iterators::Pair;
use pest::{Parser, Span};
use pest_derive::Parser;

use super::{BlockData, GenericFunction, InstData, ValueData};
use crate::function::{
    Block, Inst, InstRange, Operand, OperandConstraint, OperandKind, RematCost, Value, ValueGroup,
};

#[derive(Parser)]
#[grammar = "debug_utils/generic_function/grammar.pest"]
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
            Rule::value | Rule::physreg | Rule::regclass | Rule::inst | Rule::block
        ));
        out.push(parse_entity(pair)?);
    }
    Ok(out)
}

fn parse_value_declaration(
    pair: Pair<'_, Rule>,
    values: &mut PrimaryMap<Value, ValueData>,
    reftype_values: &mut Vec<Value>,
) -> Result<()> {
    let span = pair.as_span();
    let mut bank = None;
    let mut remat = None;
    let mut is_reftype = false;
    for pair in pair.into_inner() {
        match pair.as_rule() {
            Rule::value => {
                parse_expected_entity(pair, values.next_key())?;
            }
            Rule::regbank => {
                if bank.is_some() {
                    Err(custom_error(pair.as_span(), "duplicate attribute"))?;
                }
                bank = Some(parse_entity(pair)?);
            }
            Rule::remat => {
                if remat.is_some() {
                    Err(custom_error(pair.as_span(), "duplicate attribute"))?;
                }
                let [cost, class] = extract(pair, [Rule::remat_cost, Rule::regclass]);
                let cost = match cost.as_str() {
                    "cheaper_than_move" => RematCost::CheaperThanMove,
                    "cheaper_than_load" => RematCost::CheaperThanLoad,
                    _ => unreachable!(),
                };
                let class = parse_entity(class)?;
                remat = Some((cost, class));
            }
            Rule::reftype => {
                if is_reftype {
                    Err(custom_error(pair.as_span(), "duplicate attribute"))?;
                }
                is_reftype = true;
            }
            _ => unreachable!(),
        }
    }
    let Some(bank) = bank else {
        Err(custom_error(span, "missing bank attribute"))?
    };
    let value = values.push(ValueData { bank, remat });
    if is_reftype {
        reftype_values.push(value);
    }
    Ok(())
}

fn parse_block_label(
    pair: Pair<'_, Rule>,
    blocks: &mut PrimaryMap<Block, BlockData>,
    insts: &mut PrimaryMap<Inst, InstData>,
) -> Result<()> {
    let [block, value_list, frequency] =
        extract(pair, [Rule::block, Rule::value_list, Rule::frequency]);
    parse_expected_entity(block, blocks.next_key())?;
    let block_params_in = parse_entity_list(value_list)?;
    let [float] = extract(frequency, [Rule::float]);
    let frequency = parse_number(float)?;
    blocks.push(BlockData {
        insts: InstRange::new(insts.next_key(), insts.next_key()),
        preds: vec![],
        succs: vec![],
        frequency,
        block_params_in,
        block_params_out: vec![],
    });
    Ok(())
}

fn parse_opcode(
    pair: Pair<'_, Rule>,
    data: &mut InstData,
    block_data: &mut BlockData,
) -> Result<()> {
    let pair = pair.into_inner().next().unwrap();
    match pair.as_rule() {
        Rule::normal_inst => {}
        Rule::ret => {
            data.is_terminator = true;
        }
        Rule::jump => {
            data.is_terminator = true;
            let [block, value_list] = extract(pair, [Rule::block, Rule::value_list]);
            block_data.succs.push(parse_entity(block)?);
            block_data.block_params_out = parse_entity_list(value_list)?;
        }
        Rule::branch => {
            data.is_terminator = true;
            let [block_list] = extract(pair, [Rule::block_list]);
            block_data.succs = parse_entity_list(block_list)?;
        }
        _ => unreachable!(),
    }
    Ok(())
}

fn parse_attribute(
    pair: Pair<'_, Rule>,
    is_pure: &mut bool,
    is_safepoint: &mut bool,
) -> Result<()> {
    match pair.as_str() {
        "safepoint" => {
            if *is_safepoint {
                Err(custom_error(pair.as_span(), "duplicate attribute"))?;
            }
            *is_safepoint = true;
        }
        "pure" => {
            if *is_pure {
                Err(custom_error(pair.as_span(), "duplicate attribute"))?;
            }
            *is_pure = true;
        }
        _ => unreachable!(),
    }
    Ok(())
}

fn parse_operand(
    pair: Pair<'_, Rule>,
    groups: &mut PrimaryMap<ValueGroup, Vec<Value>>,
) -> Result<Operand> {
    let [operand_kind, value_list, constraint] = extract(
        pair,
        [Rule::operand_kind, Rule::value_list, Rule::constraint],
    );
    let values = parse_entity_list(value_list)?;
    let kind = if let [value] = values[..] {
        match operand_kind.as_str() {
            "Use" => OperandKind::Use(value),
            "Def" => OperandKind::Def(value),
            "EarlyDef" => OperandKind::EarlyDef(value),
            _ => unreachable!(),
        }
    } else {
        let group = groups.push(values);
        match operand_kind.as_str() {
            "Use" => OperandKind::UseGroup(group),
            "Def" => OperandKind::DefGroup(group),
            "EarlyDef" => OperandKind::EarlyDefGroup(group),
            _ => unreachable!(),
        }
    };
    let constraint_pair = constraint.into_inner().next().unwrap();
    let constraint = match constraint_pair.as_rule() {
        Rule::physreg => OperandConstraint::Fixed(parse_entity(constraint_pair)?),
        Rule::regclass => OperandConstraint::Class(parse_entity(constraint_pair)?),
        Rule::reuse => {
            let [number] = extract(constraint_pair, [Rule::number]);
            OperandConstraint::Reuse(parse_number(number)?)
        }
        _ => unreachable!(),
    };
    Ok(Operand::new(kind, constraint))
}

fn parse_instruction(
    pair: Pair<'_, Rule>,
    blocks: &mut PrimaryMap<Block, BlockData>,
    insts: &mut PrimaryMap<Inst, InstData>,
    groups: &mut PrimaryMap<ValueGroup, Vec<Value>>,
    safepoints: &mut Vec<Inst>,
) -> Result<()> {
    let Some((block, block_data)) = blocks.last_mut() else {
        Err(custom_error(
            pair.as_span(),
            "instruction is not inside a block",
        ))?
    };
    let mut data = InstData {
        operands: vec![],
        clobbers: vec![],
        block,
        is_terminator: false,
        is_pure: false,
    };
    let mut is_safepoint = false;
    for pair in pair.into_inner() {
        match pair.as_rule() {
            Rule::inst_label => {
                // We specifically ignore instruction labels to make it easier to edit code.
            }
            Rule::opcode => parse_opcode(pair, &mut data, block_data)?,
            Rule::attribute => parse_attribute(pair, &mut data.is_pure, &mut is_safepoint)?,
            Rule::normal_operand => data.operands.push(parse_operand(pair, groups)?),
            Rule::nonallocatable_operand => {
                let [physreg] = extract(pair, [Rule::physreg]);
                data.operands
                    .push(Operand::fixed_nonallocatable(parse_entity(physreg)?));
            }
            Rule::clobber => {
                let [unit] = extract(pair, [Rule::unit]);
                data.clobbers.push(parse_entity(unit)?);
            }
            _ => unreachable!(),
        }
    }
    let inst = insts.push(data);
    if is_safepoint {
        safepoints.push(inst);
    }
    block_data.insts.to = insts.next_key();
    Ok(())
}

fn compute_preds(blocks: &mut PrimaryMap<Block, BlockData>) {
    let mut preds = SecondaryMap::with_default(vec![]);
    for (block, data) in blocks.iter() {
        for &succ in &data.succs {
            preds[succ].push(block);
        }
    }
    for (block, preds) in preds.iter() {
        blocks[block].preds = preds.clone();
    }
}

impl GenericFunction {
    /// Parses a textual representation of a [`Function`] into a
    /// [`GenericFunction`].
    ///
    /// The text format is the same as the one generated by [`dump_function`].
    ///
    /// [`dump_function`]: crate::debug_utils::dump_function
    /// [`Function`]: crate::function::Function
    pub fn parse(input: &str) -> Result<Self> {
        let parse_result = FunctionParser::parse(Rule::function, input)?;

        let mut blocks = PrimaryMap::new();
        let mut insts = PrimaryMap::new();
        let mut values = PrimaryMap::new();
        let mut value_groups = PrimaryMap::new();
        let mut safepoints = vec![];
        let mut reftype_values = vec![];

        for pair in parse_result {
            match pair.as_rule() {
                Rule::value_declaration => {
                    parse_value_declaration(pair, &mut values, &mut reftype_values)?;
                }
                Rule::block_label => parse_block_label(pair, &mut blocks, &mut insts)?,
                Rule::instruction => parse_instruction(
                    pair,
                    &mut blocks,
                    &mut insts,
                    &mut value_groups,
                    &mut safepoints,
                )?,
                Rule::EOI => {}
                _ => unreachable!(),
            }
        }

        // Compute block predecessors since they are not encoded in the dump.
        compute_preds(&mut blocks);

        Ok(Self {
            blocks,
            insts,
            values,
            value_groups,
            safepoints,
            reftype_values,
        })
    }
}
