# regalloc3

New register allocator implementation designed as a successor to [regalloc2].
The design takes inspiration from many sources, including LLVM's greedy register allocator.

New API-level features compared to regalloc2:
- Support for 2^28 (~268M) values ("vregs" in regalloc2) per function.
- Support for up to 64 different register classes.
- Support for up to 512 registers.
- Support for compound registers and overlapping registers (e.g. `S0` / `S1` / `D0` on AArch32).
- Support for multi-register groups (e.g. `CASP` register pairs on AArch64, `LD4` SIMD vector loads on AArch64).
- Support for rematerialization of constants as an alternative to spilling.
- Support for explicit block frequencies.
- Register descriptions are described by a `RegInfo` trait.
- Functions and register descriptions can be serialized to a text format and parsed back into memory.
- Faster compilation of multiple functions by preserving and reusing memory allocations across runs.
- Validation functions to check `Function` and `RegInfo` implementations.

Refer to the extensive [design document] for more detail on the internal implementation of the allocator.

[regalloc2]: https://github.com/bytecodealliance/regalloc2
[design document]: DESIGN.md

## Flags

This crate has the following Cargo features:

- `parse`: Support for parsing a `GenericFunction` from a text representation. Requires `std`.
- `serde`: Support for serializing and de-serializing all public types.
- `arbitrary`: Support for generating randomized functions for testing.
- `trace-log`: Enables detailed logging which can be somewhat expensive and very verbose.

## Debugging register allocation issues

Regalloc3 includes many utlities for debugging issues related to register allocation. The very first thing that you should do if you encounter any issues is to check that your `Function` and `RegInfo` implementations both pass the validation functions under the `debug_utils` module.

If this doesn't solve your issues then you can dump these to a text format using `DisplayRegInfo` and `DisplayFunction` which can then be passed to `regalloc3-tool` for further debugging.

## License

Licensed under either of:

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
