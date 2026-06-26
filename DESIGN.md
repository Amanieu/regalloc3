# Introduction

This document describes the architecture of the regalloc3 register allocator. Readers are expected to be familiar with the concepts used in the public-facing API of this crate, such as values, instructions and constraints. Reading the API documentation first is strongly encouraged, in particular the one for the `function` and `reginfo` modules.

## Live ranges and uses

The *live range* of a value refers to the paths through a function from its definition to all its uses. Specifically, this covers all instructions across which a value needs to be preserved because it is later used. This is represented as a slice of `ValueSegment`, each covering a linear range between two `LiveRangePoint`.

To properly model early-def operands, it is necessary to further split each allocation into 3 slots: `Boundary`, `Early` and `Normal`. `Boundary` represents the boundary between 2 instructions and where move instructions can be inserted. `Early` is the point where an `EarlyDef` operand starts. `Normal` is the point where `Use` operands end and `Def` operands begin. It follows from this that an `EarlyDef` operand can't be assigned the same register as a `Use` operand because their live range would overlap between the `Early` and `Normal` slots (a register can only hold one value at a time).

`LiveRangePoint` encodes an instruction index and a slot into a `u32`, allowing it to be compared with other `LiveRangePoint`.

Each `ValueSegment` also contains a `UseList`: a slice of `Use` which represents all the places where the value is used or defined. Each `Use` contains the `Inst` at which the use occurs and a `UseKind` which describes the kind of use. This includes:
- Use/def as an instruction operand within a register class.
- Use/def as an instruction operand with a fixed register.
- Use that is tied to the same register as a def.
- Use as an outgoing block parameter.
- Definition as an incoming block parameter.

The `UseList` also holds some flags, notably live-in and live-out flags. A segment with the live-in flag receives an incoming value from another segment and a segment with the live-out flag passes the value to another segment at the end of its live range. A live-in/live-out `ValueSegment` must start/end on a `Boundary` slot.

The representation of a `ValueSegment` makes splitting it very efficient: a split in the middle of the segment (always at a `Boundary`) will produce 2 segments, one of which is live-out and the other live-in. The `LiveRange` and `UseList` are simply split at the chosen instruction boundary, with no data movement necessary.

## Values, value sets and virtual registers

`Value` represents an SSA value in the source function, which means that it has exactly one definition (which must be an instruction operand or an incoming block parameter) and any number of uses.

A `ValueSet` represents a group of values whose live ranges do not overlap. All the `ValueSegment` of its constituent values are concatenated and sorted by their live range.

A `VirtReg` represents a subset of a `ValueSet` which has a register class constraint and a spill weight.

## The register matrix

The register matrix tracks the live ranges allocated to each `RegUnit`. There are 2 kinds of allocations in the matrix:
- Virtual register allocations which cover the live range of a virtual register.
- Fixed definitions which start from the `Early` or `Normal` slot of an instruction to the next instruction boundary. This is also used by clobber constraints.
- Fixed uses which start from a `Boundary` to the `Normal` slot of an instruction. The matrix also tracks the `Value` of fixed uses, since this is needed for interference checks.

The matrix is frequently queried for interference checks where each `ValueSegment` of a `VirtReg` is checked for overlaps against existing allocations in a `RegUnit`. Fixed use allocations are treated specially since they *are* allowed to overlap a `VirtReg` allocation only if the fixed use is of the same `Value` as the `ValueSegment` being checked.

Because it is so critical for performance, the allocation matrix is backed by an optimized B-Tree (from the `brie-tree` crate) for each `RegUnit`.

# Allocation process

## Value live range construction

This stage constructs a vector of `ValueSegment` for each `Value`.

### Use collection

We iterate over all instructions in the function in forward order and collect a `Use` for each operand and block parameter which uses this value.

The `Use` for all values are collected in a single temporary vector along with a `next` index. This is used to construct a linked list of `Use` for each value, with each `Value` having its own linked list head and tail. This is later used to create a linear slice of `Use` for each value.

During this process, fixed-use and fixed-def constraints are handled specially: the live range is split between the fixed portion in the instruction where the use occurs (`Boundary` to `Normal` for uses, `Early`/`Normal` to `Boundary` for defs) and the variable portion (where it doesn't *have* to be kept in a particular register). The fixed portion is reserved directly in the register matrix and is not considered part of the value's live range. Instead, the 2 portions are logically joined by a move instruction.

We also record *hints* for instructions that are just after a fixed-def or just before a fixed-use: this encourages live ranges which are move-connected with the fixed use/def are allocated to the same register, which would eliminate the move. The hints take block boundaries into account: if a fixed-def occurs on the last instruction of a block, the hint is recorded at the first instruction of all successor blocks. Similarly, a fixed-use on the first instruction of a block results in a hint being recorded on the last instruction of all predecessor blocks. Hints are given a *weight* which corresponds to the frequency of the block that would contain the move instruction if it isn't eliminated.

`Reuse` constraints also need special handling because the input value may still be used after the instruction, which would preclude it from being assigned to the same register as the output value. Instead, the live range of the `Reuse` output is extended to start at the previous instruction boundary and the live range of the corresponding input value is shrunk to the preceding instruction boundary. The two are then joined with a move from the input value to the output value *before* the instruction (which may later be eliminated by coalescing).

We also have special handling for the case where the same value is used multiple times in an instruction with only one of the uses being tied to a `Reuse` def. In that case *all* of the inputs are tied to the `Reuse` output and have their live range shrunk to the preceding `Boundary`. This is necessary otherwise the non-reused inputs would interfere with the output which would prevent them from being allocated to the same register.

```
# Example where %1 is used twice but only one of the uses is reused.
inst Def(%0):reuse(1) Use(%1):class0 Use(%1):class0
```

Once all `Use` in the function have been collected into per-value linked lists, we use those linked lists to generate the final `Use` vector for each `Value`. The linked list is guaranteed to already be sorted by instruction index since it was created by forward iteration through the function. Previously this was done by collecting all `Use` in a single vector and sorting them by `Value`, but it turns out that using a linked list is actually faster in practice because it avoids the `O(n * log(n))` complexity of sorting.

### Block live-in/live-out

The `ValueSegment` must cover parts of a value's live range that don't have any uses and only "carry" the value, even across blocks. This necessitates knowing, for each block, whether the value is live-in at the start of the block and live-out at the end of the block.

The algorithm involves 2 bit sets with 1 bit per `Block`, indicating whether the value is live-in or live-out.

1. The block containing the definition of the value is marked as live-in.
2. For each use other than the definition of the value:
   1. If it is not already live-in, push it to the work-list.
   2. While the work-list is non-empty:
      1. Pop the block from the work-list.
      2. If the block is already live-in then skip it.
      3. Mark all predecessor blocks as live-out.
      4. Push any newly live-out predecessor blocks to the work-list.

Because the live-in/live-out information is immediately consumed by segment construction for the current value, there is no need for a 2D bit-set for each `Block`/`Value` combination. Instead the 2 bit-sets are simply cleared and re-used when processing the next value.

### Segment construction

The `ValueSegment` vector is constructed by starting at the definition point of the value, which the `Function` invariants guarantee appears before any uses of that value in the linear instruction order, and then walking through the uses and live-in/live-out bitsets in lock-step. If a non-live-out block is found before a use then the current `ValueSegment` is ended and a new `ValueSegment` is started at the next live-in block. This continues until all blocks where the value is live have been processed.

## Coalescing

The coalescing pass attempts to merge multiple values into a `ValueSet` where allocating all values in a set to the same register would eliminate a move instruction. There are 2 sources of implied move instructions in SSA form: block parameters and `Reuse` operands. However we must also maintain the invariant that the live ranges of values in a set do not overlap.

Initially, each value is assigned its own `ValueSet` containing the `ValueSegment` vector for that value. The mapping of `Value` to `ValueSet` is tracked using a [Union-Find data structure]. Then, each block in the function is scanned to attempt to merge value pairs from `Reuse` constraints and outgoing block parameters in that block into the same `ValueSet`.

Merging two `ValueSet` works by walking the sorted vector of `ValueSegment` in each set and checking them for overlaps. If there are no overlaps then the two sorted lists are merged and the sets are unified in the union-find data structure.

The order in which coalescing is done is important because merging two values might prevent merging with a third value. As such the order in which blocks are processed for coalescing is prioritized by block frequency since this eliminates moves that would be executed more frequently. We also prioritize eliminating moves in critical edge blocks which only contain a single jump instruction since the entire block can be eliminated by jump threading if it doesn't contain any move instructions.

[Union-Find data structure]: https://en.wikipedia.org/wiki/Disjoint-set_data_structure

## Virtual register construction

The final preparation step before allocation is to construct `VirtReg` from `ValueSet`. `VirtReg` also consists of a vector of sorted `ValueSegment` but additionally has the constraint that all uses must be satisfiable with a consistent `RegClass`. This is normally done by computing the common sub-class of all register class uses in the `ValueSet`, but when there is no common sub-class then the set must be split at the point where the conflict occurs.

When conflicts occur between instructions then the segments are split at the point between the 2 conflicting uses where the block frequency is the lowest. However it is also possible to have a conflict on a single instruction when the same value is used twice with different constraints. In that case one of the conflicting `Use` is isolated into a separate virtual register while its use in the original virtual register is turned into a `UseKind::ConstraintConflict`.

Register group constraints require special handling in this stage. When multiple values are used in group operand, they must be allocated and evicted as a single unit by the register allocator. To handle this, virtual register construction will combine multiple virtual registers which share a group use into a *virtual register group*. All group uses in a `VirtRegGroup` must have the same size and a common sub-class, and each virtual register within the group must have a consistent and distinct *index* in each group use. Virtual registers are split where these constraints cannot be satisfied.

Finally, a spill weight is computed for each virtual register. This represents the allocation priority of a virtual register, with a preference for short use-dense live ranges over long-lived sparse live ranges. The weight is computed using the formula `use_weights / (num_instr + K)` where:
- `use_weights` is the sum of all `Use` weights in the virtual register, scaled by the frequency of the block containing the instruction.
- `num_instr` is the total number of instructions covered by the live range of all `ValueSegment` in the virtual registers.
- `K` is an adjustment factor (200 by default) which avoids depending too much on exact instruction counts for short live ranges. This causes the spill weight to represent the number of uses for short ranges and use density for larger ranges.

All virtual registers in the same group share the same spill weight, which is the minimum of that of each constituent virtual register.

A virtual register which only covers a single register and whose register class doesn't allow allocation to a spillslot is assigned an infinite spill weight since that virtual register cannot be further split and *must* be allocated.

## Main allocation loop

The main allocation loop attempts to assign each `VirtReg`/`VirtRegGroup` to a `PhysReg`/`RegGroup`. The algorithm is based on LLVM's [Greedy Register Allocator], but with some modifications to make it simpler and more efficient.

The code for allocation is generic over `VirtReg`/`VirtRegGroup` so that the specificities of register groups don't affect the performance of single register handling, which is much more common in practice.

[Greedy Register Allocator]: https://blog.llvm.org/2011/09/greedy-register-allocation-in-llvm-30.html

### Allocation queue

All virtual registers (except those part of a `VirtRegGroup`) and all virtual register groups are inserted into a priority queue. As virtual registers are evicted or split, they are inserted back into the allocation queue.

Each virtual register passes through the allocation queue twice: first in the `Evict` stage, then in the `Split` stage. All evictions are performed before any splitting is attempted so that splitting can be done around interference from already allocated registers.

Each entry in the queue is a `u64` which encodes both the priority and the virtual register index. The bit encoding is designed to prioritize virtual registers in the following order:
 - Earlier allocation stages are processed first.
 - Virtual registers with a fixed-register hint are prioritized.
 - Larger groups are harder to allocate, and so are prioritized.
 - Large live ranges are harder to allocate, and so are prioritized.
 
Allocating long live ranges first allows short live ranges to fit in the gap left between the long live ranges, while the opposite may force long live ranges to be split due to fragmentation caused by many small live ranges.

### Allocation order

Once a virtual register is popped from the priority queue, the set of physical registers it can be assigned to is determined by its register class. Each register class has an *allocation order* which specifies an ordered sequence of registers to attempt to assign a virtual register of that class to.

If we have hints that indicate we should favor particular physical registers for this virtual register then these registers will be prioritized in favor of the default allocation order. There are 2 types of hints:
- Fixed register hints apply where the live ranges in the virtual register overlap a hint that was collected during the initial live range construction. This indicates that the live range connects to a fixed-use or fixed-def and selecting that register would eliminate a move instruction.
- We also track the last register allocated for each value set as a weak hint: this encourages split virtual registers to be allocated to the same register which can help eliminate redundant moves in the final move optimization stage.

Fixed register hints have a weight which indicates the frequency of the move that would be eliminated: these are summed up for all hints that apply to the virtual register's live range, resulting in a *preference weight* for a particular register.

The allocation order is then sorted by preference weight, with unhinted registers having a weight of 0.

If a virtual register group is being allocated then the allocation order consists of a sequence of `RegGroup` instead of a sequence of `PhysReg`.

### Probing

Once the allocation order has been determined, the first step, no matter which stage the virtual register is in, is to attempt to allocate it to each register in the allocation order. If the allocation order is empty (this is only allowed for register classes that are spillable) then the virtual register is immediately spilled.

The live ranges in the virtual register are checked for overlap with the live ranges already allocated to each register in the allocation order. Where a `PhysReg` consists of multiple `RegUnit`, *all* units are checked for overlap. If there is no overlap then the virtual register can be assigned to that physical register. Similarly, register groups are handled by checking each `VirtReg` in the group with the corresponding `PhysReg` in the current `RegGroup` in the allocation order.

There is a special case when the selected register has a lower preference weight than the highest preference weight in the allocation order: we *really* want to use the preferred register even if other virtual registers are already using it. In such cases we will attempt to evict any conflicting virtual registers already in the preferred register, but only if their total preference for that register is lower than that of the incoming virtual register (and there are no fixed conflicts since those cannot be evicted). This ignores the spill weight of conflicting registers, but to ensure eventual termination, each virtual register is only allowed to do this kind of eviction *once*.

If there are no available registers that can be assigned to then the virtual register proceeds to either eviction or splitting depending on its stage.

### Eviction

This pass starts with another scan for conflicting live ranges for each register in the allocation order. Even though this performs some redundant work with the initial probing, it is more efficient to keep the initial probing fast since most virtual registers will be successfully allocated on the first try.

The maximum spill weight of all interfering virtual registers is collected (conflicting fixed constraints are considered to have infinite spill weight). If this exceeds the spill weight of the incoming virtual register then eviction fails for this register.

Again there is a special case when we have a preference for a particular register: we can evict virtual registers with a higher spill weight only if their total preference weight is lower than ours and the incoming virtual register hasn't already performed an eviction for preference.

Forward-progress is guaranteed by the fact that the spill weights in the register matrix always increase, with the exception of preferred register eviction, but this only happens once per virtual register.

If no register was suitable for eviction due to the incoming virtual register's spill weight being too low then it is re-queued for the splitting stage. When multiple registers are suitable for eviction then we select the one with the lowest cost, calculated as the total preference weight followed by the maximum spill weight of the virtual registers to be evicted. If eviction succeeds then all interfering virtual registers are removed from the register matrix and re-queued at the eviction stage.

### Splitting

Splitting occurs when a virtual register has too low of a spill weight to evict conflicting virtual registers for all registers in the allocation order. At this point, the only thing that can be done is to split it into smaller pieces which either have a higher spill weight by virtue of having a shorter live range or which are spillable by virtue of not containing any non-spillable uses.

Splitting works by picking the use in a virtual register with the highest weight and building a *split region* around it. Initially the region will only consist of a single instruction and will therefore have an infinite spill weight. Then, for each register in the allocation order, we try to grow the region as long as its spill weight remains higher than any interfering virtual registers in that region. This results in a *split proposal* for each register in the allocation order. The best split proposal is selected based on the following criteria, in order:
- Maximizing the use weight covered by the region.
- If the weight is equal, prefer the proposal which covers the most instructions.
- If the number of instructions is equal, prefer the proposal with the lowest interference weight that will need to be evicted.

The region selection algorithm is kept simple for performance reasons: it only tracks 2 split points to the left and right of the initial region and doesn't take control flow into account. Parts of the virtual register between the left and right split points are considered to be within the split region. Region growth always tries to grow to the next adjacent use or the next block boundary where the next block has a lower frequency (usually a loop edge).

The above splitting algorithm only applies to single virtual registers. If a virtual register group requires splitting then all group uses are isolated into separate, single-instruction virtual registers and removed from the original virtual register. This effectively "disbands" a group into individual virtual registers.

It is possible that no suitable region can be found, for example if the virtual register has a spill weight of 0 because it doesn't contain any uses. In such cases the virtual register is spilled.

### Spilling

The last resort for a virtual register is spilling, which is a special form of splitting: all non-spillable uses are isolated into single-instruction virtual registers and the remaining gaps between uses are turned into `ValueSegment` and collected by the spill slot allocator. The single-instruction virtual registers are then re-queued for allocation.

Gaps which hold a rematerializable value (remember, virtual registers can have `ValueSegment` from different values) are discarded instead of being passed to the spill slot allocator. Move generation will automatically re-materialize the value from thin air as needed, so there is no need to keep track of those segments any more.

## Spill slot allocation

When a `ValueSegment` is spilled, the `ValueSet` that contains its value is marked as requiring a spill slot on the stack. After the allocation loop is complete and all virtual registers have been assigned to a physical register, the split slot allocator will assign a spill slot to each `ValueSet` which requires one.

Spill slot allocation works using a simplified version of the linear scan algorithm from <https://doi.org/10.1145/330249.330250> with each `ValueSet` treated as a single segment covering the union of all its constituent live ranges. This is an over-approximation of the live range since it both ignores live range gaps and reserves the spill slot even when the value is available in a register, but the move optimization stage can take advantage of this because it guarantees that, if a value is spilled, that spill slot is guaranteed to contain that value for the entire remaining lifetime of that value.

The algorithm works with 2 data structures:
- the *active set* which contains a list of `ValueSet` that are currently live.
- the *available slots* which contain the set of spill slots that are currently free.

Then, for each value set in order of increasing live range start point:
1. Remove from the active set all `ValueSet` whose live range ends before the start of the current one. Their assigned spill slots are returned to the available slots.
2. Remove an available slot and assign the current value set to it.
3. If there are no available slots then allocate a new spill slot for the current value set.
4. Add the current value set to the active set.

This pass runs separately for each spill slot size (values in a set must have the same register bank and therefore the same spill slot size).

## Move generation

At this point all values in the function have been assigned a location (register or spill slot) at each point in their live range. The only exception is rematerializable values which may not be present anywhere for parts of their live range. This information now needs to be processed to produce the output of the allocator: assigning locations for each instruction operand and generating move instructions where needed to connect live ranges. The former is relatively straightforward: we just go through all the uses in each `ValueSegment` and record the location of that value in the output allocation array. The latter is tricky because values can be moved across control-flow edges and between segments that have been split.

To handle move generation efficiently use the concept of *half-moves* from regalloc2. At each instruction boundary, a *source half-move* provides a `Value` from a source location and a *destination half-move* consumes a `Value` to a destination location. Source half-moves are tracked in a hash table of `Allocation` keyed by `(MovePosition, Value)` while destination half-moves are accumulated into a vector of `(Allocation, MovePosition, Value)` tuples which is then sorted by position.

The placement of moves across control-flow edges requires special treatment since a move instruction cannot be placed on the edge itself: it must be either in the predecessor block or the successor block. Source half-moves on single-predecessor/multiple-successor edges are placed before the first instruction of each successor block, effectively "providing" a value for the first instruction of the successor block. Moves on multiple-predecessor/single-successor edges are more complex: they are placed before the `Jump` terminator in predecessor blocks but after any moves that would normally occur before the `Jump` terminator. This is represented by having 2 move positions before an instruction: an "early" position for most moves and a "late" position just for `Jump` terminators for control-flow edge moves. Half-moves that would be inserted before the first instruction

Half-moves are generated by scanning the uses and block boundaries in each `ValueSegment` in lock-step:
- If the segment has the live-in flag then a destination half-move is emitted at the boundary before the segment.
- If the segment has the live-out flag then a source half-move is emitted at the boundary after the segment.
- If the segment crosses a block boundary then:
  - if the previous block has multiple successors then a source half-move is emitted before the first instruction of each successor block.
  - if the next block has multiple predecessors then a destination half-move is emitted at the late position on the `Jump` terminator of all predecessor blocks.
- Incoming block parameters are treated similarly to block live-ins except that the destination half moves emitted are keyed to the value of the corresponding outgoing block parameter in each predecessor.
- Outgoing block parameters are treated just like live-out values.
- For fixed-register uses, a pair of half-moves are emitted from the allocation of the segment to the fixed register before the instruction. Special handling is needed when this is the first instruction of a block or of the segment in which case only a destination half-move from the appropriate source is needed.
- Similarly, a pair of half-moves are emitted for fixed-register definitions except if it is on the last instruction of a block.
- Finally, half-moves are emitted to join the 2 parts of tied operands with the use value being copied to the location of the def value before the instruction.

## Move resolution

Once all half-moves have been emitted, the vector of destination half-moves is sorted and grouped by move location. At each move location we collect the corresponding source half moves to build the set of *parallel moves* that need to occur at this location. If a destination half-move doesn't have a corresponding source then it is treated as a rematerialization (this can only happen due to segments for rematerializable value being discarded). These need to be reified into sequential moves, which can be tricky if the destination of one move overlaps with the source of another move. These overlaps can also form cycles, in which case a temporary register is required to break the cycle (consider the pair of moves `r0 <- r1` and `r1 <- r0`).

We first collect the moves into 3 data structures:
- A vector of value rematerializations into a destination register.
- A vector of moves from a source register to a destination register.
- A mapping of register units to the index of the move (if any) that writes to it. Because the destinations of parallel moves cannot overlap, there can only be one such move. However a single move may write to a register that consists of multiple register units.

Some initial preprocessing is done while building these:
- Identity moves are eliminated.
- Moves are turned into rematerializations where this is profitable. Each rematerialization has a cost which can be either `CheaperThanMove` or `CheaperThanLoad`. This conversion is always done for `CheaperThanMove` rematerializations and for `CheaperThanLoad` rematerializations only if the move source is in memory.

The data structures above form a directed graph of moves to other moves that overwrite their source. If there are no cycles in the graph then a valid move ordering could be computed by performing a topological sort of the move graph: this ensures that no move source is overwritten before it is read. To handle cycles we used a variant of the [depth-first search algorithm](https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search) for topological sorting: instead of stopping when a cycle is detected, we perform a *diversion*. We select a scratch register and then emit a move from the scratch register to the original move destination instead of the original move. When the DFS unwinds back to the move that caused the cycle, *after* that move is emitted, the diversion is removed and we emit a move from the original move source to the scratch register. Because the topological sort emits moves in reverse order, this results in the following final output for a cycle between `r0 <- r1`, `r1 <- r2` and `r2 <- r0`:

```
move r3, r0 // r0 diversion is removed here
move r0, r1
move r1, r2
move r2, r3 // r0 is diverted to r3 here
```

Scratch registers are selected by keeping track of which registers are read and written as the moves are emitted in reverse order. A register is made available as a scratch register for earlier moves if it is the destination of a move or rematerialization. A register is made unavailable for earlier moves when it is the source of a move. Because rematerializations don't have sources, they are always placed after moves in the final order; this makes their destination register available as a scratch register. Additionally, scratch registers are available if they are not involved in moves and they are not allocated in the register matrix at the start of the next instruction (or the first instruction of the successor block for late-position moves on a `Jump` terminator). If no scratch register is available then a spill slot is used as the scratch location instead.

The final task that needs to be handled by the move resolver is eliminating stack-to-stack moves. Most architectures do not have instructions for direct memory-to-memory copies, so these must be split into 2 operations with a scratch register (which cannot be a spill slot this time!). The tricky case is when no scratch register is available: in this situation we will arbitrarily evict a register to a spill slot to free up a scratch register and then restore this register after the stack-to-stack move. The generated code is inefficient, but this situation occurs very rarely in practice.

Because of its complexity, the move resolver is fuzz tested separately from the main allocator to ensure we hit all corner cases.

## Move optimization

At this point the allocator output is functionally correct: all operands have been assigned allocations and moves have been inserted where needed. However in practice, many of the moves are unnecessary and can be eliminated. Throughout the allocation process, we make a simplifying assumption that each `Value` only lives at a single `Allocation` (register or spill slot) at any one time. This is necessary to make the allocation problem tractable, but can lead to inefficient code because we "forget" that a value is already present in a register or spill slot.

This is particularly severe when a value is repeatedly spilled and reloaded from the stack: quite often, the stack slot will already have a copy of the value so we can skip storing the value in that case. Similarly, reloading a value from memory can be unnecessary if that value is already present in a different register.

To address this, we run a general move optimization pass. The pass aims to make the following optimizations:
- Eliminate moves if the destination of the move already holds the expected value.
- Change `Use` operands that read from stack locations to read from a register if the required value is available in one.

To be able to do this, we need to know which registers contain which values at each instruction boundary. We get this information in 2 steps:
- First we need to determine what values are available in registers and spill slots at the start of each block. This is obtained by simulating execution of each block and propagating the state to successor blocks, until a fixed point is reached.
- Then, once the set of registers on block entry is definitely known, we can go through each block and optimize moves and uses.

However since this is rather expensive, by default we only track values across forward edges in the CFG. This allows move optimization to be done in a single pass. Any blocks that have incoming back edges are simply assumed to not have any live values on entry.

Even then, we can still optimize redundant spills across loop heads by taking advantage of the live range over-approximation from the spill slot allocator: because the spill slot is reserved for the full live range of a value, if a spill of a value is dominated by another spill of the same value then the spill slot *must* already hold the value since there is no chance for another value to be spilled to that slot while the value is live.

# Appendix: Comparison with regalloc2

Regalloc3 is heavily inspired by regalloc2 and retains many of its innovations. This section gives an overview of the difference for readers already familiar with regalloc2.

## Concepts

Regalloc3 and regalloc2 have similar concepts but give them different names. This table shows an approximate mapping between the two.

| regalloc2      | regalloc3        |
|----------------|------------------|
| `RegClass`     | `RegBank`        |
| `VReg`         | `Value`          |
| `LiveBundle`   | `VirtReg`        |
| `SpillSet`     | `ValueSet`       |
| `PReg`         | `PhysReg`        |
| `LiveRange`    | `ValueSegment`   |
| `ProgPoint`    | `LiveRangePoint` |
| `InstPosition` | `LiveRangeSlot`  |

## API

- Explicit block frequencies instead of inferring them from loops.
- Dominator information is provided by the client instead of computing it itself.
- Support for register groups, multi-unit registers, explicit stack slots and more flexible register classes.
- Rematerialization support.
- Stricter requirements on the input function, which simplifies the allocator implementation:
  - Block parameters are only allowed on edges with multiple predecessor blocks.
  - Return terminators cannot have def operands or clobbers.
  - Blocks must be topologically ordered wrt dominance (if A dominates B then A must come before B in the block order).
  - The entry block (which must be block 0 due to the previous property) cannot have predecessors.
- Exhaustive validators for `Function` and `RegInfo` implementations. The rest of the allocator can then assume all invariants are maintained instead of doing redundant work.
- Text format for both `Function` and `RegInfo`. This can be dumped to a log to precisely reproduce the register allocator behavior with regalloc3-tool.

## Live ranges

- `LiveRangePoint` actually represents a *point* where a live range starts or ends. This is clearer than regalloc2's `ProgPoint` which actually represents a range.
- Fixed-register defs and uses are not considered to be part of the live range of a value. These are instead represented as a separate fixed, unevictable reservation in that register which is move-connected to the rest of the value's live range. This avoids many issues that regalloc2 has around fixed-register constraints.
- We properly handle the case when the same value is used both as a normal operand and as a reused operand. The live range of the use is properly shrunk to end at the preceding instruction boundary for both operands so that both can share the same register as the reuse def operand.
- Coalescing ("bundle merging" in regalloc2) is prioritized by block frequency which helps eliminate more expensive moves first.

## Allocation loop

- No randomization of the probing order: this turned out to not help compilation speed while reducing the quality of the generated code.
- The initial probing when a virtual register is dequeued exits early if any conflict is found. It doesn't attempt to collect information about interference for the later stages. This ends up being faster since most virtual registers are immediately allocated in this stage.
- The allocation order produces weighted preferences for particular registers based on fixed-register hints. This is additionally used to force evictions above our spill weight if a virtual register has a preference for a particular physical register.
- Splitting virtual registers is deferred until all evictions are done. This allows splitting to better work around the interference patterns of already allocated registers.

## Spill slot allocation

- Linear scan allocation is used to assign spill slots. This is less precise since it treats the entire `ValueSet` as a single linear live range segment instead of individual segments with gaps. However this makes allocation much faster.
- Unlike regalloc2, we don't have a limit on the number of spillslots to attempt to assign before allocating a new one.

## Move generation & resolution

- Only 2 move priorities, with the second one only used for `Jump` terminators. This allows better move optimization since more parallel moves are resolved together.
- The parallel move resolution algorithm is more complex because it needs to deal with register units, which can result in partial overlaps between results. In turn this can result in a move graph that has multiple cycles and therefore requires multiple scratch registers to untangle.

## Move optimization

- Support for global move optimization instead of just block-local move optimization.
- Support for dominance-based spill elimination.
- We run this as a separate pass instead of doing it in the middle of move generation. In theory it could be done that way for just `MoveOptimizationLevel::Forward`, but having a separate pass allows for more optimization options.
