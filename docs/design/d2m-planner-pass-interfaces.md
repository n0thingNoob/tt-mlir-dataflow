# D2M planner-to-pass interfaces: first slice

The planner chooses a proposal. Existing passes implement it. The allocator
reports whether that implementation fits. These are different responsibilities:
adding an interface does **not** add a search algorithm or prove a speedup.

This change builds on the [planning framework](d2m-dataflow-planning.md).
It implements the **blocking and allocation** part first, in C++/MLIR.
The earlier planner still produces its temporal fallback; it does not yet
generate these requests or use this report to rank candidates.

## Which question is handled where?

| Question | Existing implementation | Interface status in this change |
| --- | --- | --- |
| Which computations share a kernel? | `GenericFusion` | Unchanged. Explicit fusion partitions remain future work. |
| Which cores and operand distributions? | `GridSelection`, `GridAnalysis` | Unchanged. A future mapping interface must validate all operand layouts, not merely override the grid. |
| How large is each block? | `ReblockGenerics`, `BlockFactorAnalysis` | **Implemented:** a per-generic request and a non-mutating C++ validator. |
| Where do buffers fit? | `Allocate` | **Implemented:** optional structured allocation feedback. Placement is still selected by the existing allocator. |
| How many backing buffers? | `MarkSynchronizedBuffers`, allocation and masking passes | Reuse the existing global `num-stream-buffers` option. Per-CB depth is not added. |
| In what order does a kernel compute? | `OpScheduler` | Unchanged. An explicit order must respect dependencies and downstream DST assumptions. |
| Who moves data and when do they wait? | `ScheduleDMA`, `DMAOptimizations` | Unchanged. Thread/NoC assignments and barrier-policy contracts remain future work. |

In particular, this is not a promise that all five optimization decisions are
now externally controllable. It is the first executable interface pair for a
future joint planner.

## Request: explicit blocking

At the input to `d2m-reblock-generics`, attach an attribute to each selected
`d2m.generic`:

```mlir
d2m.planned_block_factors = array<i64: 1, 1, 4>
```

Enable `use-explicit-block-factors=true` on that pass or the D2M pipeline.
Factors are **absolute loop block counts**, not tile sizes or core counts.
For the test matmul with a 16-tile K shard, K factor 4 means four blocks of
four K tiles. The execution grid is unchanged.

- An annotated generic uses the requested factors, overriding the buffer-size
  policy. An identity request freezes the current blocking.
- An unannotated generic uses the original policy, including its heuristics.
- With the option disabled, the attribute has no effect. Default behavior and
  output are unchanged for existing inputs without planning attributes.
- All requests in a function are validated before any generic is rebuilt.
  Invalid or unsupported requests fail the pass; there is no silent fallback.
- The request survives rebuilding, so rerunning the enabled reblock pass does
  not overwrite the chosen factors. Remove or replace it when changing plans.

`BlockFactorAnalysis::validateExplicitFactors(generic, factors, depth, reason)`
is available to C++ callers without running a pass or rewriting IR. A failed
validation returns an explanation; it does not emit an operation diagnostic.
The pass uses `GenericOp::withParallelization` to rebuild operand views, region
buffer types, blocking loops and the return view together.

The initial supported domain is deliberately conservative: verified static,
tiled, single-output affine-blocked compute generics with projected-permutation
indexing maps. Factors must be positive multiples of the current factors,
divide the remaining shard evenly, and pass the existing candidate evaluator.
That evaluator requires affected buffers to retain at least four tiles. This
last restriction is the supported search domain, **not** a universal hardware
legality rule. Coarsening, scalar/tile conversions and arbitrary mappings are
not supported by this interface.

Requests belong at the reblocking stage, after grid selection, fusion and
outer-loop generation. Attaching one to an early TTIR operation is not an
implemented transport mechanism across those transformations.

## Feedback: allocation report

Enable `emit-resource-report=true` on `d2m-allocate` or the D2M pipeline.
Each processed, non-external function receives `d2m.allocation_report`:

```mlir
{version = 1 : i64, status = "success",
 l1_capacity_bytes = 131072 : i64, l1_usage_bytes = 65536 : i64,
 dram_capacity_bytes = 1073740800 : i64, dram_usage_bytes = 0 : i64,
 l1_to_dram_count = 0 : i64,
 placements = [{id = 0 : i64, has_request = true,
                original_memory_space = "l1", memory_space = "l1",
                size_bytes = 65536 : i64, offset_bytes = 0 : i64}]}
```

- `status` is `success` only after all allocation rewrites finish. Other values
  are `failed`, `l1_capacity_exceeded`, or `dram_capacity_exceeded`.
- Capacities exclude reserved address ranges. Usage is the allocation solver's
  address-space high-water mark, including alignment/fragmentation, in the
  allocator's per-placement units. It is not total bytes across every core or
  DRAM bank, nor a measured live-byte peak.
- Unknown usage fields are absent, not zero. DRAM usage is zero if its solved
  allocation problem is empty. Placement entries are available only after both
  memory-space allocation stages succeed.
- `l1_to_dram_count` counts variables moved from L1, not variables already bound
  to DRAM. Neither it nor the placement sizes measure data-transfer traffic.
- Placement IDs are local to one report and follow the allocator's memref
  enumeration. They are not stable candidate/kernel/CB identities across IR
  rewrites. Offsets are relative to that space's usable base address. Externally
  managed values with no allocation request omit size and offset and have
  `has_request = false`.
- Enabling reporting clears old reports throughout the module before work
  starts, so a function not reached after a failure cannot retain stale success.
  With reporting disabled, the pass does not add or update reports.

Allocation is a mutating pass, not a dry-run analysis. A future search driver
must clone the relevant IR for each shortlisted candidate, run the required
stages, inspect both pass success and the report, and discard failed clones.
On failure, the report can be inspected programmatically or with
`--mlir-print-ir-after-failure`; the CLI does not produce a normal successful
output file. Successful allocation alone does not guarantee downstream
lowering, numerical correctness, or performance.

## Compiler-side use

The following is a sketch for a driver already at the reblocking stage, not
new behavior automatically performed by the existing planner:

```cpp
std::string reason;
if (failed(BlockFactorAnalysis::validateExplicitFactors(
        generic, factors, depth, reason))) {
  // Reject this candidate.
  return failure();
}
generic->setAttr(BlockFactorAnalysis::explicitFactorsAttrName,
                 builder.getDenseI64ArrayAttr(factors));
D2MReblockGenericsOptions reblock;
reblock.useExplicitBlockFactors = true;
reblock.numStreamBuffers = depth;
pm.addPass(createD2MReblockGenerics(reblock));
// Retain the normal pipeline's intervening stages and use the same global
// depth for synchronized-buffer marking, masking and allocation.
D2MAllocateOptions allocate;
allocate.numStreamBuffers = depth;
allocate.emitResourceReport = true;
pm.addPass(createD2MAllocate(allocate));
```

Read the report using `func->getAttrOfType<DictionaryAttr>(
"d2m.allocation_report")`. Read it before function outlining or subsequent
transformations invalidate the allocation-stage snapshot.

## Reproduce

```sh
ttmlir-opt test/ttmlir/Dialect/D2M/allocate/reblock_explicit_plan.mlir \
  --ttcore-register-device \
  --d2m-reblock-generics='use-explicit-block-factors=true' \
  --d2m-allocate='emit-resource-report=true test-assume-l1-capacity=8388608'
```

The fixture requests `[1, 1, 4]`, whereas the default policy selects
`[1, 1, 16]`. Lit checks the actual rewritten buffer shapes, identity requests,
repeat execution, malformed factors, and execution through allocation.
`allocation_report.mlir` checks L1/DRAM allocation, empty functions, stale
report replacement, opt-out behavior and out-of-memory feedback. All are
compiler-only tests with synthetic resource settings, not hardware timings.
