# D2M Spatial Channels: Data Edges Between `d2m.spatial` Regions

Status: design for PR1 (IR representation). PR2a/PR2b (synchronization and
lowering) are planned below.

## 1. Problem and scope

### 1.1 What `d2m.spatial` does today

`d2m.spatial` (`D2M_SpatialOp` in `include/ttmlir/Dialect/D2M/IR/D2MOps.td`)
runs several `d2m.generic` ops at the same time on disjoint core ranges of one
device:

- One region per entry of `grid_ranges`. Each region holds exactly one
  `d2m.generic`, and that generic may contain several ops.
- The op is `NoTerminator` and destination-passing style (`ins` / `outs`), with
  tensor results. The regions have no block arguments; the generics refer
  directly to values defined outside the spatial.
- `d2m.spatial_yield` (`D2MGenericRegionOps.td`) may appear at the end of a
  region. It yields the region generic's results. It is not a terminator.
  `SpatialOpNormalizeUtil` erases and regenerates it, and the TTMetal / TTNN
  lowerings skip it.
- Spatial results are the region generics' results, concatenated in region
  order. `rebuildSpatialOpInsOutsAndResultTypes` rebuilds `ins` / `outs` from
  the generics' operands. GridSelection calls this normalization after
  rewriting each generic that sits inside a spatial.
- Bufferization (`SpatialOp::bufferize`) moves the region bodies into a new
  spatial whose operands are buffers and which has no results.
- At the TTMetal level, all regions are merged into one `enqueue_program`
  (`SpatialOpRewriter`). At the TTNN level they are merged into one
  `ttnn.generic` (`convertSingleSpatial`).

Regions cannot exchange data. All regions start together and run
independently. A value that one region computes can be used by another region
only after the whole spatial ends, through DRAM or L1 outside the op.

### 1.2 What this adds

This adds an optional **edge** between regions: the producer region's generic
writes a tensor, and the consumer region's generic reads it. Both happen inside
the same spatial, and the data never leaves L1. Edges are SSA values: the
consumer region gets a block argument, and an `edges` attribute on the spatial
says which producer result feeds it.

### 1.3 Scope of the first version

| In scope | Out of scope (later) |
|---|---|
| Whole-tensor handoff: the consumer reads only after the producer has written the entire tensor | Tile-level pipelining (credits, depth, ring buffers) |
| One-to-one edges: one producer result feeds at most one consumer argument | Fan-out / multicast (1 → N) |
| Edges written by hand or by a frontend | Automatic mapping (deciding which ops go in which region) |
| Representation, verification, normalization and bufferization (PR1) | Semaphores and lowering (PR2a), multi-core layout matching (PR2b) |

A spatial without edges keeps exactly today's syntax, printed form and meaning.

## 2. IR form

### 2.1 Elements

- **Block arguments.** A consumer region declares its inputs as entry-block
  arguments: `^region1(%x: tensor<...>):`. The existing declarative assembly
  format (`$regions`) already parses and prints entry-block arguments, so the
  parser and printer do not change. A region with no arguments prints exactly
  as it does today, because MLIR omits the entry-block header when there are no
  arguments.
- **`edges` attribute.** This is an optional `ArrayAttr` of a new attribute
  `#d2m.spatial_edge<P:I -> C:J>`, meaning: result `I` of region `P`'s generic
  feeds block argument `J` of region `C`. All four fields are non-negative
  integers. The attribute sits in the op's attr-dict, so it is absent from the
  printed IR when there are no edges.
  - The source `P:I` is the same value as operand `I` of region `P`'s
    `d2m.spatial_yield` (when present). It is also the generic's DPS init `I`.
    After bufferization the generic has no results, so the verifier interprets
    `I` as an index into the generic's outputs (DPS inits). Before
    bufferization this is equivalent to result `I`.
- **Edge buffer.** The producer generic writes the edge tensor into its `I`-th
  output. That output must be one of the spatial's `outs`. It is created
  outside the spatial with `d2m.empty`, has an L1 layout, and is placed on the
  producer's cores: at a non-origin range it carries the same
  `virtualGridForwardMapping` / `virtualGridInverseMapping` as any other output
  of that region. The matching spatial result may stay unused.

### 2.2 Two-region chain

Region 0 computes `exp(in)` into `%e`, and region 1 reads `%e` and computes
`tanh` into `%o`.

```mlir
!T = tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1_layout>

%e = d2m.empty() : !T   // edge buffer: L1, on region 0's cores
%o = d2m.empty() : !T   // ordinary output
%r:2 = d2m.spatial {
    edges = [#d2m.spatial_edge<0:0 -> 1:0>],
    grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>,
                   #ttcore.core_range<(1, 1), (1, 1)>]}
    ins(%in : !T) outs(%e, %o : !T, !T) {
  ^region0:
    %g0 = d2m.generic {grid = #ttcore.grid<1x1>, ...}
        ins(%in : !T) outs(%e : !T) { ... exp ... } : !T
    d2m.spatial_yield %g0 : (!T)
}, {
  ^region1(%x: !T):                       // fed by edge 0:0 -> 1:0
    %g1 = d2m.generic {grid = #ttcore.grid<1x1, ...offset...>, ...}
        ins(%x : !T) outs(%o : !T) { ... tanh ... } : !T
    d2m.spatial_yield %g1 : (!T)
} : !T, !T
// %r#0 is the edge tensor; it may stay unused. %r#1 is the final output.
```

### 2.3 Three-region diamond

Region A writes two tensors. One goes to B. The other goes directly to C, which
also reads B's result. The edges stay one-to-one because A has two outputs.

```mlir
%a0 = d2m.empty() : !T   // A -> B
%a1 = d2m.empty() : !T   // A -> C
%b0 = d2m.empty() : !T   // B -> C
%c0 = d2m.empty() : !T   // final output
%r:4 = d2m.spatial {
    edges = [#d2m.spatial_edge<0:0 -> 1:0>,
             #d2m.spatial_edge<0:1 -> 2:0>,
             #d2m.spatial_edge<1:0 -> 2:1>],
    grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>,
                   #ttcore.core_range<(0, 1), (0, 1)>,
                   #ttcore.core_range<(0, 2), (0, 2)>]}
    ins(%in : !T) outs(%a0, %a1, %b0, %c0 : !T, !T, !T, !T) {
  ^A:
    %ga:2 = d2m.generic {...} ins(%in : !T) outs(%a0, %a1 : !T, !T) {...} : !T, !T
    d2m.spatial_yield %ga#0, %ga#1 : (!T, !T)
}, {
  ^B(%x: !T):
    %gb = d2m.generic {...} ins(%x : !T) outs(%b0 : !T) {...} : !T
    d2m.spatial_yield %gb : (!T)
}, {
  ^C(%y: !T, %z: !T):                     // y <- A:1, z <- B:0
    %gc = d2m.generic {...} ins(%y, %z : !T, !T) outs(%c0 : !T) {...} : !T
    d2m.spatial_yield %gc : (!T)
} : !T, !T, !T, !T
```

The order of `edges` does not matter. The order of `outs` / results is still
"each region's generic outputs, in region order". Because edges refer to
`(region, output index)` and not to a flat `outs` index, rebuilding `ins` /
`outs` never renumbers an edge.

### 2.4 Without edges

There is no `edges` attribute and no block arguments. Parsing, printing,
verification, normalization, bufferization and lowering all take the existing
code paths unchanged. A dedicated regression test checks the printed form
byte for byte against the output of the binary built before this change.

### 2.5 MLIR semantics this relies on

The source references below are to the LLVM checkout tt-mlir builds against
(`env/build/llvm-project-prefix/src/llvm-project/mlir`).

- **Sibling regions cannot see each other's values; outer values are visible.**
  `DominanceInfoBase::properlyDominatesImpl` (`lib/IR/Dominance.cpp:253-300`)
  walks `b` up into `a`'s region. If the two values are in sibling regions,
  `findAncestorIteratorInRegion` fails and the result is "does not dominate".
  Values from an enclosing region dominate (the `enclosingOk` case) because
  `SpatialOp` is not `IsolatedFromAbove`. This is why a value needs an explicit
  edge.
  - Observed on `main`: with a block argument and no other change, the
    GridSelection normalization (`rebuildSpatialOpInsOutsAndResultTypes`) puts
    the block argument into `ins` and produces "operand #1 does not dominate
    this use".
- **Region kind.** `SpatialOp` does not implement `RegionKindInterface`, so
  `mayHaveSSADominance` returns true (`lib/IR/RegionKindInterface.cpp:20-24`)
  and the regions are ordinary SSACFG regions. An entry block with arguments is
  legal there, which is the same thing `scf.for` and `linalg.generic` rely on.
- **Bufferization order and block arguments.**
  - One-Shot Bufferize gathers ops with a `PostOrder` walk
    (`lib/Dialect/Bufferization/Transforms/Bufferize.cpp:298`), so a region's
    generic is bufferized before its spatial. This matches the existing comment
    in `SpatialOp::bufferize`.
  - When the consumer generic calls `getBuffer(%x)`, the owner of the block
    argument is the spatial (`getOwnerOfValue`,
    `lib/Dialect/Bufferization/IR/BufferizableOpInterface.cpp:152-156`). The
    block argument's buffer type is therefore answered by
    `SpatialOp::getBufferType`, and a `to_buffer(%x)` is inserted at the start
    of the consumer block.
  - Aliasing goes through `SpatialOp::getAliasingValues`, which stays empty, so
    the analysis treats `%x` as a value with no aliases (see §3.2).
- **Python bindings.** ODS-generated Python builders make optional attributes
  keyword-only (`tools/mlir-tblgen/OpPythonBindingGen.cpp:989-1028`). The
  existing positional calls `d2m.SpatialOp(results, ins, outs, grid_ranges,
  num_regions)` in `tools/d2m-jit/_src/builder.py` and
  `tools/builder/d2m/d2m_builder.py` keep working.

## 3. Relation to existing code

| Code | Change | Why |
|---|---|---|
| `D2MOpsAttrs.td` | New `D2M_SpatialEdgeAttr` (`#d2m.spatial_edge<P:I -> C:J>`) and its array type | Readable, typed edge syntax |
| `D2M_SpatialOp` (`D2MOps.td`) | Add `OptionalAttr<...>:$edges`. Add a builder with the old signature `(results, inputs, outputs, grid_ranges, numRegions)`. Add a helper to look up an edge by consumer argument. Update the description. | Adding an optional attribute changes the generated C++ `build` signature. The old-signature builder keeps current callers source-compatible. |
| Parser / printer | None. `$regions` already round-trips entry-block arguments, and `edges` sits in the attr-dict. | Checked on `main`: a block-argument region parses and prints as `^bb0(%arg1: ...)`. |
| `SpatialOp::verify` | Existing checks unchanged. New checks run only when `edges` is present or a region has block arguments (§5). | Backward compatible |
| `SpatialOp::bufferize` | Copy `edges` to the new op. Rewire consumer block arguments to the producer's buffer, then erase the block arguments (§3.2). | §1.7 of the PR1 decision |
| `SpatialOp::getBufferType` | For a block argument, return the buffer type of the producer's edge value. Otherwise unchanged. | Both sides bufferize to the same memref type |
| `SpatialOp::isWritable` | Code unchanged (block arguments already return false); comment updated | The consumer only reads an edge |
| `SpatialOp::getAliasingValues` | Unchanged (empty) | §3.2 |
| `SpatialYieldOp::verify` | If the parent spatial has edges, require the yield operands to be exactly the region generic's results, in order | Edge index `I` and yield operand `I` must name the same value |
| `SpatialOpNormalizeUtil` | (1) Leave the spatial's own block arguments out of the rebuilt `ins`. (2) Prefix ops that use a block argument cannot exist (the verifier requires block arguments to be used only inside the region's generic), so hoisting stays valid. (3) `outs` and the result order are unchanged. Edges use `(region, output index)`, so they need no remapping. | Fixes the dominance error above. Keeps the edge numbering stable. |
| GridSelection | No code change; it reaches the code above through `normalizeSpatialOpContainingGeneric`. A consumer input that is a block argument is classified as `Kind::Skip`, so it is not rewritten. If grid selection changes the type of an edge's producer output, the verifier reports the type mismatch after the pass. Making both ends agree is PR2b. | Fails explicitly, never silently |
| Allocate, `d2m-lower-load-store-ops-to-dma`, D2MToTTMetal, D2MToTTNN | At the start of `runOnOperation`, call the new static `SpatialOp::checkNoEdges(root, passName)`, which emits "'d2m.spatial' op edges are not yet supported by <pass>" on every spatial with edges, and fail | Nothing downstream generates code for an edge before PR2. The helper lives on the op so the conversion libraries need no new link dependency. Every pass between the frontend and Allocate carries `edges` through unchanged; checked with the full pipeline in both TTNN and TTMetal modes. |
| d2m-jit builder (`tools/d2m-jit/_src/builder.py`) | None in PR1 | Python call sites stay compatible (§2.5). Creating edges from Python can come once the lowering exists. |

### 3.1 Builders

ODS will generate `build(..., ArrayAttr grid_ranges, ArrayAttr edges,
unsigned regionsCount)`. The only C++ caller is `SpatialOp::bufferize`, and it
passes `getEdgesAttr()`. An extra builder with the old signature (`edges =
nullptr`) is added so that downstream code and forks still compile.

### 3.2 Bufferization rule

`SpatialOp::bufferize` runs after every nested generic has been bufferized.
For each edge `P:I -> C:J`:

1. Let `buf` be producer generic `P`'s DPS init `I`. By now this is a memref
   (normally `to_buffer(%e)` placed right after `%e`, outside the spatial). If
   it is still a tensor, `getBuffer` is called on it. If `buf` does not
   dominate the spatial (for example, the analysis put an allocation inside
   the producer region), bufferization fails with "edge buffer must be
   bufferized in place".
2. Replace every use of consumer block argument `C:J`:
   - A `bufferization.to_buffer` of it with matching type is replaced by
     `buf` and erased.
   - Any other use goes through one `bufferization.to_tensor buf`.
3. Erase the consumer block's arguments, then build the new spatial (with
   `edges` copied) and move the region bodies as today.

After bufferization the consumer region refers to the producer's memref
directly, which is the same IR shape as the earlier shared-L1 branches. The
difference is that it is produced mechanically, and the `edges` attribute keeps
the dependency explicit for the PR2a pass.

The aliasing interface is unchanged:

- `getAliasingValues(outs operand)` stays empty, so the analysis sees no alias
  between `%e` and `%x`. That is what we want. If the analysis saw the alias,
  it would find the producer's write to `%e` and the consumer's read of `%x` in
  regions with no dominance between them. It could then decide to bufferize
  the producer out of place and break the edge.
- The consumer never writes `%x`. The verifier forbids `%x` as a DPS init, and
  `isWritable` is false, so no copy is ever inserted for it.
- Cross-region ordering is set by the edge semantics (PR2a semaphores), not by
  SSA dominance. Nothing in the One-Shot framework is changed.

### 3.3 Allocate, for PR2a

The edge buffer is a spatial operand, so its live range already covers the
whole spatial op, and PR1 reuses nothing. Allocate still needs these two
things:

- **Pin the buffer to L1.** In `Allocate.cpp` (~1008-1015) a memref that is a
  generic output with more than one generic user counts as
  `isIntermediateMemref`. With `allow-l1-output-spilling`, such a memref may be
  spilled to DRAM. An edge buffer has exactly that shape (producer output plus
  consumer input). PR2a therefore marks edge buffers `isMemspaceBound` (L1).
- **Keep it on the producer's cores and alias the producer's output.** The
  buffer is sharded over the producer grid (through the producer's VGM when the
  range is not at the origin). Allocate must not let the consumer generic alias
  it as a local operand; the consumer reads it remotely. This interacts with the
  known `canAliasOperand` issue (§6).

### 3.4 Lowering

TTMetal (`SpatialOpRewriter`) and TTNN (`convertSingleSpatial`) merge the
regions into one program and skip `spatial_yield`. Both emit the "not yet
supported" diagnostic in PR1. In PR2a, TTMetal is the first target, and the
merge stays as is: the edge synchronization is already ordinary
semaphore / DMA ops inside the generics by then.

## 4. Whole-tensor handoff protocol (PR2a; semantics only here)

For an edge `P:I -> C:J`, let the producer generic use `Np` cores and the
consumer generic use `Nc` cores.

- **Semaphore.** One local (program) semaphore per edge, with initial value 0,
  allocated on the consumer cores.
  - It is a `d2m.create_local_semaphore` created before the spatial and passed
    as an `additionalArg` to both generics, following
    `D2MPreallocateMcastSemaphores`.
  - Local rather than global: tt-metal writes program semaphores' initial
    values as part of each program launch. Every enqueue, including trace
    replay, therefore starts at 0 without an explicit reset. A
    `create_global_semaphore` would persist across launches and need a
    `reset_global_semaphore` each time. To verify on HW in PR2a: the replay
    reset.
  - **Open issue for PR2a.** The TTMetal runtime creates local semaphores per
    kernel config with that kernel's core range set
    (`runtime/lib/ttmetal/executor.cpp:381-382`). If the producer and consumer
    kernels each create the "same" semaphore, they can get different ids.
    PR2a must make one id valid on both ranges, for example by creating the
    semaphore over producer ∪ consumer cores once and passing the id to both
    kernels.
- **Producer.** After each producer core has finished its last write of the
  edge tensor, it increments the semaphore on every consumer core by 1, either
  `Nc` unicast `semaphore_inc` ops or one multicast `semaphore_inc` over the
  consumer rectangle.
  - "Finished" means the last write to this core's shard of the edge buffer is
    complete. For compute output, that is after the output CB has been
    released to the DM thread. For a `remote_store` / DMA write, it is after
    `dma_wait` or a write barrier.
  - Only then is `semaphore_inc` issued. Known `main` bug: today the inc is
    emitted before the wait (§6).
- **Consumer.** Each consumer core runs `semaphore_wait sem, Np` before its
  first read of the edge tensor. It then reads with the existing
  `remote_load` from the producer cores' L1, using the edge memref's
  layout / VGM.
- **Order independence.** Waiting for all `Np` increments means the consumer
  never needs to know the producer's iteration order or which shard it needs.
  The protocol therefore holds for any operator.
- **Regions without edges** start and run independently, as today.

## 5. Verifier rules and tests

These checks are new. All existing checks (CoreRange count = region count,
non-empty regions, one generic per region, 2D grid, grid fits in range) are
unchanged.

The new checks run only when `edges` is present or some region has block
arguments.

| # | Rule | Phase | Negative test (`spatial_channel_negative.mlir`) |
|---|---|---|---|
| R1 | Edges form a DAG over regions (no cycle) | always | 2-region cycle 0→1, 1→0 |
| R2 | No self edge (`P != C`) | always | `0:0 -> 0:0` |
| R3a | `P`, `C` < number of regions | always | `5:0 -> 1:0` |
| R3b | `I` < number of producer generic outputs | always | `0:3 -> 1:0` |
| R3c | `J` < number of consumer block arguments | tensor form | `0:0 -> 1:2` |
| R4 | Every consumer block argument has exactly one incoming edge (no missing, no duplicate) | tensor form for "missing"; always for "duplicate" | arg without edge; two edges into `1:0` |
| R4b | One-to-one: a producer output feeds at most one edge (v1) | always | `0:0 -> 1:0` and `0:0 -> 2:0` |
| R5 | Block argument type == producer generic output `I` type (shape, element type, layout) | tensor form | layout / shape mismatch |
| R6 | Producer and consumer generic grids have the same shape. Block shapes then match because the types are equal (R5). | always (see note) | 1x1 producer, 2x2 consumer |
| R7 | Producer output `I` is one of the spatial's `outs` (the edge buffer is a spatial `outs`) | always | producer writes a value not in `outs` |
| R8 | A block argument is a ranked tensor, used only inside its region's generic and never as one of its DPS inits | tensor form | use by an op outside the generic; block arg as `outs` |
| R9 | `spatial_yield` (when present) yields exactly the generic's results in order, if the spatial has edges (in `SpatialYieldOp::verify`) | tensor form | yield in different order |
| R10 | No region block arguments once the spatial is in buffer form (some `outs` is a memref) | buffer form | memref spatial with a block argument |

"Tensor form" means every `outs` is a tensor; otherwise the spatial is in buffer form. After bufferization the block
arguments are gone, so R3c, R4-missing, R5, R8 and R9 no longer apply. R1, R2,
R3a, R3b, R4-duplicate, R4b, R6 and R7 still hold.

Note on R6: the generic `grid` attribute exists at every stage. Hand-written
IR, which is the only source of edges in PR1, has its grids fixed. When
GridSelection changes a grid it also changes the operand types, so a
selection that breaks R6 also breaks R5 and fails in the verifier right after
`d2m-grid-selection`. PR2b makes GridSelection choose matching grids for both
ends.

Negative tests use `--verify-diagnostics`, one `// -----` case per rule (15 cases).

Positive tests (`spatial_channel.mlir`):

- Round-trip of the 2-region chain and the 3-region diamond (parse, verify,
  print, re-parse).
- `--d2m-grid-selection` keeps `edges` and the block arguments and does not
  put block arguments into `ins`.
- One-Shot Bufferize: consumer generics read the producer's memref, the
  spatial has no block arguments, and `edges` survives.

Regression tests:

- Existing spatial lit tests unchanged.
- `spatial_no_edges_print.mlir`: the printed IR of a no-edge spatial (2 regions,
  including `spatial_yield`) is checked line by line with `CHECK-NEXT` against
  the output of the binary built from `main` before the change.

Downstream diagnostics: `spatial_channel_unsupported.mlir` runs the chain
through the D2M pipeline up to Allocate and expects the Allocate error.
`--d2m-lower-load-store-ops-to-dma` and the TTMetal / TTNN conversions on
memref-form input expect their "not yet supported" errors.

## 6. PR2 plan

**PR2a (one core per region, TTMetal):**

- Allocate: pin edge buffers to L1 on the producer cores (§3.3).
- New pass `d2m-lower-spatial-edges`, after Allocate and run directly before
  the lowerings, with no pass in between that drops attributes or ops. Per
  edge it:
  1. allocates a local semaphore (§4) and checks the per-core budget (≤ 16
     per program, shared with multicast semaphores);
  2. inserts `semaphore_inc` on the producer after its final write and
     barrier;
  3. inserts `semaphore_wait` on the consumer before its first read of the
     edge buffer;
  4. removes the `edges` attribute once lowered, so the downstream "not yet
     supported" checks no longer fire.
- Validation: hand-written MLIR on n150, chain and diamond, PCC ≥ 0.999 vs
  torch.

**PR2b (multi-core regions):** GridSelection chooses consistent grids and block
shapes for both ends of each edge, and propagates the producer's output type to
the consumer block argument (R5 / R6 after selection).

**Known upstream bugs.** Each one gets a small commit with a lit test in PR2.
All four were checked on `main` (`f688ee694`) and are still present:

1. Offset placement.
   - Allocate `canAliasOperand` (`Allocate.cpp:1442-1452`) aliases an L1
     output without comparing the generic's grid VGM with the operand's VGM.
   - GridSelection `optimizeToLayoutGrid` (`GridSelection.cpp:176-201`)
     returns early for an already-selected or 1x1 grid, so an output produced
     by `to_layout` in an off-origin region never gets the offset VGM.
   - Reference: `feature/d2m-baseline-spatial-pipeline` (`Allocate.cpp`
     ~1447-1480 and the `placeOutput` flag in `GridSelection.cpp`).
2. DRAM bank map with spatial offset. `getMemoryMapImpl` (`Utils.cpp:696-697`,
   738-748) composes the DRAM map with the offset-carrying forward map, so a
   DRAM operand in an off-origin region selects the wrong bank. Reference:
   `feature/d2m-batch2` `f8f250c3e` (`stripDramGridOriginOffset`).
3. `semaphore_inc` before DMA wait.
   - `LowerLoadStoreOpsToDMA.cpp:247-257` emits `SemaphoreIncOp` before the
     `DMAWaitOp`.
   - `DMAOptimizations.cpp:117` lets a write barrier sink past a
     `semaphore_inc`.
   - Reference: `feature/astraia-mha-spatial-v1` `94791b03a`.
4. InsertComputeCB drops the semaphore. `eraseAliasedMarkers`
   (`InsertComputeCB.cpp:600-616`) erases an aliased `remote_store` without
   checking `getSemaphore()`, so the increment is silently lost. No reference
   fix exists. Possible fixes: do not alias a store that has a semaphore, or
   replace the marker with a bare `semaphore_inc`.

## 7. Hardware constraints PR2 must respect

- 32 CBs per core. The merged program concatenates every region's CBs, which
  `SpatialOpRewriter` already checks against `num_cbs`.
- 16 local semaphores per core per program. This caps the edges into one
  consumer range, together with the semaphores that multicast loads use.
- Two RISCs on the same core must not write on the same NoC at the same time.
- At most one multicast load per DM thread.
- Multi-source linked multicast times out: issue a write barrier before
  sending the flag.

## 8. Open questions for review

1. The verifier on `main` does not check that `grid_ranges` are disjoint (the
   op description says they must be). PR1 leaves this alone. A separate small
   commit could add it.
2. Edge index `I` refers to the producer generic's DPS output `I` in both
   forms. The alternative, a flat `outs` index, would need remapping whenever
   normalization rebuilds `outs`, so it was rejected.
