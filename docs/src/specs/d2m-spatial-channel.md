# D2M Spatial Channels: Data Edges Between `d2m.spatial` Regions

Status: PR1 (IR representation) and PR2a (single-core synchronization and
TTMetal lowering) implemented; PR2b (multi-core regions) planned.

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
| D2MToTTMetal, D2MToTTNN | Fail with "'d2m.spatial' op edges are not yet supported by <pass>" (`SpatialOp::checkNoEdges`) if an edge reaches them unlowered. PR1 also checked in Allocate and `d2m-lower-load-store-ops-to-dma`; PR2a lowers edges before both conversions and drops those two checks. | Never emit unsynchronized code |
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

### 3.3 Allocate (PR2a)

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

## 4. Whole-tensor handoff protocol (implemented in PR2a)

For an edge `P:I -> C:J`, let the producer generic use `Np` cores and `Wp` DM
threads that write the edge buffer.

- **Semaphore.** One local (program) semaphore per edge, initial value 0: a
  `d2m.create_local_semaphore` created before the spatial and appended to both
  generics' `additionalArgs` (accessed in the threads through `d2m.get_arg`
  after `d2m-normalize-thread-args`).
  - Local rather than global: tt-metal writes a program's semaphore initial
    values on every launch, so every enqueue starts at 0 without an explicit
    reset. Checked on n150: 5 submits of the same program in one process all
    pass with an exact-value wait.
  - The TTMetal runtime used to create a local semaphore separately for every
    kernel that referenced it, on that kernel's core range, so two kernels could
    get different ids (`runtime/lib/ttmetal/arguments.h`, tt-metal
    `ProgramImpl::create_semaphore` takes the lowest free id per core range).
    PR2a changes the runtime to create each local semaphore once per program
    over the union of the cores of all kernels that reference it. Without this
    change a 3-region join (two semaphores on one consumer core) hangs on n150.
- **Producer.** Every producer DM thread that DMA-writes the edge buffer
  increments the semaphore of every consumer core by 1 after its last op.
  - Allocate never aliases an edge buffer, so the producer always computes into
    a stream buffer and stores it with an explicit DMA write; every such write
    has its `dma_wait` before the end of the thread (the pass checks this).
  - Unicast only (`d2m.semaphore_inc` has no multicast form), `Nc` increments
    per writing thread.
  - `semaphore_inc` core indices are virtual coordinates of the enclosing
    generic (`d2m-generic-regions-to-funcs` maps them through the generic's
    `virt_to_physical_map`). The pass therefore maps each consumer core's
    logical coordinates through the producer grid's `physical_to_virt_map`.
- **Consumer.** Every consumer DM thread that reads the edge buffer first waits
  until the semaphore equals `Np * Wp`, then reads the producer cores' L1 with
  the existing DMA (the edge memref's layout / VGM gives the producer cores).
- **Order independence.** Waiting for all increments means the consumer never
  needs to know the producer's iteration order. The protocol holds for any
  operator.
- **Regions without edges** start and run independently, as today.
- **Pass position.** `d2m-lower-spatial-edges` runs after
  `d2m-lower-dma-to-fully-indexed-form` (threads are split and every edge
  read/write is an explicit DMA op) and before `d2m-normalize-thread-args`.
  The passes after it only lower ops; none drops the inserted semaphore ops.

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

Downstream diagnostics: `spatial_channel_unsupported.mlir` runs the TTMetal /
TTNN conversions on memref-form input with unlowered edges and expects their
"not yet supported" errors. Since PR2a the TTMetal pipeline lowers edges
(`spatial_channel_pipeline.mlir`), and in TTNN mode `d2m-lower-spatial-edges`
does nothing so that D2MToTTNN still rejects them.

## 6. PR2 plan and status

**PR2a (one core per region, TTMetal): done.**

- Allocate: edge buffers are never aliased and are bound to L1 (§3.3).
- `d2m-lower-spatial-edges` (§4), with lit tests for a 1x1 chain, a 1x2 edge
  (two increments, wait count 2), an off-origin producer (virtual
  coordinates), `ttnn-mode`, and the "no DMA write" error.
- Runtime: one id per local semaphore per program (§4).
- Validation on n150, hand-written MLIR, 1x1 regions: chain (0,0)->(1,1), pair
  (1,0)->(1,1), join (0,0),(0,1)->(1,1), chain3 (0,0)->(0,1)->(1,1). Every
  output PCC >= 0.9999 vs torch, 5 submits per process, 2 seeds.
- A diamond needs a producer generic with two outputs. Upstream D2M does not
  lower multi-output compute in one generic yet (a GenericOp
  canonicalization asserts, and DST register access fails), so the diamond is
  covered at the IR level only (PR1 tests).

**PR2b (multi-core regions):** GridSelection chooses consistent grids and block
shapes for both ends of each edge, and propagates the producer's output type to
the consumer block argument (R5 / R6 after selection).

**Known upstream bugs** (all four were present on `main` `f688ee694`):

1. Offset placement.
   - Allocate `canAliasOperand` aliased an L1 operand without comparing the
     generic's grid with the operand's VGM. **Fixed in PR2a** (an off-origin
     region reading an origin operand now streams it; an edge consumer reads
     the producer's L1).
   - GridSelection `optimizeToLayoutGrid` (`GridSelection.cpp:176-201`)
     returns early for an already-selected or 1x1 grid, so an output produced
     by `to_layout` in an off-origin region never gets the offset VGM.
     Deferred to PR2b: edge buffers come from `d2m.empty`, which gets the VGM.
2. DRAM bank map with spatial offset (`getMemoryMapImpl`, `Utils.cpp:696-697`,
   738-748). Deferred to PR2b: PR2a edges and test inputs are in L1.
   Reference: `feature/d2m-batch2` `f8f250c3e`.
3. `semaphore_inc` before DMA wait (`LowerLoadStoreOpsToDMA`) and a write
   barrier sinking past `semaphore_inc` (`DMAOptimizations`). **Fixed in
   PR2a.**
4. InsertComputeCB dropped the semaphore of an aliased `remote_store`. **Fixed
   in PR2a** by never aliasing a store that increments a semaphore (Allocate).

**Found during PR2a, not fixed:** on this n150, any spatial region at logical
core (2,2) hangs in a later program on that core, also without edges and also
with the compiler from `main`; regions at (0,1), (1,0), (0,2), (2,0) and (1,1)
work. Not investigated further; the PR2a validation avoids (2,2).

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
