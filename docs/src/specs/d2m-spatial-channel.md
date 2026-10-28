# D2M Spatial Channels: Data Edges Between `d2m.spatial` Regions

Status: PR1 (IR representation), PR2a (synchronization and TTMetal lowering),
PR2b (multi-core regions: GridSelection lays out both ends of an edge alike)
and edge fan-out with multicast signalling implemented.

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
| Fan-out: one producer result may feed several consumer regions (one block argument each); the producer signals each consumer region with one multicast increment | Pushing data to the consumers (multicast writes) and resharding on an edge |
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

Region A's result goes to B and directly to C, which also reads B's result.
With fan-out (§4), A has a single output that feeds two edges:

```mlir
edges = [#d2m.spatial_edge<0:0 -> 1:0>,   // A -> B
         #d2m.spatial_edge<0:0 -> 2:0>,   // A -> C (same output)
         #d2m.spatial_edge<1:0 -> 2:1>]   // B -> C
```

This form lowers and runs on n150 (§6). The form below, where A writes two
tensors so that every output feeds one edge, is also valid IR, but D2M does
not lower a generic with two computed outputs yet, so it is covered at the IR
level only.

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
| GridSelection | PR1: no code change; it reaches the code above through `normalizeSpatialOpContainingGeneric`. PR2b: the producer decides an edge's layout and the consumer follows. `GridAnalysis` analyzes producers before consumers and pins a consumer's edge operand to the producer output's selected grid and padding. GridSelection gives that operand `Kind::SpatialEdge` and sets the block argument's type with the same `gridAdjustedType` call the producer's output goes through. If the consumer generic would need another grid for the edge, the pass fails ("resharding an edge is not supported"). | Both ends get the same type whatever order generics are rewritten in; a layout the consumer cannot read is an error, never a silent wrong read |
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

- **Semaphore.** One local (program) semaphore per producer output, initial
  value 0, shared by all edges that leave it: a `d2m.create_local_semaphore`
  created before the spatial and appended to the producer's and every
  consumer's `additionalArgs` (accessed in the threads through `d2m.get_arg`
  after `d2m-normalize-thread-args`). One per output rather than per edge keeps
  the producer cores' semaphore count independent of the fan-out.
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
  increments the semaphore of every consumer core (of every consumer region)
  by 1 after its last op.
  - Allocate never aliases an edge buffer, so the producer always computes into
    a stream buffer and stores it with an explicit DMA write; every such write
    has its `dma_wait` before the end of the thread (the pass checks this).
  - When a consumer region's cores fill their bounding box (more than one
    core), one multicast `d2m.semaphore_inc ... mcast[h, w]` per writing thread
    covers the box; otherwise one unicast increment per consumer core. The
    multicast lowers to tt-metal `noc_semaphore_inc_multicast` with
    `num_dests = h * w`: the NoC atomic has no loopback variant, so the sender
    is outside the box and every core in it is a destination (the multicast
    `semaphore_set` and DMA write lowerings use `h * w - 1` because their
    sender is inside). The pass reports a producer core inside a consumer's
    box as an error; `d2m.spatial` documents disjoint ranges but its verifier
    does not check them (§8).
  - `semaphore_inc` core indices are virtual coordinates of the enclosing
    generic (`d2m-generic-regions-to-funcs` maps them through the generic's
    `virt_to_physical_map`; a multicast keeps its shape). The pass therefore
    maps each consumer core's (or the box corner's) logical coordinates
    through the producer grid's `physical_to_virt_map`.
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
| R4b | A producer output feeds a given consumer region at most once (it may feed several regions) | always | `0:0 -> 1:0` and `0:0 -> 1:1` |
| R5 | Block argument type == producer generic output `I` type (shape, element type, layout) | tensor form | layout / shape mismatch |
| R6 | The consumer reads the edge on the grid the producer writes it on: the shard grid of the block argument's type equals that of the producer output `I`. The two generics' own grids may differ (see note). | tensor form | producer writes on 1x1, consumer reads on 2x2 |
| R7 | Producer output `I` is one of the spatial's `outs` (the edge buffer is a spatial `outs`) | always | producer writes a value not in `outs` |
| R8 | A block argument is a ranked tensor, used only inside its region's generic and never as one of its DPS inits | tensor form | use by an op outside the generic; block arg as `outs` |
| R9 | `spatial_yield` (when present) yields exactly the generic's results in order, if the spatial has edges (in `SpatialYieldOp::verify`) | tensor form | yield in different order |
| R10 | No region block arguments once the spatial is in buffer form (some `outs` is a memref) | buffer form | memref spatial with a block argument |

"Tensor form" means every `outs` is a tensor; otherwise the spatial is in buffer form. After bufferization the block
arguments are gone, so R3c, R4-missing, R5, R6, R8 and R9 no longer apply (the
consumer reads the producer's buffer itself). R1, R2, R3a, R3b, R4-duplicate,
R4b and R7 still hold.

Note on R6: until PR2b, R6 required the producer and consumer generics to
have the same grid. That is stronger than the handoff needs: the lowering
signals every consumer core from every producer core whatever the two grids
are, and the consumer's DMA reads the producer's shards wherever they are. R6
now compares the edge operand on both ends. It is implied by R5 (equal types)
and checked first only for a clearer message. A matmul consumer is the case
it admits: its generic grid follows its MxN output (2x4 in the test) while it
reads the MxK edge on the producer's grid (2x2).

Negative tests use `--verify-diagnostics`, one `// -----` case per rule (15 cases).

Positive tests (`spatial_channel.mlir`):

- Round-trip of the 2-region chain and the 3-region diamond in both forms
  (two-output A, and single-output A with fan-out), parse, verify, print,
  re-parse; the fan-out diamond bufferizes to one buffer read by B and C.
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
- With one-to-one edges a diamond needed a producer generic with two
  outputs. Upstream D2M does not lower multi-output compute in one generic yet
  (a GenericOp canonicalization asserts, and DST register access fails), so
  the diamond stayed at the IR level until fan-out (below) made a
  single-output form possible.

**PR2b (multi-core regions): done.**

- PR2a's lowering, Allocate and runtime already handle multi-core regions: a
  hand-selected 2x2 -> 2x2 chain ran correctly without changes.
- GridSelection lays out both ends of an edge alike (§3, GridSelection row).
  Lit: 2x2 chain to an offset region, 1x4 rows, a join, a consumer region
  rewritten before its producer, and the grid-mismatch error.
- Validation on n150, pre-selection input (grid 1x1, GridSelection picks
  grids and VGMs): chain 2x2 (0,0) -> 2x2 (0,3); join 2x2 (0,0) and (3,0) ->
  2x2 (0,4); chain3 of 1x4 rows (rows 0, 1, 3); the join with DRAM input and
  DRAM consumer output. Every output PCC >= 0.9999 vs torch, 5 submits per
  process, 2 seeds. PR2a's 1x1 graphs still pass with the PR2b compiler.
- A `to_layout` result can feed only one generic when GridSelection resizes
  it (upstream assert in `recordGenericConsumer`), so two regions that read
  the same host input need one `to_layout` each.
- R6 compares the edge operand, not the generic grids (§5). Lit: a matmul
  consumer on 2x4 reading a 2x2 edge. n150: `tanh(x)` on 2x2 at (0,0) ->
  `a @ w` (64x128 @ 128x128) on 2x4 at (0,4), PCC 0.99999 for both outputs,
  5 submits, 2 seeds.
- Upstream, unrelated to edges: a matmul whose operand is multicast along a
  single row or column prints an `emitc.expression` with a repeated operand,
  which the parser rejects ("region entry argument is already in use"), so
  `ttmlir-opt -o x.mlir` followed by `ttmlir-translate x.mlir` fails. A plain
  `ttir.matmul` 64x128 @ 128x128 hits it on `main` too. Writing bytecode
  (`--emit-bytecode`) avoids the textual round trip.

**Edge multicast and fan-out: done.**

- `d2m.semaphore_inc` with `mcast` lowers to `noc_semaphore_inc_multicast`
  (it asserted before); `d2m-lower-spatial-edges` signals each consumer box
  with one multicast increment (§4).
- Fan-out: R4b is now "(producer, output, consumer) unique"; the pass gives
  each producer output one semaphore. Bufferization, GridSelection and
  Allocate needed no change (lit: the fan-out diamond, two consumers
  following a 2x2 producer, a fan-out grid mismatch, the lowering of a 2x2
  multicast plus a 1x1 unicast consumer).
- Validation on n150, pre-selection input, every output PCC >= 0.9999 vs
  torch, 5 submits per process, 2 seeds:
  - fan-out: `tanh` 2x2 (0,0) -> `exp` 2x2 (0,3) and `tanh` 2x2 (3,0);
  - the single-output diamond: A (0,0) -> B (0,3), A and B -> C (3,0) `add`
    (first diamond run on silicon);
  - fan-out into an elementwise 2x2 consumer and a 2x4 matmul consumer;
  - the PR2a and PR2b graphs, now with multicast signals (`num_dests` 4 for
    2x2 and 1x4 consumers, 8 for the 2x4 matmul; writers on NoC 0 and NoC 1).
- An edge whose consumer is a single row or column multicasts along it and
  hits the emitc text round-trip bug below; compile with `--emit-bytecode`.

**Known upstream bugs** (all four were present on `main` `f688ee694`):

1. Offset placement.
   - Allocate `canAliasOperand` aliased an L1 operand without comparing the
     generic's grid with the operand's VGM. **Fixed in PR2a** (an off-origin
     region reading an origin operand now streams it; an edge consumer reads
     the producer's L1).
   - GridSelection `optimizeToLayoutGrid` (`GridSelection.cpp:176-201`)
     returns early for an already-selected or 1x1 grid, so a `to_layout`
     input of an off-origin region stays on the origin cores. **Not fixed:**
     since the Allocate fix the region streams it from there, so this costs an
     extra transfer, not a wrong result (n150: a 1x1 region at (1,1) and a
     2x2 region at (0,2) with such inputs, PCC 0.99999). Edge buffers come
     from `d2m.empty`, which gets the VGM.
2. DRAM bank map with spatial offset (`getMemoryMapImpl`). **Fixed in
   PR2b** (same code as `feature/d2m-batch2` `f8f250c3e`): a 2x2 region at
   (0,2) writing a DRAM output read back with PCC -0.01 before, 0.99999
   after.
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
- 16 local semaphores per core per program. This caps the producer outputs
  read by one consumer range (one semaphore per output, whatever the fan-out),
  together with the semaphores that multicast loads use.
- Two RISCs on the same core must not write on the same NoC at the same time.
- At most one multicast load per DM thread.
- Multi-source linked multicast times out: issue a write barrier before
  sending the flag.

## 8. Open questions for review

1. The verifier on `main` does not check that `grid_ranges` are disjoint (the
   op description says they must be). PR1 leaves this alone; since the
   multicast signal, `d2m-lower-spatial-edges` rejects a producer core inside
   a consumer's range. A separate small commit could add the general check.
2. Edge index `I` refers to the producer generic's DPS output `I` in both
   forms. The alternative, a flat `outs` index, would need remapping whenever
   normalization rebuilds `outs`, so it was rejected.
