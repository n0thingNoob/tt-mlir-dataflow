# Automatic spatial mapping smoke test

All fixtures start from ordinary TTIR and use the same compilation and device
runner. The original `auto_diamond.mlir` is unchanged.

| Runner case | Graph and shape | Expected spatial computation |
|---|---|---|
| `diamond` | 128×128 add → {relu, neg} → add | Four stages in one program |
| `elementwise_chain` / `elementwise_rect` | add → relu → neg, 128×128 / 64×96 | Three stages in one program |
| `gemm_chain` | Two 64×64 matmuls with relu between them | Three stages in one program |
| `gemm_rect` | 64×96 @ 96×128 → relu → @ 128×96 | Three stages in one program |
| `mixed` | 64×64 diamond output is the RHS of matmul | Four-stage program followed by temporal matmul |
| `chain` | 64×64 matmul → relu → neg | Three stages in one program |

Shape variants reuse fixtures through a temporary `input.mlir` saved in the
artifact directory. The lit RUN lines compile only; hardware execution is
opt-in. The compiler driver also reuses these fixtures.

The compiler entry point is:

```bash
ttmlir-opt input.mlir \
  --ttir-to-ttmetal-pipeline="execution-strategy=spatial dump-spatial-planning=true system-desc-path=/path/to/live-device.ttsys" \
  --mlir-print-op-generic -o lowered.mlir
```

The default execution strategy remains temporal. `d2m-spatial-planning` on
its own remains analysis-only. Spatial execution adds two steps around the
existing layout/bufferization passes:

1. Select supported chain/fork-join members and dependencies, without choosing
   cores. Preserve internal edges and prevent cross-stage fusion.
2. GridSelection chooses one M-shard count P for the whole pipeline, starting
   with the largest common divisor of the stages' M tile rows that fits the
   region's worker budget. Every stage remains one generic on a logical P×1
   grid. Identical P-core rectangles are packed row-major on the device; the
   first fitting factorization (in increasing rectangle height) wins.
3. After bufferization and one-tile reblocking, move the original generics into
   one `d2m.spatial`, allocate L1 readiness counters, and insert tile waits and
   notifications. Existing lowering emits the shared multi-core program.

This baseline supports single-device, static, tile-aligned 2D BF16
add/relu/neg and standard non-transposed matmul. Each stage owns P disjoint cores;
core s computes the continuous tile-row interval `[s*M/P, (s+1)*M/P)`.
Only equal-sized M shards with matching producer/consumer boundaries are
supported. N and K are never partitioned across cores. P is not restricted to
powers of two: three tile rows may use P=3.
Matmul reads a complete 1×K tile panel and K×1 tile panel, completes the K
reduction, then writes and publishes one output tile. Only matmul's left input
may come from another stage; its full right input is prepared in L1 on that
stage's first shard core before the program. All M shards can read it through
NoC; full RHS storage is not replicated.
Batch, transpose, split-K, bias and extra epilogues are outside this baseline.

Each intermediate shard remains in its producer core's L1. Each edge has a
separate counter at every consumer shard core, written only by its matching
producer shard. With T tile columns, a consumer waits for
`local_row*T + column + 1` before reading a tile; a matmul waits for
`(local_row+1)*T` before reading its entire left input row. Physical placement
and global M offsets never enter these local cumulative thresholds.
The row may be read repeatedly for different output columns. Each producer
increments each outgoing counter once per completed tile, after the DMA write
barrier. Counters start at zero on every execution and are not reset inside
the program. No ring-buffer reuse, credit protocol or DRAM staging is needed.

Before creating any semaphore or moving stages, materialization verifies the
actual loop order/bounds, blocking, complete input panels, one unconditional
output store per iteration and exact reblocking views of shared L1 storage.
It rejects unproven traversal and wraps the entire group as ordered temporal
singletons, clearing pipeline attributes and retaining the established layouts.
Malformed preparation metadata is a compiler error. Selected L1 storage cannot
spill to DRAM; allocation failure is reported explicitly.

There is no cost model or performance claim. Selection checks the entire
region's rectangles, CB ports and conservative L1 capacity, including full RHS
storage, panel buffers, alignment, readiness counters and other tensors that
may remain live across the region. The L1 bound also respects the existing
common-address allocator; it does not assume per-core address reuse. Address
limits and capacity overrides are shared with Allocate. A resource rejection
tries smaller valid divisors, then P=1, then the temporal path before layout
mutation. Unexpected post-selection allocation exhaustion remains an error;
pipeline buffers never silently spill.

`spatial-pipeline-max-shards=0` (the compiler default) uses the device budget.
Set it to 1 to reproduce the PR14 single-core pipeline, or to a positive value
to cap P. The device's real worker grid and any grid override supply the total
region budget; stages never reserve it independently.

For an opt-in Wormhole execution, first build matching `ttmlir-opt`,
`ttmlir-translate`, and the TTMetal runtime from the same source snapshot. Set
`PYTHONPATH`, `LD_LIBRARY_PATH`, `TT_METAL_HOME`, and `TT_METAL_RUNTIME_ROOT` to
that installation. The helper reuses the runtime worker loading/query/execution
flow and requires PyTorch, the matching `ttrt` Python sources, and `fuser`.

```bash
python run_auto_spatial.py \
  --case diamond \
  --source /path/to/matched/source \
  --compiler /path/to/build/bin/ttmlir-opt \
  --translate /path/to/build/bin/ttmlir-translate \
  --metal-home /path/to/matched/tt-metal \
  --device 0000:c1:00.0 \
  --output /path/to/new/diamond-artifacts
```

Run the other cases from the table with a new output directory each time. The device argument is
an exact PCI BDF, not an index. The helper resolves its KMD index and checks for
active users before opening it. It never resets a device.

Each case queries a live descriptor, uses it for both temporal and spatial
compilation, checks complete per-core L1 allocation extents, verifies automatic
grouping, and executes each strategy three times. It requires a shared enqueue
with disjoint P-core stage ranges, exact tile-row coverage from the materialized
local traversal, cumulative waits and notifications, and no DRAM buffers.
The runner defaults to `--max-shards 1 --expect-shards 1` for baseline coverage.

Multi-core runs use the same inputs, references and thresholds:

| Extra runner arguments | Coverage |
|---|---|
| `--case gemm_rect --max-shards 0 --expect-shards 2` | Six-core GEMM chain |
| `--case diamond --max-shards 2 --expect-shards 2` | Eight-core diamond, two rows per shard |
| `--case gemm_rect --tile-rows 3 --max-shards 0 --expect-shards 3` | Nine-core GEMM chain |
| `--case gemm_rect --tile-rows 4 --max-shards 0 --expect-shards 4` | Twelve-core GEMM chain |
| `--case gemm_rect --tile-rows 4 --max-shards 2 --expect-shards 2` | GEMM with multiple local rows per shard |
| `--case gemm_rect --tile-rows 4 --max-shards 0 --expect-shards 4 --worker-grid 6,2` | Three physical 2×2 stage rectangles |

Run a corresponding `--max-shards 1 --expect-shards 1` case for each shape
variant to compare against the unchanged single-core arithmetic. Both strategies are checked
against the same seeded BF16 PyTorch reference with PCC >= 0.999, rtol = 0.02,
and atol = 0.005. Every repetition must pass; tolerances are not relaxed on
failure. Performance is not measured.

Artifacts include software/library identities and hashes, the descriptor,
selection reports, IR before GridSelection and after materialization, lowered
IR, binaries, input/reference and output tensors, per-run numerical results,
and worker logs. Failed runs retain their evidence and are not retried automatically. This is a fixed TTIR
fixture test, not live PyTorch graph capture.
