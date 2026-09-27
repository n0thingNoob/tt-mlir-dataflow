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

1. Select supported chain/fork-join candidates in stable order and reserve one
   core per stage. Record internal edges and prevent cross-stage fusion.
2. After bufferization and one-tile reblocking, move the original generics into
   one `d2m.spatial`, allocate L1 readiness counters, and insert tile waits and
   notifications. Existing lowering emits the shared program.

This baseline supports single-device, static, tile-aligned 2D BF16
add/relu/neg and standard non-transposed matmul. Each stage owns one core.
Matmul reads a complete 1×K tile panel and K×1 tile panel, completes the K
reduction, then writes and publishes one output tile. Only matmul's left input
may come from another stage; its right input must be ready before the program.
Batch, transpose, split-K, bias and extra epilogues are outside this baseline.

Full intermediate tensors remain in producer L1. With T tile columns, a
consumer waits for counter `row*T + column + 1` before reading a tile;
a matmul waits for `(row+1)*T` before reading its entire left input row.
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

There is no cost model, fusion, replication or performance tuning. Full tensor
and panel storage limits capacity, and the resource estimate is conservative.

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
compilation, checks L1 allocation addresses, verifies automatic grouping, and
executes each strategy three times. The diamond additionally requires four
compute kernels in one enqueue on `(0,0)` through `(0,3)`, cumulative waits and
notifications, and no DRAM buffers. Both strategies are checked
against the same seeded BF16 PyTorch reference with PCC >= 0.999, rtol = 0.02,
and atol = 0.005. Every repetition must pass; tolerances are not relaxed on
failure. Performance is not measured.

Artifacts include software/library identities and hashes, the descriptor,
selection reports, IR before GridSelection and after materialization, lowered
IR, binaries, input/reference and output tensors, per-run numerical results,
and worker logs. Failed runs retain their evidence and are not retried automatically. This is a fixed TTIR
fixture test, not live PyTorch graph capture.
