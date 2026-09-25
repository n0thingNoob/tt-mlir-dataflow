# Automatic spatial mapping smoke test

All fixtures start from ordinary TTIR. `auto_diamond.mlir` (32×32) and
`auto_diamond_multitile.mlir` (128×128, 16 tiles) map four stages to one spatial
program on four cores. `auto_elementwise_chain.mlir` tests a three-stage,
16-tile pipeline; `auto_chain.mlir` tests matmul's singleton temporal fallback.
They exercise planning, GridSelection, bufferization, TTMetal lowering, and
serialization. Pipeline intermediate tensors remain in producer L1. Each edge
has a cumulative ready semaphore; each consumer reads through NoC after its
ready wait. No intermediate DRAM staging or ring-buffer reuse is permitted.
The lit RUN lines compile only; they do not execute on a device.

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

This baseline supports single-device, static, same-shape, tile-aligned 2D BF16
add/relu/neg graphs. Unsupported candidates fall back to legal independent
pairs or temporal singletons. It has no cost model, fusion, replication,
ring buffers, or performance tuning. Selected L1 storage cannot spill to DRAM;
an allocation failure is reported instead. Full intermediate tensors consume
L1 for the whole program, so large graphs can fail the conservative capacity
check.

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

Run again with `--case diamond_multitile`, `--case elementwise_chain`, and
`--case chain`, each with a new output directory. The device argument is
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
IR, binaries, input/reference and output tensors, per-run numerical results, and worker logs. Failed runs
retain their evidence and are not retried automatically. This is a fixed TTIR
fixture test, not live PyTorch graph capture.
