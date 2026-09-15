# D2M planner passthrough

`enable-dataflow-planning=true` enables three frontend stages:

1. Graph planning after TTIR-to-D2M conversion: retain the temporal fallback.
2. Blocking planning before ReblockGenerics: observe current factors without
   overriding the existing selection policy or explicit user requests.
3. Allocation feedback after Allocate: read a typed resource summary, without
   ranking or retrying candidates.

`dump-dataflow-plan=true` prints each stage to stderr. It does not enable the
framework on its own. The framework remains disabled by default and rejects
TTNN mode as before.

Allocation reports generated for the framework are consumed before backend
lowering. An explicit `emit-resource-report=true` preserves them for inspection.
Failed allocation stops the pipeline; its partial report can be read through
`readDataflowAllocationFeedback` or inspected with
`--mlir-print-ir-after-failure`. Missing usage is not treated as zero.

The graph and blocking stages have separate local candidate ordinals. They do
not claim stable identities across fusion, bufferization, or other rewrites.
The allocation summary describes allocator address-space usage, not measured
transfer traffic or execution time.

## Compiler check

```sh
source env/activate
cmake --build build --target ttmlir-opt ttmlir-translate
llvm-lit -sv build/test/ttmlir/Dialect/D2M/Transforms/dataflow_passthrough_e2e.mlir
```

The test compiles a BF16 matmul/relu/neg graph with the framework off and on,
checks all three stages were reached, compares final TTMetal IR, and serializes
both outputs. It also compares the fused, single-buffer configuration and
checks allocation failure stops before feedback consumption.

This is TTIR-to-binary coverage, not hardware execution or live PyTorch capture.
There is no candidate search or performance claim in this version.

## Opt-in hardware check

Use a matched compiler, ttrt Python environment, and Metal checkout. Configure
`PYTHONPATH` and `LD_LIBRARY_PATH` for that runtime before running the command.
The helper does not install packages or reset devices.
It requires an exact PCI BDF and refuses a device reported busy by `fuser`.

```sh
python test/ttmlir/Dialect/D2M/Transforms/Inputs/run_dataflow_passthrough_hardware.py \
  --compiler build/bin/ttmlir-opt --translate build/bin/ttmlir-translate \
  --metal-home "$TT_METAL_HOME" --device "$DEVICE_BDF" \
  --output /tmp/d2m-planner-check --runs 3
```

Choose a new output directory for each invocation. The helper queries a live
descriptor, recompiles both modes, compares final IR, and executes each mode
three times against the same BF16 PyTorch reference. Every result must satisfy
PCC >= 0.999 and allclose (rtol 0.02, atol 0.005); outputs must also be identical
between modes and repetitions. Each subprocess has a timeout. A timeout fails
the test and requires investigation before another run, not an automatic reset.

Artifacts include commands' logs, descriptor, IR, binaries, per-run correctness
results, and a manifest recording device identity, source revisions, local
changes, library hashes, and relevant environment settings. Query and execution
must use the same runtime libraries, and compiler/runtime build revisions must
match before execution. This checks a checked-in TTIR graph through
device execution, not live PyTorch capture. No timings are collected.

The same check is registered with lit and skipped by default. In the matched
runtime environment, enable it explicitly with `TTMLIR_TEST_DATAFLOW_HARDWARE=1`
and `DEVICE_BDF` set, then run:

```sh
llvm-lit -sv build/test/ttmlir/Dialect/D2M/hardware/dataflow_passthrough.mlir
```
