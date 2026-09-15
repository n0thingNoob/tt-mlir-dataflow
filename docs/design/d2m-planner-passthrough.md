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
