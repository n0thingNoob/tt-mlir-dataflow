// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-fuse-elementwise=true spatial-policy=bank spatial-stream-policy=auto spatial-bank-split=true" --mlir-print-ir-after=d2m-materialize-spatial-groups --mlir-print-local-scope %s -o %t.mlir 2>&1 | FileCheck %s --check-prefix=SPATIAL
// RUN: FileCheck %s --check-prefix=METAL --input-file=%t.mlir
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="dram-args-and-results=true spatial-bank-split=true" --mlir-print-ir-before=ttir-to-d2m %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEFAULT

// spatial-bank-split on a gated MLP as frontends export it (one matmul per
// weight, transpose_b): with execution-strategy=spatial and spatial-policy=bank
// it gets the hand-split V1 structure, 12 x (gate, up, silu * up, down) + one
// hub region = 49 regions.
// SPATIAL-LABEL: IR Dump After D2MMaterializeSpatialGroups
// SPATIAL: d2m.spatial {edges = [{{.*}}], grid_ranges = [{{(#ttcore.core_range<\([0-9]+,[0-9]+\), \([0-9]+,[0-9]+\)>, ){48}()}}#ttcore.core_range<({{[0-9]+}},{{[0-9]+}}), ({{[0-9]+}},{{[0-9]+}})>]}

// Each weight is written to the device once; x once per matmul that reads it.
// METAL-COUNT-27: ttmetal.enqueue_write_buffer
// METAL-NOT: ttmetal.enqueue_write_buffer

// The default execution strategy does not split.
// DEFAULT-LABEL: IR Dump Before TTIRToD2M
// DEFAULT-NOT: ttir.slice_static
// DEFAULT-COUNT-3: "ttir.matmul"
// DEFAULT-NOT: "ttir.matmul"
func.func @glu(%x: tensor<32x64xbf16>, %wg: tensor<384x64xbf16>, %wu: tensor<384x64xbf16>, %wd: tensor<64x384xbf16>) -> tensor<32x64xbf16> {
  %g = "ttir.matmul"(%x, %wg) <{transpose_a = false, transpose_b = true}> : (tensor<32x64xbf16>, tensor<384x64xbf16>) -> tensor<32x384xbf16>
  %s = "ttir.silu"(%g) : (tensor<32x384xbf16>) -> tensor<32x384xbf16>
  %u = "ttir.matmul"(%x, %wu) <{transpose_a = false, transpose_b = true}> : (tensor<32x64xbf16>, tensor<384x64xbf16>) -> tensor<32x384xbf16>
  %h = "ttir.multiply"(%s, %u) : (tensor<32x384xbf16>, tensor<32x384xbf16>) -> tensor<32x384xbf16>
  %y = "ttir.matmul"(%h, %wd) <{transpose_a = false, transpose_b = true}> : (tensor<32x384xbf16>, tensor<64x384xbf16>) -> tensor<32x64xbf16>
  return %y : tensor<32x64xbf16>
}
