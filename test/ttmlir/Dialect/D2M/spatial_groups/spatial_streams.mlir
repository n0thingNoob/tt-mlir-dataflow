// RUN: ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-fuse-elementwise=true spatial-policy=none spatial-groups=0,0,0,0,0,0,0,1,0,1,0,1,0,2,0,2,0,2,0,3,0,3,0,3 spatial-loads=0,1,1,0,1,0,1,1,1,0,1,0,3,1,1,1,1,1 spatial-streams=0,1,2,1,1,2,2,0,1,2,1,1" --mlir-print-ir-after=d2m-materialize-spatial-groups --mlir-print-local-scope --mlir-disable-threading %s -o /dev/null 2>&1 | FileCheck %s
// RUN: ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-fuse-elementwise=true spatial-policy=none spatial-groups=0,0,0,0,0,0,0,1,0,1,0,1,0,2,0,2,0,2,0,3,0,3,0,3 spatial-streams=2,0,2,2,1,2 spatial-stream-block=1,4" --mlir-print-ir-after=d2m-materialize-spatial-groups --mlir-print-local-scope --mlir-disable-threading %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=BLOCK
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-fuse-elementwise=true spatial-policy=none spatial-groups=0,0,0,0,0,0,0,1,0,1,0,1,0,2,0,2,0,2,0,3,0,3,0,3 spatial-loads=0,1,1,0,1,0,1,1,1,0,1,0,3,1,1,1,1,1 spatial-streams=0,1,2,1,1,2" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=REMARK
// RUN: not ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-fuse-elementwise=true spatial-policy=none spatial-groups=0,0,0,0,0,0,0,1,0,1,0,1,0,2,0,2,0,2,0,3,0,3,0,3 spatial-streams=0,1" %s 2>&1 | FileCheck %s --check-prefix=ERR-COUNT
// RUN: not ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-fuse-elementwise=true spatial-policy=none spatial-groups=0,0,0,0,0,0,0,1,0,1,0,1,0,2,0,2,0,2,0,3,0,3,0,3 spatial-streams=9,1,2" %s 2>&1 | FileCheck %s --check-prefix=ERR-INDEX
// RUN: not ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-fuse-elementwise=true spatial-policy=none spatial-groups=0,0,0,0,0,0,0,1,0,1,0,1 spatial-streams=2,0,2" %s 2>&1 | FileCheck %s --check-prefix=ERR-GROUP
// RUN: not ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-fuse-elementwise=true spatial-policy=none spatial-groups=0,0,0,0,0,0,0,1,0,1,0,1,0,2,0,2,0,2,0,3,0,3,0,3 spatial-streams=2,2,2" %s 2>&1 | FileCheck %s --check-prefix=ERR-OPERAND
// RUN: not ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-fuse-elementwise=true spatial-policy=none spatial-groups=0,0,0,0,0,0,0,1,0,1,0,1,0,2,0,2,0,2,0,3,0,3,0,3 spatial-streams=2,0,9" %s 2>&1 | FileCheck %s --check-prefix=ERR-DEPTH
// RUN: not ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-fuse-elementwise=true spatial-policy=none spatial-groups=0,0,0,0,0,0,0,1,0,1,0,1,0,2,0,2,0,2,0,3,0,3,0,3 spatial-streams=2,0,2,2,0,1" %s 2>&1 | FileCheck %s --check-prefix=ERR-TWICE
// RUN: not ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-fuse-elementwise=true spatial-policy=none spatial-groups=0,0,0,0,0,0,0,1,0,1,0,1,0,2,0,2,0,2,0,3,0,3,0,3 spatial-streams=2,0,2 spatial-stream-block=1" %s 2>&1 | FileCheck %s --check-prefix=ERR-BLOCK
// RUN: not ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-fuse-elementwise=true spatial-policy=none spatial-groups=0,0,0,0,0,0,0,1,0,1,0,1,0,2,0,2,0,2,0,3,0,3,0,3 spatial-stream-block=1,4" %s 2>&1 | FileCheck %s --check-prefix=ERR-NOSTREAMS
// RUN: not ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-fuse-elementwise=true spatial-policy=none spatial-groups=0,0,0,0,0,0,0,1,0,1,0,1,0,2,0,2,0,2,0,3,0,3,0,3 spatial-streams=0,0,2" %s 2>&1 | FileCheck %s --check-prefix=ERR-NOTEDGE

// Streaming edges given by hand on the gated MLP of spatial_load_regions.mlir:
// gate #0 on (0,0), up #1 on (0,1), silu(g) * u #2 on (0,2) (fused), down #3
// on (0,3). spatial-streams names (generic, operand, depth): the weights that
// the load regions copy (input 1 of #0 and #1; the copy keeps the input
// index) and g, u into #2 (inputs 0 and 1). The other edges (h into down, Wd
// into down) stay whole-tensor handoffs.

// CHECK-LABEL: IR Dump After D2MMaterializeSpatialGroups
// CHECK: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0, stream<depth = 2>>, #d2m.spatial_edge<0:1 -> 2:0, stream<depth = 2>>, #d2m.spatial_edge<1:0 -> 3:0, stream<depth = 1>>, #d2m.spatial_edge<2:0 -> 3:1, stream<depth = 1>>, #d2m.spatial_edge<3:0 -> 5:0>, #d2m.spatial_edge<4:0 -> 5:1>]
// CHECK-NOT: d2m.spatial_streams

// BLOCK: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 2:0, stream<depth = 2, block = 1x4>>, #d2m.spatial_edge<1:0 -> 2:1, stream<depth = 2, block = 1x4>>, #d2m.spatial_edge<2:0 -> 3:0>]

// The matmuls read each weight in two K blocks of 4x8 tiles. The load region
// whose copies stream takes those blocks (d2m-reblock-generics) in the
// matmuls' order, so both load edges stream (spatial_streams_load.mlir checks
// the load itself).
// REMARK: remark: edge 0:0 -> 1:0 streams through a ring of 2 blocks of 4x8 tiles in the L1 of consumer core (0, 0)
// REMARK: remark: edge 0:1 -> 2:0 streams through a ring of 2 blocks of 4x8 tiles in the L1 of consumer core (0, 1)
// REMARK-NOT: asks for a streaming handoff

// ERR-COUNT: error: streams takes (generic, operand, depth) per input, got 2 values
// ERR-INDEX: error: streams names generic #9, but the module has only 4 generics
// ERR-GROUP: error: 'd2m.generic' op streams names generic #2, which is not in a spatial group
// ERR-OPERAND: error: 'd2m.generic' op streams names input 2 of generic #2, which has 2 inputs
// ERR-DEPTH: error: 'd2m.generic' op streams asks for depth 9 on input 0 of generic #2; it must be in [1, 8]
// ERR-TWICE: error: 'd2m.generic' op streams names input 0 of generic #2 twice
// ERR-BLOCK: error: stream-block takes two positive values (R, C)
// ERR-NOSTREAMS: error: stream-block needs streams
// ERR-NOTEDGE: error: 'd2m.generic' op d2m.spatial_streams names input 0, which does not read another member of spatial group 0
func.func @glu(%x: tensor<32x256xbf16>, %wg: tensor<256x256xbf16>, %wu: tensor<256x256xbf16>, %wd: tensor<256x256xbf16>) -> tensor<32x256xbf16> {
  %g = "ttir.matmul"(%x, %wg) : (tensor<32x256xbf16>, tensor<256x256xbf16>) -> tensor<32x256xbf16>
  %s = "ttir.silu"(%g) : (tensor<32x256xbf16>) -> tensor<32x256xbf16>
  %u = "ttir.matmul"(%x, %wu) : (tensor<32x256xbf16>, tensor<256x256xbf16>) -> tensor<32x256xbf16>
  %h = "ttir.multiply"(%s, %u) : (tensor<32x256xbf16>, tensor<32x256xbf16>) -> tensor<32x256xbf16>
  %y = "ttir.matmul"(%h, %wd) : (tensor<32x256xbf16>, tensor<256x256xbf16>) -> tensor<32x256xbf16>
  return %y : tensor<32x256xbf16>
}
