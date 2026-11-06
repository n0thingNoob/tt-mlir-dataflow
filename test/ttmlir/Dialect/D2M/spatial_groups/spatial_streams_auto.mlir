// RUN: ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-fuse-elementwise=true spatial-policy=none spatial-groups=0,0,0,0,0,0,0,1,0,1,0,1,0,2,0,2,0,2,0,3,0,3,0,3 spatial-loads=0,1,1,0,1,0,1,1,1,0,1,0,3,1,1,1,1,1 spatial-stream-policy=auto" --mlir-print-ir-after=d2m-spatial-planning --mlir-print-ir-after=d2m-materialize-spatial-groups --mlir-print-local-scope --mlir-disable-threading %s -o /dev/null 2>&1 | FileCheck %s
// RUN: ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-fuse-elementwise=true spatial-policy=none spatial-groups=0,0,0,0,0,0,0,1,0,1,0,1,0,2,0,2,0,2,0,3,0,3,0,3 spatial-loads=0,1,1,0,1,0,1,1,1,0,1,0,3,1,1,1,1,1 spatial-stream-policy=auto spatial-streams=2,0,1,2,1,1" --mlir-print-ir-after=d2m-materialize-spatial-groups --mlir-print-local-scope --mlir-disable-threading %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=MIXED
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-fuse-elementwise=true spatial-policy=none spatial-groups=0,0,0,0,0,0,0,1,0,1,0,1,0,2,0,2,0,2,0,3,0,3,0,3 spatial-loads=0,1,1,0,1,0,1,1,1,0,1,0,3,1,1,1,1,1 spatial-stream-policy=auto" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=REMARK
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-fuse-elementwise=true spatial-stream-policy=auto" %s -o /dev/null
// RUN: not ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-fuse-elementwise=true spatial-policy=none spatial-groups=0,0,0,0,0,0,0,1,0,1,0,1,0,2,0,2,0,2,0,3,0,3,0,3 spatial-loads=0,1,1,0,1,0,1,1,1,0,1,0,3,1,1,1,1,1 spatial-stream-policy=all" %s 2>&1 | FileCheck %s --check-prefix=ERR-POLICY

// spatial-stream-policy=auto on the gated MLP of spatial_streams.mlir (same
// groups and load regions): the planner marks every grouped generic
// (d2m.spatial_stream_all = depth 2, no block), so every edge gets
// stream<depth = 2>, and d2m-materialize-spatial-streams keeps the ones that
// can stream. Inputs named in spatial-streams keep their own depth. The
// greedy run (no groups given) only checks that auto compiles end to end.

// CHECK-LABEL: IR Dump After D2MSpatialPlanning
// CHECK-COUNT-4: d2m.spatial_stream_all = array<i64: 2, 0, 0>
// CHECK-LABEL: IR Dump After D2MMaterializeSpatialGroups
// CHECK: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0, stream<depth = 2>>, #d2m.spatial_edge<0:1 -> 2:0, stream<depth = 2>>, #d2m.spatial_edge<1:0 -> 3:0, stream<depth = 2>>, #d2m.spatial_edge<2:0 -> 3:1, stream<depth = 2>>, #d2m.spatial_edge<3:0 -> 5:0, stream<depth = 2>>, #d2m.spatial_edge<4:0 -> 5:1, stream<depth = 2>>]
// CHECK-NOT: d2m.spatial_stream_all

// MIXED: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0, stream<depth = 2>>, #d2m.spatial_edge<0:1 -> 2:0, stream<depth = 2>>, #d2m.spatial_edge<1:0 -> 3:0, stream<depth = 1>>, #d2m.spatial_edge<2:0 -> 3:1, stream<depth = 1>>, #d2m.spatial_edge<3:0 -> 5:0, stream<depth = 2>>, #d2m.spatial_edge<4:0 -> 5:1, stream<depth = 2>>]

// The load edges stream as in spatial_streams.mlir, and so do h into down and
// the Wd load. g and u into the fused silu * mul fall back: the matmuls write
// one block per core, the consumer reads two.
// REMARK-DAG: remark: edge 1:0 -> 3:0 stays a whole-tensor handoff: consumer core (0, 0) reads 2 blocks but the producer writes 1
// REMARK-DAG: remark: edge 2:0 -> 3:1 stays a whole-tensor handoff: consumer core (0, 0) reads 2 blocks but the producer writes 1
// REMARK-DAG: remark: edge 0:0 -> 1:0 streams through a ring of 2 blocks of 4x8 tiles in the L1 of consumer core (0, 0)
// REMARK-DAG: remark: edge 0:1 -> 2:0 streams through a ring of 2 blocks of 4x8 tiles in the L1 of consumer core (0, 1)
// REMARK-DAG: remark: edge 3:0 -> 5:0 streams through a ring of 2 blocks of 1x4 tiles in the L1 of consumer core (0, 3)
// REMARK-DAG: remark: edge 4:0 -> 5:1 streams through a ring of 2 blocks of 4x8 tiles in the L1 of consumer core (0, 3)

// ERR-POLICY: error: stream-policy must be none or auto, got all
func.func @glu(%x: tensor<32x256xbf16>, %wg: tensor<256x256xbf16>, %wu: tensor<256x256xbf16>, %wd: tensor<256x256xbf16>) -> tensor<32x256xbf16> {
  %g = "ttir.matmul"(%x, %wg) : (tensor<32x256xbf16>, tensor<256x256xbf16>) -> tensor<32x256xbf16>
  %s = "ttir.silu"(%g) : (tensor<32x256xbf16>) -> tensor<32x256xbf16>
  %u = "ttir.matmul"(%x, %wu) : (tensor<32x256xbf16>, tensor<256x256xbf16>) -> tensor<32x256xbf16>
  %h = "ttir.multiply"(%s, %u) : (tensor<32x256xbf16>, tensor<32x256xbf16>) -> tensor<32x256xbf16>
  %y = "ttir.matmul"(%h, %wd) : (tensor<32x256xbf16>, tensor<256x256xbf16>) -> tensor<32x256xbf16>
  return %y : tensor<32x256xbf16>
}
