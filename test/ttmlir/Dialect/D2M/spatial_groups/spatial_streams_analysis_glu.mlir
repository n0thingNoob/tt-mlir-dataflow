// RUN: ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-fuse-elementwise=true spatial-policy=none spatial-groups=0,0,0,0,0,0,0,1,0,1,0,1,0,2,0,2,0,2,0,3,0,3,0,3 test-buffer-size-policy=auto-mn spatial-streams=2,0,2,2,1,2,3,0,2" --mlir-print-ir-after=d2m-materialize-spatial-streams --mlir-print-local-scope --mlir-disable-threading %s -o /dev/null 2>&1 | FileCheck %s
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-fuse-elementwise=true spatial-policy=none spatial-groups=0,0,0,0,0,0,0,1,0,1,0,1,0,2,0,2,0,2,0,3,0,3,0,3 test-buffer-size-policy=auto-mn spatial-streams=2,0,2,2,1,2,3,0,2" %s -o /dev/null

// d2m-materialize-spatial-streams on one bank of the decode gated MLP at the
// main shape (H = 2048, F_j = 704 = 22 tiles), weights in DRAM: gate #0 on
// (0,0), up #1 on (0,1), silu(g) * u #2 on (0,2) (fused), down #3 on (0,3).
// spatial-streams asks for streams on g and u into #2 and on h into #3.
//
// With auto-mn the gate and up matmuls are [1, 2, 16]: two output blocks of
// 1x11 tiles, each written in every k iteration but final only after the
// last one, and #2 is [1, 2] reading the same two blocks in the same order,
// so g and u can stream. The down matmul is [1, 16, 2]: for each of its 16
// output blocks it reads both 1x11 blocks of h again (32 reads), so h stays a
// whole-tensor handoff.

// The gate and up matmuls write their output block in every k iteration;
// only the last write is final and goes into #2's ring (the second RUN line
// checks that the whole program compiles).
// CHECK-DAG: remark: edge 0:0 -> 2:0 can stream: 1 core pair, 2 blocks of 1x11 tiles each, in the producer's order
// CHECK-DAG: remark: edge 0:0 -> 2:0 streams through a ring of 2 blocks of 1x11 tiles in the L1 of consumer core (0, 2)
// CHECK-DAG: remark: edge 1:0 -> 2:1 can stream: 1 core pair, 2 blocks of 1x11 tiles each, in the producer's order
// CHECK-DAG: remark: edge 1:0 -> 2:1 streams through a ring of 2 blocks of 1x11 tiles in the L1 of consumer core (0, 2)
// CHECK-DAG: remark: edge 2:0 -> 3:0 stays a whole-tensor handoff: consumer core (0, 0) reads 32 blocks but the producer writes 2; a ring would need 2 of 2 blocks
// CHECK-LABEL: IR Dump After D2MMaterializeSpatialStreams
// CHECK: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 2:0, stream<depth = 2>>, #d2m.spatial_edge<1:0 -> 2:1, stream<depth = 2>>, #d2m.spatial_edge<2:0 -> 3:0>]
func.func @glu(%x: tensor<32x2048xbf16>, %wg: tensor<2048x704xbf16>, %wu: tensor<2048x704xbf16>, %wd: tensor<704x2048xbf16>) -> tensor<32x2048xbf16> {
  %g = "ttir.matmul"(%x, %wg) : (tensor<32x2048xbf16>, tensor<2048x704xbf16>) -> tensor<32x704xbf16>
  %s = "ttir.silu"(%g) : (tensor<32x704xbf16>) -> tensor<32x704xbf16>
  %u = "ttir.matmul"(%x, %wu) : (tensor<32x2048xbf16>, tensor<2048x704xbf16>) -> tensor<32x704xbf16>
  %h = "ttir.multiply"(%s, %u) : (tensor<32x704xbf16>, tensor<32x704xbf16>) -> tensor<32x704xbf16>
  %y = "ttir.matmul"(%h, %wd) : (tensor<32x704xbf16>, tensor<704x2048xbf16>) -> tensor<32x2048xbf16>
  return %y : tensor<32x2048xbf16>
}
