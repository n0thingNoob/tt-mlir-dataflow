// RUN: ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-fuse-elementwise=true spatial-policy=none spatial-groups=0,0,0,0,0,0,0,1,0,1,0,1,0,2,0,2,0,2,0,3,0,3,0,3 spatial-loads=0,1,1,0,1,0,1,1,1,0,1,0 test-buffer-size-policy=auto-mn spatial-streams=0,1,2,1,1,2" --mlir-print-ir-after=d2m-materialize-spatial-groups --mlir-print-ir-after=d2m-reblock-generics --mlir-print-local-scope --mlir-disable-threading %s -o /dev/null 2>&1 | FileCheck %s
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-fuse-elementwise=true spatial-policy=none spatial-groups=0,0,0,0,0,0,0,1,0,1,0,1,0,2,0,2,0,2,0,3,0,3,0,3 spatial-loads=0,1,1,0,1,0,1,1,1,0,1,0 test-buffer-size-policy=auto-mn spatial-streams=0,1,2,1,1,2" %s -o /dev/null

// One bank of the V2 decode MLP at the main shape: a load region on (1,0)
// copies Wg[:, j] and Wu[:, j] (2048x704, 64x22 tiles each) from DRAM, gate #0
// and up #1 read them as their right-hand side and stream them (spatial-
// streams names input 1 of #0 and #1). With auto-mn the matmuls are
// [1, 2, 16]: they read their weight as 4x11-tile blocks, N outer, K inner.
//
// So the load is built with its loops in that order (indexing maps
// (d0, d1) -> (d1, d0): d0 walks N, d1 walks K), d2m-reblock-generics gives it
// the matmuls' blocks ([2, 16]), and both weight edges stream through rings of
// two 4x11 blocks instead of whole copies (which do not fit in L1).

// CHECK-LABEL: IR Dump After D2MMaterializeSpatialGroups
// CHECK: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0, stream<depth = 2>>, #d2m.spatial_edge<0:1 -> 2:0, stream<depth = 2>>
// CHECK-NEXT: ins(
// CHECK-NEXT: outs(
// CHECK-NEXT: d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d1, d0)>]
// CHECK-LABEL: IR Dump After D2MReblockGenerics
// CHECK: d2m.spatial
// CHECK: d2m.generic {block_factors = [2, 16], grid = #ttcore.grid<1x1{{.*}}>, indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>
// CHECK-DAG: remark: edge 0:0 -> 1:0 streams through a ring of 2 blocks of 4x11 tiles in the L1 of consumer core (0, 0)
// CHECK-DAG: remark: edge 0:1 -> 2:0 streams through a ring of 2 blocks of 4x11 tiles in the L1 of consumer core (0, 1)
func.func @glu(%x: tensor<32x2048xbf16>, %wg: tensor<2048x704xbf16>, %wu: tensor<2048x704xbf16>, %wd: tensor<704x2048xbf16>) -> tensor<32x2048xbf16> {
  %g = "ttir.matmul"(%x, %wg) : (tensor<32x2048xbf16>, tensor<2048x704xbf16>) -> tensor<32x704xbf16>
  %s = "ttir.silu"(%g) : (tensor<32x704xbf16>) -> tensor<32x704xbf16>
  %u = "ttir.matmul"(%x, %wu) : (tensor<32x2048xbf16>, tensor<2048x704xbf16>) -> tensor<32x704xbf16>
  %h = "ttir.multiply"(%s, %u) : (tensor<32x704xbf16>, tensor<32x704xbf16>) -> tensor<32x704xbf16>
  %y = "ttir.matmul"(%h, %wd) : (tensor<32x704xbf16>, tensor<704x2048xbf16>) -> tensor<32x2048xbf16>
  return %y : tensor<32x2048xbf16>
}
