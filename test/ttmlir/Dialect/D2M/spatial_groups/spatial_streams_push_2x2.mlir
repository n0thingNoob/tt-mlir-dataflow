// RUN: ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-policy=none spatial-groups=0,0,0,0,1,1,0,1,0,3,1,4 spatial-streams=1,0,2 spatial-stream-block=1,1" --mlir-print-ir-after=d2m-reblock-generics --mlir-print-ir-after=d2m-materialize-spatial-streams --mlir-print-local-scope --mlir-disable-threading %s -o /dev/null 2>&1 | FileCheck %s
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-policy=none spatial-groups=0,0,0,0,1,1,0,1,0,3,1,4 spatial-streams=1,0,2 spatial-stream-block=1,1" --mlir-print-ir-after=d2m-lower-spatial-edges --mlir-print-local-scope --mlir-disable-threading %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=LOWER

// A streaming edge between two 2x2 regions whose producer is a reduction:
// a = x @ w (64x256 @ 256x128) on (0,0)-(1,1), neg(a) on (0,3)-(1,4).
// Each matmul core computes a 1x2-tile output shard; block = 1x1 asks for
// blocks of one tile, so ReblockGenerics gives the matmul two N blocks and
// neg two blocks, and matmul core k streams its two output tiles to neg
// core k.

// CHECK-LABEL: IR Dump After D2MReblockGenerics
// CHECK: d2m.spatial
// CHECK: d2m.generic {block_factors = [1, 2, {{[0-9]+}}], grid = #ttcore.grid<2x2>
// CHECK: d2m.generic {block_factors = [1, 2], grid = #ttcore.grid<2x2,
// CHECK: remark: edge 0:0 -> 1:0 can stream: 4 core pairs, 2 blocks of 1x1 tiles each, in the producer's order
// CHECK: remark: edge 0:0 -> 1:0 streams through a ring of 2 blocks of 1x1 tiles in the L1 of each of the 4 consumer cores, from (0, 3)
// CHECK-LABEL: IR Dump After D2MMaterializeSpatialStreams
// CHECK: memref.alloc() {d2m.stream_ring, {{.*}}} : memref<2x2x2x1x!ttcore.tile<32x32, bf16>

// The matmul's writer writes the ring only in the last k iteration; earlier
// iterations yield a null transaction. Core k signals core k + (0, 3), and
// neg core k returns its credit to core k + (0, -3).
// LOWER-LABEL: IR Dump After D2MLowerSpatialEdges
// LOWER: %[[LAST:.*]] = arith.subi %c2, %c1{{.*}} : index
// LOWER-NEXT: %[[FINAL:.*]] = arith.cmpi eq, %{{.*}}, %[[LAST]] : index
// LOWER-NEXT: scf.if %[[FINAL]] -> (!d2m.mem_tx<write>) {
// LOWER: d2m.semaphore_wait %{{.*}}, %{{.*}} min
// LOWER-NEXT: %[[TX:.*]] = d2m.dma_write
// LOWER-NEXT: %[[B:.*]] = d2m.null_tx : !d2m.mem_tx<write>
// LOWER-NEXT: d2m.dma_wait %[[B]]
// LOWER: %[[C1:.*]] = d2m.core_index(1)
// LOWER-NEXT: %[[X:.*]] = arith.addi %[[C1]], %c3
// LOWER-NEXT: d2m.semaphore_inc %{{.*}}, %{{.*}}, core[%{{.*}}, %[[X]]]
// LOWER-NEXT: scf.yield %[[TX]]
// LOWER-NEXT: } else {
// LOWER-NEXT: %[[N:.*]] = d2m.null_tx : !d2m.mem_tx<write>
// LOWER-NEXT: scf.yield %[[N]]
// LOWER: d2m.semaphore_wait %{{.*}}, %c2{{.*}} min
// LOWER: %[[RX:.*]] = arith.addi %core1{{.*}}, %c-3
// LOWER-NEXT: d2m.semaphore_inc %{{.*}}, %{{.*}}, core[%{{.*}}, %[[RX]]]
func.func @mm_neg(%x: tensor<64x256xbf16>, %w: tensor<256x128xbf16>) -> tensor<64x128xbf16> {
  %a = "ttir.matmul"(%x, %w) <{transpose_a = false, transpose_b = false}> : (tensor<64x256xbf16>, tensor<256x128xbf16>) -> tensor<64x128xbf16>
  %n = "ttir.neg"(%a) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
  return %n : tensor<64x128xbf16>
}
