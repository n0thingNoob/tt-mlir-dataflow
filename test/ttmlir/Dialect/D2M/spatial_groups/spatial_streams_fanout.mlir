// RUN: ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-policy=none spatial-groups=0,0,0,0,0,0,0,1,0,1,0,1,0,2,1,0,1,0,0,3,1,1,1,1 spatial-streams=1,0,2,2,0,2,3,0,2,3,1,2" --mlir-print-ir-after=d2m-materialize-spatial-streams --mlir-print-local-scope --mlir-disable-threading %s -o /dev/null 2>&1 | FileCheck %s
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-policy=none spatial-groups=0,0,0,0,0,0,0,1,0,1,0,1,0,2,1,0,1,0,0,3,1,1,1,1 spatial-streams=1,0,2,2,0,2,3,0,2,3,1,2" --mlir-print-ir-after=d2m-lower-spatial-edges --mlir-print-local-scope --mlir-disable-threading %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=LOWER

// A diamond of 1x1 regions, 256x256: tanh #0 on (0,0) feeds exp #1 on (0,1)
// and neg #2 on (1,0) (fan-out); add #3 on (1,1) reads both. Every input asks
// to stream.
//
// The fan-out streams: one ring per consumer region. exp -> add and
// neg -> add could stream too, but each would leave the other as a
// whole-tensor edge inside the same streaming tree (add would wait for the
// end of one producer, which waits for add to drain the other), so both keep
// the whole-tensor handoff.

// CHECK-DAG: remark: edge 0:0 -> 1:0 streams through a ring of 2 blocks of 1x4 tiles in the L1 of consumer core (0, 1)
// CHECK-DAG: remark: edge 0:0 -> 2:0 streams through a ring of 2 blocks of 1x4 tiles in the L1 of consumer core (1, 0)
// CHECK-DAG: remark: edge 1:0 -> 3:0 stays a whole-tensor handoff: with the streaming edges before it, a whole-tensor edge would join two regions of one streaming tree, which could deadlock
// CHECK-DAG: remark: edge 2:0 -> 3:1 stays a whole-tensor handoff: with the streaming edges before it, a whole-tensor edge would join two regions of one streaming tree, which could deadlock
// CHECK-LABEL: IR Dump After D2MMaterializeSpatialStreams
// CHECK-DAG: memref.alloc() {d2m.stream_ring = array<i64: 0, 0, 1, 0>,
// CHECK-DAG: memref.alloc() {d2m.stream_ring = array<i64: 0, 0, 2, 0>,
// CHECK: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0, stream<depth = 2>>, #d2m.spatial_edge<0:0 -> 2:0, stream<depth = 2>>, #d2m.spatial_edge<1:0 -> 3:0>, #d2m.spatial_edge<2:0 -> 3:1>]

// tanh's writer waits for both consumers' credits (2 per block), writes the
// block into exp's ring on (0,1) and the same block into neg's ring on (1,0)
// (the copy is moved by (+1, -1)), and signals both cores. It finally waits
// for 2 x 16 credits.
// LOWER-LABEL: IR Dump After D2MLowerSpatialEdges
// LOWER: %[[K:.*]] = arith.muli %{{[0-9]+}}, %c2_{{[0-9]+}} : index
// LOWER-NEXT: d2m.semaphore_wait %[[S:.*]], %[[K]] min
// LOWER-NEXT: d2m.dma_write %{{.*}}, %{{.*}}[%c0{{.*}}, %c1{{.*}}, %{{.*}}]
// LOWER: %[[Y:.*]] = arith.addi %c0{{.*}}, %c1{{.*}} : index
// LOWER: %[[X:.*]] = arith.addi %c1{{.*}}, %c-1 : index
// LOWER-NEXT: d2m.dma_write %{{.*}}, %{{.*}}[%[[Y]], %[[X]], %{{.*}}]
// LOWER-NEXT: %[[B:.*]] = d2m.null_tx : !d2m.mem_tx<write>
// LOWER-NEXT: d2m.dma_wait %[[B]]
// LOWER: d2m.semaphore_inc %[[S]], %{{.*}}, core[%c0{{.*}}, %c1{{.*}}]
// LOWER: d2m.semaphore_inc %[[S]], %{{.*}}, core[%c1{{.*}}, %c0{{.*}}]
// LOWER: d2m.semaphore_wait %[[S]], %c32 min
func.func @diamond(%x: tensor<256x256xbf16>) -> tensor<256x256xbf16> {
  %t = "ttir.tanh"(%x) : (tensor<256x256xbf16>) -> tensor<256x256xbf16>
  %e = "ttir.exp"(%t) : (tensor<256x256xbf16>) -> tensor<256x256xbf16>
  %n = "ttir.neg"(%t) : (tensor<256x256xbf16>) -> tensor<256x256xbf16>
  %a = "ttir.add"(%e, %n) : (tensor<256x256xbf16>, tensor<256x256xbf16>) -> tensor<256x256xbf16>
  return %a : tensor<256x256xbf16>
}
