// RUN: ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-policy=none spatial-groups=0,0,0,0,0,0,0,1,1,1,1,1,0,2,0,2,0,2 spatial-streams=1,0,2,2,0,2" --mlir-print-ir-after=d2m-materialize-spatial-streams --mlir-print-local-scope --mlir-disable-threading %s -o /dev/null 2>&1 | FileCheck %s
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-policy=none spatial-groups=0,0,0,0,0,0,0,1,1,1,1,1,0,2,0,2,0,2 spatial-streams=1,0,2,2,0,2" --mlir-print-ir-after=d2m-lower-spatial-edges --mlir-print-local-scope --mlir-disable-threading %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=LOWER

// Streaming edges between 1x1 regions: tanh #0 on (0,0), exp #1 on (1,1),
// add #2 on (0,2), 256x256 bf16. ReblockGenerics gives each generic 16 blocks
// of 1x4 tiles, written and read row-major, so both edges stream. Each gets a
// ring of 2 blocks (depth 2) in its consumer's L1; the producer writes into it
// remotely.

// CHECK-DAG: remark: edge 0:0 -> 1:0 streams through a ring of 2 blocks of 1x4 tiles in the L1 of consumer core (1, 1)
// CHECK-DAG: remark: edge 1:0 -> 2:0 streams through a ring of 2 blocks of 1x4 tiles in the L1 of consumer core (0, 2)
// CHECK-LABEL: IR Dump After D2MMaterializeSpatialStreams
// The rings replace the edge buffers: 2 x (1x4) tiles on the consumer's core.
// CHECK: %[[R0:.*]] = memref.alloc() {d2m.stream_ring, d2m.virtualGridForwardMapping = affine_map<(d0, d1, d2, d3) -> (d0 + 1, d1 + 1, d2, d3)>, d2m.virtualGridInverseMapping = affine_map<(d0, d1) -> (0, d0 - 1, d1 - 1)>} : memref<1x1x2x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<8192x2048, 1>, #ttcore.memory_space<l1>>
// CHECK: %[[R1:.*]] = memref.alloc() {d2m.stream_ring, d2m.virtualGridForwardMapping = affine_map<(d0, d1, d2, d3) -> (d0, d1 + 2, d2, d3)>, d2m.virtualGridInverseMapping = affine_map<(d0, d1) -> (0, d0, d1 - 2)>} : memref<1x1x2x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<8192x2048, 1>, #ttcore.memory_space<l1>>
// CHECK: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0, stream<depth = 2>>, #d2m.spatial_edge<1:0 -> 2:0, stream<depth = 2>>]
// CHECK-NEXT: ins(
// CHECK-NEXT: outs(%[[R0]], %[[R1]], %{{.*}} :
// Block (d0, d1) of the 8x2 block grid is block 2 * d0 + d1, in slot (2 * d0 + d1) mod 2.
// CHECK: d2m.view_layout %[[R0]] remapping = affine_map<(d0, d1, d2, d3) -> (0, 0, d1 mod 2 + d2, d3)>

// The producer's writer waits for a free slot (credits >= n - 1), writes the
// block to the consumer's core, waits for the write and bumps the consumer's
// count; it finally waits for all 16 credits. The consumer's reader waits for
// block n (count >= n + 1), reads it from its own core and returns a credit to
// the producer's core, given in its own virtual coordinates.
// LOWER-LABEL: IR Dump After D2MLowerSpatialEdges
// LOWER: %[[S:.*]] = d2m.create_local_semaphore
// LOWER: d2m.generic
// LOWER: arith.maxsi
// LOWER-NEXT: d2m.semaphore_wait %{{.*}}, %{{.*}} min
// LOWER-NEXT: %[[TX:.*]] = d2m.dma_write %{{.*}}, %{{.*}}[%c1{{.*}}, %c1{{.*}}, %{{.*}}]
// LOWER-NEXT: %[[B:.*]] = d2m.null_tx : !d2m.mem_tx<write>
// LOWER-NEXT: d2m.dma_wait %[[B]] : !d2m.mem_tx<write>
// LOWER: d2m.semaphore_inc %{{.*}}, %c1{{.*}}, core[%c1{{.*}}, %c1{{.*}}]
// LOWER: d2m.semaphore_wait %{{.*}}, %c16 min
// LOWER: d2m.semaphore_wait %{{.*}}, %{{.*}} min
// LOWER-NEXT: d2m.dma_read %{{.*}}[%c1{{.*}}, %c1{{.*}}, %{{.*}}]
// LOWER-NEXT: %[[RB:.*]] = d2m.null_tx : !d2m.mem_tx<read>
// LOWER-NEXT: d2m.dma_wait %[[RB]] : !d2m.mem_tx<read>
// LOWER: d2m.semaphore_inc %{{.*}}, %c1{{.*}}, core[%c-1, %c-1{{.*}}]
func.func @chain(%x: tensor<256x256xbf16>, %z: tensor<256x256xbf16>) -> tensor<256x256xbf16> {
  %t = "ttir.tanh"(%x) : (tensor<256x256xbf16>) -> tensor<256x256xbf16>
  %e = "ttir.exp"(%t) : (tensor<256x256xbf16>) -> tensor<256x256xbf16>
  %a = "ttir.add"(%e, %z) : (tensor<256x256xbf16>, tensor<256x256xbf16>) -> tensor<256x256xbf16>
  return %a : tensor<256x256xbf16>
}
