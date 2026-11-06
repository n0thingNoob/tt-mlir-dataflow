// RUN: ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-policy=none spatial-fuse-elementwise=true spatial-groups=0,0,0,0,0,0,0,1,0,1,0,1,0,2,0,2,0,2,0,3,0,3,0,3,0,4,1,0,1,0 spatial-streams=4,0,2,4,1,2,4,2,2,4,3,2 spatial-stream-block=1,1" --mlir-print-ir-after=d2m-materialize-spatial-streams --mlir-print-local-scope --mlir-disable-threading %s -o /dev/null 2>&1 | FileCheck %s
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-policy=none spatial-fuse-elementwise=true spatial-groups=0,0,0,0,0,0,0,1,0,1,0,1,0,2,0,2,0,2,0,3,0,3,0,3,0,4,1,0,1,0 spatial-streams=4,0,2,4,1,2,4,2,2,4,3,2 spatial-stream-block=1,1" --mlir-print-ir-after=d2m-lower-spatial-edges --mlir-print-local-scope --mlir-disable-threading %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=LOWER

// A join: four matmuls #0..#3 on (0,0)..(0,3) stream their 1x8-tile outputs,
// as blocks of one tile, into a fused four-input add #4 (the hub) on (1,0).
// Each edge has its own ring on the hub core and its own semaphore; the four
// producers are independent, so the streaming edges form a tree and all stay.

// CHECK-COUNT-4: streams through a ring of 2 blocks of 1x1 tiles in the L1 of consumer core (1, 0)
// CHECK-LABEL: IR Dump After D2MMaterializeSpatialStreams
// CHECK-DAG: memref.alloc() {d2m.stream_ring = array<i64: 0, 0, 4, 0>,
// CHECK-DAG: memref.alloc() {d2m.stream_ring = array<i64: 1, 0, 4, 1>,
// CHECK-DAG: memref.alloc() {d2m.stream_ring = array<i64: 2, 0, 4, 2>,
// CHECK-DAG: memref.alloc() {d2m.stream_ring = array<i64: 3, 0, 4, 3>,

// The hub waits for block n of each input on that input's semaphore and
// returns the credit to that input's producer, (-1, k) in its virtual
// coordinates, which is core (0, k).
// LOWER-LABEL: IR Dump After D2MLowerSpatialEdges
// LOWER: %[[S0:.*]] = d2m.create_local_semaphore
// LOWER: %[[S1:.*]] = d2m.create_local_semaphore
// LOWER: %[[S2:.*]] = d2m.create_local_semaphore
// LOWER: %[[S3:.*]] = d2m.create_local_semaphore
// LOWER-DAG: d2m.semaphore_inc %[[S0]], %{{.*}}, core[%c-1{{.*}}, %c0{{.*}}]
// LOWER-DAG: d2m.semaphore_inc %[[S1]], %{{.*}}, core[%c-1{{.*}}, %c1{{.*}}]
// LOWER-DAG: d2m.semaphore_inc %[[S2]], %{{.*}}, core[%c-1{{.*}}, %c2{{.*}}]
// LOWER-DAG: d2m.semaphore_inc %[[S3]], %{{.*}}, core[%c-1{{.*}}, %c3{{.*}}]
func.func @hub4(%x: tensor<32x256xbf16>, %w0: tensor<256x256xbf16>, %w1: tensor<256x256xbf16>, %w2: tensor<256x256xbf16>, %w3: tensor<256x256xbf16>) -> tensor<32x256xbf16> {
  %0 = "ttir.matmul"(%x, %w0) <{transpose_a = false, transpose_b = false}> : (tensor<32x256xbf16>, tensor<256x256xbf16>) -> tensor<32x256xbf16>
  %1 = "ttir.matmul"(%x, %w1) <{transpose_a = false, transpose_b = false}> : (tensor<32x256xbf16>, tensor<256x256xbf16>) -> tensor<32x256xbf16>
  %2 = "ttir.add"(%0, %1) : (tensor<32x256xbf16>, tensor<32x256xbf16>) -> tensor<32x256xbf16>
  %3 = "ttir.matmul"(%x, %w2) <{transpose_a = false, transpose_b = false}> : (tensor<32x256xbf16>, tensor<256x256xbf16>) -> tensor<32x256xbf16>
  %4 = "ttir.add"(%2, %3) : (tensor<32x256xbf16>, tensor<32x256xbf16>) -> tensor<32x256xbf16>
  %5 = "ttir.matmul"(%x, %w3) <{transpose_a = false, transpose_b = false}> : (tensor<32x256xbf16>, tensor<256x256xbf16>) -> tensor<32x256xbf16>
  %6 = "ttir.add"(%4, %5) : (tensor<32x256xbf16>, tensor<32x256xbf16>) -> tensor<32x256xbf16>
  return %6 : tensor<32x256xbf16>
}
