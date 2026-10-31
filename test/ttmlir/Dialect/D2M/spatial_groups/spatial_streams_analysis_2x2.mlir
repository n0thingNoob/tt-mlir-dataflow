// RUN: ttmlir-opt --split-input-file --d2m-fe-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-policy=none spatial-groups=0,0,0,0,1,1,0,1,0,2,1,3 spatial-streams=1,0,2" --mlir-print-ir-after=d2m-materialize-spatial-streams --mlir-print-local-scope --mlir-disable-threading %s -o /dev/null 2>&1 | FileCheck %s

// d2m-materialize-spatial-streams on two 2x2 regions, the producer at
// (0,0)-(1,1) and the consumer at (0,2)-(1,3). First tanh -> exp, where exp
// core (y, x) reads only the shard of tanh core (y, x): four core pairs.

// 1024x1024: ReblockGenerics gives both generics [8, 8], so every core
// writes and reads 64 blocks of 2x2 tiles in the same row-major order.
// CHECK: remark: edge 0:0 -> 1:0 can stream: 4 core pairs, 64 blocks of 2x2 tiles each, in the producer's order
// CHECK-LABEL: IR Dump After D2MMaterializeSpatialStreams
// CHECK: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0, stream<depth = 2>>]
func.func @chain_1024(%x: tensor<1024x1024xbf16>) -> tensor<1024x1024xbf16> {
  %t = "ttir.tanh"(%x) : (tensor<1024x1024xbf16>) -> tensor<1024x1024xbf16>
  %e = "ttir.exp"(%t) : (tensor<1024x1024xbf16>) -> tensor<1024x1024xbf16>
  return %e : tensor<1024x1024xbf16>
}

// -----

// 128x128: [1, 1], one 2x2 block per core, so there is nothing to stream.
// CHECK: remark: edge 0:0 -> 1:0 stays a whole-tensor handoff: one block per core pair, nothing to stream
// CHECK-LABEL: IR Dump After D2MMaterializeSpatialStreams
// CHECK: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>]
func.func @chain_128(%x: tensor<128x128xbf16>) -> tensor<128x128xbf16> {
  %t = "ttir.tanh"(%x) : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
  %e = "ttir.exp"(%t) : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
  return %e : tensor<128x128xbf16>
}

// -----

// A 2x2 matmul consumer loads its left operand with a multicast along each
// row, so one consumer core's read feeds others: not a one-to-one core
// pairing, so the edge stays a whole-tensor handoff.
// CHECK: remark: edge 0:0 -> 1:0 stays a whole-tensor handoff: the consumer loads it with a multicast inside its region
// CHECK-LABEL: IR Dump After D2MMaterializeSpatialStreams
// CHECK: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>]
func.func @tanh_matmul(%x: tensor<64x128xbf16>, %w: tensor<128x128xbf16>) -> tensor<64x128xbf16> {
  %t = "ttir.tanh"(%x) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
  %m = "ttir.matmul"(%t, %w) <{transpose_a = false, transpose_b = false}> : (tensor<64x128xbf16>, tensor<128x128xbf16>) -> tensor<64x128xbf16>
  return %m : tensor<64x128xbf16>
}
