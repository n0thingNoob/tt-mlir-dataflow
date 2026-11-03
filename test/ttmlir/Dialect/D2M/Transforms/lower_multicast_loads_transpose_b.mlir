// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline --mlir-print-ir-after=d2m-lower-multicast-loads --mlir-print-local-scope --mlir-disable-threading %s -o /dev/null 2>&1 | FileCheck %s

// matmul with transpose_b: B's indexing map is (d1, d2) instead of (d2, d1),
// so B's parallel dim d1 is result 0 of B's map but result 1 of the output
// map. A load is multicast along the compute grid dim the output does not
// fix, so the unit-group check must use the output position.

// One tile row (grid 1x8): B would go to a group of one core, so it stays a
// unicast load; A is multicast along the row.
// CHECK-LABEL: func.func @transpose_b_one_row
// CHECK: d2m.remote_load {{.*}} mcore[
// CHECK-NOT: mcore[
// CHECK-LABEL: func.func @transpose_b_two_rows
func.func @transpose_b_one_row(%x: tensor<32x256xbf16>, %w: tensor<128x256xbf16>) -> tensor<32x128xbf16> {
  %0 = "ttir.matmul"(%x, %w) <{transpose_a = false, transpose_b = true}> : (tensor<32x256xbf16>, tensor<128x256xbf16>) -> tensor<32x128xbf16>
  return %0 : tensor<32x128xbf16>
}

// Two tile rows (grid 2x8): both rows need the same B tiles, so B is
// multicast down each column, A along each row.
// CHECK-COUNT-2: d2m.remote_load {{.*}} mcore[
// CHECK-NOT: mcore[
func.func @transpose_b_two_rows(%x: tensor<64x256xbf16>, %w: tensor<128x256xbf16>) -> tensor<64x128xbf16> {
  %0 = "ttir.matmul"(%x, %w) <{transpose_a = false, transpose_b = true}> : (tensor<64x256xbf16>, tensor<128x256xbf16>) -> tensor<64x128xbf16>
  return %0 : tensor<64x128xbf16>
}
