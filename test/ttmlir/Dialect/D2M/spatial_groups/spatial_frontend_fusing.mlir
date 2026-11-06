// RUN: ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-fuse-elementwise=true spatial-policy=bank-load spatial-remarks=true" --mlir-print-local-scope %s -o /dev/null 2>&1 | FileCheck %s
// RUN: ttmlir-opt --d2m-fe-pipeline="dram-args-and-results=true" --mlir-print-local-scope %s | FileCheck %s --check-prefix=TEMPORAL
// RUN: ttmlir-opt --d2m-fe-pipeline="dram-args-and-results=true frontend-fusing=false" --mlir-print-local-scope %s | FileCheck %s --check-prefix=UNFUSED

// A gated MLP as tt-xla exports nn.Linear and silu from PyTorch: each weight is
// stored [out, in] and transposed before a dot_general, and silu is
// sigmoid(g) * g. The frontend folds the transposes into transpose_b and the
// sigmoid into silu (ttir-fusing d2m-patterns-only), so the spatial planner
// sees the same four generics as for
// matmul + silu: load (7,3), gate (6,3), up (7,2), silu * u (5,3), down (7,4)
// (spatial_planning_bank.mlir has the derivation).
// CHECK: remark: policy=bank-load gives groups=0,0,6,3,6,3,0,1,7,2,7,2,0,2,5,3,5,3,0,3,7,4,7,4 loads=0,1,7,3,7,3,1,1,7,3,7,3

// The temporal strategy lowers the same folded TTIR: no transpose generics,
// one silu instead of sigmoid and a multiply.
// TEMPORAL-NOT: d2m.tile_transpose
// TEMPORAL-NOT: d2m.tile_sigmoid
// TEMPORAL: d2m.tile_silu
// TEMPORAL-NOT: d2m.tile_transpose
// TEMPORAL-NOT: d2m.tile_sigmoid

// frontend-fusing=false lowers the TTIR as written: the transposes stay
// generics of their own.
// UNFUSED-COUNT-3: d2m.tile_transpose

func.func @glu_xla(%wd: tensor<256x128xbf16> {d2m.dram_bank = 0 : i64},
                   %wu: tensor<128x256xbf16> {d2m.dram_bank = 0 : i64},
                   %x: tensor<32x256xbf16>,
                   %wg: tensor<128x256xbf16> {d2m.dram_bank = 0 : i64}) -> tensor<32x256xbf16> {
  %0 = "ttir.permute"(%wg) <{permutation = array<i64: 1, 0>}> : (tensor<128x256xbf16>) -> tensor<256x128xbf16>
  %1 = "ttir.dot_general"(%x, %0) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x256xbf16>, tensor<256x128xbf16>) -> tensor<32x128xbf16>
  %2 = "ttir.sigmoid"(%1) : (tensor<32x128xbf16>) -> tensor<32x128xbf16>
  %3 = "ttir.multiply"(%1, %2) : (tensor<32x128xbf16>, tensor<32x128xbf16>) -> tensor<32x128xbf16>
  %4 = "ttir.permute"(%wu) <{permutation = array<i64: 1, 0>}> : (tensor<128x256xbf16>) -> tensor<256x128xbf16>
  %5 = "ttir.dot_general"(%x, %4) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x256xbf16>, tensor<256x128xbf16>) -> tensor<32x128xbf16>
  %6 = "ttir.multiply"(%3, %5) : (tensor<32x128xbf16>, tensor<32x128xbf16>) -> tensor<32x128xbf16>
  %7 = "ttir.permute"(%wd) <{permutation = array<i64: 1, 0>}> : (tensor<256x128xbf16>) -> tensor<128x256xbf16>
  %8 = "ttir.dot_general"(%6, %7) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x128xbf16>, tensor<128x256xbf16>) -> tensor<32x256xbf16>
  return %8 : tensor<32x256xbf16>
}
