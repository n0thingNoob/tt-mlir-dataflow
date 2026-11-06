// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m="dram-args-and-results=true" --canonicalize --mlir-print-local-scope %s | FileCheck %s
// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m="default-input-memspace=dram" --canonicalize --mlir-print-local-scope %s | FileCheck %s --check-prefix=INDRAM

// dram-args-and-results: function arguments and returned results in DRAM,
// every other tensor in L1.

// CHECK-LABEL: func.func @chain
// CHECK: d2m.to_layout %arg0, %{{.*}} : tensor<64x128xbf16> into tensor<{{.*}}logical_shape = 64x128,{{.*}}, dram, sharded>>
// CHECK: d2m.to_layout %arg1, %{{.*}} : tensor<128x128xbf16> into tensor<{{.*}}logical_shape = 128x128,{{.*}}, dram, sharded>>
// The matmul result is an intermediate: L1.
// CHECK: d2m.generic
// CHECK: outs(%{{.*}} : tensor<1x1x2x4x!ttcore.tile<32x32, bf16>, {{.*}}, l1, sharded>>)
// CHECK: d2m.tile_matmul
// The exp result is returned: DRAM.
// CHECK: d2m.generic
// CHECK: outs(%{{.*}} : tensor<1x1x2x4x!ttcore.tile<32x32, bf16>, {{.*}}, dram, sharded>>)
// CHECK: d2m.tile_exp
// CHECK: return
func.func @chain(%x: tensor<64x128xbf16>, %w: tensor<128x128xbf16>) -> tensor<64x128xbf16> {
  %0 = "ttir.matmul"(%x, %w) : (tensor<64x128xbf16>, tensor<128x128xbf16>) -> tensor<64x128xbf16>
  %1 = "ttir.exp"(%0) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
  return %1 : tensor<64x128xbf16>
}

// A view of an argument keeps the argument's memory space (a view cannot
// change it); the add reads it through a copy into L1. With only the inputs
// in DRAM (INDRAM) the view used to take the output's memory space and fail
// the verifier ("view cannot change memory space").
// CHECK-LABEL: func.func @reshape_arg
// CHECK: %[[A:.*]] = d2m.to_layout %arg0, %{{.*}} : tensor<128xbf16> into tensor<{{.*}}, dram, sharded>>
// CHECK: %[[V:.*]] = d2m.view_layout %[[A]] {{.*}} -> tensor<1x1x32x128xbf16, {{.*}}logical_shape = 1x128,{{.*}}, dram, sharded>>
// CHECK: d2m.to_layout %arg1, %{{.*}} : tensor<64x128xbf16> into tensor<{{.*}}, dram, sharded>>
// CHECK: d2m.to_layout %[[V]], %{{.*}} into tensor<{{.*}}, l1, sharded>>
// CHECK: d2m.generic
// CHECK: outs(%{{.*}} : tensor<{{.*}}, dram, sharded>>)
// CHECK: return
// INDRAM-LABEL: func.func @reshape_arg
// INDRAM: d2m.view_layout {{.*}} -> tensor<1x1x32x128xbf16, {{.*}}logical_shape = 1x128,{{.*}}, dram, sharded>>
// INDRAM: return
func.func @reshape_arg(%x: tensor<128xbf16>, %y: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
  %0 = "ttir.reshape"(%x) <{shape = [1 : i32, 128 : i32]}> : (tensor<128xbf16>) -> tensor<1x128xbf16>
  %1 = "ttir.add"(%y, %0) : (tensor<64x128xbf16>, tensor<1x128xbf16>) -> tensor<64x128xbf16>
  return %1 : tensor<64x128xbf16>
}
