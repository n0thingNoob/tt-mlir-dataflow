// RUN: ttmlir-opt --ttcore-register-device --d2m-lower-to-layout %s | FileCheck %s

// A scalar DRAM view tilized into a DRAM buffer with the same placement is one
// tilize generic that reads through the view; no L1 buffer and no rebuffer.

// CHECK: #[[L1:layout[0-9]*]] = #ttcore.metal_layout<logical_shape = 64x64, {{.*}}, l1, sharded>

#ci = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>
#whole_rows = #ttcore.metal_layout<logical_shape = 128x64, dim_alignments = 32x32, collapsed_intervals = #ci, dram, sharded>
#whole_cols = #ttcore.metal_layout<logical_shape = 64x128, dim_alignments = 32x32, collapsed_intervals = #ci, dram, sharded>
#slice = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = #ci, dram, sharded>

// CHECK-LABEL: func.func @row_slice
func.func @row_slice(%arg0: tensor<128x64xbf16>) -> tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #slice> {
  // CHECK: %[[STAGE:.*]] = d2m.to_device %arg0
  // CHECK: %[[VIEW:.*]] = d2m.view_layout %[[STAGE]]
  // CHECK: %[[OUT:.*]] = d2m.empty() {d2m.dram_bank = 3 : i64}
  // CHECK-NOT: d2m.empty
  // CHECK: %[[RES:.*]] = d2m.generic
  // CHECK-NEXT: ins(%[[VIEW]] :
  // CHECK-NEXT: outs(%[[OUT]] :
  // CHECK: d2m.remote_load {{.*}} %[[VIEW]]
  // CHECK: d2m.tile_tilize_block
  // CHECK: d2m.remote_store %[[OUT]]
  // CHECK-NOT: d2m.generic
  // CHECK: return %[[RES]]
  %0 = d2m.empty() : tensor<1x1x128x64xbf16, #whole_rows>
  %1 = d2m.to_layout %arg0, %0 : tensor<128x64xbf16> into tensor<1x1x128x64xbf16, #whole_rows> -> tensor<1x1x128x64xbf16, #whole_rows>
  %v = d2m.view_layout %1 remapping = affine_map<(d0, d1, d2, d3) -> (0, 0, d2 + 64, d3)> : tensor<1x1x128x64xbf16, #whole_rows> -> tensor<1x1x64x64xbf16, #slice>
  %2 = d2m.empty() {d2m.dram_bank = 3 : i64} : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #slice>
  %3 = d2m.to_layout %v, %2 : tensor<1x1x64x64xbf16, #slice> into tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #slice> -> tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #slice>
  return %3 : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #slice>
}

// CHECK-LABEL: func.func @column_slice
func.func @column_slice(%arg0: tensor<64x128xbf16>) -> tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #slice> {
  // CHECK: %[[STAGE:.*]] = d2m.to_device %arg0
  // CHECK: %[[VIEW:.*]] = d2m.view_layout %[[STAGE]] remapping = #{{.*}} :
  // CHECK: %[[OUT:.*]] = d2m.empty() {d2m.dram_bank = 5 : i64}
  // CHECK-NOT: d2m.empty
  // CHECK: %[[RES:.*]] = d2m.generic
  // CHECK-NEXT: ins(%[[VIEW]] :
  // CHECK-NEXT: outs(%[[OUT]] :
  // CHECK: d2m.tile_tilize_block
  // CHECK-NOT: d2m.generic
  // CHECK: return %[[RES]]
  %0 = d2m.empty() : tensor<1x1x64x128xbf16, #whole_cols>
  %1 = d2m.to_layout %arg0, %0 : tensor<64x128xbf16> into tensor<1x1x64x128xbf16, #whole_cols> -> tensor<1x1x64x128xbf16, #whole_cols>
  %v = d2m.view_layout %1 remapping = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3 + 64)> : tensor<1x1x64x128xbf16, #whole_cols> -> tensor<1x1x64x64xbf16, #slice>
  %2 = d2m.empty() {d2m.dram_bank = 5 : i64} : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #slice>
  %3 = d2m.to_layout %v, %2 : tensor<1x1x64x64xbf16, #slice> into tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #slice> -> tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #slice>
  return %3 : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #slice>
}

// A grid change is not a pure format conversion: the view still bounces
// through L1.
// CHECK-LABEL: func.func @grid_change_bounces
func.func @grid_change_bounces(%arg0: tensor<128x64xbf16>) -> tensor<1x2x2x1x!ttcore.tile<32x32, bf16>, #slice> {
  // CHECK: d2m.view_layout
  // CHECK: d2m.empty() : tensor<1x1x64x64xbf16, #[[L1]]>
  // CHECK: d2m.generic
  // CHECK: d2m.generic
  // CHECK: d2m.tile_tilize_block
  // CHECK: d2m.generic
  %0 = d2m.empty() : tensor<1x1x128x64xbf16, #whole_rows>
  %1 = d2m.to_layout %arg0, %0 : tensor<128x64xbf16> into tensor<1x1x128x64xbf16, #whole_rows> -> tensor<1x1x128x64xbf16, #whole_rows>
  %v = d2m.view_layout %1 remapping = affine_map<(d0, d1, d2, d3) -> (0, 0, d2 + 64, d3)> : tensor<1x1x128x64xbf16, #whole_rows> -> tensor<1x1x64x64xbf16, #slice>
  %2 = d2m.empty() : tensor<1x2x2x1x!ttcore.tile<32x32, bf16>, #slice>
  %3 = d2m.to_layout %v, %2 : tensor<1x1x64x64xbf16, #slice> into tensor<1x2x2x1x!ttcore.tile<32x32, bf16>, #slice> -> tensor<1x2x2x1x!ttcore.tile<32x32, bf16>, #slice>
  return %3 : tensor<1x2x2x1x!ttcore.tile<32x32, bf16>, #slice>
}
