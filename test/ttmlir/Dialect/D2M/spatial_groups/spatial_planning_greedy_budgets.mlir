// RUN: ttmlir-opt --split-input-file --verify-diagnostics -mlir-print-op-on-diagnostic=false --ttcore-register-device --ttir-to-d2m --d2m-spatial-planning="policy=greedy exclude-cores=2,2 max-cbs=5 remarks=true" %s -o /dev/null
// RUN: ttmlir-opt --split-input-file --verify-diagnostics -mlir-print-op-on-diagnostic=false --ttcore-register-device --ttir-to-d2m --d2m-spatial-planning="policy=greedy exclude-cores=2,2 max-cbs=5 remarks=true" %s | FileCheck %s --check-prefix=CB

// Budgets of the greedy planner on a chain of 64x64 unary ops (each a 2x2
// generic with 2 CBs). max-cbs=5: two members use 4 CBs, a third would need 6.
// CB-LABEL: func.func @chain
// CB: d2m.generic {block_factors = [1, 1], d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(0,0), (1,1)>
// CB: d2m.generic {block_factors = [1, 1], d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(0,2), (1,3)>
// CB: d2m.generic {block_factors = [1, 1], d2m.spatial_group = 1 : i64, d2m.spatial_range = #ttcore.core_range<(0,0), (1,1)>
// CB: d2m.generic {block_factors = [1, 1], d2m.spatial_group = 1 : i64, d2m.spatial_range = #ttcore.core_range<(0,2), (1,3)>
func.func @chain(%x: tensor<64x64xf32>) -> tensor<64x64xf32> {
  // expected-remark @+1 {{spatial group 0: generics #0 #1, 1 edges, 8 cores, 4 CBs, 1 semaphores per core at most}}
  %0 = "ttir.tanh"(%x) : (tensor<64x64xf32>) -> tensor<64x64xf32>
  %1 = "ttir.exp"(%0) : (tensor<64x64xf32>) -> tensor<64x64xf32>
  // expected-remark @+2 {{generic #2 starts a new group: the spatial would need 6 CBs (limit 5)}}
  // expected-remark @+1 {{spatial group 1: generics #2 #3, 1 edges, 8 cores, 4 CBs, 1 semaphores per core at most}}
  %2 = "ttir.tanh"(%1) : (tensor<64x64xf32>) -> tensor<64x64xf32>
  %3 = "ttir.exp"(%2) : (tensor<64x64xf32>) -> tensor<64x64xf32>
  return %3 : tensor<64x64xf32>
}
