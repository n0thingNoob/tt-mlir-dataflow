// RUN: ttmlir-opt --split-input-file --verify-diagnostics -mlir-print-op-on-diagnostic=false --ttcore-register-device --ttir-to-d2m --d2m-spatial-planning="policy=greedy exclude-cores=2,2 max-cbs=2 remarks=true" %s -o /dev/null
// RUN: ttmlir-opt --split-input-file --verify-diagnostics -mlir-print-op-on-diagnostic=false --ttcore-register-device --ttir-to-d2m --d2m-spatial-planning="policy=greedy exclude-cores=2,2 max-cbs=2 remarks=true" %s | FileCheck %s --check-prefix=CB

// The CB budget is per core: each region's CBs are placed on its own cores
// and every core runs one region. A chain of 64x64 unary ops (each a 2x2
// generic with 2 CBs) fits max-cbs=2 as one group although the spatial uses
// 8 CBs in total; the add after it needs 3 CBs on its cores and is cut.
// CB-LABEL: func.func @chain
// CB: d2m.generic {block_factors = [1, 1], d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(0,0), (1,1)>
// CB: d2m.generic {block_factors = [1, 1], d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(0,2), (1,3)>
// CB: d2m.generic {block_factors = [1, 1], d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(0,4), (1,5)>
// CB: d2m.generic {block_factors = [1, 1], d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(0,6), (1,7)>
// CB-NOT: d2m.spatial_group
func.func @chain(%x: tensor<64x64xf32>, %y: tensor<64x64xf32>) -> tensor<64x64xf32> {
  // expected-remark @+1 {{spatial group 0: generics #0 #1 #2 #3, 3 edges, 16 cores, 8 CBs, 2 semaphores per core at most}}
  %0 = "ttir.tanh"(%x) : (tensor<64x64xf32>) -> tensor<64x64xf32>
  %1 = "ttir.exp"(%0) : (tensor<64x64xf32>) -> tensor<64x64xf32>
  %2 = "ttir.tanh"(%1) : (tensor<64x64xf32>) -> tensor<64x64xf32>
  %3 = "ttir.exp"(%2) : (tensor<64x64xf32>) -> tensor<64x64xf32>
  // expected-remark @+1 {{generic #4 starts a new group: a core would need 3 CBs (limit 2)}}
  %4 = "ttir.add"(%3, %y) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
  return %4 : tensor<64x64xf32>
}
