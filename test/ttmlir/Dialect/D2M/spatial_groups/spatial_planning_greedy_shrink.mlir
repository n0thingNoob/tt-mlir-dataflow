// RUN: ttmlir-opt --split-input-file --verify-diagnostics -mlir-print-op-on-diagnostic=false --d2m-fe-pipeline="execution-strategy=spatial spatial-remarks=true" %s | FileCheck %s

// Groups that only fit when the whole group takes a smaller GridAnalysis
// target. Every member's region is the grid GridAnalysis picks for that
// target, GridSelection picks exactly that grid in the region, and the edges
// between members (eltwise to eltwise, eltwise to matmul, join) keep one
// layout at both ends.

// Three 8x8 generics: an 8x8 range always contains the excluded core (2,2).
// The first member alone fits at target 8x4; with the second member the group
// takes target 4x4 (8x4, 4x8 do not fit twice); the third fits there too.
// CHECK-LABEL: func.func @chain8x8
// CHECK: d2m.spatial {{.*}}grid_ranges = [#ttcore.core_range<(0,3), (3,6)>, #ttcore.core_range<(4,0), (7,3)>, #ttcore.core_range<(4,4), (7,7)>]
// CHECK: d2m.generic {{.*}}grid = #ttcore.grid<4x4{{[,>]}}
// CHECK: }, {
// CHECK: d2m.generic {{.*}}grid = #ttcore.grid<4x4{{[,>]}}
// CHECK: }, {
// CHECK: d2m.generic {{.*}}grid = #ttcore.grid<4x4{{[,>]}}
func.func @chain8x8(%x: tensor<256x256xbf16>) -> tensor<256x256xbf16> {
  // expected-remark @+2 {{spatial group 0 shrinks to target 4x4: #0 8x8->4x4 #1 8x8->4x4 #2 8x8->4x4}}
  // expected-remark @+1 {{spatial group 0: generics #0 #1 #2, 2 edges, 48 cores, 6 CBs, 2 semaphores per core at most}}
  %0 = "ttir.exp"(%x) : (tensor<256x256xbf16>) -> tensor<256x256xbf16>
  %1 = "ttir.tanh"(%0) : (tensor<256x256xbf16>) -> tensor<256x256xbf16>
  %2 = "ttir.exp"(%1) : (tensor<256x256xbf16>) -> tensor<256x256xbf16>
  return %2 : tensor<256x256xbf16>
}

// -----

// Fan-out and join on 4x8 grids: the second 4x8 does not fit, so all four
// members are placed again at target 8x2 (4x2 grids). The tanh output feeds
// an eltwise and a matmul consumer; the join reads both.
// CHECK-LABEL: func.func @fanout4x8
// CHECK: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>, #d2m.spatial_edge<0:0 -> 2:0>, #d2m.spatial_edge<1:0 -> 3:0>, #d2m.spatial_edge<2:0 -> 3:1>], grid_ranges = [#ttcore.core_range<(0,0), (3,1)>, #ttcore.core_range<(0,3), (3,4)>, #ttcore.core_range<(0,5), (3,6)>, #ttcore.core_range<(4,0), (7,1)>]
// CHECK: d2m.generic {{.*}}grid = #ttcore.grid<4x2{{[,>]}}
// CHECK: }, {
// CHECK: d2m.generic {{.*}}grid = #ttcore.grid<4x2{{[,>]}}
// CHECK: }, {
// CHECK: d2m.generic {{.*}}grid = #ttcore.grid<4x2{{[,>]}}
// CHECK: }, {
// CHECK: d2m.generic {{.*}}grid = #ttcore.grid<4x2{{[,>]}}
func.func @fanout4x8(%x: tensor<128x256xbf16>, %y: tensor<256x256xbf16>) -> tensor<128x256xbf16> {
  // expected-remark @+2 {{spatial group 0 shrinks to target 8x2: #0 4x8->4x2 #1 4x8->4x2 #2 4x8->4x2 #3 4x8->4x2}}
  // expected-remark @+1 {{spatial group 0: generics #0 #1 #2 #3, 4 edges, 32 cores, 10 CBs, 6 semaphores per core at most}}
  %0 = "ttir.tanh"(%x) : (tensor<128x256xbf16>) -> tensor<128x256xbf16>
  %1 = "ttir.exp"(%0) : (tensor<128x256xbf16>) -> tensor<128x256xbf16>
  %2 = "ttir.matmul"(%0, %y) : (tensor<128x256xbf16>, tensor<256x256xbf16>) -> tensor<128x256xbf16>
  %3 = "ttir.add"(%1, %2) : (tensor<128x256xbf16>, tensor<128x256xbf16>) -> tensor<128x256xbf16>
  return %3 : tensor<128x256xbf16>
}
