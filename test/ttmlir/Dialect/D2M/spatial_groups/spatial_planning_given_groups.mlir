// RUN: ttmlir-opt --verify-diagnostics -mlir-print-op-on-diagnostic=false --d2m-fe-pipeline="execution-strategy=spatial spatial-remarks=true spatial-groups=0,0,6,0,7,1,0,1,6,6,7,7" %s | FileCheck %s --check-prefixes=CHECK,GREEDY
// RUN: ttmlir-opt -mlir-print-op-on-diagnostic=false --d2m-fe-pipeline="execution-strategy=spatial spatial-policy=none spatial-groups=0,0,6,0,7,1,0,1,6,6,7,7" %s | FileCheck %s --check-prefixes=CHECK,NONE
// RUN: not ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial spatial-groups=0,0,6,0,7" %s 2>&1 | FileCheck %s --check-prefix=ERR-COUNT
// RUN: not ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial spatial-groups=0,9,6,0,7,1" %s 2>&1 | FileCheck %s --check-prefix=ERR-INDEX
// RUN: not ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial spatial-groups=0,1,6,0,7,1,1,1,0,0,0,0" %s 2>&1 | FileCheck %s --check-prefix=ERR-TWICE

// spatial-groups puts tanh (#0) and exp (#1) in group 0 on the given ranges,
// away from where the planner would put them. With the greedy policy the
// planner leaves them alone and numbers its own group (matmul #2, add #3)
// after it: group 1, not a second group 0. With policy none only the given
// group is built.

// CHECK-LABEL: func.func @fanout
// CHECK: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>], grid_ranges = [#ttcore.core_range<(6,0), (7,1)>, #ttcore.core_range<(6,6), (7,7)>]}
// GREEDY: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>], grid_ranges = [#ttcore.core_range<(0,0), (1,1)>, #ttcore.core_range<(0,2), (1,3)>]}
// NONE-NOT: d2m.spatial
// CHECK: return
func.func @fanout(%x: tensor<64x64xbf16>, %y: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  // expected-remark @+2 {{generic #0 is in given group 0}}
  // expected-remark @+1 {{generic #0 is not grouped: it already has a group}}
  %0 = "ttir.tanh"(%x) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
  // expected-remark @+2 {{generic #1 is in given group 0}}
  // expected-remark @+1 {{generic #1 is not grouped: it already has a group}}
  %1 = "ttir.exp"(%0) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
  // expected-remark @+1 {{spatial group 1: generics #2 #3, 1 edges, 8 cores, 6 CBs, 5 semaphores per core at most}}
  %2 = "ttir.matmul"(%0, %y) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
  %3 = "ttir.add"(%1, %2) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
  return %3 : tensor<64x64xbf16>
}

// ERR-COUNT: error: groups takes (group, generic, y0, x0, y1, x1) per generic, got 5 values
// ERR-INDEX: error: groups names generic #9, but the module has only 4 generics
// ERR-TWICE: error: groups names generic #1 more than once
