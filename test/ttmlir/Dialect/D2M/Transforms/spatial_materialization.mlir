// RUN: ttmlir-opt %s --ttcore-register-device -o %t.input
// RUN: ttmlir-opt %t.input --d2m-spatial-planning="materialize=true dump-regions=true" -o %t.once 2> %t.report
// RUN: FileCheck %s --input-file=%t.report
// RUN: FileCheck %s --check-prefix=REJECT --input-file=%t.report
// RUN: ttmlir-opt %t.once --d2m-spatial-planning="materialize=true" -o %t.twice
// RUN: diff %t.once %t.twice
// RUN: ttmlir-opt %t.input --d2m-spatial-planning="materialize=true dump-regions=true" -o %t.repeat 2> %t.repeat-report
// RUN: diff %t.report %t.repeat-report
// RUN: diff %t.once %t.repeat
// RUN: not ttmlir-opt %s --d2m-spatial-planning="materialize=true" 2>&1 | FileCheck %s --check-prefix=MISSING
// MISSING: spatial materialization requires a registered device

// RUN: sed 's/workerGrid = #ttcore.grid<8x8,/workerGrid = #ttcore.grid<2x1,/' %t.input > %t.vertical
// RUN: ttmlir-opt %t.vertical --d2m-spatial-planning="materialize=true dump-regions=true" -o /dev/null 2> %t.vertical-report
// RUN: FileCheck %s --check-prefix=VERTICAL --input-file=%t.vertical-report
// VERTICAL: selected S0 members=[G0, G1]
// VERTICAL-SAME: ranges=[#ttcore.core_range<(0,0), (0,0)>, #ttcore.core_range<(1,0), (1,0)>]
// RUN: sed 's/workerGrid = #ttcore.grid<8x8,/workerGrid = #ttcore.grid<1x1,/' %t.input > %t.small
// RUN: ttmlir-opt %t.small --d2m-spatial-planning="materialize=true dump-regions=true" -o /dev/null 2> %t.small-report
// RUN: FileCheck %s --check-prefix=SMALL --input-file=%t.small-report --implicit-check-not='reason=parallel'
// SMALL: two grids do not fit side by side
// SMALL: selected S0 members=[G0] reason=temporal fallback:
// SMALL: selected S1 members=[G1] reason=temporal fallback:
// RUN: sed -e 's/grid = #ttcore.grid<1x1>/grid = #ttcore.grid<1x9>/' -e 's/tensor<1x1x1x1x/tensor<1x9x1x1x/g' %t.input > %t.illegal
// RUN: not ttmlir-opt %t.illegal --d2m-spatial-planning="materialize=true" --mlir-print-ir-after-failure --mlir-disable-threading -o /dev/null 2> %t.illegal-report
// RUN: FileCheck %s --check-prefix=ILLEGAL --input-file=%t.illegal-report --implicit-check-not='d2m.spatial {'
// ILLEGAL: spatial materialization supports only static tensor generics with ordinary 2D grids fitting the device

#layout = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
!grid = tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
!shard = tensor<1x1x!ttcore.tile<32x32, f32>>
#id = affine_map<(d0, d1) -> (d0, d1)>
#p = #ttcore.iterator_type<parallel>

// CHECK-LABEL: spatial_selected_groups @pair
// CHECK: selected S0 members=[G0, G1] reason=parallel:
// CHECK-SAME: ranges=[#ttcore.core_range<(0,0), (0,0)>, #ttcore.core_range<(0,1), (0,1)>]
func.func @pair(%a: !grid) -> (!grid, !grid) {
  %o0 = d2m.empty() : !grid
  %g0 = d2m.generic {test.id = 0 : i64, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
      ins(%a : !grid) outs(%o0 : !grid) {
    %empty = tensor.empty() : !shard
    d2m.yield %empty : (!shard)
  } : !grid
  %o1 = d2m.empty() : !grid
  %g1 = d2m.generic {test.id = 1 : i64, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
      ins(%a : !grid) outs(%o1 : !grid) {
    %empty = tensor.empty() : !shard
    d2m.yield %empty : (!shard)
  } : !grid
  return %g0, %g1 : !grid, !grid
}

// CHECK-LABEL: spatial_selected_groups @diamond
// CHECK: selected S0 members=[G0] reason=temporal fallback:
// CHECK: selected S1 members=[G1, G2] reason=parallel:
// CHECK-SAME: ranges=[#ttcore.core_range<(0,0), (0,0)>, #ttcore.core_range<(0,1), (0,1)>]
// CHECK: selected S2 members=[G3] reason=temporal fallback:
func.func @diamond(%a: !grid) -> (!grid, !grid, !grid, !grid) {
  %o0 = d2m.empty() : !grid
  %g0 = d2m.generic {test.id = 0 : i64, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
      ins(%a : !grid) outs(%o0 : !grid) {
    %empty = tensor.empty() : !shard
    d2m.yield %empty : (!shard)
  } : !grid
  %o1 = d2m.empty() : !grid
  %g1 = d2m.generic {test.id = 1 : i64, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
      ins(%g0 : !grid) outs(%o1 : !grid) {
    %empty = tensor.empty() : !shard
    d2m.yield %empty : (!shard)
  } : !grid
  %o2 = d2m.empty() : !grid
  %g2 = d2m.generic {test.id = 2 : i64, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
      ins(%g0 : !grid) outs(%o2 : !grid) {
    %empty = tensor.empty() : !shard
    d2m.yield %empty : (!shard)
  } : !grid
  %o3 = d2m.empty() : !grid
  %g3 = d2m.generic {test.id = 3 : i64, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
      ins(%g1, %g2 : !grid, !grid) outs(%o3 : !grid) {
    %empty = tensor.empty() : !shard
    d2m.yield %empty : (!shard)
  } : !grid
  return %g0, %g1, %g2, %g3 : !grid, !grid, !grid, !grid
}

// CHECK-LABEL: spatial_selected_groups @chain
// CHECK: selected S0 members=[G0] reason=temporal fallback:
// CHECK: selected S1 members=[G1] reason=temporal fallback:
// CHECK: selected S2 members=[G2] reason=temporal fallback:
func.func @chain(%a: !grid) -> (!grid, !grid, !grid) {
  %o0 = d2m.empty() : !grid
  %g0 = d2m.generic {test.id = 0 : i64, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
      ins(%a : !grid) outs(%o0 : !grid) {
    %empty = tensor.empty() : !shard
    d2m.yield %empty : (!shard)
  } : !grid
  %o1 = d2m.empty() : !grid
  %g1 = d2m.generic {test.id = 1 : i64, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
      ins(%g0 : !grid) outs(%o1 : !grid) {
    %empty = tensor.empty() : !shard
    d2m.yield %empty : (!shard)
  } : !grid
  %o2 = d2m.empty() : !grid
  %g2 = d2m.generic {test.id = 2 : i64, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
      ins(%g1 : !grid) outs(%o2 : !grid) {
    %empty = tensor.empty() : !shard
    d2m.yield %empty : (!shard)
  } : !grid
  return %g0, %g1, %g2 : !grid, !grid, !grid
}

// CHECK-LABEL: spatial_selected_groups @three_branches
// CHECK: selected S0 members=[G0, G1] reason=parallel:
// CHECK-SAME: ranges=[#ttcore.core_range<(0,0), (0,0)>, #ttcore.core_range<(0,1), (0,1)>]
// CHECK: selected S1 members=[G2] reason=temporal fallback:
func.func @three_branches(%a: !grid) -> (!grid, !grid, !grid) {
  %o0 = d2m.empty() : !grid
  %g0 = d2m.generic {test.id = 0 : i64, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
      ins(%a : !grid) outs(%o0 : !grid) {
    %empty = tensor.empty() : !shard
    d2m.yield %empty : (!shard)
  } : !grid
  %o1 = d2m.empty() : !grid
  %g1 = d2m.generic {test.id = 1 : i64, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
      ins(%a : !grid) outs(%o1 : !grid) {
    %empty = tensor.empty() : !shard
    d2m.yield %empty : (!shard)
  } : !grid
  %o2 = d2m.empty() : !grid
  %g2 = d2m.generic {test.id = 2 : i64, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
      ins(%a : !grid) outs(%o2 : !grid) {
    %empty = tensor.empty() : !shard
    d2m.yield %empty : (!shard)
  } : !grid
  return %g0, %g1, %g2 : !grid, !grid, !grid
}

// CHECK-LABEL: spatial_selected_groups @shared_output
// REJECT: shared or unknown destination storage
// CHECK: selected S0 members=[G0] reason=temporal fallback:
// CHECK: selected S1 members=[G1] reason=temporal fallback:
func.func @shared_output(%a: !grid) -> (!grid, !grid) {
  %o0 = d2m.empty() : !grid
  %g0 = d2m.generic {test.id = 0 : i64, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
      ins(%a : !grid) outs(%o0 : !grid) {
    %empty = tensor.empty() : !shard
    d2m.yield %empty : (!shard)
  } : !grid
  %o1 = d2m.empty() : !grid
  %g1 = d2m.generic {test.id = 1 : i64, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
      ins(%a : !grid) outs(%o0 : !grid) {
    %empty = tensor.empty() : !shard
    d2m.yield %empty : (!shard)
  } : !grid
  return %g0, %g1 : !grid, !grid
}

// CHECK-LABEL: spatial_selected_groups @unsafe_preparation
// REJECT: input preparation cannot be safely hoisted
// CHECK: selected S0 members=[G0] reason=temporal fallback:
// CHECK: selected S1 members=[G1] reason=temporal fallback:
func.func @unsafe_preparation(%a: !grid) -> (!grid, !grid) {
  %o0 = d2m.empty() : !grid
  %g0 = d2m.generic {test.id = 0 : i64, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
      ins(%a : !grid) outs(%o0 : !grid) {
    %empty = tensor.empty() : !shard
    d2m.yield %empty : (!shard)
  } : !grid
  %o1 = d2m.empty() : !grid
  %true = arith.constant true
  %view = arith.select %true, %a, %a : !grid
  %g1 = d2m.generic {test.id = 1 : i64, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
      ins(%view : !grid) outs(%o1 : !grid) {
    %empty = tensor.empty() : !shard
    d2m.yield %empty : (!shard)
  } : !grid
  return %g0, %g1 : !grid, !grid
}

// CHECK-LABEL: spatial_selected_groups @effect_boundary
// CHECK: selected S0 members=[G0] reason=temporal fallback:
// CHECK: selected S1 members=[G1] reason=temporal fallback:
func.func @effect_boundary(%a: !grid) -> (!grid, !grid) {
  %o0 = d2m.empty() : !grid
  %g0 = d2m.generic {test.id = 0 : i64, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
      ins(%a : !grid) outs(%o0 : !grid) {
    %empty = tensor.empty() : !shard
    d2m.yield %empty : (!shard)
  } : !grid
  %o1 = d2m.empty() : !grid
  func.call @effect() : () -> ()
  %g1 = d2m.generic {test.id = 1 : i64, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
      ins(%a : !grid) outs(%o1 : !grid) {
    %empty = tensor.empty() : !shard
    d2m.yield %empty : (!shard)
  } : !grid
  return %g0, %g1 : !grid, !grid
}

// CHECK-LABEL: spatial_selected_groups @local_view
// CHECK: selected S0 members=[G0, G1] reason=parallel:
// CHECK-SAME: ranges=[#ttcore.core_range<(0,0), (0,0)>, #ttcore.core_range<(0,1), (0,1)>]
func.func @local_view(%a: !grid) -> (!grid, !grid) {
  %o0 = d2m.empty() : !grid
  %g0 = d2m.generic {test.id = 0 : i64, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
      ins(%a : !grid) outs(%o0 : !grid) {
    %empty = tensor.empty() : !shard
    d2m.yield %empty : (!shard)
  } : !grid
  %o1 = d2m.empty() : !grid
  %view = d2m.view_layout %a remapping = affine_map<(a,b,c,d)->(a,b,c,d)> : !grid -> !grid
  %g1 = d2m.generic {test.id = 1 : i64, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
      ins(%view : !grid) outs(%o1 : !grid) {
    %empty = tensor.empty() : !shard
    d2m.yield %empty : (!shard)
  } : !grid
  return %g0, %g1 : !grid, !grid
}

func.func private @effect()

// CHECK-LABEL: spatial_selected_groups @multi_result
// CHECK: selected S0 members=[G0, G1] reason=parallel:
// RUN: FileCheck %s --check-prefix=SSA --input-file=%t.once
// SSA-LABEL: func.func @multi_result
// SSA: %[[GROUP:[a-z0-9]+]]:3 = d2m.spatial
// SSA: %[[FIRST:[a-z0-9]+]]:2 = d2m.generic
// SSA-SAME: test.id = 10 : i64
// SSA: d2m.spatial_yield %[[FIRST]]#0, %[[FIRST]]#1
// SSA: %[[SECOND:[a-z0-9]+]] = d2m.generic
// SSA-SAME: test.id = 11 : i64
// SSA: d2m.spatial_yield %[[SECOND]]
// SSA: return %[[GROUP]]#0, %[[GROUP]]#1, %[[GROUP]]#2
func.func @multi_result(%a: !grid) -> (!grid, !grid, !grid) {
  %o0 = d2m.empty() : !grid
  %o1 = d2m.empty() : !grid
  %g0:2 = d2m.generic {test.id = 10 : i64, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
      ins(%a : !grid) outs(%o0, %o1 : !grid, !grid) {
    %e0 = tensor.empty() : !shard
    %e1 = tensor.empty() : !shard
    d2m.yield %e0, %e1 : (!shard, !shard)
  } : !grid, !grid
  %o2 = d2m.empty() : !grid
  %g1 = d2m.generic {test.id = 11 : i64, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
      ins(%a : !grid) outs(%o2 : !grid) {
    %e = tensor.empty() : !shard
    d2m.yield %e : (!shard)
  } : !grid
  return %g0#0, %g0#1, %g1 : !grid, !grid, !grid
}
