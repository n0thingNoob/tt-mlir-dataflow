// RUN: ttmlir-opt --split-input-file --ttcore-register-device --d2m-grid-selection --mlir-print-local-scope %s | FileCheck %s

// The producer lays the edge out on a 2x2 grid, but the consumer region is
// 1x4: its generic needs the edge on a 1x2 grid. The block argument keeps the
// producer's 2x2 layout and the consumer reads it through a view reblocked to
// 1x2 (each consumer core pulls its blocks from the producer cores holding
// them). This used to be an error ("resharding an edge is not supported").
// CHECK-LABEL: func.func @edge_grid_mismatch
// CHECK: ^bb0(%[[E:.*]]: tensor<2x2x1x1x!ttcore.tile<32x32, f32>
// CHECK-NEXT: %[[V:.*]] = d2m.view_layout %[[E]] remapping = affine_map<(d0, d1, d2, d3) -> (d2 + d1 floordiv 2, d1 mod 2, 0, 0)> : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, {{.*}}> -> tensor<1x2x2x1x!ttcore.tile<32x32, f32>
// CHECK-NEXT: d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x2, virt_to_physical_map = (d0, d1) -> (0, d0 + 2, d1)
// CHECK-NEXT: ins(%[[V]] :
#layout = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
func.func @edge_grid_mismatch() {
  %o0 = d2m.empty() : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
  %o1 = d2m.empty() : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
  %r:2 = d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>], grid_ranges = [#ttcore.core_range<(0, 0), (1, 1)>, #ttcore.core_range<(2, 0), (2, 3)>]}
      ins() outs(%o0, %o1 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>) {
      ^region0:
        %g = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins() outs(%o0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>) {
          %b = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
          d2m.yield %b : (tensor<2x2x!ttcore.tile<32x32, f32>>)
        } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
        d2m.spatial_yield %g : (tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>)
    }, {
      ^region1(%a0: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>):
        %g = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%a0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>) outs(%o1 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>) {
          %b = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
          d2m.yield %b : (tensor<2x2x!ttcore.tile<32x32, f32>>)
        } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
        d2m.spatial_yield %g : (tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>)
  } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
  return
}

// -----

// Fan-out: region 1 (2x2) reads the 2x2 edge directly, region 2 (1x4) through
// a view reblocked to its 1x2 grid.
// CHECK-LABEL: func.func @edge_fan_out_grid_mismatch
// CHECK: ^bb0(%[[E1:.*]]: tensor<2x2x1x1x!ttcore.tile<32x32, f32>
// CHECK-NEXT: d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x2, virt_to_physical_map = (d0, d1) -> (0, d0, d1 + 3)
// CHECK-NEXT: ins(%[[E1]] :
// CHECK: ^bb0(%[[E2:.*]]: tensor<2x2x1x1x!ttcore.tile<32x32, f32>
// CHECK-NEXT: %[[V2:.*]] = d2m.view_layout %[[E2]]
// CHECK-NEXT: d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x2, virt_to_physical_map = (d0, d1) -> (0, d0 + 3, d1)
// CHECK-NEXT: ins(%[[V2]] :
#layout = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
func.func @edge_fan_out_grid_mismatch() {
  %o0 = d2m.empty() : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
  %o1 = d2m.empty() : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
  %o2 = d2m.empty() : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
  %r:3 = d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>, #d2m.spatial_edge<0:0 -> 2:0>], grid_ranges = [#ttcore.core_range<(0, 0), (1, 1)>, #ttcore.core_range<(0, 3), (1, 4)>, #ttcore.core_range<(3, 0), (3, 3)>]}
      ins() outs(%o0, %o1, %o2 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>) {
      ^region0:
        %g = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins() outs(%o0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>) {
          %b = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
          d2m.yield %b : (tensor<2x2x!ttcore.tile<32x32, f32>>)
        } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
        d2m.spatial_yield %g : (tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>)
    }, {
      ^region1(%a0: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>):
        %g = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%a0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>) outs(%o1 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>) {
          %b = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
          d2m.yield %b : (tensor<2x2x!ttcore.tile<32x32, f32>>)
        } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
        d2m.spatial_yield %g : (tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>)
    }, {
      ^region2(%a0: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>):
        %g = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%a0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>) outs(%o2 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>) {
          %b = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
          d2m.yield %b : (tensor<2x2x!ttcore.tile<32x32, f32>>)
        } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
        d2m.spatial_yield %g : (tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>)
  } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
  return
}
