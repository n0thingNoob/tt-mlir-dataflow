// RUN: ttmlir-opt --split-input-file --ttcore-register-device --d2m-grid-selection %s | FileCheck %s

// d2m-grid-selection on d2m.spatial edges: the producer's output grid pins the
// consumer's edge operand, and the consumer region's block argument takes the
// producer output's new type. Inputs are in pre-selection form (grid 1x1).
// 2x2 producer at the origin -> 2x2 consumer at (0, 2).
#layout = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
// CHECK-LABEL: func.func @edge_chain_offset
func.func @edge_chain_offset() {
  %o0 = d2m.empty() : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
  %o1 = d2m.empty() : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
  // CHECK: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>]
  // CHECK: d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x2>,
  // CHECK: outs(%{{.*}} : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>)
  // CHECK: ^bb0(%[[ARG:.*]]: tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>):
  // CHECK-NEXT: d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x2, virt_to_physical_map = (d0, d1) -> (0, d0, d1 + 2), physical_to_virt_map = (d0, d1) -> (0, d0, d1 - 2)>,
  // CHECK-NEXT: ins(%[[ARG]] : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>)
  // CHECK: } : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>, tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>
  %r:2 = d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>], grid_ranges = [#ttcore.core_range<(0, 0), (1, 1)>, #ttcore.core_range<(0, 2), (1, 3)>]}
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

// 1x4 producer on row 0 -> 1x4 consumer on row 1.
#layout = #ttcore.metal_layout<logical_shape = 32x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
// CHECK-LABEL: func.func @edge_row_1x4
func.func @edge_row_1x4() {
  %o0 = d2m.empty() : tensor<1x1x1x4x!ttcore.tile<32x32, f32>, #layout>
  %o1 = d2m.empty() : tensor<1x1x1x4x!ttcore.tile<32x32, f32>, #layout>
  // CHECK: d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x4>,
  // CHECK: ^bb0(%[[ARG:.*]]: tensor<1x4x1x1x!ttcore.tile<32x32, f32>, #layout>):
  // CHECK-NEXT: d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x4, virt_to_physical_map = (d0, d1) -> (0, d0 + 1, d1), physical_to_virt_map = (d0, d1) -> (0, d0 - 1, d1)>,
  // CHECK-NEXT: ins(%[[ARG]] : tensor<1x4x1x1x!ttcore.tile<32x32, f32>, #layout>)
  %r:2 = d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>], grid_ranges = [#ttcore.core_range<(0, 0), (0, 3)>, #ttcore.core_range<(1, 0), (1, 3)>]}
      ins() outs(%o0, %o1 : tensor<1x1x1x4x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x1x4x!ttcore.tile<32x32, f32>, #layout>) {
      ^region0:
        %g = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins() outs(%o0 : tensor<1x1x1x4x!ttcore.tile<32x32, f32>, #layout>) {
          %b = tensor.empty() : tensor<1x4x!ttcore.tile<32x32, f32>>
          d2m.yield %b : (tensor<1x4x!ttcore.tile<32x32, f32>>)
        } : tensor<1x1x1x4x!ttcore.tile<32x32, f32>, #layout>
        d2m.spatial_yield %g : (tensor<1x1x1x4x!ttcore.tile<32x32, f32>, #layout>)
    }, {
      ^region1(%a0: tensor<1x1x1x4x!ttcore.tile<32x32, f32>, #layout>):
        %g = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%a0 : tensor<1x1x1x4x!ttcore.tile<32x32, f32>, #layout>) outs(%o1 : tensor<1x1x1x4x!ttcore.tile<32x32, f32>, #layout>) {
          %b = tensor.empty() : tensor<1x4x!ttcore.tile<32x32, f32>>
          d2m.yield %b : (tensor<1x4x!ttcore.tile<32x32, f32>>)
        } : tensor<1x1x1x4x!ttcore.tile<32x32, f32>, #layout>
        d2m.spatial_yield %g : (tensor<1x1x1x4x!ttcore.tile<32x32, f32>, #layout>)
  } : tensor<1x1x1x4x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x1x4x!ttcore.tile<32x32, f32>, #layout>
  return
}

// -----

// Two 2x2 producers -> one 2x2 consumer with two edges.
#layout = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
// CHECK-LABEL: func.func @edge_join
func.func @edge_join() {
  %o0 = d2m.empty() : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
  %o1 = d2m.empty() : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
  %o2 = d2m.empty() : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
  // CHECK: d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x2>,
  // CHECK: d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x2, virt_to_physical_map = (d0, d1) -> (0, d0 + 3, d1), physical_to_virt_map = (d0, d1) -> (0, d0 - 3, d1)>,
  // CHECK: ^bb0(%[[A0:.*]]: tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>, %[[A1:.*]]: tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>):
  // CHECK-NEXT: d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x2, virt_to_physical_map = (d0, d1) -> (0, d0, d1 + 4), physical_to_virt_map = (d0, d1) -> (0, d0, d1 - 4)>,
  // CHECK-NEXT: ins(%[[A0]], %[[A1]] : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>, tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>)
  %r:3 = d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 2:0>, #d2m.spatial_edge<1:0 -> 2:1>], grid_ranges = [#ttcore.core_range<(0, 0), (1, 1)>, #ttcore.core_range<(3, 0), (4, 1)>, #ttcore.core_range<(0, 4), (1, 5)>]}
      ins() outs(%o0, %o1, %o2 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>) {
      ^region0:
        %g = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins() outs(%o0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>) {
          %b = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
          d2m.yield %b : (tensor<2x2x!ttcore.tile<32x32, f32>>)
        } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
        d2m.spatial_yield %g : (tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>)
    }, {
      ^region1:
        %g = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins() outs(%o1 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>) {
          %b = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
          d2m.yield %b : (tensor<2x2x!ttcore.tile<32x32, f32>>)
        } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
        d2m.spatial_yield %g : (tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>)
    }, {
      ^region2(%a0: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>, %a1: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>):
        %g = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%a0, %a1 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>) outs(%o2 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>) {
          %b = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
          d2m.yield %b : (tensor<2x2x!ttcore.tile<32x32, f32>>)
        } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
        d2m.spatial_yield %g : (tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>)
  } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
  return
}

// -----

// The consumer is region 0 and is rewritten before its producer: the result
// does not depend on the order in which generics are processed.
#layout = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
// CHECK-LABEL: func.func @edge_consumer_region_first
func.func @edge_consumer_region_first() {
  %o0 = d2m.empty() : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
  %o1 = d2m.empty() : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
  // CHECK: ^bb0(%[[ARG:.*]]: tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>):
  // CHECK-NEXT: d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x2, virt_to_physical_map = (d0, d1) -> (0, d0, d1 + 2), physical_to_virt_map = (d0, d1) -> (0, d0, d1 - 2)>,
  // CHECK-NEXT: ins(%[[ARG]] : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>)
  // CHECK: d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x2>,
  // CHECK: } : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>, tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>
  %r:2 = d2m.spatial {edges = [#d2m.spatial_edge<1:0 -> 0:0>], grid_ranges = [#ttcore.core_range<(0, 2), (1, 3)>, #ttcore.core_range<(0, 0), (1, 1)>]}
      ins() outs(%o0, %o1 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>) {
      ^region0(%a0: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>):
        %g = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%a0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>) outs(%o0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>) {
          %b = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
          d2m.yield %b : (tensor<2x2x!ttcore.tile<32x32, f32>>)
        } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
        d2m.spatial_yield %g : (tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>)
    }, {
      ^region1:
        %g = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins() outs(%o1 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>) {
          %b = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
          d2m.yield %b : (tensor<2x2x!ttcore.tile<32x32, f32>>)
        } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
        d2m.spatial_yield %g : (tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>)
  } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
  return
}

// -----

// Matmul consumer: A (0,0)-(1,1) writes the 64x128 edge on its 2x2 grid; the
// matmul in B (0,4)-(1,7) has a 2x4 grid (its 64x128 output) and reads the edge
// as its MxK operand on that same 2x2 grid.
#lx = #ttcore.metal_layout<logical_shape = 64x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#lw = #ttcore.metal_layout<logical_shape = 128x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#mk = affine_map<(d0, d1, d2) -> (d0, d2)>
#kn = affine_map<(d0, d1, d2) -> (d2, d1)>
#mn = affine_map<(d0, d1, d2) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>
// CHECK-LABEL: func.func @edge_matmul_consumer
func.func @edge_matmul_consumer() {
  %w = d2m.empty() : tensor<1x1x4x4x!ttcore.tile<32x32, f32>, #lw>
  %o0 = d2m.empty() : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #lx>
  %o1 = d2m.empty() : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #lx>
  // CHECK: d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x2>,
  // CHECK: ^bb0(%[[ARG:.*]]: tensor<2x2x1x2x!ttcore.tile<32x32, f32>, #layout{{[0-9]*}}>):
  // CHECK-NEXT: d2m.generic {block_factors = [1, 1, 2], grid = #ttcore.grid<2x4, virt_to_physical_map = (d0, d1) -> (0, d0, d1 + 4), physical_to_virt_map = (d0, d1) -> (0, d0, d1 - 4)>,
  // CHECK-NEXT: ins(%[[ARG]], %{{.*}} : tensor<2x2x1x2x!ttcore.tile<32x32, f32>, #layout{{[0-9]*}}>,
  %r:2 = d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>], grid_ranges = [#ttcore.core_range<(0, 0), (1, 1)>, #ttcore.core_range<(0, 4), (1, 7)>]}
      ins(%w : tensor<1x1x4x4x!ttcore.tile<32x32, f32>, #lw>) outs(%o0, %o1 : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #lx>, tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #lx>) {
      ^region0:
        %g = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins() outs(%o0 : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #lx>) {
          %b = tensor.empty() : tensor<2x4x!ttcore.tile<32x32, f32>>
          d2m.yield %b : (tensor<2x4x!ttcore.tile<32x32, f32>>)
        } : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #lx>
        d2m.spatial_yield %g : (tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #lx>)
    }, {
      ^region1(%a0: tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #lx>):
        %g = d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#mk, #kn, #mn], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<unified>]}
            ins(%a0, %w : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #lx>, tensor<1x1x4x4x!ttcore.tile<32x32, f32>, #lw>) outs(%o1 : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #lx>) {
          %b = tensor.empty() : tensor<2x4x!ttcore.tile<32x32, f32>>
          d2m.yield %b : (tensor<2x4x!ttcore.tile<32x32, f32>>)
        } : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #lx>
        d2m.spatial_yield %g : (tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #lx>)
  } : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #lx>, tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #lx>
  return
}
