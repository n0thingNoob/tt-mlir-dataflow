// RUN: ttmlir-opt --split-input-file --ttcore-register-device --d2m-materialize-spatial-groups %s | FileCheck %s

#l = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#lu = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>
#lo = #ttcore.metal_layout<logical_shape = 50x50, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
!T = tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #l>
!U = tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #lu>
!O = tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #lo>
!R = tensor<1x1x64x64xf32, #l>
!H = tensor<64x64xf32>
!S = tensor<2x2x!ttcore.tile<32x32, f32>>
// A two-member chain: the round trip between the members folds into an edge,
// the members get fresh d2m.empty outputs, and the ungrouped consumer reads
// the spatial result through its own round trip.
// CHECK-LABEL: func.func @chain
// CHECK: %[[X:.*]] = d2m.to_layout %arg0
// CHECK: %[[E0:.*]] = d2m.empty() : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
// CHECK: %[[E1:.*]] = d2m.empty() : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
// CHECK: %[[S:.*]]:2 = d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>], grid_ranges = [#ttcore.core_range<(0,0), (1,1)>, #ttcore.core_range<(0,3), (1,4)>]}
// CHECK-NEXT: ins(%[[X]] : {{.*}})
// CHECK-NEXT: outs(%[[E0]], %[[E1]] : {{.*}})
// CHECK: d2m.generic {block_factors
// CHECK-NEXT: ins(%[[X]] : {{.*}})
// CHECK-NEXT: outs(%[[E0]] : {{.*}})
// CHECK: d2m.spatial_yield
// CHECK: ^bb0(%[[ARG:.*]]: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>):
// CHECK: d2m.generic {block_factors
// CHECK-NEXT: ins(%[[ARG]] : {{.*}})
// CHECK-NEXT: outs(%[[E1]] : {{.*}})
// CHECK: d2m.remote_load %{{.*}} %[[ARG]][
// CHECK: d2m.spatial_yield
// CHECK: d2m.to_layout %[[S]]#1
// CHECK: d2m.generic {block_factors
// CHECK-NOT: d2m.spatial_group
// CHECK: return
func.func @chain(%arg0: !H) -> (!H) {
  %x_d = d2m.empty() : !T
  %x = d2m.to_layout %arg0, %x_d : !H into !T -> !T
  %o0 = d2m.empty() : !T
  %a = d2m.generic {d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(0, 0), (1, 1)>, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%x : !T) outs(%o0 : !T) {
  %blk = tensor.empty() : !S
  d2m.yield %blk : (!S)
  } : !T
  %a2_h = d2m.empty() : !H
  %a2_host = d2m.to_layout %a, %a2_h : !T into !H -> !H
  %a2_d = d2m.empty() : !T
  %a2 = d2m.to_layout %a2_host, %a2_d : !H into !T -> !T
  %o1 = d2m.empty() : !T
  %b = d2m.generic {d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(0, 3), (1, 4)>, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%a2 : !T) outs(%o1 : !T) {
  %i0 = d2m.block_index(0) : index
  %i1 = d2m.block_index(1) : index
  %t = tensor.empty() : !S
  %blk = d2m.remote_load %t %a2[%i0, %i1] : !S, !T -> !S
  d2m.yield %blk : (!S)
  } : !T
  %b2_h = d2m.empty() : !H
  %b2_host = d2m.to_layout %b, %b2_h : !T into !H -> !H
  %b2_d = d2m.empty() : !T
  %b2 = d2m.to_layout %b2_host, %b2_d : !H into !T -> !T
  %o2 = d2m.empty() : !T
  %c = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%b2 : !T) outs(%o2 : !T) {
  %blk = tensor.empty() : !S
  d2m.yield %blk : (!S)
  } : !T
  %out_h = d2m.empty() : !H
  %out = d2m.to_layout %c, %out_h : !T into !H -> !H
  return %out : !H
}

// -----

#l = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#lu = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>
#lo = #ttcore.metal_layout<logical_shape = 50x50, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
!T = tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #l>
!U = tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #lu>
!O = tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #lo>
!R = tensor<1x1x64x64xf32, #l>
!H = tensor<64x64xf32>
!S = tensor<2x2x!ttcore.tile<32x32, f32>>
// Fan-out and join: one producer output feeds two regions (one block argument
// each), and the join region gets one block argument per producer.
// CHECK-LABEL: func.func @diamond
// CHECK: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>, #d2m.spatial_edge<0:0 -> 2:0>, #d2m.spatial_edge<1:0 -> 2:1>]
// CHECK: ^bb0(%{{.*}}: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>):
// CHECK: ^bb0(%[[Y:.*]]: tensor<{{.*}}>, %[[Z:.*]]: tensor<{{.*}}>):
// CHECK: ins(%[[Y]], %[[Z]] :
func.func @diamond(%arg0: !H) -> (!H) {
  %x_d = d2m.empty() : !T
  %x = d2m.to_layout %arg0, %x_d : !H into !T -> !T
  %o0 = d2m.empty() : !T
  %a = d2m.generic {d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(0, 0), (1, 1)>, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%x : !T) outs(%o0 : !T) {
  %blk = tensor.empty() : !S
  d2m.yield %blk : (!S)
  } : !T
  %ab_h = d2m.empty() : !H
  %ab_host = d2m.to_layout %a, %ab_h : !T into !H -> !H
  %ab_d = d2m.empty() : !T
  %ab = d2m.to_layout %ab_host, %ab_d : !H into !T -> !T
  %o1 = d2m.empty() : !T
  %b = d2m.generic {d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(0, 3), (1, 4)>, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%ab : !T) outs(%o1 : !T) {
  %blk = tensor.empty() : !S
  d2m.yield %blk : (!S)
  } : !T
  %ac_h = d2m.empty() : !H
  %ac_host = d2m.to_layout %a, %ac_h : !T into !H -> !H
  %ac_d = d2m.empty() : !T
  %ac = d2m.to_layout %ac_host, %ac_d : !H into !T -> !T
  %bc_h = d2m.empty() : !H
  %bc_host = d2m.to_layout %b, %bc_h : !T into !H -> !H
  %bc_d = d2m.empty() : !T
  %bc = d2m.to_layout %bc_host, %bc_d : !H into !T -> !T
  %o2 = d2m.empty() : !T
  %c = d2m.generic {d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(3, 0), (4, 1)>, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%ac, %bc : !T, !T) outs(%o2 : !T) {
  %blk = tensor.empty() : !S
  d2m.yield %blk : (!S)
  } : !T
  %out_h = d2m.empty() : !H
  %out = d2m.to_layout %c, %out_h : !T into !H -> !H
  return %out : !H
}

// -----

#l = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#lu = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>
#lo = #ttcore.metal_layout<logical_shape = 50x50, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
!T = tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #l>
!U = tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #lu>
!O = tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #lo>
!R = tensor<1x1x64x64xf32, #l>
!H = tensor<64x64xf32>
!S = tensor<2x2x!ttcore.tile<32x32, f32>>
// A consumer that reads one producer output on two operands gets one block
// argument (an edge is unique per producer output and consumer region).
// CHECK-LABEL: func.func @same_edge_twice
// CHECK: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>]
// CHECK: ^bb0(%[[A:.*]]: tensor<{{.*}}>):
// CHECK: ins(%[[A]], %[[A]] :
func.func @same_edge_twice(%arg0: !H) -> (!H) {
  %x_d = d2m.empty() : !T
  %x = d2m.to_layout %arg0, %x_d : !H into !T -> !T
  %o0 = d2m.empty() : !T
  %a = d2m.generic {d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(0, 0), (1, 1)>, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%x : !T) outs(%o0 : !T) {
  %blk = tensor.empty() : !S
  d2m.yield %blk : (!S)
  } : !T
  %a1_h = d2m.empty() : !H
  %a1_host = d2m.to_layout %a, %a1_h : !T into !H -> !H
  %a1_d = d2m.empty() : !T
  %a1 = d2m.to_layout %a1_host, %a1_d : !H into !T -> !T
  %a2_h = d2m.empty() : !H
  %a2_host = d2m.to_layout %a, %a2_h : !T into !H -> !H
  %a2_d = d2m.empty() : !T
  %a2 = d2m.to_layout %a2_host, %a2_d : !H into !T -> !T
  %o1 = d2m.empty() : !T
  %b = d2m.generic {d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(0, 3), (1, 4)>, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%a1, %a2 : !T, !T) outs(%o1 : !T) {
  %blk = tensor.empty() : !S
  d2m.yield %blk : (!S)
  } : !T
  %out_h = d2m.empty() : !H
  %out = d2m.to_layout %b, %out_h : !T into !H -> !H
  return %out : !H
}

// -----

#l = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#lu = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>
#lo = #ttcore.metal_layout<logical_shape = 50x50, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
!T = tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #l>
!U = tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #lu>
!O = tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #lo>
!R = tensor<1x1x64x64xf32, #l>
!H = tensor<64x64xf32>
!S = tensor<2x2x!ttcore.tile<32x32, f32>>
// A member result used inside the group (edge) and returned (spatial result).
// CHECK-LABEL: func.func @used_outside
// CHECK: %[[S:.*]]:2 = d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>]
// CHECK: %[[AH:.*]] = d2m.to_layout %[[S]]#0
// CHECK: d2m.to_layout %[[S]]#1
// CHECK: return %[[AH]],
func.func @used_outside(%arg0: !H) -> (!H, !H) {
  %x_d = d2m.empty() : !T
  %x = d2m.to_layout %arg0, %x_d : !H into !T -> !T
  %o0 = d2m.empty() : !T
  %a = d2m.generic {d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(0, 0), (1, 1)>, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%x : !T) outs(%o0 : !T) {
  %blk = tensor.empty() : !S
  d2m.yield %blk : (!S)
  } : !T
  %ah_h = d2m.empty() : !H
  %ah = d2m.to_layout %a, %ah_h : !T into !H -> !H
  %a2_d = d2m.empty() : !T
  %a2 = d2m.to_layout %ah, %a2_d : !H into !T -> !T
  %o1 = d2m.empty() : !T
  %b = d2m.generic {d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(0, 3), (1, 4)>, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%a2 : !T) outs(%o1 : !T) {
  %blk = tensor.empty() : !S
  d2m.yield %blk : (!S)
  } : !T
  %out1_h = d2m.empty() : !H
  %out1 = d2m.to_layout %b, %out1_h : !T into !H -> !H
  return %ah, %out1 : !H, !H
}

// -----

#l = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#lu = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>
#lo = #ttcore.metal_layout<logical_shape = 50x50, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
!T = tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #l>
!U = tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #lu>
!O = tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #lo>
!R = tensor<1x1x64x64xf32, #l>
!H = tensor<64x64xf32>
!S = tensor<2x2x!ttcore.tile<32x32, f32>>
// The producer writes #l (collapsed) and the consumer reads #lu (no collapse):
// same device shape, so the consumer operand takes the producer's type. Its
// output keeps #lu.
// CHECK-LABEL: func.func @collapse_only
// CHECK: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>]
// CHECK: ^bb0(%[[A:.*]]: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>):
// CHECK-NEXT: d2m.generic
// CHECK-NEXT: ins(%[[A]] : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>)
// CHECK-NEXT: outs(%{{.*}} : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout1>)
func.func @collapse_only(%arg0: !H) -> (!H) {
  %x_d = d2m.empty() : !T
  %x = d2m.to_layout %arg0, %x_d : !H into !T -> !T
  %o0 = d2m.empty() : !T
  %a = d2m.generic {d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(0, 0), (1, 1)>, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%x : !T) outs(%o0 : !T) {
  %blk = tensor.empty() : !S
  d2m.yield %blk : (!S)
  } : !T
  %a2_h = d2m.empty() : !H
  %a2_host = d2m.to_layout %a, %a2_h : !T into !H -> !H
  %a2_d = d2m.empty() : !U
  %a2 = d2m.to_layout %a2_host, %a2_d : !H into !U -> !U
  %o1 = d2m.empty() : !U
  %b = d2m.generic {d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(0, 3), (1, 4)>, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%a2 : !U) outs(%o1 : !U) {
  %blk = tensor.empty() : !S
  d2m.yield %blk : (!S)
  } : !U
  %out_h = d2m.empty() : !H
  %out = d2m.to_layout %b, %out_h : !U into !H -> !H
  return %out : !H
}

// -----

#l = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#lu = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>
#lo = #ttcore.metal_layout<logical_shape = 50x50, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
!T = tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #l>
!U = tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #lu>
!O = tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #lo>
!R = tensor<1x1x64x64xf32, #l>
!H = tensor<64x64xf32>
!S = tensor<2x2x!ttcore.tile<32x32, f32>>
// A mask on a tile-aligned tensor folds to its input, so the chain is still an
// edge.
// CHECK-LABEL: func.func @aligned_mask
// CHECK-NOT: d2m.mask
// CHECK: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>]
func.func @aligned_mask(%arg0: !H) -> (!H) {
  %x_d = d2m.empty() : !T
  %x = d2m.to_layout %arg0, %x_d : !H into !T -> !T
  %o0 = d2m.empty() : !T
  %a = d2m.generic {d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(0, 0), (1, 1)>, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%x : !T) outs(%o0 : !T) {
  %blk = tensor.empty() : !S
  d2m.yield %blk : (!S)
  } : !T
  %a2_h = d2m.empty() : !H
  %a2_host = d2m.to_layout %a, %a2_h : !T into !H -> !H
  %a2_d = d2m.empty() : !T
  %a2 = d2m.to_layout %a2_host, %a2_d : !H into !T -> !T
  %m0 = d2m.empty() : !T
  %a3 = d2m.mask %a2, %m0 logical_shape = [64, 64] fill_value = <zero> : !T into !T -> !T
  %o1 = d2m.empty() : !T
  %b = d2m.generic {d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(0, 3), (1, 4)>, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%a3 : !T) outs(%o1 : !T) {
  %blk = tensor.empty() : !S
  d2m.yield %blk : (!S)
  } : !T
  %out_h = d2m.empty() : !H
  %out = d2m.to_layout %b, %out_h : !T into !H -> !H
  return %out : !H
}

// -----

#l = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#lu = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>
#lo = #ttcore.metal_layout<logical_shape = 50x50, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
!T = tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #l>
!U = tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #lu>
!O = tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #lo>
!R = tensor<1x1x64x64xf32, #l>
!H = tensor<64x64xf32>
!S = tensor<2x2x!ttcore.tile<32x32, f32>>
// The input to_layout %x feeds a member and an ungrouped generic: the member
// gets its own copy.
// CHECK-LABEL: func.func @to_layout_copy
// CHECK: %[[X:.*]] = d2m.to_layout %arg0
// CHECK: %[[C:.*]] = d2m.to_layout %arg0
// CHECK: d2m.spatial
// CHECK-NEXT: ins(%[[C]] :
// CHECK: d2m.spatial_yield
// CHECK: d2m.generic {block_factors
// CHECK-NEXT: ins(%[[X]] :
func.func @to_layout_copy(%arg0: !H) -> (!H, !H) {
  %x_d = d2m.empty() : !T
  %x = d2m.to_layout %arg0, %x_d : !H into !T -> !T
  %o0 = d2m.empty() : !T
  %a = d2m.generic {d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(0, 0), (1, 1)>, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%x : !T) outs(%o0 : !T) {
  %blk = tensor.empty() : !S
  d2m.yield %blk : (!S)
  } : !T
  %o1 = d2m.empty() : !T
  %b = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%x : !T) outs(%o1 : !T) {
  %blk = tensor.empty() : !S
  d2m.yield %blk : (!S)
  } : !T
  %out0_h = d2m.empty() : !H
  %out0 = d2m.to_layout %a, %out0_h : !T into !H -> !H
  %out1_h = d2m.empty() : !H
  %out1 = d2m.to_layout %b, %out1_h : !T into !H -> !H
  return %out0, %out1 : !H, !H
}

// -----

#l = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#lu = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>
#lo = #ttcore.metal_layout<logical_shape = 50x50, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
!T = tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #l>
!U = tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #lu>
!O = tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #lo>
!R = tensor<1x1x64x64xf32, #l>
!H = tensor<64x64xf32>
!S = tensor<2x2x!ttcore.tile<32x32, f32>>
// %c reads %a but does not feed the group, and sits between %a and %b: it
// moves after the spatial.
// CHECK-LABEL: func.func @move_after
// CHECK: %[[S:.*]]:2 = d2m.spatial
// CHECK: d2m.to_layout %[[S]]#0
// CHECK: d2m.generic {block_factors
// CHECK: return
func.func @move_after(%arg0: !H) -> (!H, !H) {
  %x_d = d2m.empty() : !T
  %x = d2m.to_layout %arg0, %x_d : !H into !T -> !T
  %o0 = d2m.empty() : !T
  %a = d2m.generic {d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(0, 0), (1, 1)>, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%x : !T) outs(%o0 : !T) {
  %blk = tensor.empty() : !S
  d2m.yield %blk : (!S)
  } : !T
  %ac_h = d2m.empty() : !H
  %ac_host = d2m.to_layout %a, %ac_h : !T into !H -> !H
  %ac_d = d2m.empty() : !T
  %ac = d2m.to_layout %ac_host, %ac_d : !H into !T -> !T
  %o2 = d2m.empty() : !T
  %c = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%ac : !T) outs(%o2 : !T) {
  %blk = tensor.empty() : !S
  d2m.yield %blk : (!S)
  } : !T
  %ab_h = d2m.empty() : !H
  %ab_host = d2m.to_layout %a, %ab_h : !T into !H -> !H
  %ab_d = d2m.empty() : !T
  %ab = d2m.to_layout %ab_host, %ab_d : !H into !T -> !T
  %o1 = d2m.empty() : !T
  %b = d2m.generic {d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(0, 3), (1, 4)>, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%ab : !T) outs(%o1 : !T) {
  %blk = tensor.empty() : !S
  d2m.yield %blk : (!S)
  } : !T
  %out0_h = d2m.empty() : !H
  %out0 = d2m.to_layout %b, %out0_h : !T into !H -> !H
  %out1_h = d2m.empty() : !H
  %out1 = d2m.to_layout %c, %out1_h : !T into !H -> !H
  return %out0, %out1 : !H, !H
}

// -----

#l = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#lu = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>
#lo = #ttcore.metal_layout<logical_shape = 50x50, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
!T = tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #l>
!U = tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #lu>
!O = tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #lo>
!R = tensor<1x1x64x64xf32, #l>
!H = tensor<64x64xf32>
!S = tensor<2x2x!ttcore.tile<32x32, f32>>
// Two groups, the second reading the first's result from outside.
// CHECK-LABEL: func.func @two_groups
// CHECK: %[[S0:.*]]:2 = d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>]
// CHECK: d2m.to_layout %[[S0]]#1
// CHECK: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>], grid_ranges = [#ttcore.core_range<(3,0), (4,1)>, #ttcore.core_range<(3,3), (4,4)>]}
func.func @two_groups(%arg0: !H) -> (!H) {
  %x_d = d2m.empty() : !T
  %x = d2m.to_layout %arg0, %x_d : !H into !T -> !T
  %o0 = d2m.empty() : !T
  %a = d2m.generic {d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(0, 0), (1, 1)>, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%x : !T) outs(%o0 : !T) {
  %blk = tensor.empty() : !S
  d2m.yield %blk : (!S)
  } : !T
  %ab_h = d2m.empty() : !H
  %ab_host = d2m.to_layout %a, %ab_h : !T into !H -> !H
  %ab_d = d2m.empty() : !T
  %ab = d2m.to_layout %ab_host, %ab_d : !H into !T -> !T
  %o1 = d2m.empty() : !T
  %b = d2m.generic {d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(0, 3), (1, 4)>, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%ab : !T) outs(%o1 : !T) {
  %blk = tensor.empty() : !S
  d2m.yield %blk : (!S)
  } : !T
  %bc_h = d2m.empty() : !H
  %bc_host = d2m.to_layout %b, %bc_h : !T into !H -> !H
  %bc_d = d2m.empty() : !T
  %bc = d2m.to_layout %bc_host, %bc_d : !H into !T -> !T
  %o2 = d2m.empty() : !T
  %c = d2m.generic {d2m.spatial_group = 1 : i64, d2m.spatial_range = #ttcore.core_range<(3, 0), (4, 1)>, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%bc : !T) outs(%o2 : !T) {
  %blk = tensor.empty() : !S
  d2m.yield %blk : (!S)
  } : !T
  %cd_h = d2m.empty() : !H
  %cd_host = d2m.to_layout %c, %cd_h : !T into !H -> !H
  %cd_d = d2m.empty() : !T
  %cd = d2m.to_layout %cd_host, %cd_d : !H into !T -> !T
  %o3 = d2m.empty() : !T
  %d = d2m.generic {d2m.spatial_group = 1 : i64, d2m.spatial_range = #ttcore.core_range<(3, 3), (4, 4)>, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%cd : !T) outs(%o3 : !T) {
  %blk = tensor.empty() : !S
  d2m.yield %blk : (!S)
  } : !T
  %out_h = d2m.empty() : !H
  %out = d2m.to_layout %d, %out_h : !T into !H -> !H
  return %out : !H
}

// -----

#l = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#lu = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>
#lo = #ttcore.metal_layout<logical_shape = 50x50, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
!T = tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #l>
!U = tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #lu>
!O = tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #lo>
!R = tensor<1x1x64x64xf32, #l>
!H = tensor<64x64xf32>
!S = tensor<2x2x!ttcore.tile<32x32, f32>>
// A group with one member is a one-region spatial without edges.
// CHECK-LABEL: func.func @single
// CHECK: d2m.spatial {grid_ranges = [#ttcore.core_range<(0,3), (1,4)>]}
func.func @single(%arg0: !H) -> (!H) {
  %x_d = d2m.empty() : !T
  %x = d2m.to_layout %arg0, %x_d : !H into !T -> !T
  %o0 = d2m.empty() : !T
  %a = d2m.generic {d2m.spatial_group = 7 : i64, d2m.spatial_range = #ttcore.core_range<(0, 3), (1, 4)>, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%x : !T) outs(%o0 : !T) {
  %blk = tensor.empty() : !S
  d2m.yield %blk : (!S)
  } : !T
  %out_h = d2m.empty() : !H
  %out = d2m.to_layout %a, %out_h : !T into !H -> !H
  return %out : !H
}
