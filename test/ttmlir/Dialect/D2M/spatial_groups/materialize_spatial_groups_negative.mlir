// RUN: ttmlir-opt --split-input-file --ttcore-register-device --d2m-materialize-spatial-groups="exclude-cores=2,2" --verify-diagnostics %s

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
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
// %b sits between the two members of group 0.
func.func @not_convex(%arg0: !H) -> (!H) {
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
  %b = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%ab : !T) outs(%o1 : !T) {
  %blk = tensor.empty() : !S
  d2m.yield %blk : (!S)
  } : !T
  %bc_h = d2m.empty() : !H
  %bc_host = d2m.to_layout %b, %bc_h : !T into !H -> !H
  %bc_d = d2m.empty() : !T
  %bc = d2m.to_layout %bc_host, %bc_d : !H into !T -> !T
  %o2 = d2m.empty() : !T
  // expected-error @+1 {{spatial group 0 is not convex: a generic outside the group lies on a path into this member}}
  %c = d2m.generic {d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(0, 3), (1, 4)>, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%bc : !T) outs(%o2 : !T) {
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
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
func.func @overlap(%arg0: !H) -> (!H, !H) {
  %x_d = d2m.empty() : !T
  %x = d2m.to_layout %arg0, %x_d : !H into !T -> !T
  %y_d = d2m.empty() : !T
  %y = d2m.to_layout %arg0, %y_d : !H into !T -> !T
  %o0 = d2m.empty() : !T
  %a = d2m.generic {d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(0, 0), (1, 1)>, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%x : !T) outs(%o0 : !T) {
  %blk = tensor.empty() : !S
  d2m.yield %blk : (!S)
  } : !T
  %o1 = d2m.empty() : !T
  // expected-error @+1 {{core range #ttcore.core_range<(1,1), (2,1)> overlaps #ttcore.core_range<(0,0), (1,1)> of group member 0}}
  %b = d2m.generic {d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(1, 1), (2, 1)>, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%y : !T) outs(%o1 : !T) {
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
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
func.func @excluded_core(%arg0: !H) -> (!H) {
  %x_d = d2m.empty() : !T
  %x = d2m.to_layout %arg0, %x_d : !H into !T -> !T
  %o0 = d2m.empty() : !T
  // expected-error @+1 {{core range #ttcore.core_range<(2,2), (3,3)> contains excluded core (2, 2)}}
  %a = d2m.generic {d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(2, 2), (3, 3)>, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%x : !T) outs(%o0 : !T) {
  %blk = tensor.empty() : !S
  d2m.yield %blk : (!S)
  } : !T
  %out_h = d2m.empty() : !H
  %out = d2m.to_layout %a, %out_h : !T into !H -> !H
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
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
func.func @outside_grid(%arg0: !H) -> (!H) {
  %x_d = d2m.empty() : !T
  %x = d2m.to_layout %arg0, %x_d : !H into !T -> !T
  %o0 = d2m.empty() : !T
  // expected-error @+1 {{core range #ttcore.core_range<(7,7), (8,8)> is empty or outside the worker grid}}
  %a = d2m.generic {d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(7, 7), (8, 8)>, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%x : !T) outs(%o0 : !T) {
  %blk = tensor.empty() : !S
  d2m.yield %blk : (!S)
  } : !T
  %out_h = d2m.empty() : !H
  %out = d2m.to_layout %a, %out_h : !T into !H -> !H
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
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
func.func @missing_range(%arg0: !H) -> (!H) {
  %x_d = d2m.empty() : !T
  %x = d2m.to_layout %arg0, %x_d : !H into !T -> !T
  %o0 = d2m.empty() : !T
  // expected-error @+1 {{spatial group member needs a 'd2m.spatial_range' core range}}
  %a = d2m.generic {d2m.spatial_group = 0 : i64, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%x : !T) outs(%o0 : !T) {
  %blk = tensor.empty() : !S
  d2m.yield %blk : (!S)
  } : !T
  %out_h = d2m.empty() : !H
  %out = d2m.to_layout %a, %out_h : !T into !H -> !H
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
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
// The consumer reads the producer as row-major: a tilize, not an edge.
func.func @format_change(%arg0: !H) -> (!H) {
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
  %a2_d = d2m.empty() : !R
  %a2 = d2m.to_layout %a2_host, %a2_d : !H into !R -> !R
  %o1 = d2m.empty() : !T
  // expected-error @+1 {{input 0 needs a layout change from its producer in spatial group 0; resharding an edge is not supported}}
  %b = d2m.generic {d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(0, 3), (1, 4)>, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%a2 : !R) outs(%o1 : !T) {
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
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
// A mask over real padding (logical 50x50) lowers to its own generic later.
func.func @real_mask(%arg0: !H) -> (!H) {
  %x_d = d2m.empty() : !O
  %x = d2m.to_layout %arg0, %x_d : !H into !O -> !O
  %o0 = d2m.empty() : !O
  %a = d2m.generic {d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(0, 0), (1, 1)>, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%x : !O) outs(%o0 : !O) {
  %blk = tensor.empty() : !S
  d2m.yield %blk : (!S)
  } : !O
  %m0 = d2m.empty() : !O
  %a3 = d2m.mask %a, %m0 logical_shape = [50, 50] fill_value = <zero> : !O into !O -> !O
  %o1 = d2m.empty() : !O
  // expected-error @+1 {{input 0 reads another member of spatial group 0 through ops that cannot be an edge (only to_layout, no-op mask and a composite_view of such chains)}}
  %b = d2m.generic {d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(0, 3), (1, 4)>, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%a3 : !O) outs(%o1 : !O) {
  %blk = tensor.empty() : !S
  d2m.yield %blk : (!S)
  } : !O
  %out_h = d2m.empty() : !H
  %out = d2m.to_layout %b, %out_h : !O into !H -> !H
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
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
// A view between two members (e.g. a transpose).
func.func @view(%arg0: !H) -> (!H) {
  %x_d = d2m.empty() : !T
  %x = d2m.to_layout %arg0, %x_d : !H into !T -> !T
  %o0 = d2m.empty() : !T
  %a = d2m.generic {d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(0, 0), (1, 1)>, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%x : !T) outs(%o0 : !T) {
  %blk = tensor.empty() : !S
  d2m.yield %blk : (!S)
  } : !T
  %v = d2m.view_layout %a remapping = #id4 : !T -> !T
  %o1 = d2m.empty() : !T
  // expected-error @+1 {{input 0 reads another member of spatial group 0 through ops that cannot be an edge (only to_layout, no-op mask and a composite_view of such chains)}}
  %b = d2m.generic {d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(0, 3), (1, 4)>, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%v : !T) outs(%o1 : !T) {
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
#ld = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, dram, sharded>
!D = tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #ld>
#id4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
// Both members' tensors are in DRAM: an edge buffer must stay in L1.
func.func @dram(%arg0: !H) -> (!H) {
  %x_d = d2m.empty() : !D
  %x = d2m.to_layout %arg0, %x_d : !H into !D -> !D
  %o0 = d2m.empty() : !D
  %a = d2m.generic {d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(0, 0), (1, 1)>, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%x : !D) outs(%o0 : !D) {
  %blk = tensor.empty() : !S
  d2m.yield %blk : (!S)
  } : !D
  %o1 = d2m.empty() : !D
  // expected-error @+1 {{input 0 reads another member of spatial group 0 outside L1; an edge buffer must stay in L1}}
  %b = d2m.generic {d2m.spatial_group = 0 : i64, d2m.spatial_range = #ttcore.core_range<(0, 3), (1, 4)>, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%a : !D) outs(%o1 : !D) {
  %blk = tensor.empty() : !S
  d2m.yield %blk : (!S)
  } : !D
  %out_h = d2m.empty() : !H
  %out = d2m.to_layout %b, %out_h : !D into !H -> !H
  return %out : !H
}
