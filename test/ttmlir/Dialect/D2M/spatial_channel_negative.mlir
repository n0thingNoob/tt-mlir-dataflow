// RUN: ttmlir-opt --split-input-file --verify-diagnostics %s
// Negative tests for d2m.spatial edges (docs/src/specs/d2m-spatial-channel.md, section 5).
// One case per verifier rule.

#l1 = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// R1 cycle: edges 0:0 -> 1:0 and 1:0 -> 0:0
func.func @r1_cycle(%in: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
  %e0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  %e1 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  // expected-error @+1 {{edges must not form a cycle between regions}}
  %r:2 = d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>, #d2m.spatial_edge<1:0 -> 0:0>], grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(0, 1), (0, 1)>]}
      ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
      outs(%e0, %e1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
      ^bb0(%y: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>):
        %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%y : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%e0 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %g0 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    }, {
      ^bb0(%x: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>):
        %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%x : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%e1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %g1 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  return
}

// -----

#l1 = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// R2 self edge
func.func @r2_self_edge(%in: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
  %e0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  %e1 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  // expected-error @+1 {{edge #d2m.spatial_edge<0:0 -> 0:0> connects region 0 to itself}}
  %r:2 = d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 0:0>], grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(0, 1), (0, 1)>]}
      ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
      outs(%e0, %e1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
      ^bb0(%y: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>):
        %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%y : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%e0 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %g0 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    }, {
        %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%e1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %g1 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  return
}

// -----

#l1 = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// R3a region out of range
func.func @r3a_region_out_of_range(%in: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
  %e0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  %e1 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  // expected-error @+1 {{edge #d2m.spatial_edge<5:0 -> 1:0> refers to a region out of range (2 regions)}}
  %r:2 = d2m.spatial {edges = [#d2m.spatial_edge<5:0 -> 1:0>], grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(0, 1), (0, 1)>]}
      ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
      outs(%e0, %e1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
        %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%e0 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %g0 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    }, {
      ^bb0(%x: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>):
        %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%x : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%e1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %g1 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  return
}

// -----

#l1 = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// R3b producer output out of range
func.func @r3b_producer_output_out_of_range(%in: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
  %e0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  %e1 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  // expected-error @+1 {{edge #d2m.spatial_edge<0:3 -> 1:0>: producer generic in region 0 has 1 outputs}}
  %r:2 = d2m.spatial {edges = [#d2m.spatial_edge<0:3 -> 1:0>], grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(0, 1), (0, 1)>]}
      ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
      outs(%e0, %e1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
        %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%e0 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %g0 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    }, {
      ^bb0(%x: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>):
        %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%x : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%e1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %g1 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  return
}

// -----

#l1 = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// R3c consumer argument out of range
func.func @r3c_consumer_argument_out_of_range(%in: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
  %e0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  %e1 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  // expected-error @+1 {{edge #d2m.spatial_edge<0:0 -> 1:2>: region 1 has 1 block arguments}}
  %r:2 = d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:2>], grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(0, 1), (0, 1)>]}
      ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
      outs(%e0, %e1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
        %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%e0 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %g0 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    }, {
      ^bb0(%x: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>):
        %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%x : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%e1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %g1 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  return
}

// -----

#l1 = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// R4 block argument without edge
func.func @r4_block_argument_without_edge(%in: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
  %e0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  %e1 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  // expected-error @+1 {{block argument 0 of region 1 has no incoming edge}}
  %r:2 = d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(0, 1), (0, 1)>]}
      ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
      outs(%e0, %e1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
        %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%e0 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %g0 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    }, {
      ^bb0(%x: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>):
        %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%x : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%e1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %g1 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  return
}

// -----

#l1 = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// R4 two edges into one argument
func.func @r4_two_edges_into_one_argument(%in: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
  %e0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  %e1 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  %e2 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  // expected-error @+1 {{block argument 0 of region 1 has more than one incoming edge}}
  %r:3 = d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>, #d2m.spatial_edge<0:1 -> 1:0>], grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(0, 1), (0, 1)>]}
      ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
      outs(%e0, %e1, %e2 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
        %g0:2 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%e0, %e1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          %t1 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t0, %t1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %g0#0, %g0#1 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    }, {
      ^bb0(%x: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>):
        %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%x : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%e2 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %g1 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  return
}

// -----

#l1 = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// R4b fan-out from one output
func.func @r4b_fan_out_from_one_output(%in: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
  %e0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  %e1 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  %e2 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  // expected-error @+1 {{output 0 of region 0 feeds more than one edge; only one-to-one edges are supported}}
  %r:3 = d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>, #d2m.spatial_edge<0:0 -> 2:0>], grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(0, 1), (0, 1)>, #ttcore.core_range<(0, 2), (0, 2)>]}
      ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
      outs(%e0, %e1, %e2 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
        %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%e0 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %g0 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    }, {
      ^bb0(%x: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>):
        %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%x : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%e1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %g1 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    }, {
      ^bb0(%z: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>):
        %g2 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%z : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%e2 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %g2 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  return
}

// -----

#l1 = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// R5 type mismatch
func.func @r5_type_mismatch(%in: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
  %e0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  %e1 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #l1>
  // expected-error @+1 {{edge #d2m.spatial_edge<0:0 -> 1:0>: block argument type}}
  %r:2 = d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>], grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(0, 1), (0, 1)>]}
      ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
      outs(%e0, %e1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #l1>) {
        %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%e0 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %g0 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    }, {
      ^bb0(%x: tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #l1>):
        %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%x : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #l1>)
            outs(%e1 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #l1>) {
          %t0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #l1>
        d2m.spatial_yield %g1 : (tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #l1>)
    } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #l1>
  return
}

// -----

#l1 = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// R6 grid mismatch
func.func @r6_grid_mismatch(%in: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
  %e0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  %e1 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  // expected-error @+1 {{edge #d2m.spatial_edge<0:0 -> 1:0>: producer grid [1, 1] and consumer grid [1, 2] must match}}
  %r:2 = d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>], grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(1, 0), (1, 1)>]}
      ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
      outs(%e0, %e1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
        %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%e0 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %g0 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    }, {
      ^bb0(%x: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>):
        %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x2>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%x : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%e1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %g1 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  return
}

// -----

#l1 = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// R7 edge buffer not in outs
func.func @r7_edge_buffer_not_in_outs(%in: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
  %e0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  %e1 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  // expected-error @+1 {{edge #d2m.spatial_edge<0:0 -> 1:0>: producer output must be one of the spatial outs}}
  %r = d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>], grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(0, 1), (0, 1)>]}
      ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
      outs(%e1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
        %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%e0 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %g0 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    }, {
      ^bb0(%x: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>):
        %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%x : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%e1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %g1 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  return
}

// -----

#l1 = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// R8 block argument used outside the generic
func.func @r8_block_argument_used_outside_the_generic(%in: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
  %e0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  %e1 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  // expected-error @+1 {{block argument 0 of region 1 may only be used inside the region's d2m.generic}}
  %r:2 = d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>], grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(0, 1), (0, 1)>]}
      ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
      outs(%e0, %e1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
        %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%e0 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %g0 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    }, {
      ^bb0(%x: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>):
        %rank = tensor.rank %x : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%x : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%e1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %g1 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  return
}

// -----

#l1 = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// R8 block argument as generic output
func.func @r8_block_argument_as_generic_output(%in: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
  %e0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  // expected-error @+1 {{block argument 0 of region 1 must not be an output of the region's d2m.generic}}
  %r = d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>], grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(0, 1), (0, 1)>]}
      ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
      outs(%e0 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
        %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%e0 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %g0 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    }, {
      ^bb0(%x: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>):
        %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%x : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
    } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  return
}

// -----

#l1 = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// R9 spatial_yield order
func.func @r9_spatial_yield_order(%in: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
  %e0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  %e1 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  %e2 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  %r:3 = d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>], grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(0, 1), (0, 1)>]}
      ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
      outs(%e0, %e1, %e2 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
        %g0:2 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%e0, %e1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          %t1 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t0, %t1 : (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        // expected-error @+1 {{'d2m.spatial_yield' op in a d2m.spatial with edges must yield the region generic's results in order}}
        d2m.spatial_yield %g0#1, %g0#0 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    }, {
      ^bb0(%x: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>):
        %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%x : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%e2 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t0 : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %g1 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  return
}

// -----

#l1 = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// Bufferized spatial keeps block arguments
func.func @bufferized_spatial_keeps_block_arguments(%in: memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #ttcore.memory_space<l1>>) {
  %e0 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #ttcore.memory_space<l1>>
  %e1 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #ttcore.memory_space<l1>>
  // expected-error @+1 {{region 1 has block arguments after bufferization}}
  d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>], grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(0, 1), (0, 1)>]}
      ins(%in : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #ttcore.memory_space<l1>>)
      outs(%e0, %e1 : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #ttcore.memory_space<l1>>, memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #ttcore.memory_space<l1>>) {
        d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%in : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #ttcore.memory_space<l1>>)
            outs(%e0 : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #ttcore.memory_space<l1>>) {
        }
    }, {
      ^bb0(%x: memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #ttcore.memory_space<l1>>):
        d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%x : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #ttcore.memory_space<l1>>)
            outs(%e1 : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #ttcore.memory_space<l1>>) {
        }
    }
  return
}
