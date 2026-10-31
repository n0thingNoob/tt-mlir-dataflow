// RUN: ttmlir-opt --split-input-file --verify-diagnostics %s
// Negative tests for streaming spatial edges (R11-R13 in
// docs/src/specs/d2m-spatial-channel.md, section 5).

#l1 = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// R11: the two edges that leave A:0 use different protocols.
func.func @r11_mixed_protocols(%in: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) -> tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1> {
  %a0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  %b0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  %c0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  // expected-error @+1 {{edges #d2m.spatial_edge<0:0 -> 1:0, stream<depth = 2>> and #d2m.spatial_edge<0:0 -> 2:0> leave the same output but use different protocols}}
  %r:3 = d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0, stream<depth = 2>>, #d2m.spatial_edge<0:0 -> 2:0>, #d2m.spatial_edge<1:0 -> 2:1>],
                      grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(0, 1), (0, 1)>, #ttcore.core_range<(0, 2), (0, 2)>]}
      ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
      outs(%a0, %b0, %c0 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
      ^A:
        %ga = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%a0 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %ga : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    }, {
      ^B(%x: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>):
        %gb = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%x : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%b0 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %gb : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    }, {
      ^C(%y: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, %z: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>):
        %gc = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%y, %z : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%c0 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %gc : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  return %r#2 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
}

// -----

#l1 = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// R11: same depth, different blocks.
func.func @r11_mixed_blocks(%in: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) -> tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1> {
  %a0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  %b0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  %c0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  // expected-error @+1 {{leave the same output but use different protocols}}
  %r:3 = d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0, stream<depth = 2, block = 1x1>>, #d2m.spatial_edge<0:0 -> 2:0, stream<depth = 2>>, #d2m.spatial_edge<1:0 -> 2:1>],
                      grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(0, 1), (0, 1)>, #ttcore.core_range<(0, 2), (0, 2)>]}
      ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
      outs(%a0, %b0, %c0 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
      ^A:
        %ga = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%a0 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %ga : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    }, {
      ^B(%x: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>):
        %gb = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%x : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%b0 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %gb : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    }, {
      ^C(%y: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, %z: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>):
        %gc = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%y, %z : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%c0 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %gc : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  return %r#2 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
}

// -----

#l1 = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// R12: depth above the maximum.
func.func @r12_depth_too_large() attributes {
  // expected-error @+1 {{stream depth must be in [1, 8], got 9}}
  d2m.edge = #d2m.spatial_edge<0:0 -> 1:0, stream<depth = 9>>} {
  return
}

// -----

#l1 = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// R12: depth 0 is the whole-tensor handoff, written without `stream`.
func.func @r12_depth_zero() attributes {
  // expected-error @+1 {{stream depth must be in [1, 8], got 0}}
  d2m.edge = #d2m.spatial_edge<0:0 -> 1:0, stream<depth = 0>>} {
  return
}

// -----

#l1 = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// R13: a block has two dimensions.
func.func @r13_block_rank() attributes {
  // expected-error @+1 {{stream block must be RxC tiles with R, C > 0}}
  d2m.edge = #d2m.spatial_edge<0:0 -> 1:0, stream<depth = 2, block = 1x1x1>>} {
  return
}

// -----

#l1 = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// R13: block dimensions are positive.
func.func @r13_block_zero() attributes {
  // expected-error @+1 {{stream block must be RxC tiles with R, C > 0}}
  d2m.edge = #d2m.spatial_edge<0:0 -> 1:0, stream<depth = 2, block = 0x1>>} {
  return
}
