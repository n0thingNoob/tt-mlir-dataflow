// RUN: ttmlir-opt %s | FileCheck %s
// RUN: ttmlir-opt %s | ttmlir-opt | FileCheck %s
// RUN: ttmlir-opt --ttcore-register-device --ttcore-one-shot-bufferize %s | FileCheck %s --check-prefix=BUF

// Streaming edges (stream<depth = D[, block = RxC]>) parse, print, round-trip
// and survive bufferization. A: 0 feeds B and C (fan-out, one protocol for
// both), B -> C streams with depth 1, no block.

// CHECK-LABEL: func.func @diamond_stream
// CHECK: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0, stream<depth = 2, block = 1x1>>, #d2m.spatial_edge<0:0 -> 2:0, stream<depth = 2, block = 1x1>>, #d2m.spatial_edge<1:0 -> 2:1, stream<depth = 1>>]
// BUF-LABEL: func.func @diamond_stream
// BUF: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0, stream<depth = 2, block = 1x1>>, #d2m.spatial_edge<0:0 -> 2:0, stream<depth = 2, block = 1x1>>, #d2m.spatial_edge<1:0 -> 2:1, stream<depth = 1>>]
#l1 = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
func.func @diamond_stream(%in: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) -> tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1> {
  %a0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  %b0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  %c0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  %r:3 = d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0, stream<depth = 2, block = 1x1>>, #d2m.spatial_edge<0:0 -> 2:0, stream<depth = 2, block = 1x1>>, #d2m.spatial_edge<1:0 -> 2:1, stream<depth = 1>>],
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
