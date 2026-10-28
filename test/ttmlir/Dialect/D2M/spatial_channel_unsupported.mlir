// RUN: not ttmlir-opt --ttcore-register-device --ttcore-one-shot-bufferize --convert-d2m-to-ttmetal %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=METAL
// RUN: not ttmlir-opt --ttcore-register-device --ttcore-one-shot-bufferize --convert-d2m-to-ttnn %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=TTNN

// Edges are synchronized by d2m-lower-spatial-edges (TTMetal only). An edge
// that reaches the TTMetal or TTNN lowering unlowered must fail with an
// explicit diagnostic instead of emitting unsynchronized code.

// METAL: error: 'd2m.spatial' op edges are not yet supported by convert-d2m-to-ttmetal
// TTNN: error: 'd2m.spatial' op edges are not yet supported by convert-d2m-to-ttnn

#layout = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1) -> (0, d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0 + 1, d1 + 1, d2, d3)>
#map3 = affine_map<(d0, d1) -> (0, d0 - 1, d1 - 1)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
module {
  func.func @chain_tanh_exp(%arg0: tensor<32x32xbf16>) -> (tensor<32x32xbf16>, tensor<32x32xbf16>) attributes {tt.function_type = "forward_device"} {
    %0 = d2m.empty() {virtualGridForwardMapping = #map, virtualGridInverseMapping = #map1} : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
    %1 = d2m.to_layout %arg0, %0 : tensor<32x32xbf16> into tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout> -> tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
    %view_in_0 = d2m.view_layout %1 remapping = #map : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout> -> tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
    %o0 = d2m.empty() {virtualGridForwardMapping = #map, virtualGridInverseMapping = #map1} : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
    %view_out0 = d2m.view_layout %o0 remapping = #map : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout> -> tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
    %o1 = d2m.empty() {virtualGridForwardMapping = #map2, virtualGridInverseMapping = #map3} : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
    %view_out1 = d2m.view_layout %o1 remapping = #map : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout> -> tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
    %r:2 = d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>], grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(1, 1), (1, 1)>]}
        ins(%view_in_0 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>)
        outs(%view_out0, %view_out1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>) {
      ^region0():
        %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map4, #map4], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%view_in_0 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>)
            outs(%view_out0 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>) {
          %block0 = d2m.block_index(0) : index
          %block1 = d2m.block_index(1) : index
          %t0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          %ld = d2m.remote_load %t0 %view_in_0[%block0, %block1] : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          %t1 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          %ev = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%ld : tensor<1x1x!ttcore.tile<32x32, bf16>>) outs(%t1 : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
          ^bb0(%in: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
            %t = "d2m.tile_tanh"(%in) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            linalg.yield %t : !ttcore.tile<32x32, bf16>
          } -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          %b0 = d2m.block_index(0) : index
          %b1 = d2m.block_index(1) : index
          %st = d2m.remote_store %view_out0[%b0, %b1] %ev : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
          d2m.yield %st : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
        d2m.spatial_yield %g0 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>)
    }, {
      ^region1(%x: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>):
        %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map4, #map4], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%x : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>)
            outs(%view_out1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>) {
          %block0 = d2m.block_index(0) : index
          %block1 = d2m.block_index(1) : index
          %t0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          %ld = d2m.remote_load %t0 %x[%block0, %block1] : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          %t1 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          %ev = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%ld : tensor<1x1x!ttcore.tile<32x32, bf16>>) outs(%t1 : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
          ^bb0(%in: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
            %e = "d2m.tile_exp"(%in) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            linalg.yield %e : !ttcore.tile<32x32, bf16>
          } -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          %b0 = d2m.block_index(0) : index
          %b1 = d2m.block_index(1) : index
          %st = d2m.remote_store %view_out1[%b0, %b1] %ev : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
          d2m.yield %st : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
        d2m.spatial_yield %g1 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>)
    } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
    %e0 = d2m.empty() : tensor<32x32xbf16>
    %e1 = d2m.empty() : tensor<32x32xbf16>
    %out0 = d2m.to_layout %r#0, %e0 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout> into tensor<32x32xbf16> -> tensor<32x32xbf16>
    %out1 = d2m.to_layout %r#1, %e1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout> into tensor<32x32xbf16> -> tensor<32x32xbf16>
    return %out0, %out1 : tensor<32x32xbf16>, tensor<32x32xbf16>
  }
}
