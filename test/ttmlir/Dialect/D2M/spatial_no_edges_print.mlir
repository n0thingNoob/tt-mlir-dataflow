// RUN: ttmlir-opt %s | FileCheck %s --strict-whitespace --match-full-lines
// A d2m.spatial without edges prints exactly as it did before edges were
// added. The CHECK lines are the verbatim output of ttmlir-opt built from main
// (f688ee694) without the edge support, so any change to the no-edge printed
// form fails here.

#l1 = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
module {
  func.func @no_edges(%in: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) -> (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
    %e = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
    %o = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
    %r:2 = d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(1, 1), (1, 1)>]}
        ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
        outs(%e, %o : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
      ^region0:
        %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%e : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %g0 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    }, {
      ^region1():
        %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%o : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %g1 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
    return %r#0, %r#1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  }
}

// CHECK:#layout = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<{{\[\[}}0, 1], [1, 2{{\]\]}}> : tensor<2x2xi64>, l1, sharded>
// CHECK-NEXT:#map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-NEXT:#parallel = #ttcore.iterator_type<parallel>
// CHECK-NEXT:module {
// CHECK-NEXT:  func.func @no_edges(%arg0: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>) -> (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>) {
// CHECK-NEXT:    %0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
// CHECK-NEXT:    %1 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
// CHECK-NEXT:    %2:2 = d2m.spatial {grid_ranges = [#ttcore.core_range<(0,0), (0,0)>, #ttcore.core_range<(1,1), (1,1)>]}
// CHECK-NEXT:        ins(%arg0 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>)
// CHECK-NEXT:        outs(%0, %1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>)  {
// CHECK-NEXT:      %3 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
// CHECK-NEXT:        ins(%arg0 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>)
// CHECK-NEXT:        outs(%0 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>)
// CHECK-NEXT:        {
// CHECK-NEXT:        %4 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
// CHECK-NEXT:        d2m.yield %4 : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
// CHECK-NEXT:      } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
// CHECK-NEXT:      d2m.spatial_yield %3 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>)
// CHECK-NEXT:    }, {
// CHECK-NEXT:      %3 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
// CHECK-NEXT:        ins(%arg0 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>)
// CHECK-NEXT:        outs(%1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>)
// CHECK-NEXT:        {
// CHECK-NEXT:        %4 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
// CHECK-NEXT:        d2m.yield %4 : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
// CHECK-NEXT:      } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
// CHECK-NEXT:      d2m.spatial_yield %3 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>)
// CHECK-NEXT:    } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
// CHECK-NEXT:    return %2#0, %2#1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
// CHECK-NEXT:  }
// CHECK-NEXT:}
