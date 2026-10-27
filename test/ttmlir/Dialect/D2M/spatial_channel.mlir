// RUN: ttmlir-opt --split-input-file %s | FileCheck %s
// RUN: ttmlir-opt --split-input-file %s | ttmlir-opt --split-input-file | FileCheck %s
// RUN: ttmlir-opt --split-input-file --ttcore-register-device --d2m-grid-selection %s | FileCheck %s --check-prefix=GS
// RUN: ttmlir-opt --split-input-file --ttcore-register-device --ttcore-one-shot-bufferize %s | FileCheck %s --check-prefix=BUF
// RUN: ttmlir-opt --split-input-file --ttcore-register-device --ttcore-one-shot-bufferize %s | FileCheck %s --check-prefix=NOTENSOR

// d2m.spatial region-to-region edges: parse, verify, round-trip, and survive
// the normalization that d2m-grid-selection runs on every spatial generic.
// Bufferization replaces each consumer block argument with the producer's
// buffer and keeps `edges`.

// NOTENSOR-NOT: bufferization.to_tensor
// NOTENSOR-NOT: ^bb0(

#l1 = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// Two-region chain: region 0 writes %e, region 1 reads it as block arg %x.
// CHECK-LABEL: func.func @chain
// CHECK: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>]
// CHECK-NEXT: ins(%arg0 :
// CHECK-NEXT: outs(%[[E:.*]], %[[O:.*]] :
// CHECK: ^bb0(%[[X:.*]]: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>):
// CHECK-NEXT: d2m.generic
// CHECK-NEXT: ins(%[[X]] :
// CHECK-NEXT: outs(%[[O]] :
// GS-LABEL: func.func @chain
// GS: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>]
// GS-NEXT: ins(%arg0 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>)
// GS: ^bb0(%[[X:.*]]: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>):
// GS-NEXT: d2m.generic
// GS-NEXT: ins(%[[X]] :
// GS: d2m.spatial_yield
// BUF-LABEL: func.func @chain
// BUF: %[[E:.*]] = memref.alloc()
// BUF: %[[O:.*]] = memref.alloc()
// BUF: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>]
// BUF-NEXT: ins(%arg0 :
// BUF-NEXT: outs(%[[E]], %[[O]] :
// BUF-NEXT: d2m.generic
// BUF-NEXT: ins(%arg0 :
// BUF-NEXT: outs(%[[E]] :
// BUF: d2m.generic
// BUF-NEXT: ins(%[[E]] :
// BUF-NEXT: outs(%[[O]] :
module {
  func.func @chain(%in: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) -> tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1> {
    %e = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
    %o = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
    %r:2 = d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>],
                        grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(1, 1), (1, 1)>]}
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
      ^region1(%x: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>):
        %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%x : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%o : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t : (tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %g1 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
    } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
    return %r#1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  }
}

// -----

#l1 = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// Three-region diamond: A -> B, A -> C, B -> C. A has two outputs so every
// edge stays one-to-one; C has two block arguments fed by A:1 and B:0. The
// edges are listed out of order on purpose.
// CHECK-LABEL: func.func @diamond
// CHECK: d2m.spatial {edges = [#d2m.spatial_edge<1:0 -> 2:1>, #d2m.spatial_edge<0:0 -> 1:0>, #d2m.spatial_edge<0:1 -> 2:0>]
// CHECK: ^bb0(%[[X:.*]]: tensor
// CHECK-NEXT: d2m.generic
// CHECK-NEXT: ins(%[[X]] :
// CHECK: ^bb0(%[[Y:.*]]: tensor<{{.*}}>, %[[Z:.*]]: tensor<{{.*}}>):
// CHECK-NEXT: d2m.generic
// CHECK-NEXT: ins(%[[Y]], %[[Z]] :
// GS-LABEL: func.func @diamond
// GS: d2m.spatial {edges = [#d2m.spatial_edge<1:0 -> 2:1>, #d2m.spatial_edge<0:0 -> 1:0>, #d2m.spatial_edge<0:1 -> 2:0>]
// GS-NEXT: ins(%arg0 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>)
// GS-NEXT: outs(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} :
// GS: ^bb0(%[[X:.*]]: tensor
// GS-NEXT: d2m.generic
// GS-NEXT: ins(%[[X]] :
// GS: ^bb0(%[[Y:.*]]: tensor<{{.*}}>, %[[Z:.*]]: tensor<{{.*}}>):
// GS-NEXT: d2m.generic
// GS-NEXT: ins(%[[Y]], %[[Z]] :
// BUF-LABEL: func.func @diamond
// BUF: %[[A0:.*]] = memref.alloc()
// BUF: %[[A1:.*]] = memref.alloc()
// BUF: %[[B0:.*]] = memref.alloc()
// BUF: %[[C0:.*]] = memref.alloc()
// BUF: d2m.spatial {edges = [#d2m.spatial_edge<1:0 -> 2:1>, #d2m.spatial_edge<0:0 -> 1:0>, #d2m.spatial_edge<0:1 -> 2:0>]
// BUF: outs(%[[A0]], %[[A1]] :
// BUF: ins(%[[A0]] :
// BUF-NEXT: outs(%[[B0]] :
// BUF: ins(%[[A1]], %[[B0]] :
// BUF-NEXT: outs(%[[C0]] :
module {
  func.func @diamond(%in: tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) -> tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1> {
    %a0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
    %a1 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
    %b0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
    %c0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
    %r:4 = d2m.spatial {edges = [#d2m.spatial_edge<1:0 -> 2:1>, #d2m.spatial_edge<0:0 -> 1:0>, #d2m.spatial_edge<0:1 -> 2:0>],
                        grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(0, 1), (0, 1)>, #ttcore.core_range<(0, 2), (0, 2)>]}
        ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
        outs(%a0, %a1, %b0, %c0 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
      ^A:
        %ga:2 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%a0, %a1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>) {
          %t = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          %u = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
          d2m.yield %t, %u : (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
        d2m.spatial_yield %ga#0, %ga#1 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>)
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
    } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
    return %r#3 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #l1>
  }
}
