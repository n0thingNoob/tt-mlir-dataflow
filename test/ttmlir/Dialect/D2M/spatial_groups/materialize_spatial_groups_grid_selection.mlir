// RUN: ttmlir-opt --ttcore-register-device --d2m-materialize-spatial-groups --d2m-grid-selection %s | FileCheck %s

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
// A collapse-only edge after GridSelection: both regions get 2x2 grids, the
// consumer's block argument and edge operand follow the producer's 2x2x1x1
// layout, and the consumer region is placed at its range's origin.
// CHECK-LABEL: func.func @collapse_then_gs
// CHECK: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>]
// CHECK: d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x2>
// CHECK-NEXT: ins(%{{.*}} : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>)
// CHECK: ^bb0(%[[A:.*]]: tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>):
// CHECK: d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x2, virt_to_physical_map = (d0, d1) -> (0, d0, d1 + 3)
// CHECK-NEXT: ins(%[[A]] : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>)
// CHECK-NEXT: outs(%{{.*}} : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout1>)
func.func @collapse_then_gs(%arg0: !H) -> (!H) {
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
  %i0 = d2m.block_index(0) : index
  %i1 = d2m.block_index(1) : index
  %t = tensor.empty() : !S
  %blk = d2m.remote_load %t %a2[%i0, %i1] : !S, !U -> !S
  d2m.yield %blk : (!S)
  } : !U
  %out_h = d2m.empty() : !H
  %out = d2m.to_layout %b, %out_h : !U into !H -> !H
  return %out : !H
}
