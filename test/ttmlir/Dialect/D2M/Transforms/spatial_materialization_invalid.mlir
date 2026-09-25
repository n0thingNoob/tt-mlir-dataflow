// RUN: not ttmlir-opt %s --ttcore-register-device --d2m-spatial-planning="materialize=true" --mlir-print-ir-after-failure --mlir-disable-threading -o /dev/null 2> %t.report
// RUN: FileCheck %s --input-file=%t.report --implicit-check-not='d2m.spatial {'
// A later unsupported function must fail before the earlier valid function
// has been modified. Hand-mapped grids are outside automatic placement scope.
// CHECK: spatial materialization supports only static tensor generics with ordinary 2D grids fitting the device
// CHECK: IR Dump After
// CHECK: func.func @pair
// CHECK: func.func @manual

#layout = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
!grid = tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
!shard = tensor<1x1x!ttcore.tile<32x32, f32>>
#id = affine_map<(d0, d1) -> (d0, d1)>
#p = #ttcore.iterator_type<parallel>

func.func @pair(%a: !grid) -> (!grid, !grid) {
  %o0 = d2m.empty() {virtualGridForwardMapping = affine_map<(a,b,c,d) -> (a,b,c,d)>, virtualGridInverseMapping = affine_map<(a,b) -> (0,a,b)>} : !grid
  %g0 = d2m.generic {test.id = 0 : i64, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
      ins(%a : !grid) outs(%o0 : !grid) {
    %empty = tensor.empty() : !shard
    d2m.yield %empty : (!shard)
  } : !grid
  %o1 = d2m.empty() {virtualGridForwardMapping = affine_map<(a,b,c,d) -> (a,b,c,d)>, virtualGridInverseMapping = affine_map<(a,b) -> (0,a,b)>} : !grid
  %g1 = d2m.generic {test.id = 1 : i64, block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
      ins(%a : !grid) outs(%o1 : !grid) {
    %empty = tensor.empty() : !shard
    d2m.yield %empty : (!shard)
  } : !grid
  return %g0, %g1 : !grid, !grid
}
func.func @manual(%a: !grid) -> (!grid, !grid) {
  %o0 = d2m.empty() {virtualGridForwardMapping = affine_map<(a,b,c,d) -> (a,b,c,d)>, virtualGridInverseMapping = affine_map<(a,b) -> (0,a,b)>} : !grid
  %g0 = d2m.generic {test.id = 0 : i64, block_factors = [1, 1], grid = #ttcore.grid<1x1, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
      ins(%a : !grid) outs(%o0 : !grid) {
    %empty = tensor.empty() : !shard
    d2m.yield %empty : (!shard)
  } : !grid
  %o1 = d2m.empty() {virtualGridForwardMapping = affine_map<(a,b,c,d) -> (a,b,c,d)>, virtualGridInverseMapping = affine_map<(a,b) -> (0,a,b)>} : !grid
  %g1 = d2m.generic {test.id = 1 : i64, block_factors = [1, 1], grid = #ttcore.grid<1x1, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
      ins(%a : !grid) outs(%o1 : !grid) {
    %empty = tensor.empty() : !shard
    d2m.yield %empty : (!shard)
  } : !grid
  return %g0, %g1 : !grid, !grid
}
