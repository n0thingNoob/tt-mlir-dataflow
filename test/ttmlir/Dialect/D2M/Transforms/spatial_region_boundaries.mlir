// RUN: ttmlir-opt %s -o %t.input
// RUN: ttmlir-opt %t.input --d2m-spatial-planning -o %t.silent 2> %t.stderr
// RUN: test ! -s %t.stderr
// RUN: diff %t.input %t.silent
// RUN: ttmlir-opt %t.input --d2m-spatial-planning="dump-regions=true" -o %t.output 2> %t.report
// RUN: FileCheck %s --input-file=%t.report
// RUN: diff %t.input %t.output
// RUN: ttmlir-opt %t.input --d2m-spatial-planning="dump-regions=true" -o /dev/null 2> %t.repeat
// RUN: diff %t.report %t.repeat
// RUN: ttmlir-opt %t.input --pass-pipeline='builtin.module(d2m-spatial-planning{dump-regions=true},d2m-spatial-planning{dump-regions=true})' -o %t.twice 2> %t.twice-report
// RUN: cat %t.report %t.report > %t.expected-twice
// RUN: diff %t.expected-twice %t.twice-report
// RUN: diff %t.input %t.twice

// Synthetic tensor generics isolate graph structure from conversion policies.
// Their empty bodies intentionally exercise unknown compute classification.
#layout = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
!grid = tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
!shard = tensor<1x1x!ttcore.tile<32x32, f32>>
#id = affine_map<(d0, d1) -> (d0, d1)>
#p = #ttcore.iterator_type<parallel>

// CHECK-LABEL: spatial_region_candidates @capture block0
// CHECK: G0 type=unknown interval=0 producers=[] consumers=[G1]
// CHECK: G1 type=unknown interval=0 producers=[G0] consumers=[]
// CHECK: captures=[%{{[0-9]+}} : tensor
// CHECK: G0 -> G1.operand0 {{.*}}connection=direct
// CHECK: members=[G0, G1] kinds=[producer_consumer_chain]
func.func @capture(%a: !grid, %init: !grid) -> !grid {
    %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%a : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%g0 : !grid) outs(%init : !grid) {
      %c0 = arith.constant 0 : index
      %size = tensor.dim %g0, %c0 : !grid
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    return %g1 : !grid
}

// CHECK-LABEL: spatial_region_candidates @multiple_results block0
// CHECK: G0 type=unknown interval=0 producers=[] consumers=[G1]
// CHECK: G1 type=unknown interval=0 producers=[G0] consumers=[]
// CHECK: G0 -> G1.operand0 source=%{{[0-9]+}}#1
// CHECK: G0 -> G1.operand1 source=%{{[0-9]+}}#0
// CHECK: members=[G0, G1] kinds=[producer_consumer_chain]
func.func @multiple_results(%a: !grid, %init: !grid) -> (!grid, !grid) {
    %pair:2 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%a : !grid) outs(%init, %init : !grid, !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty, %empty : (!shard, !shard)
    } : !grid, !grid
    %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%pair#1, %pair#0 : !grid, !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    return %pair#0, %g1 : !grid, !grid
}

// CHECK-LABEL: spatial_region_candidates @memory_barrier block0 candidate_only {
// CHECK: barriers = [memref.store
// CHECK: G0 type=unknown interval=0 producers=[] consumers=[G1]
// CHECK: G1 type=unknown interval=1 producers=[G0] consumers=[]
// CHECK: members=[G0] kinds=[singleton]
// CHECK: members=[G1] kinds=[singleton]
func.func @memory_barrier(%a: !grid, %b: !grid, %c: !grid, %init: !grid, %buffer: memref<1xi32>) -> !grid {
    %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%a : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %zero = arith.constant 0 : index
    %value = arith.constant 1 : i32
    memref.store %value, %buffer[%zero] : memref<1xi32>
    %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%g0 : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    return %g1 : !grid
}

// CHECK-LABEL: spatial_region_candidates @control_flow_barrier block0 candidate_only {
// CHECK: barriers = [scf.if
// CHECK: G1 type=unknown interval=1 producers=[G0] consumers=[]
// CHECK-NOT: G2 type=
// CHECK: members=[G0] kinds=[singleton]
// CHECK: members=[G1] kinds=[singleton]
func.func @control_flow_barrier(%a: !grid, %b: !grid, %c: !grid, %init: !grid, %condition: i1) -> !grid {
    %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%a : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    scf.if %condition {
    %g100 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%g0 : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    }
    %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%g0 : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    return %g1 : !grid
}

// CHECK-LABEL: spatial_region_candidates @spatial_barrier block0 candidate_only {
// CHECK: barriers = [d2m.spatial
// CHECK: G1 type=unknown interval=1 producers=[G0] consumers=[]
// CHECK-NOT: G2 type=
// CHECK: members=[G0] kinds=[singleton]
// CHECK: members=[G1] kinds=[singleton]
func.func @spatial_barrier(%a: !grid, %b: !grid, %c: !grid, %init: !grid) -> !grid {
    %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%a : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %spatial = d2m.spatial {grid_ranges = [#ttcore.core_range<(0,0), (0,0)>]} ins(%g0 : !grid) outs(%init : !grid) {
    %g100 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%g0 : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
      d2m.spatial_yield %g100 : (!grid)
    } : !grid
    %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%g0 : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    return %g1 : !grid
}

// CHECK-LABEL: spatial_region_candidates @fresh_tensor block0 candidate_only {
// CHECK: G1 type=unknown interval=0 producers=[G0] consumers=[]
// CHECK: members=[G0, G1] kinds=[producer_consumer_chain]
func.func @fresh_tensor(%a: !grid, %b: !grid, %c: !grid, %init: !grid) -> !grid {
    %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%a : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %fresh = d2m.empty() : !grid
    %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%g0 : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    return %g1 : !grid
}

// CHECK-LABEL: spatial_region_candidates @blocks block0
// CHECK: G0 type=unknown interval=0 producers=[] consumers=[]
// CHECK: members=[G0] kinds=[singleton]
// CHECK-LABEL: spatial_region_candidates @blocks block1
// CHECK: G0 type=unknown interval=0 producers=[] consumers=[]
// CHECK: members=[G0] kinds=[singleton]
func.func @blocks(%a: !grid, %init: !grid) -> !grid {
    %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%a : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    cf.br ^next(%g0 : !grid)
  ^next(%block_input: !grid):
    %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%block_input : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    return %g1 : !grid
}

// CHECK-LABEL: spatial_region_candidates @mixed block0
// CHECK: G0 type=mixed
// CHECK: compute_ops=[d2m.tile_matmul, d2m.tile_relu] resource_hints=[FPU, SFPU]
// CHECK: members=[G0] kinds=[singleton]
func.func @mixed(%a: !grid, %init: !grid) -> !grid {
    %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%a : !grid) outs(%init : !grid) {
      %c0 = arith.constant 0 : index
      %shard = tensor.empty() : !shard
      %tile = tensor.extract %shard[%c0, %c0] : !shard
      %mm = "d2m.tile_matmul"(%tile, %tile, %tile) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      %activation = "d2m.tile_relu"(%mm) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    return %g0 : !grid
}

// CHECK-LABEL: spatial_region_candidates @external_capture block0
// CHECK: G0 type=unknown interval=0 producers=[] consumers=[]
// CHECK: captures=[%arg2 : tensor
// CHECK: G1 type=unknown interval=0 producers=[] consumers=[]
// CHECK: external -> G0.capture0 source=%arg2
// CHECK: external -> G1.capture0 source=%arg2
// CHECK: members=[G0, G1] kinds=[parallel_branches]
// CHECK: context={ external_input=%arg2 }
func.func @external_capture(%a: !grid, %b: !grid, %captured: !grid, %init: !grid) -> !grid {
    %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%a : !grid) outs(%init : !grid) {
      %c0 = arith.constant 0 : index
      %size = tensor.dim %captured, %c0 : !grid
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%b : !grid) outs(%init : !grid) {
      %c0 = arith.constant 0 : index
      %size = tensor.dim %captured, %c0 : !grid
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    return %g1 : !grid
}

// CHECK-LABEL: spatial_region_candidates @matmul_block block0
// CHECK: G0 type=matmul
// CHECK: compute_ops=[d2m.tile_matmul_block] resource_hints=[FPU]
// CHECK: members=[G0] kinds=[singleton]
func.func @matmul_block(%a: !grid, %init: !grid) -> !grid {
  %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
      ins(%a : !grid) outs(%init : !grid) {
    %shard = tensor.empty() : !shard
    "d2m.tile_matmul_block"(%shard, %shard, %shard) : (!shard, !shard, !shard) -> ()
    d2m.yield %shard : (!shard)
  } : !grid
  return %g0 : !grid
}

// CHECK-LABEL: spatial_region_candidates @bufferized block0
// CHECK-NOT: G0 type=
// CHECK-NOT: candidate R
func.func @bufferized(%output: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>) {
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
      ins() outs(%output : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>) {
    ^bb0:
  }
  return
}
