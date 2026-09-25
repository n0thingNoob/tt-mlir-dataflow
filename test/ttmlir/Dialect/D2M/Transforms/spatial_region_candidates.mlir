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

// CHECK-LABEL: spatial_region_candidates @chain block0 candidate_only {
// CHECK: G0 type=unknown interval=0 producers=[] consumers=[G1]
// CHECK: G1 type=unknown interval=0 producers=[G0] consumers=[G2]
// CHECK: G2 type=unknown interval=0 producers=[G1] consumers=[]
// CHECK: G0 -> G1.operand0 {{.*}}connection=direct via=[]
// CHECK: user=func.return.operand0
// CHECK: members=[G0, G1, G2] kinds=[producer_consumer_chain]
// CHECK: independent=[]
// CHECK: incoming=[{{.+}}] outgoing=[] external_uses=[{{.+}}]
// CHECK-NOT: candidate R1
func.func @chain(%a: !grid, %b: !grid, %c: !grid, %init: !grid) -> !grid {
    %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%a : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%g0 : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %g2 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%g1 : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    return %g2 : !grid
}

// CHECK-LABEL: spatial_region_candidates @diamond block0 candidate_only {
// CHECK: G0 type=unknown interval=0 producers=[] consumers=[G1, G2]
// CHECK: G3 type=unknown interval=0 producers=[G1, G2] consumers=[]
// CHECK: candidate R0
// CHECK: members=[G0, G1, G2, G3] kinds=[fork_join]
// CHECK: independent=[(G1, G2)]
// CHECK: context={ producer=G0 consumer=G3 }
// CHECK: candidate R1
// CHECK: members=[G1, G2] kinds=[parallel_branches]
// CHECK: context={ producer=G0 }
// CHECK: context={ consumer=G3 }
// CHECK-NOT: candidate R2
func.func @diamond(%a: !grid, %b: !grid, %c: !grid, %init: !grid) -> !grid {
    %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%a : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%g0 : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %g2 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%g0 : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %g3 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%g1, %g2 : !grid, !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    return %g3 : !grid
}

// CHECK-LABEL: spatial_region_candidates @extended_diamond block0 candidate_only {
// CHECK: members=[G0, G1, G2, G3, G4, G5] kinds=[fork_join]
// CHECK: independent=[(G1, G3), (G1, G4), (G2, G3), (G2, G4)]
// CHECK: members=[G1, G2] kinds=[producer_consumer_chain]
// CHECK: members=[G1, G3] kinds=[parallel_branches]
// CHECK: members=[G2, G4] kinds=[parallel_branches]
// CHECK: members=[G3, G4] kinds=[producer_consumer_chain]
func.func @extended_diamond(%a: !grid, %b: !grid, %c: !grid, %init: !grid) -> !grid {
    %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%a : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%g0 : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %g2 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%g1 : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %g3 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%g0 : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %g4 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%g3 : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %g5 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%g2, %g4 : !grid, !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    return %g5 : !grid
}

// CHECK-LABEL: spatial_region_candidates @fork_only block0 candidate_only {
// CHECK-NOT: kinds=[fork_join]
// CHECK: members=[G0] kinds=[singleton]
// CHECK: members=[G1, G2] kinds=[parallel_branches]
// CHECK-NOT: kinds=[fork_join]
func.func @fork_only(%a: !grid, %b: !grid, %c: !grid, %init: !grid) -> !grid {
    %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%a : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%g0 : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %g2 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%g0 : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    return %g2 : !grid
}

// CHECK-LABEL: spatial_region_candidates @join_only block0 candidate_only {
// CHECK-NOT: kinds=[fork_join]
// CHECK: members=[G0, G1] kinds=[parallel_branches]
// CHECK: context={ consumer=G2 }
// CHECK: members=[G2] kinds=[singleton]
// CHECK-NOT: kinds=[fork_join]
func.func @join_only(%a: !grid, %b: !grid, %c: !grid, %init: !grid) -> !grid {
    %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%a : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%b : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %g2 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%g0, %g1 : !grid, !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    return %g2 : !grid
}

// CHECK-LABEL: spatial_region_candidates @transitive_dependency block0 candidate_only {
// CHECK: G1 type=unknown interval=0 producers=[G0] consumers=[G3, G4]
// CHECK: members=[G0] kinds=[singleton]
// CHECK: members=[G1, G2] kinds=[parallel_branches]
// CHECK-NOT: members=[G1, G3]
// CHECK-NOT: kinds=[fork_join]
// CHECK: members=[G3] kinds=[singleton]
// CHECK: members=[G4] kinds=[singleton]
func.func @transitive_dependency(%a: !grid, %b: !grid, %c: !grid, %init: !grid) -> !grid {
    %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%a : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%g0 : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %g2 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%g0 : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %g3 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%g1 : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %g4 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id, #id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%g3, %g2, %g1 : !grid, !grid, !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    return %g4 : !grid
}

// CHECK-LABEL: spatial_region_candidates @shared_input block0 candidate_only {
// CHECK: members=[G0, G1] kinds=[parallel_branches]
// CHECK: context={ external_input=%arg0 }
// CHECK: members=[G2] kinds=[singleton]
// CHECK-NOT: candidate R2
func.func @shared_input(%a: !grid, %b: !grid, %c: !grid, %init: !grid) -> !grid {
    %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%a : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%a : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %g2 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%b : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    return %g2 : !grid
}

// CHECK-LABEL: spatial_region_candidates @disconnected block0 candidate_only {
// CHECK: members=[G0] kinds=[singleton]
// CHECK: members=[G1] kinds=[singleton]
// CHECK: members=[G2] kinds=[singleton]
func.func @disconnected(%a: !grid, %b: !grid, %c: !grid, %init: !grid) -> !grid {
    %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%a : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%b : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %g2 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%c : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    return %g2 : !grid
}

// CHECK-LABEL: spatial_region_candidates @repeated_operand block0 candidate_only {
// CHECK: G0 type=unknown interval=0 producers=[] consumers=[G1]
// CHECK: G1 type=unknown interval=0 producers=[G0] consumers=[]
// CHECK: G0 -> G1.operand0
// CHECK: G0 -> G1.operand1
// CHECK: members=[G0, G1] kinds=[producer_consumer_chain]
func.func @repeated_operand(%a: !grid, %b: !grid, %c: !grid, %init: !grid) -> !grid {
    %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%a : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%g0, %g0 : !grid, !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    return %g1 : !grid
}

// CHECK-LABEL: spatial_region_candidates @through_view block0 candidate_only {
// CHECK: G1 type=unknown interval=0 producers=[G0] consumers=[]
// CHECK: G0 -> G1.operand0 {{.*}}connection=projected via=[d2m.view_layout]
// CHECK: user=d2m.view_layout.operand0
// CHECK: members=[G0, G1] kinds=[producer_consumer_chain]
func.func @through_view(%a: !grid, %b: !grid, %c: !grid, %init: !grid) -> !grid {
    %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%a : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %view = d2m.view_layout %g0 remapping = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)> : !grid -> !grid
    %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%view : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    return %g1 : !grid
}

// CHECK-LABEL: spatial_region_candidates @call_barrier block0 candidate_only {
// CHECK: barriers = [func.call
// CHECK: G0 type=unknown interval=0 producers=[] consumers=[G1, G2]
// CHECK: G1 type=unknown interval=1 producers=[G0] consumers=[]
// CHECK: G2 type=unknown interval=1 producers=[G0] consumers=[]
// CHECK: members=[G0] kinds=[singleton]
// CHECK: members=[G1, G2] kinds=[parallel_branches]
func.func @call_barrier(%a: !grid, %b: !grid, %c: !grid, %init: !grid) -> !grid {
    %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%a : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    func.call @side_effect() : () -> ()
    %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%g0 : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %g2 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%g0 : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    return %g2 : !grid
}

func.func private @side_effect()

// CHECK-LABEL: spatial_region_candidates @nested_fork block0 candidate_only {
// CHECK-NOT: members=[G0, G1, G2, G3, G4, G5, G6]
// CHECK: members=[G0] kinds=[singleton]
// CHECK: members=[G1, G2] kinds=[parallel_branches]
// CHECK: members=[G1, G3, G4, G5] kinds=[fork_join]
// CHECK: members=[G2, G5] kinds=[parallel_branches]
// CHECK: members=[G3, G4] kinds=[parallel_branches]
// CHECK: members=[G6] kinds=[singleton]
func.func @nested_fork(%a: !grid, %b: !grid, %c: !grid, %init: !grid) -> !grid {
    %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%a : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%g0 : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %g2 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%g0 : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %g3 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%g1 : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %g4 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%g1 : !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %g5 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%g3, %g4 : !grid, !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    %g6 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id, #id, #id], iterator_types = [#p, #p], threads = [#d2m.thread<unified>]}
        ins(%g2, %g5 : !grid, !grid) outs(%init : !grid) {
      %empty = tensor.empty() : !shard
      d2m.yield %empty : (!shard)
    } : !grid
    return %g6 : !grid
}

// CHECK-LABEL: spatial_region_candidates @empty block0 candidate_only {
// CHECK-NOT: candidate R
func.func @empty() {
  return
}
