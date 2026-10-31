// RUN: ttmlir-opt --ttcore-register-device --d2m-allocate %s | FileCheck %s
// RUN: ttmlir-opt --ttcore-register-device --d2m-allocate="allow-l1-output-spilling=true" %s | FileCheck %s --check-prefix=SPILL

// d2m-allocate with a d2m.spatial edge: the edge buffer %e is never aliased,
// also when the generics reach it through views (@edge_view).
// The producer (core (0,0)) computes into a stream buffer and stores it into
// %e explicitly, so that its DM thread has a last write to signal after; the
// consumer (core (1,1)) loads %e remotely into its own buffer. %e stays in L1
// even when output spilling is allowed.

#l1 = #ttcore.memory_space<l1>
#s = #ttcore.shard<4096x4096, 1>
#m = affine_map<(d0, d1) -> (d0, d1)>
#par = #ttcore.iterator_type<parallel>
#off = affine_map<(d0, d1, d2, d3) -> (d0 + 1, d1 + 1, d2, d3)>
#offinv = affine_map<(d0, d1) -> (0, d0 - 1, d1 - 1)>

module {
  // CHECK-LABEL: func.func @edge
  // CHECK: %[[E:.*]] = memref.alloc() {{.*}}address = {{.*}} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
  // CHECK: d2m.spatial
  // CHECK-NOT: d2m.operand_alias %[[E]] :
  // CHECK: d2m.remote_store %[[E]][%{{.*}}, %{{.*}}] %[[PBUF:[^ ]*]] :
  // CHECK: }, {
  // CHECK-NOT: d2m.operand_alias %[[E]] :
  // CHECK: d2m.remote_load %[[CBUF:[^ ]*]] %[[E]][
  // CHECK-NOT: d2m.operand_alias %[[E]] :
  // CHECK: }
  // SPILL-LABEL: func.func @edge
  // SPILL: %[[E:.*]] = memref.alloc() {{.*}}address = {{.*}} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
  // SPILL: d2m.spatial {edges
  // SPILL-NEXT: ins(
  // SPILL-NEXT: outs(%[[E]],
  func.func @edge(%x: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>) {
    %e = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>
    %o = memref.alloc() {d2m.virtualGridForwardMapping = #off, d2m.virtualGridInverseMapping = #offinv} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>
    d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>], grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(1, 1), (1, 1)>]}
        ins(%x : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>) outs(%e, %o : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>) {
      d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#m, #m], iterator_types = [#par, #par], threads = [#d2m.thread<unified>]}
          ins(%x : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>) outs(%e : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>) {
      ^bb0:
        %u = d2m.get_block_factor(0) : index
        %v = d2m.get_block_factor(1) : index
        affine.for %i = 0 to %u {
          affine.for %j = 0 to %v {
            %l = memref.alloc() {d2m.synchronized_buffer = 2} : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            d2m.remote_load %l %x[%i, %j] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>
            %lo = memref.alloc() {d2m.synchronized_buffer = 2} : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            linalg.generic {indexing_maps = [#m, #m], iterator_types = ["parallel", "parallel"]} ins(%l : memref<1x1x!ttcore.tile<32x32, f32>, #l1>) outs(%lo : memref<1x1x!ttcore.tile<32x32, f32>, #l1>) {
            ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
              %r = "d2m.tile_exp"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              linalg.yield %r : !ttcore.tile<32x32, f32>
            }
            d2m.remote_store %e[%i, %j] %lo : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>
          } {d2m.blocking_loop = 1}
        } {d2m.blocking_loop = 0}
      }
    }, {
      d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1, virt_to_physical_map = (d0, d1) -> (0, d0 + 1, d1 + 1), physical_to_virt_map = (d0, d1) -> (0, d0 - 1, d1 - 1)>, indexing_maps = [#m, #m], iterator_types = [#par, #par], threads = [#d2m.thread<unified>]}
          ins(%e : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>) outs(%o : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>) {
      ^bb0:
        %u = d2m.get_block_factor(0) : index
        %v = d2m.get_block_factor(1) : index
        affine.for %i = 0 to %u {
          affine.for %j = 0 to %v {
            %l = memref.alloc() {d2m.synchronized_buffer = 2} : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            d2m.remote_load %l %e[%i, %j] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>
            %lo = memref.alloc() {d2m.synchronized_buffer = 2} : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            linalg.generic {indexing_maps = [#m, #m], iterator_types = ["parallel", "parallel"]} ins(%l : memref<1x1x!ttcore.tile<32x32, f32>, #l1>) outs(%lo : memref<1x1x!ttcore.tile<32x32, f32>, #l1>) {
            ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
              %r = "d2m.tile_tanh"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              linalg.yield %r : !ttcore.tile<32x32, f32>
            }
            d2m.remote_store %o[%i, %j] %lo : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>
          } {d2m.blocking_loop = 1}
        } {d2m.blocking_loop = 0}
      }
    }
    return
  }

  // A reblocked producer and consumer reach %e through views of it: still no
  // alias, an explicit store through the producer's view, a remote load
  // through the consumer's view, and %e stays in L1.
  // CHECK-LABEL: func.func @edge_view
  // CHECK: %[[E:.*]] = memref.alloc() {{.*}}address = {{.*}} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
  // CHECK: d2m.spatial
  // CHECK: %[[EV:.*]] = d2m.view_layout %[[E]]
  // CHECK-NOT: d2m.operand_alias %[[EV]] :
  // CHECK: d2m.remote_store %[[EV]][%{{.*}}, %{{.*}}] %{{[^ ]*}} :
  // CHECK: }, {
  // CHECK: %[[EV2:.*]] = d2m.view_layout %[[E]]
  // CHECK-NOT: d2m.operand_alias %[[EV2]] :
  // CHECK: d2m.remote_load %{{[^ ]*}} %[[EV2]][
  // SPILL-LABEL: func.func @edge_view
  // SPILL: %[[E:.*]] = memref.alloc() {{.*}}address = {{.*}} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
  // SPILL: d2m.spatial {edges
  // SPILL-NEXT: ins(
  // SPILL-NEXT: outs(%[[E]],
  func.func @edge_view(%x: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>) {
    %e = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>
    %o = memref.alloc() {d2m.virtualGridForwardMapping = #off, d2m.virtualGridInverseMapping = #offinv} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>
    d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>], grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(1, 1), (1, 1)>]}
        ins(%x : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>) outs(%e, %o : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>) {
      %ev = d2m.view_layout %e remapping = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)> : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>
      d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#m, #m], iterator_types = [#par, #par], threads = [#d2m.thread<unified>]}
          ins(%x : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>) outs(%ev : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>) {
      ^bb0:
        %u = d2m.get_block_factor(0) : index
        %v = d2m.get_block_factor(1) : index
        affine.for %i = 0 to %u {
          affine.for %j = 0 to %v {
            %l = memref.alloc() {d2m.synchronized_buffer = 2} : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            d2m.remote_load %l %x[%i, %j] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>
            %lo = memref.alloc() {d2m.synchronized_buffer = 2} : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            linalg.generic {indexing_maps = [#m, #m], iterator_types = ["parallel", "parallel"]} ins(%l : memref<1x1x!ttcore.tile<32x32, f32>, #l1>) outs(%lo : memref<1x1x!ttcore.tile<32x32, f32>, #l1>) {
            ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
              %r = "d2m.tile_exp"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              linalg.yield %r : !ttcore.tile<32x32, f32>
            }
            d2m.remote_store %ev[%i, %j] %lo : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>
          } {d2m.blocking_loop = 1}
        } {d2m.blocking_loop = 0}
      }
    }, {
      %ev2 = d2m.view_layout %e remapping = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)> : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>
      d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1, virt_to_physical_map = (d0, d1) -> (0, d0 + 1, d1 + 1), physical_to_virt_map = (d0, d1) -> (0, d0 - 1, d1 - 1)>, indexing_maps = [#m, #m], iterator_types = [#par, #par], threads = [#d2m.thread<unified>]}
          ins(%ev2 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>) outs(%o : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>) {
      ^bb0:
        %u = d2m.get_block_factor(0) : index
        %v = d2m.get_block_factor(1) : index
        affine.for %i = 0 to %u {
          affine.for %j = 0 to %v {
            %l = memref.alloc() {d2m.synchronized_buffer = 2} : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            d2m.remote_load %l %ev2[%i, %j] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>
            %lo = memref.alloc() {d2m.synchronized_buffer = 2} : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
            linalg.generic {indexing_maps = [#m, #m], iterator_types = ["parallel", "parallel"]} ins(%l : memref<1x1x!ttcore.tile<32x32, f32>, #l1>) outs(%lo : memref<1x1x!ttcore.tile<32x32, f32>, #l1>) {
            ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
              %r = "d2m.tile_tanh"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              linalg.yield %r : !ttcore.tile<32x32, f32>
            }
            d2m.remote_store %o[%i, %j] %lo : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>
          } {d2m.blocking_loop = 1}
        } {d2m.blocking_loop = 0}
      }
    }
    return
  }
}
