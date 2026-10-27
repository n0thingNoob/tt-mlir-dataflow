// RUN: ttmlir-opt --ttcore-register-device --d2m-allocate %s | FileCheck %s

// d2m-allocate never aliases a remote_store that increments a semaphore: an
// aliased store has no DMA to wait on before the increment, and
// d2m-insert-compute-cb later erases its marker together with the increment.
// The same store without a semaphore is still aliased.

#l1 = #ttcore.memory_space<l1>
#s = #ttcore.shard<4096x4096, 1>
#m = affine_map<(d0, d1) -> (d0, d1)>
#par = #ttcore.iterator_type<parallel>

module {
  // CHECK-LABEL: func.func @store_with_semaphore
  // CHECK: %[[O:.*]] = memref.alloc() {{.*}}address
  // CHECK-NOT: d2m.operand_alias %[[O]] :
  // CHECK: d2m.remote_store %[[O]][%{{.*}}, %{{.*}}] %[[BUF:[^ ]*]] semaphore increment
  func.func @store_with_semaphore(%a: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>, %sem: !d2m.global_semaphore) {
    %o = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#m, #m], iterator_types = [#par, #par], threads = [#d2m.thread<unified>]}
        ins(%a : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>) outs(%o : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>)
          additionalArgs(%sem : !d2m.global_semaphore) {
    ^bb0:
      %u = d2m.get_block_factor(0) : index
      %v = d2m.get_block_factor(1) : index
      %z = arith.constant 0 : index
      affine.for %i = 0 to %u {
        affine.for %j = 0 to %v {
          %l = memref.alloc() {d2m.synchronized_buffer = 2} : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
          d2m.remote_load %l %a[%i, %j] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>
          %lo = memref.alloc() {d2m.synchronized_buffer = 2} : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
          linalg.generic {indexing_maps = [#m, #m], iterator_types = ["parallel", "parallel"]} ins(%l : memref<1x1x!ttcore.tile<32x32, f32>, #l1>) outs(%lo : memref<1x1x!ttcore.tile<32x32, f32>, #l1>) {
          ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
            %r = "d2m.tile_exp"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %r : !ttcore.tile<32x32, f32>
          }
          d2m.remote_store %o[%i, %j] %lo semaphore increment %sem[%z, %z] : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }

  // CHECK-LABEL: func.func @store_without_semaphore
  // CHECK: %[[O:.*]] = memref.alloc() {{.*}}address
  // CHECK: %[[ALIAS:.*]] = d2m.operand_alias %[[O]] :
  // CHECK: d2m.remote_store %[[O]][%{{.*}}, %{{.*}}] %[[ALIAS]] :
  func.func @store_without_semaphore(%a: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>) {
    %o = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#m, #m], iterator_types = [#par, #par], threads = [#d2m.thread<unified>]}
        ins(%a : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>) outs(%o : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>) {
    ^bb0:
      %u = d2m.get_block_factor(0) : index
      %v = d2m.get_block_factor(1) : index
      %z = arith.constant 0 : index
      affine.for %i = 0 to %u {
        affine.for %j = 0 to %v {
          %l = memref.alloc() {d2m.synchronized_buffer = 2} : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
          d2m.remote_load %l %a[%i, %j] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>
          %lo = memref.alloc() {d2m.synchronized_buffer = 2} : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
          linalg.generic {indexing_maps = [#m, #m], iterator_types = ["parallel", "parallel"]} ins(%l : memref<1x1x!ttcore.tile<32x32, f32>, #l1>) outs(%lo : memref<1x1x!ttcore.tile<32x32, f32>, #l1>) {
          ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
            %r = "d2m.tile_exp"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %r : !ttcore.tile<32x32, f32>
          }
          d2m.remote_store %o[%i, %j] %lo : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }
}
