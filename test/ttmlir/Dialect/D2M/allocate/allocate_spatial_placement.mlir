// RUN: ttmlir-opt --ttcore-register-device --d2m-allocate %s | FileCheck %s

// d2m-allocate aliases an L1 operand only when its shards live on the
// generic's own cores. A d2m.spatial region off the origin (core (1,1)) that
// reads an operand allocated at the origin must load it with a DMA; an operand
// whose virtual grid mapping matches the region is still aliased.

#l1 = #ttcore.memory_space<l1>
#s = #ttcore.shard<4096x4096, 1>
#m = affine_map<(d0, d1) -> (d0, d1)>
#par = #ttcore.iterator_type<parallel>
#off = affine_map<(d0, d1, d2, d3) -> (d0 + 1, d1 + 1, d2, d3)>
#offinv = affine_map<(d0, d1) -> (0, d0 - 1, d1 - 1)>

module {
  // CHECK-LABEL: func.func @origin_input_offset_region
  // CHECK-SAME: (%[[A:[^:]*]]:
  // CHECK: d2m.spatial
  // CHECK-NOT: d2m.operand_alias %[[A]] :
  // CHECK: d2m.remote_load %{{[^ ]*}} %[[A]][
  func.func @origin_input_offset_region(%a: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>) {
    %o = memref.alloc() {d2m.virtualGridForwardMapping = #off, d2m.virtualGridInverseMapping = #offinv} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>
    d2m.spatial {grid_ranges = [#ttcore.core_range<(1, 1), (1, 1)>]}
        ins(%a : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>) outs(%o : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>) {
      d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1, virt_to_physical_map = (d0, d1) -> (0, d0 + 1, d1 + 1), physical_to_virt_map = (d0, d1) -> (0, d0 - 1, d1 - 1)>, indexing_maps = [#m, #m], iterator_types = [#par, #par], threads = [#d2m.thread<unified>]}
          ins(%a : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>) outs(%o : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>) {
      ^bb0:
        %u = d2m.get_block_factor(0) : index
        %v = d2m.get_block_factor(1) : index
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
    }
    return
  }

  // CHECK-LABEL: func.func @placed_input_offset_region
  // CHECK: %[[A:.*]] = memref.alloc() {{.*}}d2m.virtualGridForwardMapping{{.*}} : memref<1x1x1x1x!ttcore.tile<32x32, f32>
  // CHECK: d2m.spatial
  // CHECK: %[[ALIAS:.*]] = d2m.operand_alias %[[A]]
  // CHECK: d2m.remote_load %[[ALIAS]] %[[A]][
  func.func @placed_input_offset_region() {
    %a = memref.alloc() {d2m.virtualGridForwardMapping = #off, d2m.virtualGridInverseMapping = #offinv} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>
    %o = memref.alloc() {d2m.virtualGridForwardMapping = #off, d2m.virtualGridInverseMapping = #offinv} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>
    d2m.spatial {grid_ranges = [#ttcore.core_range<(1, 1), (1, 1)>]}
        ins(%a : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>) outs(%o : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>) {
      d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1, virt_to_physical_map = (d0, d1) -> (0, d0 + 1, d1 + 1), physical_to_virt_map = (d0, d1) -> (0, d0 - 1, d1 - 1)>, indexing_maps = [#m, #m], iterator_types = [#par, #par], threads = [#d2m.thread<unified>]}
          ins(%a : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>) outs(%o : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #s, #l1>) {
      ^bb0:
        %u = d2m.get_block_factor(0) : index
        %v = d2m.get_block_factor(1) : index
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
    }
    return
  }
}
