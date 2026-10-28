// RUN: ttmlir-opt --ttcore-register-device --d2m-lower-load-store-ops-to-dma %s | FileCheck %s --check-prefix=LOWER
// RUN: ttmlir-opt --ttcore-register-device --d2m-optimize-dma %s | FileCheck %s --check-prefix=OPT

// A semaphore increment announces that a write has landed, so it must follow
// the write's completion:
// - d2m-lower-load-store-ops-to-dma emits the semaphore_inc of a
//   `remote_store ... semaphore increment` after the store's dma_wait;
// - d2m-optimize-dma must not defer a write barrier past a semaphore_inc.

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module {
  // LOWER-LABEL: func.func @remote_store_with_semaphore
  // LOWER: %[[TX:.*]] = d2m.dma_write
  // LOWER-NEXT: d2m.dma_wait %[[TX]]
  // LOWER-NEXT: d2m.semaphore_inc %{{.*}}, %{{.*}}, core[
  // LOWER-NEXT: d2m.pop
  func.func @remote_store_with_semaphore(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>, %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, %sem: !d2m.global_semaphore) {
    %stream = d2m.view_layout %arg1 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%arg0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%stream : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        additionalArgs(%sem : !d2m.global_semaphore) {
    ^datamovement0:
      %cb1 = d2m.get_cb(1) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %c0 = arith.constant 0 : index
      d2m.remote_store %stream[%core0, %core1] from %cb1 semaphore increment %sem[%c0, %c0] : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> from !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
    }, {
    ^compute0:
    }
    return
  }

  // OPT-LABEL: func.func @write_barrier_before_semaphore_inc
  // OPT: %[[TX:.*]] = d2m.dma_write
  // OPT-NEXT: d2m.dma_wait %[[TX]] : !d2m.mem_tx<write>
  // OPT-NEXT: d2m.semaphore_inc
  func.func @write_barrier_before_semaphore_inc(
      %arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
      %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %sem = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    %stream0 = d2m.view_layout %arg0 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %stream1 = d2m.view_layout %arg1 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%stream0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%stream1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        additionalArgs(%sem : !d2m.local_semaphore) {
    ^datamovement0:
      %cb0 = d2m.get_cb(0) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %cb1 = d2m.get_cb(1) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %gi = d2m.core_index(0) : index
      %gj = d2m.core_index(1) : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.for %iv = %c0 to %c8 step %c1 {
        %local_in = d2m.reserve %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
        %tx_r = d2m.dma_read %stream0[%gi, %gj], %local_in, <0> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
        d2m.dma_wait %tx_r : !d2m.mem_tx<read>
        d2m.push %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
        %local_out = d2m.wait %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
        %tx_w = d2m.dma_write %local_out, %stream1[%gi, %gj], <0> : (memref<2x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>) -> !d2m.mem_tx<write>
        d2m.dma_wait %tx_w : !d2m.mem_tx<write>
        d2m.semaphore_inc %sem, %c1, core[%c0, %c0] : !d2m.local_semaphore
        d2m.pop %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      }
    }, {
    ^compute0:
    }
    return
  }
}
