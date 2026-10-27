// RUN: ttmlir-opt --split-input-file --d2m-lower-spatial-edges %s | FileCheck %s
// RUN: ttmlir-opt --split-input-file --d2m-lower-spatial-edges="ttnn-mode=true" %s | FileCheck %s --check-prefix=TTNN
// d2m-lower-spatial-edges on split DM/compute threads: one local semaphore
// per edge, producer increments after its last op, consumer waits first.

#l1 = #ttcore.memory_space<l1>

// CHECK-LABEL: func.func @chain
// CHECK: %[[SEM:.*]] = d2m.create_local_semaphore <{initialValue = 0 : ui32}>
// CHECK: d2m.spatial {grid_ranges
// CHECK: d2m.generic
// CHECK: additionalArgs(%{{.*}}, %[[SEM]] :
// CHECK: d2m.dma_write
// CHECK: d2m.dma_wait
// CHECK: d2m.pop
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[Y:.*]] = arith.constant 1 : index
// CHECK: %[[X:.*]] = arith.constant 1 : index
// CHECK: d2m.semaphore_inc %[[SEM]], %[[C1]], core[%[[Y]], %[[X]]]
// CHECK-NEXT: }, {
// CHECK: d2m.generic
// CHECK: additionalArgs(%{{.*}}, %[[SEM]] :
// CHECK-NEXT: {
// CHECK-NEXT: %[[N:.*]] = arith.constant 1 : index
// CHECK-NEXT: d2m.semaphore_wait %[[SEM]], %[[N]]
// CHECK: d2m.dma_read
// TTNN-LABEL: func.func @chain
// TTNN: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>]
// TTNN-NOT: semaphore
func.func @chain(%in: memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
  %e = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>
  %o = memref.alloc() {d2m.virtualGridForwardMapping = affine_map<(d0, d1, d2, d3) -> (d0 + 1, d1 + 1, d2, d3)>, d2m.virtualGridInverseMapping = affine_map<(d0, d1) -> (0, d0 - 1, d1 - 1)>} : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>
  %cb0 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
  %cb1 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
  d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>], grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(1, 1), (1, 1)>]}
      ins(%in : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) outs(%e, %o : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
      d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
          ins(%in : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
          outs(%e : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
          additionalArgs(%cb0 : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>) {
        %cb = d2m.get_cb(2) resolution_stage = compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>>
        %y = d2m.core_index(0) : index
        %x = d2m.core_index(1) : index
        %c0 = arith.constant 0 : index
        %src = d2m.wait %cb : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
        %tx = d2m.dma_write %src[%c0], %e[%y, %x, %c0], <1> : (memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) -> !d2m.mem_tx<write>
        d2m.dma_wait %tx : !d2m.mem_tx<write>
        d2m.pop %cb : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>>
      }, {
        %unused = arith.constant 0 : index
      }
  }, {
      d2m.generic {block_factors = [], grid = #ttcore.grid<1x1, virt_to_physical_map = (d0, d1) -> (0, d0 + 1, d1 + 1), physical_to_virt_map = (d0, d1) -> (0, d0 - 1, d1 - 1)>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
          ins(%e : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
          outs(%o : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
          additionalArgs(%cb1 : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>) {
        %cb = d2m.get_cb(2) resolution_stage = compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>>
        %y = d2m.core_index(0) : index
        %x = d2m.core_index(1) : index
        %c0 = arith.constant 0 : index
        %dst = d2m.reserve %cb : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
        %tx = d2m.dma_read %e[%y, %x, %c0], %dst[%c0], <1> : (memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>) -> !d2m.mem_tx<read>
        d2m.dma_wait %tx : !d2m.mem_tx<read>
        d2m.push %cb : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>>
      }, {
        %unused = arith.constant 0 : index
      }
  }
  return
}

// -----

#l1 = #ttcore.memory_space<l1>

// Producer and consumer on 1x2 grids: every producer core bumps both consumer
// cores (logical (2,3) and (2,4)); each consumer core waits for 2 signals.
// CHECK-LABEL: func.func @wide
// CHECK: d2m.dma_wait
// CHECK: d2m.pop
// CHECK: d2m.semaphore_inc %{{.*}}, %{{.*}}, core[%[[Y0:.*]], %[[X0:.*]]]
// CHECK: d2m.semaphore_inc %{{.*}}, %{{.*}}, core[%[[Y1:.*]], %[[X1:.*]]]
// CHECK: %[[N:.*]] = arith.constant 2 : index
// CHECK-NEXT: d2m.semaphore_wait %{{.*}}, %[[N]]
func.func @wide(%in: memref<1x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
  %e = memref.alloc() : memref<1x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>
  %o = memref.alloc() {d2m.virtualGridForwardMapping = affine_map<(d0, d1, d2, d3) -> (d0 + 2, d1 + 3, d2, d3)>, d2m.virtualGridInverseMapping = affine_map<(d0, d1) -> (0, d0 - 2, d1 - 3)>} : memref<1x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>
  %cb0 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
  %cb1 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
  d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>], grid_ranges = [#ttcore.core_range<(0, 0), (0, 1)>, #ttcore.core_range<(2, 3), (2, 4)>]}
      ins(%in : memref<1x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) outs(%e, %o : memref<1x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>, memref<1x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
      d2m.generic {block_factors = [], grid = #ttcore.grid<1x2>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
          ins(%in : memref<1x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
          outs(%e : memref<1x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
          additionalArgs(%cb0 : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>) {
        %cb = d2m.get_cb(2) resolution_stage = compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>>
        %y = d2m.core_index(0) : index
        %x = d2m.core_index(1) : index
        %c0 = arith.constant 0 : index
        %src = d2m.wait %cb : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
        %tx = d2m.dma_write %src[%c0], %e[%y, %x, %c0], <1> : (memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>, memref<1x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) -> !d2m.mem_tx<write>
        d2m.dma_wait %tx : !d2m.mem_tx<write>
        d2m.pop %cb : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>>
      }, {
        %unused = arith.constant 0 : index
      }
  }, {
      d2m.generic {block_factors = [], grid = #ttcore.grid<1x2, virt_to_physical_map = (d0, d1) -> (0, d0 + 2, d1 + 3), physical_to_virt_map = (d0, d1) -> (0, d0 - 2, d1 - 3)>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
          ins(%e : memref<1x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
          outs(%o : memref<1x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
          additionalArgs(%cb1 : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>) {
        %cb = d2m.get_cb(2) resolution_stage = compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>>
        %y = d2m.core_index(0) : index
        %x = d2m.core_index(1) : index
        %c0 = arith.constant 0 : index
        %dst = d2m.reserve %cb : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
        %tx = d2m.dma_read %e[%y, %x, %c0], %dst[%c0], <1> : (memref<1x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>) -> !d2m.mem_tx<read>
        d2m.dma_wait %tx : !d2m.mem_tx<read>
        d2m.push %cb : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>>
      }, {
        %unused = arith.constant 0 : index
      }
  }
  return
}

// -----

#l1 = #ttcore.memory_space<l1>

// Producer off the origin (core (1,1)), consumer on core (2,2): semaphore_inc
// core indices are virtual coordinates of the producer generic, so the target
// is (1,1) (= physical (2,2) through the producer's virt_to_physical_map).
// CHECK-LABEL: func.func @offset_producer
// CHECK: d2m.pop
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: %[[Y:.*]] = arith.constant 1 : index
// CHECK: %[[X:.*]] = arith.constant 1 : index
// CHECK: d2m.semaphore_inc %{{.*}}, %[[ONE]], core[%[[Y]], %[[X]]]
func.func @offset_producer(%in: memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
  %e = memref.alloc() {d2m.virtualGridForwardMapping = affine_map<(d0, d1, d2, d3) -> (d0 + 1, d1 + 1, d2, d3)>, d2m.virtualGridInverseMapping = affine_map<(d0, d1) -> (0, d0 - 1, d1 - 1)>} : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>
  %o = memref.alloc() {d2m.virtualGridForwardMapping = affine_map<(d0, d1, d2, d3) -> (d0 + 2, d1 + 2, d2, d3)>, d2m.virtualGridInverseMapping = affine_map<(d0, d1) -> (0, d0 - 2, d1 - 2)>} : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>
  %cb0 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
  %cb1 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
  d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>], grid_ranges = [#ttcore.core_range<(1, 1), (1, 1)>, #ttcore.core_range<(2, 2), (2, 2)>]}
      ins(%in : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) outs(%e, %o : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
      d2m.generic {block_factors = [], grid = #ttcore.grid<1x1, virt_to_physical_map = (d0, d1) -> (0, d0 + 1, d1 + 1), physical_to_virt_map = (d0, d1) -> (0, d0 - 1, d1 - 1)>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
          ins(%in : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
          outs(%e : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
          additionalArgs(%cb0 : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>) {
        %cb = d2m.get_cb(2) resolution_stage = compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>>
        %y = d2m.core_index(0) : index
        %x = d2m.core_index(1) : index
        %c0 = arith.constant 0 : index
        %src = d2m.wait %cb : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
        %tx = d2m.dma_write %src[%c0], %e[%y, %x, %c0], <1> : (memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) -> !d2m.mem_tx<write>
        d2m.dma_wait %tx : !d2m.mem_tx<write>
        d2m.pop %cb : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>>
      }, {
        %unused = arith.constant 0 : index
      }
  }, {
      d2m.generic {block_factors = [], grid = #ttcore.grid<1x1, virt_to_physical_map = (d0, d1) -> (0, d0 + 2, d1 + 2), physical_to_virt_map = (d0, d1) -> (0, d0 - 2, d1 - 2)>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
          ins(%e : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
          outs(%o : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
          additionalArgs(%cb1 : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>) {
        %cb = d2m.get_cb(2) resolution_stage = compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>>
        %y = d2m.core_index(0) : index
        %x = d2m.core_index(1) : index
        %c0 = arith.constant 0 : index
        %dst = d2m.reserve %cb : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
        %tx = d2m.dma_read %e[%y, %x, %c0], %dst[%c0], <1> : (memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>) -> !d2m.mem_tx<read>
        d2m.dma_wait %tx : !d2m.mem_tx<read>
        d2m.push %cb : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>>
      }, {
        %unused = arith.constant 0 : index
      }
  }
  return
}
