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
// cores (logical (2,3) and (2,4)) with one multicast increment; each consumer
// core waits for 2 signals.
// CHECK-LABEL: func.func @wide
// CHECK: d2m.dma_wait
// CHECK: d2m.pop
// CHECK: %[[Y:.*]] = arith.constant 2 : index
// CHECK-NEXT: %[[X:.*]] = arith.constant 3 : index
// CHECK-NEXT: %[[H:.*]] = arith.constant 1 : index
// CHECK-NEXT: %[[W:.*]] = arith.constant 2 : index
// CHECK-NEXT: d2m.semaphore_inc %{{.*}}, %{{.*}}, core[%[[Y]], %[[X]]] mcast[%[[H]], %[[W]]]
// CHECK-NOT: d2m.semaphore_inc
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

// -----

#l1 = #ttcore.memory_space<l1>

// Producer 2x2 at (3,3), consumer 2x2 at the origin: one multicast
// increment per writer thread. Its start is the consumer's corner in the
// producer's virtual coordinates, (-3,-3); the shape stays 2x2.
// CHECK-LABEL: func.func @offset_consumer_mcast
// CHECK: d2m.pop
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK-NEXT: %[[Y:.*]] = arith.constant -3 : index
// CHECK-NEXT: %[[X:.*]] = arith.constant -3 : index
// CHECK-NEXT: %[[H:.*]] = arith.constant 2 : index
// CHECK-NEXT: %[[W:.*]] = arith.constant 2 : index
// CHECK-NEXT: d2m.semaphore_inc %{{.*}}, %[[ONE]], core[%[[Y]], %[[X]]] mcast[%[[H]], %[[W]]]
// CHECK-NOT: d2m.semaphore_inc
// CHECK: %[[N:.*]] = arith.constant 4 : index
// CHECK-NEXT: d2m.semaphore_wait %{{.*}}, %[[N]]
func.func @offset_consumer_mcast(%in: memref<2x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
  %e = memref.alloc() {d2m.virtualGridForwardMapping = affine_map<(d0, d1, d2, d3) -> (d0 + 3, d1 + 3, d2, d3)>, d2m.virtualGridInverseMapping = affine_map<(d0, d1) -> (0, d0 - 3, d1 - 3)>} : memref<2x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>
  %o = memref.alloc() : memref<2x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>
  %cb0 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
  %cb1 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
  d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>], grid_ranges = [#ttcore.core_range<(3, 3), (4, 4)>, #ttcore.core_range<(0, 0), (1, 1)>]}
      ins(%in : memref<2x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) outs(%e, %o : memref<2x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>, memref<2x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
      d2m.generic {block_factors = [], grid = #ttcore.grid<2x2, virt_to_physical_map = (d0, d1) -> (0, d0 + 3, d1 + 3), physical_to_virt_map = (d0, d1) -> (0, d0 - 3, d1 - 3)>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
          ins(%in : memref<2x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
          outs(%e : memref<2x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
          additionalArgs(%cb0 : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>) {
        %cb = d2m.get_cb(2) resolution_stage = compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>>
        %y = d2m.core_index(0) : index
        %x = d2m.core_index(1) : index
        %c0 = arith.constant 0 : index
        %src = d2m.wait %cb : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
        %tx = d2m.dma_write %src[%c0], %e[%y, %x, %c0], <1> : (memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>, memref<2x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) -> !d2m.mem_tx<write>
        d2m.dma_wait %tx : !d2m.mem_tx<write>
        d2m.pop %cb : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>>
      }, {
        %unused = arith.constant 0 : index
      }
  }, {
      d2m.generic {block_factors = [], grid = #ttcore.grid<2x2>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
          ins(%e : memref<2x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
          outs(%o : memref<2x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
          additionalArgs(%cb1 : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>) {
        %cb = d2m.get_cb(2) resolution_stage = compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>>
        %y = d2m.core_index(0) : index
        %x = d2m.core_index(1) : index
        %c0 = arith.constant 0 : index
        %dst = d2m.reserve %cb : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
        %tx = d2m.dma_read %e[%y, %x, %c0], %dst[%c0], <1> : (memref<2x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>) -> !d2m.mem_tx<read>
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

// One producer output feeds two regions: one semaphore for both edges. The
// producer bumps B (2x2, multicast) and C (one core, unicast); B and C each
// wait for the 4 producer cores.
// CHECK-LABEL: func.func @fan_out
// CHECK: %[[SEM:.*]] = d2m.create_local_semaphore
// CHECK-NOT: d2m.create_local_semaphore
// CHECK: d2m.spatial {grid_ranges
// CHECK: additionalArgs(%{{.*}}, %[[SEM]] :
// CHECK: d2m.pop
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK-NEXT: %[[BY:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[BX:.*]] = arith.constant 3 : index
// CHECK-NEXT: %[[BH:.*]] = arith.constant 2 : index
// CHECK-NEXT: %[[BW:.*]] = arith.constant 2 : index
// CHECK-NEXT: d2m.semaphore_inc %[[SEM]], %[[ONE]], core[%[[BY]], %[[BX]]] mcast[%[[BH]], %[[BW]]]
// CHECK-NEXT: %[[CY:.*]] = arith.constant 3 : index
// CHECK-NEXT: %[[CX:.*]] = arith.constant 0 : index
// CHECK-NEXT: d2m.semaphore_inc %[[SEM]], %[[ONE]], core[%[[CY]], %[[CX]]]
// CHECK-NEXT: }, {
// CHECK: additionalArgs(%{{.*}}, %[[SEM]] :
// CHECK-NEXT: {
// CHECK-NEXT: %[[N:.*]] = arith.constant 4 : index
// CHECK-NEXT: d2m.semaphore_wait %[[SEM]], %[[N]]
// CHECK: additionalArgs(%{{.*}}, %[[SEM]] :
// CHECK-NEXT: {
// CHECK-NEXT: %[[N2:.*]] = arith.constant 4 : index
// CHECK-NEXT: d2m.semaphore_wait %[[SEM]], %[[N2]]
func.func @fan_out(%in: memref<2x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
  %e = memref.alloc() : memref<2x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>
  %o1 = memref.alloc() {d2m.virtualGridForwardMapping = affine_map<(d0, d1, d2, d3) -> (d0 + 0, d1 + 3, d2, d3)>, d2m.virtualGridInverseMapping = affine_map<(d0, d1) -> (0, d0 - 0, d1 - 3)>} : memref<2x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>
  %o2 = memref.alloc() {d2m.virtualGridForwardMapping = affine_map<(d0, d1, d2, d3) -> (d0 + 3, d1 + 0, d2, d3)>, d2m.virtualGridInverseMapping = affine_map<(d0, d1) -> (0, d0 - 3, d1 - 0)>} : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>
  %cb0 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
  %cb1 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
  %cb2 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
  d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>, #d2m.spatial_edge<0:0 -> 2:0>], grid_ranges = [#ttcore.core_range<(0, 0), (1, 1)>, #ttcore.core_range<(0, 3), (1, 4)>, #ttcore.core_range<(3, 0), (3, 0)>]}
      ins(%in : memref<2x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) outs(%e, %o1, %o2 : memref<2x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>, memref<2x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
      d2m.generic {block_factors = [], grid = #ttcore.grid<2x2>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
          ins(%in : memref<2x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
          outs(%e : memref<2x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
          additionalArgs(%cb0 : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>) {
        %cb = d2m.get_cb(2) resolution_stage = compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>>
        %y = d2m.core_index(0) : index
        %x = d2m.core_index(1) : index
        %c0 = arith.constant 0 : index
        %src = d2m.wait %cb : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
        %tx = d2m.dma_write %src[%c0], %e[%y, %x, %c0], <1> : (memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>, memref<2x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) -> !d2m.mem_tx<write>
        d2m.dma_wait %tx : !d2m.mem_tx<write>
        d2m.pop %cb : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>>
      }, {
        %unused = arith.constant 0 : index
      }
  }, {
      d2m.generic {block_factors = [], grid = #ttcore.grid<2x2, virt_to_physical_map = (d0, d1) -> (0, d0 + 0, d1 + 3), physical_to_virt_map = (d0, d1) -> (0, d0 - 0, d1 - 3)>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
          ins(%e : memref<2x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
          outs(%o1 : memref<2x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
          additionalArgs(%cb1 : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>) {
        %cb = d2m.get_cb(2) resolution_stage = compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>>
        %y = d2m.core_index(0) : index
        %x = d2m.core_index(1) : index
        %c0 = arith.constant 0 : index
        %dst = d2m.reserve %cb : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
        %tx = d2m.dma_read %e[%y, %x, %c0], %dst[%c0], <1> : (memref<2x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>) -> !d2m.mem_tx<read>
        d2m.dma_wait %tx : !d2m.mem_tx<read>
        d2m.push %cb : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>>
      }, {
        %unused = arith.constant 0 : index
      }
  }, {
      d2m.generic {block_factors = [], grid = #ttcore.grid<1x1, virt_to_physical_map = (d0, d1) -> (0, d0 + 3, d1 + 0), physical_to_virt_map = (d0, d1) -> (0, d0 - 3, d1 - 0)>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
          ins(%e : memref<2x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
          outs(%o2 : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
          additionalArgs(%cb2 : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>) {
        %cb = d2m.get_cb(2) resolution_stage = compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>>
        %y = d2m.core_index(0) : index
        %x = d2m.core_index(1) : index
        %c0 = arith.constant 0 : index
        %dst = d2m.reserve %cb : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
        %tx = d2m.dma_read %e[%y, %x, %c0], %dst[%c0], <1> : (memref<2x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>) -> !d2m.mem_tx<read>
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

// The producer writes the edge buffer inside a loop and waits for each write
// one iteration later and for the last one after the loop (the form a blocked
// matmul output takes): the increment goes after the final wait.
// CHECK-LABEL: func.func @pipelined_write
// CHECK: %[[SEM:.*]] = d2m.create_local_semaphore
// CHECK: %[[LAST:.*]] = scf.for
// CHECK: d2m.dma_wait %[[LAST]]
// CHECK: d2m.pop
// CHECK: d2m.semaphore_inc %[[SEM]]
// CHECK-NEXT: }, {
func.func @pipelined_write(%in: memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
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
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %init = d2m.null_tx : !d2m.mem_tx<write>
        %last = scf.for %k = %c0 to %c2 step %c1 iter_args(%prev = %init) -> (!d2m.mem_tx<write>) {
          %notFirst = arith.cmpi ne, %k, %c0 : index
          scf.if %notFirst {
            d2m.dma_wait %prev : !d2m.mem_tx<write>
            d2m.pop %cb : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>>
          }
          %src = d2m.wait %cb : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
          %tx = d2m.dma_write %src[%c0], %e[%y, %x, %c0], <1> : (memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) -> !d2m.mem_tx<write>
          scf.yield %tx : !d2m.mem_tx<write>
        }
        d2m.dma_wait %last : !d2m.mem_tx<write>
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
