// RUN: ttmlir-opt %s --ttcore-register-device --d2m-optimize-dma | FileCheck %s

#l1 = #ttcore.memory_space<l1>
!local = memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
!remote = memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>

// Publishing readiness must fence the tile write. Neither barrier sinking nor
// loop barrier deferral may move the wait after semaphore_inc.
// CHECK-LABEL: func.func @publish_tile
// CHECK-NOT: d2m.null_tx
// CHECK: scf.for
// CHECK: [[WRITE:%.+]] = d2m.dma_write
// CHECK-NEXT: d2m.dma_wait [[WRITE]] : !d2m.mem_tx<write>
// CHECK-NEXT: d2m.semaphore_inc
func.func @publish_tile(%out: !remote, %sem: !d2m.global_semaphore) {
  "d2m.generic"(%out, %sem) <{
    block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [],
    iterator_types = [], operandSegmentSizes = array<i32: 0, 1, 1>,
    threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
  }> ({
    %cb = d2m.get_cb(0) : !d2m.cb<!local>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    scf.for %i = %c0 to %c16 step %c1 {
      %tile = d2m.wait %cb : <!local> -> !local
      %tx = d2m.dma_write %tile, %out[%c0, %c0], <0> : (!local, !remote) -> !d2m.mem_tx<write>
      d2m.dma_wait %tx : !d2m.mem_tx<write>
      d2m.semaphore_inc %sem, %c1, core[%c0, %c1] : !d2m.global_semaphore
      d2m.pop %cb : <!local>
    }
  }, {
  ^bb0:
  }) : (!remote, !d2m.global_semaphore) -> ()
  return
}
