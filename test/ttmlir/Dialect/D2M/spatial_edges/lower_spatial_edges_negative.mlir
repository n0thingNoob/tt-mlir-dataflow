// RUN: ttmlir-opt --split-input-file --verify-diagnostics --d2m-lower-spatial-edges %s

#l1 = #ttcore.memory_space<l1>

func.func @aliased(%in: memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
  %e = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>
  %o = memref.alloc() {d2m.virtualGridForwardMapping = affine_map<(d0, d1, d2, d3) -> (d0 + 1, d1 + 1, d2, d3)>, d2m.virtualGridInverseMapping = affine_map<(d0, d1) -> (0, d0 - 1, d1 - 1)>} : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>
  %cb0 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
  %cb1 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
// The producer never DMA-writes the edge buffer (as if it were aliased).
// expected-error @+1 {{expected DMA writes of the edge buffer in the producer and DMA reads of it in the consumer}}
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

func.func @overlap(%in: memref<1x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
  %e = memref.alloc() : memref<1x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>
  %o = memref.alloc() {d2m.virtualGridForwardMapping = affine_map<(d0, d1, d2, d3) -> (d0 + 0, d1 + 1, d2, d3)>, d2m.virtualGridInverseMapping = affine_map<(d0, d1) -> (0, d0 - 0, d1 - 1)>} : memref<1x2x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>
  %cb0 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
  %cb1 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
// The consumer's range (0,1) is also a producer core.
// expected-error @+1 {{edge #d2m.spatial_edge<0:0 -> 1:0>: producer core (0, 1) lies inside the consumer's core range}}
  d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>], grid_ranges = [#ttcore.core_range<(0, 0), (0, 1)>, #ttcore.core_range<(0, 1), (0, 1)>]}
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
      d2m.generic {block_factors = [], grid = #ttcore.grid<1x1, virt_to_physical_map = (d0, d1) -> (0, d0 + 0, d1 + 1), physical_to_virt_map = (d0, d1) -> (0, d0 - 0, d1 - 1)>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
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

// The last write in the loop is never waited on.
// expected-error @+6 {{edge #d2m.spatial_edge<0:0 -> 1:0>: producer DMA write of the edge buffer has no wait}}
func.func @unwaited_loop_write(%in: memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
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

// The write is carried out of the loop through scf.if and the only write
// wait in the thread comes before the loop, so nothing covers it.
// expected-error @+6 {{edge #d2m.spatial_edge<0:0 -> 1:0>: producer DMA write of the edge buffer has no wait}}
func.func @write_wait_before_loop(%in: memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
  %e = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>
  %o = memref.alloc() {d2m.virtualGridForwardMapping = affine_map<(d0, d1, d2, d3) -> (d0 + 1, d1 + 1, d2, d3)>, d2m.virtualGridInverseMapping = affine_map<(d0, d1) -> (0, d0 - 1, d1 - 1)>} : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>
  %cb0 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
  %cb1 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
  d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>], grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(1, 1), (1, 1)>]}
      ins(%in : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) outs(%e, %o : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
      %ev = d2m.view_layout %e remapping = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)> : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>
      d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
          ins(%in : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
          outs(%ev : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
          additionalArgs(%cb0 : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>) {
        %cb = d2m.get_cb(2) resolution_stage = compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>>
        %y = d2m.core_index(0) : index
        %x = d2m.core_index(1) : index
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %init = d2m.null_tx : !d2m.mem_tx<write>
        d2m.dma_wait %init : !d2m.mem_tx<write>
        %last = scf.for %k = %c0 to %c2 step %c1 iter_args(%prev = %init) -> (!d2m.mem_tx<write>) {
          %isLast = arith.cmpi eq, %k, %c1 : index
          %t = scf.if %isLast -> (!d2m.mem_tx<write>) {
            %src = d2m.wait %cb : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
            %tx = d2m.dma_write %src[%c0], %ev[%y, %x, %c0], <1> : (memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) -> !d2m.mem_tx<write>
            scf.yield %tx : !d2m.mem_tx<write>
          } else {
            scf.yield %prev : !d2m.mem_tx<write>
          }
          scf.yield %t : !d2m.mem_tx<write>
        }
        d2m.pop %cb : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>>
      }, {
        %unused = arith.constant 0 : index
      }
  }, {
      %ev2 = d2m.view_layout %e remapping = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)> : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>
      d2m.generic {block_factors = [], grid = #ttcore.grid<1x1, virt_to_physical_map = (d0, d1) -> (0, d0 + 1, d1 + 1), physical_to_virt_map = (d0, d1) -> (0, d0 - 1, d1 - 1)>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
          ins(%ev2 : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
          outs(%o : memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
          additionalArgs(%cb1 : memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>) {
        %cb = d2m.get_cb(2) resolution_stage = compile : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>>
        %y = d2m.core_index(0) : index
        %x = d2m.core_index(1) : index
        %c0 = arith.constant 0 : index
        %dst = d2m.reserve %cb : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>> -> memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>
        %tx = d2m.dma_read %ev2[%y, %x, %c0], %dst[%c0], <1> : (memref<1x1x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>) -> !d2m.mem_tx<read>
        d2m.dma_wait %tx : !d2m.mem_tx<read>
        d2m.push %cb : <memref<1x1x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<2048x2048, 2>, #l1>>
      }, {
        %unused = arith.constant 0 : index
      }
  }
  return
}
