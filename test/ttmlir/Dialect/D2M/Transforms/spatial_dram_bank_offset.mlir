// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="ttnn-mode=false" --mlir-print-ir-after=d2m-lower-dma-to-fully-indexed-form -o /dev/null %s 2>&1 | FileCheck %s

// A d2m.spatial region placed at cores (0, 2)-(1, 3) writes a DRAM output.
// GridSelection records the region's core offset in the output buffer's
// virtual grid forward mapping. DRAM shard -> bank placement must not depend
// on it: the host places shard i in bank i % 12 starting from bank 0, so the
// kernel's bank index must be the offset-free `(d0 * 2 + d1 + ...) mod 12`
// and must not add the region's x offset back
// (`- ((d0 * 2 + d1 + d2 + d3 + 2) floordiv 12) * 12 + 2`).

// The offset is still recorded (it places the generic on cores 0x2).
// CHECK: memref.alloc() {{.*}}d2m.virtualGridForwardMapping = affine_map<(d0, d1, d2, d3) -> (d0, d1 + 2, d2, d3)>{{.*}}#ttcore.memory_space<dram>
// CHECK: d2m.generic {{.*}}grid = #ttcore.grid<2x2, virt_to_physical_map = (d0, d1) -> (0, d0, d1 + 2)
// CHECK-NOT: d2m.generic
// CHECK: affine.apply affine_map<(d0, d1, d2, d3) -> ((d0 * 2 + d1 + d2 + d3) mod 12)>
// CHECK-NEXT: affine.apply affine_map<(d0, d1, d2, d3) -> (((d0 * 2 + d1 + d2 + d3) floordiv 12) * 2048)>
// CHECK: d2m.dma_write {{.*}}#ttcore.memory_space<dram>

#l1 = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#dram = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, dram, sharded>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
func.func @offset_region_dram_output(%arg0: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  %0 = d2m.empty() : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #l1>
  %in = d2m.to_layout %arg0, %0 : tensor<64x64xbf16> into tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #l1> -> tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #l1>
  %o = d2m.empty() : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #dram>
  %r = d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 2), (1, 3)>]}
      ins(%in : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #l1>)
      outs(%o : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #dram>) {
    ^region0:
      %g = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
          ins(%in : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #l1>)
          outs(%o : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #dram>) {
        %b0 = d2m.block_index(0) : index
        %b1 = d2m.block_index(1) : index
        %t0 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, bf16>>
        %ld = d2m.remote_load %t0 %in[%b0, %b1] : tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #l1> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
        %t1 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, bf16>>
        %e = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]} ins(%ld : tensor<2x2x!ttcore.tile<32x32, bf16>>) outs(%t1 : tensor<2x2x!ttcore.tile<32x32, bf16>>) {
        ^bb0(%x: !ttcore.tile<32x32, bf16>, %y: !ttcore.tile<32x32, bf16>):
          %v = "d2m.tile_exp"(%x) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
          linalg.yield %v : !ttcore.tile<32x32, bf16>
        } -> tensor<2x2x!ttcore.tile<32x32, bf16>>
        %st = d2m.remote_store %o[%b0, %b1] %e : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #dram>, tensor<2x2x!ttcore.tile<32x32, bf16>> -> tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #dram>
        d2m.yield %st : (tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #dram>)
      } : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #dram>
      d2m.spatial_yield %g : (tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #dram>)
  } : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #dram>
  %h = d2m.empty() : tensor<64x64xbf16>
  %out = d2m.to_layout %r, %h : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #dram> into tensor<64x64xbf16> -> tensor<64x64xbf16>
  return %out : tensor<64x64xbf16>
}
