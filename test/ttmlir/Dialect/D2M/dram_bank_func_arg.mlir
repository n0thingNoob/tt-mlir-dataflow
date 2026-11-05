// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="ttnn-mode=false" --mlir-print-ir-after=d2m-lower-dma-to-fully-indexed-form -o /dev/null %s 2>&1 | FileCheck %s

// A tiled DRAM buffer passed in as a function argument (a resident parameter)
// with d2m.dram_bank = 5: its shard i is in bank (5 + i) mod 12, so the
// kernel that reads it rotates the bank index by 5, as for a buffer the
// function allocates itself (dram_bank.mlir).

// CHECK: IR Dump After D2MLowerDMAToFullyIndexedForm
// CHECK: affine.apply affine_map<{{.*}}mod 12 + 5) floordiv 12) * 12 + 5)>
#l1 = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#dram = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, dram, sharded>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
func.func @run(%w: tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #dram> {d2m.dram_bank = 5 : i64}) -> tensor<64x64xbf16> {
  %o = d2m.empty() : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #l1>
  %g = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#id2, #id2], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%w : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #dram>)
      outs(%o : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #l1>) {
    %b0 = d2m.block_index(0) : index
    %b1 = d2m.block_index(1) : index
    %t0 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, bf16>>
    %ld = d2m.remote_load %t0 %w[%b0, %b1] : tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #dram> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    %t1 = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, bf16>>
    %e = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]} ins(%ld : tensor<2x2x!ttcore.tile<32x32, bf16>>) outs(%t1 : tensor<2x2x!ttcore.tile<32x32, bf16>>) {
    ^bb0(%x: !ttcore.tile<32x32, bf16>, %y: !ttcore.tile<32x32, bf16>):
      %v = "d2m.tile_exp"(%x) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      linalg.yield %v : !ttcore.tile<32x32, bf16>
    } -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    %st = d2m.remote_store %o[%b0, %b1] %e : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #l1>, tensor<2x2x!ttcore.tile<32x32, bf16>> -> tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #l1>
    d2m.yield %st : (tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #l1>)
  } : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #l1>
  %h = d2m.empty() : tensor<64x64xbf16>
  %out = d2m.to_layout %g, %h : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, #l1> into tensor<64x64xbf16> -> tensor<64x64xbf16>
  return %out : tensor<64x64xbf16>
}
