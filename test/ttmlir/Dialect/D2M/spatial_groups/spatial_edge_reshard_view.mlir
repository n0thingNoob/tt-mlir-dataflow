// RUN: ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial spatial-policy=none spatial-groups=0,0,0,0,0,0,0,1,2,0,2,3" --mlir-print-ir-after=d2m-grid-selection --mlir-print-local-scope --mlir-disable-threading %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=GS
// RUN: ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial spatial-policy=none spatial-groups=0,0,0,0,0,0,0,1,2,0,2,3" %s | FileCheck %s --check-prefix=FE

// An edge whose two ends have different grids. The producer (exp) runs on one
// core, (0,0), and lays its 2x4-tile output out on a 1x1 grid; the consumer
// (tanh) runs on the 1x4 range (2,0)-(2,3). The edge keeps the producer's
// layout, and the consumer reads it through a view reblocked to its own 1x4
// grid: each consumer core pulls its column of tiles from the producer core.
// (Before, grid selection failed: "resharding an edge is not supported".)

// GS-LABEL: IR Dump After D2MGridSelection
// GS: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>], grid_ranges = [#ttcore.core_range<(0,0), (0,0)>, #ttcore.core_range<(2,0), (2,3)>]}
// GS: d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>
// GS: ^bb0(%[[E:.*]]: tensor<1x1x2x4x!ttcore.tile<32x32, bf16>
// GS-NEXT: %[[V:.*]] = d2m.view_layout %[[E]] remapping = affine_map<(d0, d1, d2, d3) -> (0, 0, (d2 + d1 floordiv 4) mod 2, d1 mod 4)> : tensor<1x1x2x4x!ttcore.tile<32x32, bf16>, {{.*}}> -> tensor<1x4x2x1x!ttcore.tile<32x32, bf16>
// GS-NEXT: d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x4, virt_to_physical_map = (d0, d1) -> (0, d0 + 2, d1)
// GS-NEXT: ins(%[[V]] :

// After bufferization the view reads the producer's L1 buffer inside the
// consumer region.
// FE: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>]
// FE: d2m.view_layout %[[BUF:.*]] remapping = #{{.*}} : memref<1x1x2x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<8192x2048, 1>, #l1> -> memref<2x1x1x4x
// FE: d2m.view_layout %[[BUF2:.*]] remapping = #{{.*}} : memref<1x1x2x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<8192x2048, 1>, #l1> -> memref<1x4x2x1x!ttcore.tile<32x32, bf16>, #ttcore.view<4>, #l1>
// FE-NEXT: d2m.generic {block_factors = [], grid = #ttcore.grid<1x4, virt_to_physical_map = (d0, d1) -> (0, d0 + 2, d1)
func.func @chain(%x: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
  %0 = "ttir.exp"(%x) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
  %1 = "ttir.tanh"(%0) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
  return %1 : tensor<64x128xbf16>
}
