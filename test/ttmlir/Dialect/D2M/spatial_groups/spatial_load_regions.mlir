// RUN: ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-fuse-elementwise=true spatial-policy=none spatial-groups=0,0,0,0,0,0,0,1,0,1,0,1,0,2,0,2,0,2,0,3,0,3,0,3 spatial-loads=0,1,1,0,1,0,1,1,1,0,1,0,3,1,1,1,1,1" --mlir-print-ir-after=d2m-materialize-spatial-groups --mlir-print-local-scope --mlir-disable-threading %s -o /dev/null 2>&1 | FileCheck %s
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-fuse-elementwise=true spatial-policy=none spatial-groups=0,0,0,0,0,0,0,1,0,1,0,1,0,2,0,2,0,2,0,3,0,3,0,3 spatial-loads=0,1,1,0,1,0,1,1,1,0,1,0,3,1,1,1,1,1" %s | FileCheck %s --check-prefix=METAL
// RUN: not ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-policy=none spatial-loads=0,1,1,0,1,0" %s 2>&1 | FileCheck %s --check-prefix=ERR-GROUP
// RUN: not ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial spatial-policy=none spatial-groups=0,0,0,0,0,0,0,1,0,1,0,1 spatial-loads=0,1,1,0,1,0" %s 2>&1 | FileCheck %s --check-prefix=ERR-DRAM
// RUN: not ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-policy=none spatial-groups=0,0,0,0,0,0,0,1,0,1,0,1 spatial-loads=0,0,1,0,1,0,0,1,1,0,1,0" %s 2>&1 | FileCheck %s --check-prefix=ERR-TYPES
// RUN: not ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-policy=none spatial-groups=0,0,0,0,0,0,0,1,0,1,0,1 spatial-loads=0,2,1,0,1,0" %s 2>&1 | FileCheck %s --check-prefix=ERR-OPERAND
// RUN: not ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-policy=none spatial-groups=0,0,0,0,0,0,0,1,0,1,0,1 spatial-loads=9,1,1,0,1,0" %s 2>&1 | FileCheck %s --check-prefix=ERR-INDEX
// RUN: not ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-policy=none spatial-groups=0,0,0,0,0,0,0,1,0,1,0,1 spatial-loads=0,1,1,0,1" %s 2>&1 | FileCheck %s --check-prefix=ERR-COUNT

// A gated MLP with its weights in DRAM: gate #0 on (0,0), up #1 on (0,1),
// silu(g) * u #2 on (0,2) (fused), down #3 on (0,3). spatial-loads adds load
// regions: Wg (input 1 of #0) and Wu (input 1 of #1) are copied into L1 on
// (1,0) by one data-movement generic with two outputs, Wd (input 1 of #3) on
// (1,1). Each consumer reads its weight over an edge. Wu's to_layout comes
// after the gate matmul in the function; it moves up so that the load, which
// goes before its first consumer, can use it.

// CHECK-LABEL: IR Dump After D2MMaterializeSpatialGroups
// CHECK: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>, #d2m.spatial_edge<0:1 -> 2:0>, #d2m.spatial_edge<1:0 -> 3:0>, #d2m.spatial_edge<2:0 -> 3:1>, #d2m.spatial_edge<3:0 -> 5:0>, #d2m.spatial_edge<4:0 -> 5:1>], grid_ranges = [#ttcore.core_range<(1,0), (1,0)>, #ttcore.core_range<(0,0), (0,0)>, #ttcore.core_range<(0,1), (0,1)>, #ttcore.core_range<(0,2), (0,2)>, #ttcore.core_range<(1,1), (1,1)>, #ttcore.core_range<(0,3), (0,3)>]}
// The load on (1,0): DRAM in, L1 out, a remote_load / remote_store per input.
// CHECK: d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>
// CHECK-NEXT: ins(%{{.*}}, %{{.*}} : tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #ttcore.metal_layout<{{.*}}dram, sharded>>, tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #ttcore.metal_layout<{{.*}}dram, sharded>>)
// CHECK-NEXT: outs(%{{.*}}, %{{.*}} : tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #ttcore.metal_layout<{{.*}}l1, sharded>>, tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #ttcore.metal_layout<{{.*}}l1, sharded>>)
// CHECK: d2m.remote_load
// CHECK: d2m.remote_store
// CHECK: d2m.remote_load
// CHECK: d2m.remote_store
// CHECK: d2m.spatial_yield %[[L:.*]]#0, %[[L]]#1
// The gate reads Wg from its block argument.
// CHECK: ^bb0(%[[WG:.*]]: tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #ttcore.metal_layout<{{.*}}l1, sharded>>):
// CHECK: d2m.remote_load %{{.*}} %[[WG]]

// All six regions run in one program.
// METAL: "ttmetal.enqueue_program"{{.*}}#ttmetal.core_range<1x0, 1x1>{{.*}}#ttmetal.core_range<1x1, 1x1>

// ERR-GROUP: error: 'd2m.generic' op loads names generic #0, which is not in a spatial group
// ERR-DRAM: error: 'd2m.generic' op loads names input 1 of generic #0, which is not in DRAM
// ERR-TYPES: error: 'd2m.generic' op loads puts inputs of different types on the same cores #ttcore.core_range<(1,0), (1,0)>
// ERR-OPERAND: error: 'd2m.generic' op loads names input 2 of generic #0, which has 2 inputs
// ERR-INDEX: error: loads names generic #9, but the module has only 5 generics
// ERR-COUNT: error: loads takes (generic, operand, y0, x0, y1, x1) per input, got 5 values
func.func @glu(%x: tensor<32x256xbf16>, %wg: tensor<256x256xbf16>, %wu: tensor<256x256xbf16>, %wd: tensor<256x256xbf16>) -> tensor<32x256xbf16> {
  %g = "ttir.matmul"(%x, %wg) : (tensor<32x256xbf16>, tensor<256x256xbf16>) -> tensor<32x256xbf16>
  %s = "ttir.silu"(%g) : (tensor<32x256xbf16>) -> tensor<32x256xbf16>
  %u = "ttir.matmul"(%x, %wu) : (tensor<32x256xbf16>, tensor<256x256xbf16>) -> tensor<32x256xbf16>
  %h = "ttir.multiply"(%s, %u) : (tensor<32x256xbf16>, tensor<32x256xbf16>) -> tensor<32x256xbf16>
  %y = "ttir.matmul"(%h, %wd) : (tensor<32x256xbf16>, tensor<256x256xbf16>) -> tensor<32x256xbf16>
  return %y : tensor<32x256xbf16>
}
