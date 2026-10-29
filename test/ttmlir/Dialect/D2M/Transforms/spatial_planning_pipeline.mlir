// RUN: ttmlir-opt %s --d2m-fe-pipeline --mlir-print-ir-before=d2m-grid-selection --mlir-disable-threading -o %t.default 2> %t.default.before-grid
// RUN: ttmlir-opt %s --d2m-fe-pipeline="execution-strategy=temporal" --mlir-print-ir-before=d2m-grid-selection --mlir-disable-threading -o %t.temporal 2> %t.temporal.before-grid
// RUN: ttmlir-opt %s --d2m-fe-pipeline="execution-strategy=spatial" --mlir-print-ir-before=d2m-grid-selection --mlir-disable-threading -o %t.spatial 2> %t.spatial.before-grid
// RUN: diff %t.default.before-grid %t.temporal.before-grid
// RUN: FileCheck %s --check-prefix=PLANNED --input-file=%t.spatial.before-grid
// RUN: ttmlir-opt %t.default --d2m-spatial-planning -o %t.bufferized
// RUN: diff %t.default %t.bufferized
// RUN: ttmlir-opt %s --d2m-fe-pipeline="execution-strategy=spatial" --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=PIPELINE
// RUN: ttmlir-opt %s --d2m-fe-pipeline="execution-strategy=temporal" --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=TEMPORAL --implicit-check-not=d2m-spatial-planning
// RUN: ttmlir-opt %s --ttir-to-ttmetal-pipeline -o %t.metal.default
// RUN: ttmlir-opt %s --ttir-to-ttmetal-pipeline="execution-strategy=spatial" -o %t.metal.spatial
// RUN: FileCheck %s --check-prefix=METAL --input-file=%t.metal.default --implicit-check-not=d2m.spatial --implicit-check-not=d2m.generic
// RUN: FileCheck %s --check-prefix=METAL --input-file=%t.metal.spatial --implicit-check-not=d2m.spatial --implicit-check-not=d2m.generic
// RUN: not ttmlir-opt %s --d2m-fe-pipeline="execution-strategy=invalid" 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not ttmlir-opt %s --d2m-fe-pipeline="execution-strategy=auto" 2>&1 | FileCheck %s --check-prefix=AUTO

// Planning and group materialization run right before grid selection, after
// the passes that still add or remove generics (ScalarizeConstTensors drops
// splat fills, MaterializeViewReturns adds view copies), so the groups name
// the generics GridSelection sees.
// PIPELINE: ttir-to-d2m
// PIPELINE-NEXT: d2m-scalarize-const-tensors,
// PIPELINE-NEXT: d2m-materialize-view-returns,
// PIPELINE-NEXT: d2m-spatial-planning{ max-cbs=32 max-semaphores=16 policy=greedy remarks=false  ttnn-mode=false},
// PIPELINE-NEXT: d2m-materialize-spatial-groups{},
// PIPELINE-NEXT: d2m-grid-selection
// TEMPORAL: ttir-to-d2m
// INVALID: Cannot find option named 'invalid'
// AUTO: Cannot find option named 'auto'
// METAL: "ttmetal.enqueue_write_buffer"
// METAL: "ttmetal.enqueue_program"
// METAL: "ttmetal.enqueue_read_buffer"
// METAL: emitc.call_opaque

// Compare exact IR before grid selection: the existing allocator can choose
// different addresses between identical compilations. Full lowering is checked
// separately for both strategies.

// Temporal and default produce the same IR. With execution-strategy=spatial
// the greedy policy puts the whole diamond in one spatial: the shared producer
// fans out to both branches (the multiply reads it twice through one edge) and
// the join reads both branches.
// PLANNED: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>, #d2m.spatial_edge<0:0 -> 2:0>, #d2m.spatial_edge<1:0 -> 3:0>, #d2m.spatial_edge<2:0 -> 3:1>], grid_ranges = [#ttcore.core_range<(0,0), (1,1)>, #ttcore.core_range<(0,2), (1,3)>, #ttcore.core_range<(0,4), (1,5)>, #ttcore.core_range<(0,6), (1,7)>]}

// A diamond with a repeated operand exercises shared dependencies.
module {
  func.func @diamond(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %0 = "ttir.add"(%a, %b) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    %1 = "ttir.multiply"(%0, %0) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    %2 = "ttir.add"(%0, %b) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    %3 = "ttir.add"(%1, %2) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %3 : tensor<64x64xf32>
  }
}
