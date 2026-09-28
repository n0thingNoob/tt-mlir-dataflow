// The same TTIR fixtures are used by the opt-in device runner. Rectangular
// Shapes: 64x96 @ 96x128 -> relu -> @ 128x96 (K is never just one tile).
// RUN: ttmlir-opt %S/../../../Silicon/TTMetal/n150/spatial/auto_gemm_chain.mlir --d2m-fe-pipeline="execution-strategy=spatial spatial-pipeline-max-shards=1 dump-spatial-planning=true" --mlir-print-ir-after=d2m-materialize-spatial-pipelines --mlir-disable-threading --mlir-print-op-generic -o %t.fe 2> %t.report
// RUN: FileCheck %s --check-prefix=GEMM --input-file=%t.report
// RUN: ttmlir-opt %S/../../../Silicon/TTMetal/n150/spatial/auto_gemm_chain.mlir --ttir-to-ttmetal-pipeline="execution-strategy=spatial spatial-pipeline-max-shards=1" --mlir-print-op-generic -o %t.metal
// RUN: FileCheck %s --check-prefix=METAL --input-file=%t.metal --implicit-check-not='memory_space<dram>'
// RUN: ttmlir-translate %t.metal --ttmetal-to-flatbuffer -o %t.ttm
// RUN: sed -E 's/[0-9]+x[0-9]+xbf16/64x64xbf16/g' %S/../../../Silicon/TTMetal/n150/spatial/auto_gemm_chain.mlir | ttmlir-opt --ttir-to-ttmetal-pipeline="execution-strategy=spatial spatial-pipeline-max-shards=1 dump-spatial-planning=true" --mlir-print-op-generic -o %t.square 2> %t.square.report
// RUN: FileCheck %S/../../../Silicon/TTMetal/n150/spatial/auto_gemm_chain.mlir --input-file=%t.square.report
// RUN: ttmlir-translate %t.square --ttmetal-to-flatbuffer -o %t.square.ttm

// GEMM: selected pipeline members=[G0, G1, G2]
// GEMM: emitted L1 tile pipeline
// GEMM: block_factors = [2, 4, 1]
// GEMM: memref<1x3x!ttcore.tile<32x32, bf16>>
// GEMM: memref<3x1x!ttcore.tile<32x32, bf16>>
// GEMM: d2m.pipeline_stage = 0
// GEMM: block_factors = [2, 4]
// GEMM: "d2m.semaphore_wait"
// GEMM: d2m.pipeline_stage = 1
// GEMM: block_factors = [2, 3, 1]
// GEMM: [[COLS:%.+]] = "arith.constant"() <{value = 4 : index}>
// GEMM: [[ONE:%.+]] = "arith.constant"() <{value = 1 : index}>
// GEMM: [[ROW:%.+]] = "arith.addi"({{%.+}}, [[ONE]])
// GEMM: [[READY:%.+]] = "arith.muli"([[ROW]], [[COLS]])
// GEMM: "d2m.semaphore_wait"({{%.+}}, [[READY]])
// GEMM: memref<1x4x!ttcore.tile<32x32, bf16>>
// GEMM: memref<4x1x!ttcore.tile<32x32, bf16>>
// GEMM: d2m.pipeline_stage = 2
// METAL: "ttmetal.enqueue_program"{{.*}}#ttmetal.compute_config<@{{[^,]+}}, #ttmetal.core_range<0x0, 1x1>{{.*}}#ttmetal.compute_config<@{{[^,]+}}, #ttmetal.core_range<0x1, 1x1>{{.*}}#ttmetal.compute_config<@{{[^,]+}}, #ttmetal.core_range<0x2, 1x1>
// METAL: callee = "noc_semaphore_inc"
// METAL: callee = "experimental::matmul_block"
// METAL: callee = "experimental::semaphore_wait_min"

// Supported elementwise chain: square and 2x3 tile rectangular variants.
// RUN: ttmlir-opt %S/../../../Silicon/TTMetal/n150/spatial/auto_elementwise_chain.mlir --ttir-to-ttmetal-pipeline="execution-strategy=spatial spatial-pipeline-max-shards=1 dump-spatial-planning=true" --mlir-print-op-generic -o %t.eltwise 2> %t.eltwise.report
// RUN: FileCheck %S/../../../Silicon/TTMetal/n150/spatial/auto_elementwise_chain.mlir --input-file=%t.eltwise.report
// RUN: ttmlir-translate %t.eltwise --ttmetal-to-flatbuffer -o %t.eltwise.ttm
// RUN: sed 's/128x128/64x96/g' %S/../../../Silicon/TTMetal/n150/spatial/auto_elementwise_chain.mlir | ttmlir-opt --ttir-to-ttmetal-pipeline="execution-strategy=spatial spatial-pipeline-max-shards=1 dump-spatial-planning=true" --mlir-print-op-generic -o %t.rect 2> %t.rect.report
// RUN: FileCheck %S/../../../Silicon/TTMetal/n150/spatial/auto_elementwise_chain.mlir --input-file=%t.rect.report
// RUN: ttmlir-translate %t.rect --ttmetal-to-flatbuffer -o %t.rect.ttm

// One spatial diamond followed by one temporal matmul with an internal RHS.
// RUN: ttmlir-opt %S/../../../Silicon/TTMetal/n150/spatial/auto_mixed.mlir --ttir-to-ttmetal-pipeline="execution-strategy=spatial spatial-pipeline-max-shards=1 dump-spatial-planning=true" --mlir-print-op-generic -o %t.mixed 2> %t.mixed.report
// RUN: FileCheck %S/../../../Silicon/TTMetal/n150/spatial/auto_mixed.mlir --input-file=%t.mixed.report
// RUN: FileCheck %s --check-prefix=MIXED --input-file=%t.mixed.report
// RUN: ttmlir-translate %t.mixed --ttmetal-to-flatbuffer -o %t.mixed.ttm
// MIXED: selected pipeline members=[G0, G1, G2, G3]
// MIXED: selected S1 members=[G4]
// MIXED: emitted L1 tile pipeline
