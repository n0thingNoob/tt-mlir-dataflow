// RUN: ttmlir-opt %s --d2m-fe-pipeline="execution-strategy=spatial dump-spatial-planning=true" --mlir-print-op-generic -o %t.mlir 2> %t.report
// RUN: FileCheck %s --check-prefix=PLAN --input-file=%t.report
// RUN: FileCheck %s --check-prefix=IR --input-file=%t.mlir
// RUN: ttmlir-opt %t.mlir --d2m-materialize-spatial-pipelines -o %t.again --mlir-print-op-generic
// RUN: diff %t.mlir %t.again
// RUN: ttmlir-opt %s --ttir-to-ttmetal-pipeline="execution-strategy=spatial" --mlir-print-op-generic -o %t.lowered
// RUN: FileCheck %s --check-prefix=LOWER --input-file=%t.lowered --implicit-check-not='memory_space<dram>'
// RUN: ttmlir-translate %t.lowered --ttmetal-to-flatbuffer -o %t.ttm

// PLAN: selected pipeline members=[G0, G1, G2, G3]
// PLAN: emitted L1 tile pipeline
// PLAN-COUNT-4: block_factors = [4, 4]
// IR: "d2m.create_global_semaphore"
// IR: "d2m.reset_global_semaphore"
// IR: "d2m.spatial"
// IR: d2m.pipeline_stage = 0
// IR: "d2m.semaphore_wait"
// IR: d2m.pipeline_signals = [array<i64: 1, 0, 2>]
// IR-SAME: d2m.pipeline_stage = 1
// IR: "d2m.semaphore_wait"
// IR: d2m.pipeline_signals = [array<i64: 1, 0, 1>]
// IR-SAME: d2m.pipeline_stage = 2
// IR-COUNT-2: "d2m.semaphore_wait"
// IR: d2m.pipeline_stage = 3
// LOWER: "ttmetal.enqueue_program"{{.*}}#ttmetal.compute_config<@{{[^,]+}}, #ttmetal.core_range<0x0, 1x1>{{.*}}#ttmetal.compute_config<@{{[^,]+}}, #ttmetal.core_range<0x3, 1x1>
// LOWER: callee = "noc_semaphore_inc"
// LOWER: callee = "experimental::semaphore_wait_min"

func.func @main(%a: tensor<128x128xbf16>, %b: tensor<128x128xbf16>) -> tensor<128x128xbf16> {
  %0 = "ttir.add"(%a, %b) : (tensor<128x128xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>
  %1 = "ttir.relu"(%0) : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
  %2 = "ttir.neg"(%0) : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
  %3 = "ttir.add"(%1, %2) : (tensor<128x128xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>
  return %3 : tensor<128x128xbf16>
}
