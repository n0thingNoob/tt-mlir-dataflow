// RUN: ttmlir-opt %s --ttir-to-ttmetal-pipeline="execution-strategy=spatial dump-spatial-planning=true" --mlir-print-op-generic -o %t.mlir 2> %t.report
// RUN: FileCheck %s --check-prefix=REPORT --input-file=%t.report
// RUN: FileCheck %s --check-prefix=LOWER --input-file=%t.mlir
// RUN: ttmlir-translate %t.mlir --ttmetal-to-flatbuffer -o %t.ttm

// The complete diamond runs in one program. External tilize/untilize programs
// are permitted, but every intermediate buffer remains in L1.
// REPORT: spatial_selected_groups @diamond
// REPORT: selected pipeline members=[G0, G1, G2, G3]
// REPORT: spatial_selected_groups @chain
// REPORT: selected S0 members=[G0] reason=temporal fallback:
// REPORT: selected S1 members=[G1] reason=temporal fallback:
// REPORT: selected S2 members=[G2] reason=temporal fallback:
// REPORT: emitted L1 tile pipeline
// REPORT: d2m.pipeline_stage = 0
// REPORT: d2m.pipeline_stage = 1
// REPORT: d2m.pipeline_stage = 2
// REPORT: d2m.pipeline_stage = 3
// LOWER: "ttmetal.create_global_semaphore"
// LOWER: "ttmetal.reset_global_semaphore"
// LOWER: "ttmetal.enqueue_program"{{.*}}#ttmetal.compute_config<@{{[^,]+}}, #ttmetal.core_range<0x0, 1x1>{{.*}}#ttmetal.compute_config<@{{[^,]+}}, #ttmetal.core_range<0x1, 1x1>{{.*}}#ttmetal.compute_config<@{{[^,]+}}, #ttmetal.core_range<0x2, 1x1>{{.*}}#ttmetal.compute_config<@{{[^,]+}}, #ttmetal.core_range<0x3, 1x1>
// LOWER: noc0.async_write_barrier()
// LOWER: callee = "noc_semaphore_inc"
// LOWER: callee = "experimental::semaphore_wait_min"
// LOWER: noc0.async_read(

func.func @diamond(%a: tensor<32x32xbf16>, %b: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  %0 = "ttir.add"(%a, %b) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %1 = "ttir.relu"(%0) : (tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %2 = "ttir.neg"(%0) : (tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %3 = "ttir.add"(%1, %2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %3 : tensor<32x32xbf16>
}

func.func @chain(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  %0 = "ttir.matmul"(%a, %b) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
  %1 = "ttir.relu"(%0) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
  %2 = "ttir.neg"(%1) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
  return %2 : tensor<64x64xbf16>
}
