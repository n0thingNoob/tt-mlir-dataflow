// RUN: ttmlir-opt %s --ttir-to-ttmetal-pipeline="execution-strategy=spatial dump-spatial-planning=true" --mlir-print-op-generic -o %t.lowered 2> %t.report
// RUN: FileCheck %s --input-file=%t.report
// RUN: FileCheck %s --check-prefix=LOWER --input-file=%t.lowered --implicit-check-not='memory_space<dram>'
// Stage 8 crosses a worker row. Semaphore destinations are relative to the
// sending stage, so outlining applies its placement offset only once.
// CHECK: selected pipeline members=[G0, G1, G2, G3, G4, G5, G6, G7, G8]
// CHECK: #ttcore.core_range<(1,0), (1,0)>
// CHECK: d2m.pipeline_signals = [array<i64: 1, 1, -7>]
// CHECK-SAME: d2m.pipeline_stage = 7
// LOWER: "ttmetal.enqueue_program"{{.*}}#ttmetal.compute_config<@{{[^,]+}}, #ttmetal.core_range<0x0, 1x1>{{.*}}#ttmetal.compute_config<@{{[^,]+}}, #ttmetal.core_range<1x0, 1x1>

func.func @chain(%a: tensor<32x32xbf16>, %b: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  %0 = "ttir.add"(%a, %b) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %1 = "ttir.add"(%0, %b) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %2 = "ttir.add"(%1, %b) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %3 = "ttir.add"(%2, %b) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %4 = "ttir.add"(%3, %b) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %5 = "ttir.add"(%4, %b) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %6 = "ttir.add"(%5, %b) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %7 = "ttir.add"(%6, %b) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %8 = "ttir.add"(%7, %b) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %8 : tensor<32x32xbf16>
}
