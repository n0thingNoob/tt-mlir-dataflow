// RUN: ttmlir-opt %s --ttir-to-ttmetal-pipeline --mlir-print-op-generic -o %t.off.mlir
// RUN: ttmlir-opt %s --ttir-to-ttmetal-pipeline='enable-dataflow-planning=true dump-dataflow-plan=true' --mlir-print-op-generic -o %t.on.mlir 2>&1 | FileCheck %s --check-prefix=TRACE
// RUN: diff %t.off.mlir %t.on.mlir
// RUN: FileCheck %s --input-file=%t.on.mlir --check-prefix=METAL
// RUN: ttmlir-translate %t.off.mlir --ttmetal-to-flatbuffer -o %t.off.ttm
// RUN: ttmlir-translate %t.on.mlir --ttmetal-to-flatbuffer -o %t.on.ttm
// RUN: test -s %t.off.ttm
// RUN: test -s %t.on.ttm
// RUN: ttmlir-opt %s --ttir-to-ttmetal-pipeline='enable-elementwise-fusion=true num-stream-buffers=1' -o %t.fused-off.mlir
// RUN: ttmlir-opt %s --ttir-to-ttmetal-pipeline='enable-elementwise-fusion=true num-stream-buffers=1 enable-dataflow-planning=true' -o %t.fused-on.mlir
// RUN: diff %t.fused-off.mlir %t.fused-on.mlir
// RUN: not ttmlir-opt %s --ttir-to-ttmetal-pipeline='enable-dataflow-planning=true dump-dataflow-plan=true test-assume-l1-capacity=16' -o /dev/null 2>&1 | FileCheck %s --check-prefix=OOM

// TRACE: d2m-dataflow-plan function=@main
// TRACE: d2m-dataflow-blocking function=@main
// TRACE: d2m-dataflow-allocation function=@main status=success
// METAL-NOT: d2m.allocation_report
// METAL: "ttmetal.enqueue_program"
// METAL-NOT: d2m.allocation_report
// OOM: d2m-dataflow-blocking function=@main
// OOM: exceeds memory capacity
// OOM-NOT: d2m-dataflow-allocation

// Shared by the compiler test and the opt-in hardware test. Hardware runs
// must compile it again using a descriptor queried from their runtime.
// Generic assembly avoids EmitC expression custom-printer round-trip issues.
func.func @main(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  %0 = "ttir.matmul"(%a, %b) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
  %1 = "ttir.relu"(%0) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
  %2 = "ttir.neg"(%1) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
  return %2 : tensor<64x64xbf16>
}
