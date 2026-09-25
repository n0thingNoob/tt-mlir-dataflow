// RUN: ttmlir-opt %s --ttir-to-ttmetal-pipeline="execution-strategy=spatial dump-spatial-planning=true system-desc-path=%system_desc_path%" --mlir-print-op-generic -o %t.mlir 2> %t.report
// RUN: FileCheck %s --input-file=%t.report
// RUN: ttmlir-translate %t.mlir --ttmetal-to-flatbuffer -o %t.ttm

// Compile with a live descriptor for device execution; run_auto_spatial.py
// additionally checks the generated mapping, repeated execution and numerics.
// CHECK: selected S0 members=[G0] reason=temporal fallback:
// CHECK: selected S1 members=[G1] reason=temporal fallback:
// CHECK: selected S2 members=[G2] reason=temporal fallback:
func.func @main(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  %0 = "ttir.matmul"(%a, %b) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
  %1 = "ttir.relu"(%0) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
  %2 = "ttir.neg"(%1) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
  return %2 : tensor<64x64xbf16>
}
