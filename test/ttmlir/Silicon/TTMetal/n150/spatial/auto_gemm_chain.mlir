// RUN: ttmlir-opt %s --ttir-to-ttmetal-pipeline="execution-strategy=spatial dump-spatial-planning=true system-desc-path=%system_desc_path%" --mlir-print-op-generic -o %t.mlir 2> %t.report
// RUN: FileCheck %s --input-file=%t.report
// RUN: ttmlir-translate %t.mlir --ttmetal-to-flatbuffer -o %t.ttm
// CHECK: selected pipeline members=[G0, G1, G2]
// CHECK: emitted L1 tile pipeline
func.func @main(%a: tensor<64x96xbf16>, %b: tensor<96x128xbf16>, %c: tensor<128x96xbf16>) -> tensor<64x96xbf16> {
  %0 = "ttir.matmul"(%a, %b) : (tensor<64x96xbf16>, tensor<96x128xbf16>) -> tensor<64x128xbf16>
  %1 = "ttir.relu"(%0) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
  %2 = "ttir.matmul"(%1, %c) : (tensor<64x128xbf16>, tensor<128x96xbf16>) -> tensor<64x96xbf16>
  return %2 : tensor<64x96xbf16>
}
