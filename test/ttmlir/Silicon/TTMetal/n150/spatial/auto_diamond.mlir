// RUN: ttmlir-opt %s --ttir-to-ttmetal-pipeline="execution-strategy=spatial dump-spatial-planning=true system-desc-path=%system_desc_path%" --mlir-print-op-generic -o %t.mlir 2> %t.report
// RUN: FileCheck %s --input-file=%t.report
// RUN: ttmlir-translate %t.mlir --ttmetal-to-flatbuffer -o %t.ttm

// CHECK: selected pipeline members=[G0, G1, G2, G3]
// CHECK: emitted L1 tile pipeline
func.func @main(%a: tensor<128x128xbf16>, %b: tensor<128x128xbf16>) -> tensor<128x128xbf16> {
  %0 = "ttir.add"(%a, %b) : (tensor<128x128xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>
  %1 = "ttir.relu"(%0) : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
  %2 = "ttir.neg"(%0) : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
  %3 = "ttir.add"(%1, %2) : (tensor<128x128xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>
  return %3 : tensor<128x128xbf16>
}
