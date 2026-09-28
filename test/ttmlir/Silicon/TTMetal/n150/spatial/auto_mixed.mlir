// RUN: ttmlir-opt %s --ttir-to-ttmetal-pipeline="execution-strategy=spatial spatial-pipeline-max-shards=1 dump-spatial-planning=true system-desc-path=%system_desc_path%" --mlir-print-op-generic -o %t.mlir 2> %t.report
// RUN: FileCheck %s --input-file=%t.report
// RUN: ttmlir-translate %t.mlir --ttmetal-to-flatbuffer -o %t.ttm

// CHECK: selected pipeline members=[G0, G1, G2, G3]
// CHECK: selected S1 members=[G4] reason=temporal fallback:
// CHECK: emitted L1 tile pipeline
func.func @main(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>, %w: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  %0 = "ttir.add"(%a, %b) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
  %1 = "ttir.relu"(%0) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
  %2 = "ttir.neg"(%0) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
  %3 = "ttir.add"(%1, %2) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
  %4 = "ttir.matmul"(%w, %3) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
  return %4 : tensor<64x64xbf16>
}
