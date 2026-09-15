// RUN: ttmlir-opt %s --d2m-fe-pipeline -o %t.first
// RUN: ttmlir-opt %s --d2m-fe-pipeline -o %t.second
// RUN: diff %t.first %t.second
// RUN: ttmlir-opt %s --d2m-fe-pipeline -o %t.third
// RUN: diff %t.first %t.third
// RUN: ttmlir-opt %s --d2m-fe-pipeline -o %t.fourth
// RUN: diff %t.first %t.fourth

// Independent compiler processes must retain the same solver variable order,
// allocation addresses, and deallocation order, regardless of pointer hashes.
func.func @main(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  %0 = "ttir.matmul"(%a, %b) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
  %1 = "ttir.relu"(%0) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
  %2 = "ttir.neg"(%1) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
  return %2 : tensor<64x64xbf16>
}
