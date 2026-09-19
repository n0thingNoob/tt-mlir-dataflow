// RUN: ttmlir-opt %s --ttcore-register-device --ttir-to-d2m -o %t.dag
// RUN: ttmlir-opt %t.dag --d2m-spatial-planning -o %t.planned
// RUN: diff %t.dag %t.planned
// RUN: ttmlir-opt %t.planned --d2m-spatial-planning -o %t.repeated
// RUN: diff %t.planned %t.repeated
// RUN: ttmlir-opt %S/grid_selection_spatial.mlir --split-input-file -o %t.existing
// RUN: ttmlir-opt %S/grid_selection_spatial.mlir --split-input-file --d2m-spatial-planning -o %t.existing.planned
// RUN: diff %t.existing %t.existing.planned

// Empty and external functions are legal. Functions form separate DAGs.
module {
  func.func private @external()
  func.func @empty() {
    return
  }

  // Independent branches must not receive invented groups or core ranges.
  func.func @independent(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>) {
    %0 = "ttir.add"(%a, %b) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    %1 = "ttir.multiply"(%a, %b) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %0, %1 : tensor<64x64xf32>, tensor<64x64xf32>
  }

  // Trace producer dependencies through a representational layout view.
  func.func @through_view(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>) -> tensor<32x128xf32> {
    %0 = "ttir.add"(%a, %b) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    %1 = "ttir.reshape"(%0) {shape = [32 : i32, 128 : i32]} : (tensor<64x64xf32>) -> tensor<32x128xf32>
    %2 = "ttir.add"(%1, %1) : (tensor<32x128xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>
    return %2 : tensor<32x128xf32>
  }
}
