// RUN: ttmlir-opt --ttir-fusing="d2m-patterns-only=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

// d2m-patterns-only folds a transpose of matmul input B into transpose_b and
// x * sigmoid(x) into silu, and nothing else.

module {
  // y = x W^T as frontends write it for nn.Linear.
  // CHECK-LABEL: func.func @permute_b
  // CHECK-NOT: "ttir.permute"
  // CHECK: "ttir.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = true}>
  func.func @permute_b(%x: tensor<32x256xbf16>, %w: tensor<128x256xbf16>) -> tensor<32x128xbf16> {
    %0 = "ttir.permute"(%w) <{permutation = array<i64: 1, 0>}> : (tensor<128x256xbf16>) -> tensor<256x128xbf16>
    %1 = "ttir.matmul"(%x, %0) : (tensor<32x256xbf16>, tensor<256x128xbf16>) -> tensor<32x128xbf16>
    return %1 : tensor<32x128xbf16>
  }

  // D2M takes no transpose_a: a transpose of input A stays.
  // CHECK-LABEL: func.func @permute_a
  // CHECK: "ttir.permute"
  // CHECK: "ttir.matmul"
  // CHECK-SAME: transpose_a = false
  func.func @permute_a(%a: tensor<256x32xbf16>, %w: tensor<256x128xbf16>) -> tensor<32x128xbf16> {
    %0 = "ttir.permute"(%a) <{permutation = array<i64: 1, 0>}> : (tensor<256x32xbf16>) -> tensor<32x256xbf16>
    %1 = "ttir.matmul"(%0, %w) : (tensor<32x256xbf16>, tensor<256x128xbf16>) -> tensor<32x128xbf16>
    return %1 : tensor<32x128xbf16>
  }

  // CHECK-LABEL: func.func @silu
  // CHECK-NOT: "ttir.sigmoid"
  // CHECK: %[[S:.*]] = "ttir.silu"(%arg0)
  // CHECK-NOT: "ttir.multiply"
  // CHECK: return %[[S]]
  func.func @silu(%x: tensor<32x128xbf16>) -> tensor<32x128xbf16> {
    %0 = "ttir.sigmoid"(%x) : (tensor<32x128xbf16>) -> tensor<32x128xbf16>
    %1 = "ttir.multiply"(%x, %0) : (tensor<32x128xbf16>, tensor<32x128xbf16>) -> tensor<32x128xbf16>
    return %1 : tensor<32x128xbf16>
  }

  // The full pass turns matmul + add into linear, which D2M does not lower;
  // here both stay.
  // CHECK-LABEL: func.func @bias
  // CHECK-NOT: "ttir.linear"
  // CHECK: "ttir.matmul"
  // CHECK: "ttir.add"
  func.func @bias(%x: tensor<32x256xbf16>, %w: tensor<256x128xbf16>, %b: tensor<32x128xbf16>) -> tensor<32x128xbf16> {
    %0 = "ttir.matmul"(%x, %w) : (tensor<32x256xbf16>, tensor<256x128xbf16>) -> tensor<32x128xbf16>
    %1 = "ttir.add"(%0, %b) : (tensor<32x128xbf16>, tensor<32x128xbf16>) -> tensor<32x128xbf16>
    return %1 : tensor<32x128xbf16>
  }
}
