// RUN: ttmlir-opt --verify-diagnostics -mlir-print-op-on-diagnostic=false --ttcore-register-device --ttir-to-d2m --d2m-spatial-planning="policy=greedy exclude-cores=0,1,0,2,0,3,0,4,0,5,0,6,0,7,1,0,1,1,1,2,1,3,1,4,1,5,1,6,1,7,2,0,2,1,2,2,2,3,2,4,2,5,2,6,2,7,3,0,3,1,3,2,3,3,3,4,3,5,3,6,3,7,4,0,4,1,4,2,4,3,4,4,4,5,4,6,4,7,5,0,5,1,5,2,5,3,5,4,5,5,5,6,5,7,6,0,6,1,6,2,6,3,6,4,6,5,6,6,6,7,7,0,7,1,7,2,7,3,7,4,7,5,7,6,7,7 remarks=true" %s -o /dev/null

// Only core (0,0) is free: the first 1x1 generic takes it, and the second
// fits at no target, so the group is cut. The remark lists every target tried.
func.func @tiny(%x: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = "ttir.tanh"(%x) : (tensor<32x32xf32>) -> tensor<32x32xf32>
  // expected-remark @+1 {{generic #1 starts a new group: no free 1x1 core range is left (nor with the group at targets 4x8, 8x4, 4x4, 2x8, 8x2, 2x4, 4x2, 1x8, 8x1, 2x2, 1x4, 4x1, 1x2, 2x1, 1x1)}}
  %1 = "ttir.exp"(%0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}
