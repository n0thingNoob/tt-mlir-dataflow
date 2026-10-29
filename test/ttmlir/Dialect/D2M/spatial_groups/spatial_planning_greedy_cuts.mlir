// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-spatial-planning="policy=greedy max-semaphores=1 remarks=true" -mlir-print-op-on-diagnostic=false %s 2>&1 >/dev/null | FileCheck %s --check-prefix=SEM
// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-spatial-planning="policy=greedy exclude-cores=0,0,0,2,0,4,0,6,2,0,2,2,2,4,2,6,4,0,4,2,4,4,4,6,6,0,6,2,6,4,6,6 remarks=true" -mlir-print-op-on-diagnostic=false %s 2>&1 >/dev/null | FileCheck %s --check-prefix=ROOM
// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-spatial-planning="policy=greedy skip-generics=1 remarks=true" -mlir-print-op-on-diagnostic=false %s 2>&1 >/dev/null | FileCheck %s --check-prefix=SKIP
// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-spatial-planning="policy=greedy ttnn-mode=true remarks=true" %s 2>&1 | FileCheck %s --check-prefix=TTNN

// Cuts of the greedy planner on a chain of 64x64 unary ops (each a 2x2
// generic). The remarks are checked as text: expected-remark directives would
// be verified on every RUN line.

// max-semaphores=1: the middle of a chain holds an incoming and an outgoing
// edge semaphore.
// SEM: remark: generic #2 starts a new group: a core would need 2 local semaphores (limit 1)

// Excluding every other core of rows 0, 2, 4, 6 leaves no 2x2 square free:
// the group takes the first smaller target whose grids fit, 1x8 (1x2 grids
// in the free odd rows).
// ROOM: remark: spatial group 0 shrinks to target 1x8: #0 2x2->1x2 #1 2x2->1x2 #2 2x2->1x2 #3 2x2->1x2
// ROOM: remark: spatial group 0: generics #0 #1 #2 #3, 3 edges, 8 cores

// Skipping the second generic: it reads the first and feeds the third, so
// the first and third cannot share a group.
// SKIP: remark: generic #1 is not grouped: it is in skip-generics
// SKIP: remark: generic #2 starts a new group: not convex: it reads generic #1, which is outside the group and reads it
// SKIP: remark: spatial group 0: generics #2 #3

// TTNN mode does not lower spatial edges: no groups.
// TTNN-NOT: d2m.spatial_group
// TTNN-NOT: remark
func.func @chain(%x: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = "ttir.tanh"(%x) : (tensor<64x64xf32>) -> tensor<64x64xf32>
  %1 = "ttir.exp"(%0) : (tensor<64x64xf32>) -> tensor<64x64xf32>
  %2 = "ttir.tanh"(%1) : (tensor<64x64xf32>) -> tensor<64x64xf32>
  %3 = "ttir.exp"(%2) : (tensor<64x64xf32>) -> tensor<64x64xf32>
  return %3 : tensor<64x64xf32>
}
