// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="ttnn-mode=false execution-strategy=spatial" %s | FileCheck %s

// Three 256x512 eltwise generics grouped at target 4x4 (three 4x4 regions,
// 2x4 tiles per core). d2m-reblock-generics rebuilds them on views of their
// operands, so both ends of each edge reach the edge buffer through a view.
// The full pipeline still lowers both edges: one local semaphore each, shared
// by the producer and consumer kernels of the one spatial program.

// CHECK-LABEL: func.func @reblocked_chain
// CHECK: %[[S0:.*]] = "ttmetal.create_local_semaphore"() <{initial_value = 0 : ui32}>
// CHECK: %[[S1:.*]] = "ttmetal.create_local_semaphore"() <{initial_value = 0 : ui32}>
// CHECK: "ttmetal.enqueue_program"({{.*}}%[[S0]]{{.*}}%[[S0]], %[[S1]]{{.*}}%[[S1]]
// CHECK-SAME: core_range<0x3, 4x4>
// CHECK-SAME: core_range<4x0, 4x4>
// CHECK-SAME: core_range<4x4, 4x4>
// CHECK-NOT: d2m.spatial
func.func @reblocked_chain(%x: tensor<256x512xbf16>) -> tensor<256x512xbf16> {
  %0 = "ttir.exp"(%x) : (tensor<256x512xbf16>) -> tensor<256x512xbf16>
  %1 = "ttir.tanh"(%0) : (tensor<256x512xbf16>) -> tensor<256x512xbf16>
  %2 = "ttir.exp"(%1) : (tensor<256x512xbf16>) -> tensor<256x512xbf16>
  return %2 : tensor<256x512xbf16>
}
