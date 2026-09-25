// RUN: ttmlir-opt %s --ttcore-register-device --ttir-to-d2m -o %t.input
// RUN: ttmlir-opt %t.input --d2m-spatial-planning="dump-regions=true" -o %t.output 2> %t.report
// RUN: FileCheck %s --input-file=%t.report
// RUN: diff %t.input %t.output

// Exercise real conversion output, including layout changes, output tensor
// allocations between generics, and tensor-form compute operations. Grouping is
// structural: this chain is not reported as fused or assigned cores.
// CHECK-LABEL: spatial_region_candidates @metadata block0 candidate_only {
// CHECK: G0 type=matmul interval=0 producers=[] consumers=[G1]
// CHECK: compute_ops=[d2m.tile_matmul] resource_hints=[FPU]
// CHECK: inputs=[{{.*}}tensor<{{.*}}!ttcore.tile<32x32, f32>{{.*}}#ttcore.metal_layout
// CHECK: output_inits=[
// CHECK: results=[
// CHECK: captures=[
// CHECK: G1 type=activation interval=0 producers=[G0] consumers=[G2]
// CHECK: compute_ops=[d2m.tile_relu] resource_hints=[SFPU]
// CHECK: G2 type=activation interval=0 producers=[G1] consumers=[G3]
// CHECK: compute_ops=[d2m.tile_sigmoid] resource_hints=[SFPU]
// CHECK: G3 type=activation interval=0 producers=[G2] consumers=[G4]
// CHECK: compute_ops=[d2m.tile_gelu] resource_hints=[SFPU]
// CHECK: G4 type=elementwise interval=0 producers=[G3] consumers=[G5]
// CHECK: compute_ops=[d2m.tile_add] resource_hints=[FPU_or_SFPU]
// CHECK: G5 type=unknown interval=0 producers=[G4] consumers=[]
// CHECK: compute_ops=[d2m.tile_exp] resource_hints=[unknown]
// CHECK: G0 -> G1.operand0 {{.*}}connection=projected via=[d2m.to_layout, d2m.to_layout]
// CHECK: members=[G0, G1, G2, G3, G4, G5] kinds=[producer_consumer_chain]
// CHECK-NOT: candidate R1
func.func @metadata(%a: tensor<32x32xf32>, %b: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = "ttir.matmul"(%a, %b) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  %1 = "ttir.relu"(%0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %2 = "ttir.sigmoid"(%1) : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %3 = "ttir.gelu"(%2) : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %4 = "ttir.add"(%3, %b) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  %5 = "ttir.exp"(%4) : (tensor<32x32xf32>) -> tensor<32x32xf32>
  return %5 : tensor<32x32xf32>
}
