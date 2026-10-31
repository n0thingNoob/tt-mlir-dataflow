// RUN: ttmlir-opt --split-input-file --verify-diagnostics -mlir-print-op-on-diagnostic=false --d2m-fe-pipeline="execution-strategy=spatial spatial-remarks=true spatial-fuse-elementwise=true" %s | FileCheck %s
// RUN: ttmlir-opt --split-input-file -mlir-print-op-on-diagnostic=false --d2m-fe-pipeline="execution-strategy=spatial spatial-remarks=true" %s 2>&1 | FileCheck %s --check-prefix=NOFUSE

// spatial-fuse-elementwise fuses elementwise chains before spatial planning,
// so each chain takes one region.

// A split-K style hub: four matmuls feed a chain of three adds. Fused, the
// adds are one generic with four inputs, each an edge from one matmul.
// CHECK-LABEL: func.func @hub4
// CHECK: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 4:0>, #d2m.spatial_edge<1:0 -> 4:1>, #d2m.spatial_edge<2:0 -> 4:2>, #d2m.spatial_edge<3:0 -> 4:3>]
// NOFUSE: remark: spatial group 0: generics #0 #1 #2 #3 #4 #5 #6, 6 edges, 56 cores, 21 CBs, 5 semaphores per core at most
func.func @hub4(%x: tensor<32x256xbf16>, %w0: tensor<256x256xbf16>, %w1: tensor<256x256xbf16>, %w2: tensor<256x256xbf16>, %w3: tensor<256x256xbf16>) -> tensor<32x256xbf16> {
  // expected-remark @+1 {{spatial group 0: generics #0 #1 #2 #3 #4, 4 edges, 40 cores, 17 CBs, 5 semaphores per core at most}}
  %p0 = "ttir.matmul"(%x, %w0) : (tensor<32x256xbf16>, tensor<256x256xbf16>) -> tensor<32x256xbf16>
  %p1 = "ttir.matmul"(%x, %w1) : (tensor<32x256xbf16>, tensor<256x256xbf16>) -> tensor<32x256xbf16>
  %p2 = "ttir.matmul"(%x, %w2) : (tensor<32x256xbf16>, tensor<256x256xbf16>) -> tensor<32x256xbf16>
  %p3 = "ttir.matmul"(%x, %w3) : (tensor<32x256xbf16>, tensor<256x256xbf16>) -> tensor<32x256xbf16>
  %s1 = "ttir.add"(%p0, %p1) : (tensor<32x256xbf16>, tensor<32x256xbf16>) -> tensor<32x256xbf16>
  %s2 = "ttir.add"(%s1, %p2) : (tensor<32x256xbf16>, tensor<32x256xbf16>) -> tensor<32x256xbf16>
  %s3 = "ttir.add"(%s2, %p3) : (tensor<32x256xbf16>, tensor<32x256xbf16>) -> tensor<32x256xbf16>
  return %s3 : tensor<32x256xbf16>
}

// -----

// Gated MLP: silu(g) * u is one generic (one region) between the gate/up
// matmuls and the down matmul.
// CHECK-LABEL: func.func @glu_silu
// CHECK: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 2:0>, #d2m.spatial_edge<1:0 -> 2:1>, #d2m.spatial_edge<2:0 -> 3:0>]
// CHECK: d2m.tile_silu
// CHECK-NEXT: linalg.yield
// CHECK: d2m.tile_mul
// NOFUSE: remark: spatial group 0: generics #0 #1 #2 #3 #4, 4 edges, 40 cores, 14 CBs, 5 semaphores per core at most
func.func @glu_silu(%x: tensor<32x256xbf16>, %wg: tensor<256x512xbf16>, %wu: tensor<256x512xbf16>, %wd: tensor<512x256xbf16>) -> tensor<32x256xbf16> {
  // expected-remark @+1 {{spatial group 0: generics #0 #1 #2 #3, 3 edges, 32 cores, 12 CBs, 5 semaphores per core at most}}
  %g = "ttir.matmul"(%x, %wg) : (tensor<32x256xbf16>, tensor<256x512xbf16>) -> tensor<32x512xbf16>
  %u = "ttir.matmul"(%x, %wu) : (tensor<32x256xbf16>, tensor<256x512xbf16>) -> tensor<32x512xbf16>
  %a = "ttir.silu"(%g) : (tensor<32x512xbf16>) -> tensor<32x512xbf16>
  %h = "ttir.multiply"(%a, %u) : (tensor<32x512xbf16>, tensor<32x512xbf16>) -> tensor<32x512xbf16>
  %y = "ttir.matmul"(%h, %wd) : (tensor<32x512xbf16>, tensor<512x256xbf16>) -> tensor<32x256xbf16>
  return %y : tensor<32x256xbf16>
}
