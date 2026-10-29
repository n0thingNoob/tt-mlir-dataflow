// RUN: ttmlir-opt --split-input-file --verify-diagnostics -mlir-print-op-on-diagnostic=false --d2m-fe-pipeline="execution-strategy=spatial spatial-remarks=true" %s | FileCheck %s

// Greedy spatial planning on small real programs. The remarks are the
// budget record of each group: cores (worker grid 8x8, no core excluded),
// CBs (32 per spatial) and local semaphores per core (16). Every group keeps
// the default grids (any shrink would be an unexpected remark); shrinking is
// covered by spatial_planning_gre// MLP: three 2x8 generics in rows 0-1, 2-3, 4-5 and the last matmul's 2x4
// in rows 6-7.
// CHECK-LABEL: func.func @mlp
// CHECK: d2m.spatial {{.*}}grid_ranges = [#ttcore.core_range<(0,0), (1,7)>, #ttcore.core_range<(2,0), (3,7)>, #ttcore.core_range<(4,0), (5,7)>, #ttcore.core_range<(6,0), (7,3)>]
func.func @mlp(%x: tensor<64x128xbf16>, %w1: tensor<128x256xbf16>, %b1: tensor<64x256xbf16>, %w2: tensor<256x128xbf16>) -> tensor<64x128xbf16> {
  // expected-remark @+1 {{spatial group 0: generics #0 #1 #2 #3, 3 edges, 56 cores, 11 CBs, 5 semaphores per core at most}}
  %0 = "ttir.matmul"(%x, %w1) : (tensor<64x128xbf16>, tensor<128x256xbf16>) -> tensor<64x256xbf16>
  %1 = "ttir.add"(%0, %b1) : (tensor<64x256xbf16>, tensor<64x256xbf16>) -> tensor<64x256xbf16>
  %2 = "ttir.relu"(%1) : (tensor<64x256xbf16>) -> tensor<64x256xbf16>
  %3 = "ttir.matmul"(%2, %w2) : (tensor<64x256xbf16>, tensor<256x128xbf16>) -> tensor<64x128xbf16>
  return %3 : tensor<64x128xbf16>
}

// -----

// Fan-out and join: one group, the four 2x2 regions side by side in rows 0-1.
// CHECK-LABEL: func.func @fanout
// CHECK: d2m.spatial {{.*}}grid_ranges = [#ttcore.core_range<(0,0), (1,1)>, #ttcore.core_range<(0,2), (1,3)>, #ttcore.core_range<(0,4), (1,5)>, #ttcore.core_range<(0,6), (1,7)>]
func.func @fanout(%x: tensor<64x64xbf16>, %y: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  // expected-remark @+1 {{spatial group 0: generics #0 #1 #2 #3, 4 edges, 16 cores, 10 CBs, 6 semaphores per core at most}}
  %0 = "ttir.tanh"(%x) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
  %1 = "ttir.exp"(%0) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
  %2 = "ttir.matmul"(%0, %y) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
  %3 = "ttir.add"(%1, %2) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
  return %3 : tensor<64x64xbf16>
}

// -----

// Attention: the scaler fill stays outside; collapse-only edges are grouped.
// CHECK-LABEL: func.func @attn
// CHECK: d2m.spatial {{.*}}grid_ranges = [#ttcore.core_range<(0,0), (1,1)>, #ttcore.core_range<(0,2), (1,3)>, #ttcore.core_range<(0,4), (1,4)>, #ttcore.core_range<(0,5), (1,6)>, #ttcore.core_range<(2,0), (3,1)>]
func.func @attn(%q: tensor<64x64xbf16>, %kt: tensor<64x64xbf16>, %v: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  // expected-remark @+1 {{spatial group 0: generics #0 #1 #3 #4 #5, 5 edges, 18 cores, 14 CBs, 5 semaphores per core at most}}
  %0 = "ttir.matmul"(%q, %kt) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
  // expected-remark @+1 {{generic #2 is not grouped: it has no inputs (a fill stays outside, as an input)}}
  %1 = "ttir.softmax"(%0) <{dimension = 1 : si32}> : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
  %2 = "ttir.matmul"(%1, %v) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
  return %2 : tensor<64x64xbf16>
}

// -----

// RMSNorm -> matmul: the gamma copy is row-major, so the tilize in front of
// the gamma multiply cuts the chain.
// CHECK-LABEL: func.func @rms_mm
// CHECK: d2m.spatial {{.*}}grid_ranges = [#ttcore.core_range<(0,0), (1,3)>, #ttcore.core_range<(0,4), (1,4)>, #ttcore.core_range<(0,5), (1,5)>, #ttcore.core_range<(0,6), (1,6)>, #ttcore.core_range<(2,0), (3,3)>, #ttcore.core_range<(2,4), (2,7)>]
// CHECK: d2m.spatial {{.*}}grid_ranges = [#ttcore.core_range<(0,0), (1,3)>, #ttcore.core_range<(0,4), (1,7)>]
func.func @rms_mm(%x: tensor<64x128xbf16>, %g: tensor<128xbf16>, %w: tensor<128x128xbf16>) -> tensor<64x128xbf16> {
  // expected-remark @+4 {{generic #1 is not grouped: it has no inputs (a fill stays outside, as an input)}}
  // expected-remark @+3 {{generic #7 starts a new group: input 1 needs a layout change from its producer}}
  // expected-remark @+2 {{spatial group 0: generics #0 #2 #3 #4 #5 #6, 4 edges, 26 cores, 15 CBs, 2 semaphores per core at most}}
  // expected-remark @+1 {{spatial group 1: generics #7 #8, 1 edges, 16 cores, 6 CBs, 5 semaphores per core at most}}
  %0 = "ttir.rms_norm"(%x, %g) <{normalized_shape = array<i64: 128>, epsilon = 1.0e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<64x128xbf16>, tensor<128xbf16>) -> tensor<64x128xbf16>
  %1 = "ttir.matmul"(%0, %w) : (tensor<64x128xbf16>, tensor<128x128xbf16>) -> tensor<64x128xbf16>
  return %1 : tensor<64x128xbf16>
}

// -----

// Decode matmul chain (M = 32): three 1x8 rows.
// CHECK-LABEL: func.func @decode_mm
// CHECK: d2m.spatial {{.*}}grid_ranges = [#ttcore.core_range<(0,0), (0,7)>, #ttcore.core_range<(1,0), (1,7)>, #ttcore.core_range<(2,0), (2,7)>]
func.func @decode_mm(%x: tensor<32x256xbf16>, %w1: tensor<256x512xbf16>, %w2: tensor<512x256xbf16>, %w3: tensor<256x256xbf16>) -> tensor<32x256xbf16> {
  // expected-remark @+1 {{spatial group 0: generics #0 #1 #2, 2 edges, 24 cores, 9 CBs, 6 semaphores per core at most}}
  %0 = "ttir.matmul"(%x, %w1) : (tensor<32x256xbf16>, tensor<256x512xbf16>) -> tensor<32x512xbf16>
  %1 = "ttir.matmul"(%0, %w2) : (tensor<32x512xbf16>, tensor<512x256xbf16>) -> tensor<32x256xbf16>
  %2 = "ttir.matmul"(%1, %w3) : (tensor<32x256xbf16>, tensor<256x256xbf16>) -> tensor<32x256xbf16>
  return %2 : tensor<32x256xbf16>
}

// -----

// Decode gated MLP: six 1x8 rows (rows 0-5).
// CHECK-LABEL: func.func @decode_glu
// CHECK: d2m.spatial {{.*}}grid_ranges = [#ttcore.core_range<(0,0), (0,7)>, #ttcore.core_range<(1,0), (1,7)>, #ttcore.core_range<(2,0), (2,7)>, #ttcore.core_range<(3,0), (3,7)>, #ttcore.core_range<(4,0), (4,7)>, #ttcore.core_range<(5,0), (5,7)>]
func.func @decode_glu(%x: tensor<32x256xbf16>, %wg: tensor<256x512xbf16>, %wu: tensor<256x512xbf16>, %wd: tensor<512x256xbf16>) -> tensor<32x256xbf16> {
  // expected-remark @+1 {{spatial group 0: generics #0 #1 #2 #3 #4 #5, 6 edges, 48 cores, 17 CBs, 5 semaphores per core at most}}
  %g = "ttir.matmul"(%x, %wg) : (tensor<32x256xbf16>, tensor<256x512xbf16>) -> tensor<32x512xbf16>
  %u = "ttir.matmul"(%x, %wu) : (tensor<32x256xbf16>, tensor<256x512xbf16>) -> tensor<32x512xbf16>
  %s = "ttir.sigmoid"(%g) : (tensor<32x512xbf16>) -> tensor<32x512xbf16>
  %a = "ttir.multiply"(%g, %s) : (tensor<32x512xbf16>, tensor<32x512xbf16>) -> tensor<32x512xbf16>
  %h = "ttir.multiply"(%a, %u) : (tensor<32x512xbf16>, tensor<32x512xbf16>) -> tensor<32x512xbf16>
  %o = "ttir.matmul"(%h, %wd) : (tensor<32x512xbf16>, tensor<512x256xbf16>) -> tensor<32x256xbf16>
  return %o : tensor<32x256xbf16>
}

// -----

// Attention with an unaligned length: the real mask between exp and the
// row sum cuts the group.
// CHECK-LABEL: func.func @attn_odd
// CHECK: d2m.spatial {{.*}}grid_ranges = [#ttcore.core_range<(0,0), (1,1)>, #ttcore.core_range<(0,2), (1,3)>]
// CHECK: d2m.spatial {{.*}}grid_ranges = [#ttcore.core_range<(0,0), (1,0)>, #ttcore.core_range<(0,1), (1,2)>, #ttcore.core_range<(0,3), (1,4)>]
func.func @attn_odd(%q: tensor<50x64xbf16>, %kt: tensor<64x50xbf16>, %v: tensor<50x64xbf16>) -> tensor<50x64xbf16> {
  // expected-remark @+1 {{spatial group 0: generics #0 #1, 1 edges, 8 cores, 5 CBs, 5 semaphores per core at most}}
  %0 = "ttir.matmul"(%q, %kt) : (tensor<50x64xbf16>, tensor<64x50xbf16>) -> tensor<50x50xbf16>
  // expected-remark @+3 {{generic #2 is not grouped: it has no inputs (a fill stays outside, as an input)}}
  // expected-remark @+2 {{generic #3 starts a new group: input 0 reads the group through ops that cannot be an edge}}
  // expected-remark @+1 {{spatial group 1: generics #3 #4 #5, 2 edges, 10 cores, 9 CBs, 5 semaphores per core at most}}
  %1 = "ttir.softmax"(%0) <{dimension = 1 : si32}> : (tensor<50x50xbf16>) -> tensor<50x50xbf16>
  %2 = "ttir.matmul"(%1, %v) : (tensor<50x50xbf16>, tensor<50x64xbf16>) -> tensor<50x64xbf16>
  return %2 : tensor<50x64xbf16>
}

// -----

// Transpose and reshape are views: both cut.
// CHECK-LABEL: func.func @tm
// CHECK: d2m.spatial {{.*}}grid_ranges = [#ttcore.core_range<(0,0), (3,1)>, #ttcore.core_range<(0,2), (3,3)>, #ttcore.core_range<(0,4), (3,5)>]
func.func @tm(%x: tensor<64x128xbf16>, %w: tensor<128x128xbf16>, %y: tensor<64x64xbf16>) -> tensor<64x128xbf16> {
  %0 = "ttir.matmul"(%x, %w) : (tensor<64x128xbf16>, tensor<128x128xbf16>) -> tensor<64x128xbf16>
  // expected-remark @+2 {{generic #1 starts a new group: input 0 reads the group through ops that cannot be an edge}}
  // expected-remark @+1 {{spatial group 0: generics #1 #2 #3, 2 edges, 24 cores, 7 CBs, 6 semaphores per core at most}}
  %1 = "ttir.transpose"(%0) <{dim0 = 0 : si32, dim1 = 1 : si32}> : (tensor<64x128xbf16>) -> tensor<128x64xbf16>
  %2 = "ttir.matmul"(%1, %y) : (tensor<128x64xbf16>, tensor<64x64xbf16>) -> tensor<128x64xbf16>
  %3 = "ttir.reshape"(%2) <{shape = [64 : i32, 128 : i32]}> : (tensor<128x64xbf16>) -> tensor<64x128xbf16>
  // expected-remark @+1 {{generic #4 starts a new group: input 0 reads the group through ops that cannot be an edge}}
  %4 = "ttir.relu"(%3) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
  return %4 : tensor<64x128xbf16>
}
