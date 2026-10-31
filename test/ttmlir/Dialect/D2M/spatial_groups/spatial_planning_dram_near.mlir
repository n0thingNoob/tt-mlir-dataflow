// RUN: ttmlir-opt --split-input-file --verify-diagnostics -mlir-print-op-on-diagnostic=false --d2m-fe-pipeline="execution-strategy=spatial spatial-remarks=true default-input-memspace=dram" %s | FileCheck %s

// Inputs in DRAM, outputs in L1 (default-input-memspace=dram): every generic
// writes L1 and reads its inputs as DRAM tensors. Inside a group, a producer's
// L1 output read as the same tensor in DRAM becomes an L1 edge: the consumer
// reads the producer's L1 buffer. Inputs from outside the group (the function
// arguments here) stay DRAM. A member that reads such a DRAM input is placed
// on the free range covering the most DRAM-near cores of the system desc's
// dram_bank_to_logical_worker_noc0 table (the mock table here); the other
// members keep first fit.

// Fan-out and join. #0 (tanh of x) and #2 (matmul with y) read DRAM and move
// next to the DRAM-near cores; #1 and #3 read only edges and take the first
// free ranges.
// CHECK-LABEL: func.func @fanout
// CHECK: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>, #d2m.spatial_edge<0:0 -> 2:0>, #d2m.spatial_edge<1:0 -> 3:0>, #d2m.spatial_edge<2:0 -> 3:1>], grid_ranges = [#ttcore.core_range<(0,3), (1,4)>, #ttcore.core_range<(0,0), (1,1)>, #ttcore.core_range<(3,0), (4,1)>, #ttcore.core_range<(0,5), (1,6)>]}
// CHECK-NEXT: ins(%[[X:[^,]*]], %[[Y:[^ ]*]] : memref<{{.*}}#dram>, memref<{{.*}}#dram>)
// CHECK-NEXT: outs(%[[E0:[^,]*]], %[[E1:[^,]*]], %[[E2:[^,]*]], %{{[^ ]*}} : memref<{{.*}}#l1>, memref<{{.*}}#l1>, memref<{{.*}}#l1>, memref<{{.*}}#l1>)
// CHECK: d2m.generic
// CHECK-NEXT: ins(%[[X]] : memref<{{.*}}#dram>)
// CHECK-NEXT: outs(%[[E0]] : memref<{{.*}}#l1>)
// CHECK: }, {
// CHECK: d2m.generic
// CHECK-NEXT: ins(%[[E0]] : memref<{{.*}}#l1>)
// CHECK: }, {
// CHECK: d2m.generic
// CHECK-NEXT: ins(%[[E0]], %[[Y]] : memref<{{.*}}#l1>, memref<{{.*}}#dram>)
// CHECK: }, {
// CHECK: d2m.generic
// CHECK-NEXT: ins(%[[E1]], %[[E2]] : memref<{{.*}}#l1>, memref<{{.*}}#l1>)
func.func @fanout(%x: tensor<64x64xbf16>, %y: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  // expected-remark @+2 {{spatial group 0: generic #0 reads DRAM: region (0,3) 2x2 covers 2 of 12 DRAM-near cores (first fit (0,0) 2x2 covers 1)}}
  // expected-remark @+1 {{spatial group 0: generics #0 #1 #2 #3, 4 edges, 16 cores, 10 CBs, 6 semaphores per core at most}}
  %0 = "ttir.tanh"(%x) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
  %1 = "ttir.exp"(%0) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
  // expected-remark @+1 {{spatial group 0: generic #2 reads DRAM: region (3,0) 2x2 covers 2 of 12 DRAM-near cores (first fit (0,5) 2x2 covers 0)}}
  %2 = "ttir.matmul"(%0, %y) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
  %3 = "ttir.add"(%1, %2) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
  return %3 : tensor<64x64xbf16>
}

// -----

// The placement preference with a group that shrinks: three 8x8 generics, the
// group takes target 4x4; #0 and #2 read DRAM. The ranges do not overlap and
// GridSelection picks exactly their shapes.
// CHECK-LABEL: func.func @chain8x8
// CHECK: d2m.spatial {{.*}}grid_ranges = [#ttcore.core_range<(0,3), (3,6)>, #ttcore.core_range<(4,0), (7,3)>, #ttcore.core_range<(4,4), (7,7)>]
// CHECK: d2m.generic {{.*}}grid = #ttcore.grid<4x4{{[,>]}}
// CHECK: }, {
// CHECK: d2m.generic {{.*}}grid = #ttcore.grid<4x4{{[,>]}}
// CHECK: }, {
// CHECK: d2m.generic {{.*}}grid = #ttcore.grid<4x4{{[,>]}}
func.func @chain8x8(%x: tensor<256x256xbf16>, %y: tensor<256x256xbf16>) -> tensor<256x256xbf16> {
  // expected-remark @+3 {{spatial group 0: generic #0 reads DRAM: region (0,3) 4x4 covers 4 of 12 DRAM-near cores (first fit (0,0) 4x4 covers 2)}}
  // expected-remark @+2 {{spatial group 0 shrinks to target 4x4: #0 8x8->4x4 #1 8x8->4x4 #2 8x8->4x4}}
  // expected-remark @+1 {{spatial group 0: generics #0 #1 #2, 2 edges, 48 cores, 7 CBs, 2 semaphores per core at most}}
  %0 = "ttir.exp"(%x) : (tensor<256x256xbf16>) -> tensor<256x256xbf16>
  %1 = "ttir.tanh"(%0) : (tensor<256x256xbf16>) -> tensor<256x256xbf16>
  // expected-remark @+1 {{spatial group 0: generic #2 reads DRAM: region (4,4) 4x4 covers 4 of 12 DRAM-near cores (first fit (4,4) 4x4 covers 4)}}
  %2 = "ttir.add"(%1, %y) : (tensor<256x256xbf16>, tensor<256x256xbf16>) -> tensor<256x256xbf16>
  return %2 : tensor<256x256xbf16>
}

// -----

// Two groups cut by an unaligned mask: the connection between them still goes
// through DRAM (group 1 reads group 0's result as a DRAM input), while each
// group's own edges stay in L1.
// CHECK-LABEL: func.func @attn_odd
// CHECK: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>]
// CHECK: d2m.spatial {edges = [#d2m.spatial_edge<0:0 -> 1:0>, #d2m.spatial_edge<1:0 -> 2:0>]
// CHECK-NEXT: ins({{.*}} : memref<{{.*}}#dram>, memref<{{.*}}#dram>, memref<{{.*}}#dram>, memref<{{.*}}#dram>)
// CHECK: }, {
// CHECK: d2m.generic
// CHECK-NEXT: ins(%{{[^,]*}}, %{{[^ ]*}} : memref<{{.*}}#dram>, memref<{{.*}}#l1>)
func.func @attn_odd(%q: tensor<50x64xbf16>, %kt: tensor<64x50xbf16>, %v: tensor<50x64xbf16>) -> tensor<50x64xbf16> {
  // expected-remark @+2 {{spatial group 0: generic #0 reads DRAM: region (0,3) 2x2 covers 2 of 12 DRAM-near cores (first fit (0,0) 2x2 covers 1)}}
  // expected-remark @+1 {{spatial group 0: generics #0 #1, 1 edges, 8 cores, 5 CBs, 5 semaphores per core at most}}
  %0 = "ttir.matmul"(%q, %kt) : (tensor<50x64xbf16>, tensor<64x50xbf16>) -> tensor<50x50xbf16>
  // expected-remark @+5 {{generic #2 is not grouped: it has no inputs (a fill stays outside, as an input)}}
  // expected-remark @+4 {{generic #3 starts a new group: input 0 reads the group through ops that cannot be an edge}}
  // expected-remark @+3 {{spatial group 1: generic #3 reads DRAM: region (0,4) 2x1 covers 2 of 12 DRAM-near cores (first fit (0,0) 2x1 covers 1)}}
  // expected-remark @+2 {{spatial group 1: generic #4 reads DRAM: region (3,0) 2x2 covers 2 of 12 DRAM-near cores (first fit (0,0) 2x2 covers 1)}}
  // expected-remark @+1 {{spatial group 1: generics #3 #4 #5, 2 edges, 10 cores, 9 CBs, 5 semaphores per core at most}}
  %1 = "ttir.softmax"(%0) <{dimension = 1 : si32}> : (tensor<50x50xbf16>) -> tensor<50x50xbf16>
  // expected-remark @+1 {{spatial group 1: generic #5 reads DRAM: region (3,3) 2x2 covers 2 of 12 DRAM-near cores (first fit (0,0) 2x2 covers 1)}}
  %2 = "ttir.matmul"(%1, %v) : (tensor<50x50xbf16>, tensor<50x64xbf16>) -> tensor<50x64xbf16>
  return %2 : tensor<50x64xbf16>
}
