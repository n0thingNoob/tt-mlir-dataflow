// RUN: ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-fuse-elementwise=true spatial-policy=bank spatial-remarks=true" --mlir-print-ir-after=d2m-materialize-spatial-groups --mlir-print-local-scope --mlir-disable-threading %s -o /dev/null 2>&1 | FileCheck %s
// RUN: ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-fuse-elementwise=true spatial-policy=bank-load spatial-remarks=true" --mlir-print-ir-after=d2m-materialize-spatial-groups --mlir-print-local-scope --mlir-disable-threading %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=LOAD
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-fuse-elementwise=true spatial-policy=bank-load spatial-stream-policy=auto" %s | FileCheck %s --check-prefix=METAL
// RUN: not ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-policy=bank spatial-groups=0,0,0,0,0,0" %s 2>&1 | FileCheck %s --check-prefix=ERR-GROUPS
// RUN: not ttmlir-opt --d2m-fe-pipeline="execution-strategy=spatial spatial-policy=nearest" %s 2>&1 | FileCheck %s --check-prefix=ERR-POLICY

// A gated MLP split over two DRAM banks (d2m.dram_bank on the weights): per
// bank j, gate #4j and up #4j+1 read Wg_j / Wu_j, silu(g) * u #4j+2 (fused)
// reads both, down #4j+3 reads Wd_j; #8 sums the two partial results. The
// mock system desc puts the worker core of bank 0 at (7,3) and of bank 1 at
// (0,0).
//
// policy=bank: the clusters are {#0..#3} (bank 0) and {#4..#7} (bank 1);
// #8 reads both, so it is in none. Round robin, generics that read the bank
// first: gate, up, down, then silu * u. Each takes the free core nearest its
// bank's core (row-major on ties):
//   bank 0: gate (7,3), up (6,3), down (7,2), glu (7,4)
//   bank 1: gate (0,0), up (0,1), down (1,0), glu (0,2)
// #8 takes the free core with the shortest total distance to (7,2) and (1,0)
// (8 hops: rows 1-7, columns 0-2), the first being (1,1).
// CHECK: remark: policy=bank gives groups=0,0,7,3,7,3,0,1,6,3,6,3,0,2,7,4,7,4,0,3,7,2,7,2,0,4,0,0,0,0,0,5,0,1,0,1,0,6,0,2,0,2,0,7,1,0,1,0,0,8,1,1,1,1 loads={{$}}
// CHECK: remark: generic #8 is in bank-placed group 0
// CHECK-LABEL: IR Dump After D2MMaterializeSpatialGroups
// CHECK: d2m.spatial {edges = [{{.*}}], grid_ranges = [#ttcore.core_range<(7,3), (7,3)>, #ttcore.core_range<(6,3), (6,3)>, #ttcore.core_range<(7,4), (7,4)>, #ttcore.core_range<(7,2), (7,2)>, #ttcore.core_range<(0,0), (0,0)>, #ttcore.core_range<(0,1), (0,1)>, #ttcore.core_range<(0,2), (0,2)>, #ttcore.core_range<(1,0), (1,0)>, #ttcore.core_range<(1,1), (1,1)>]}

// policy=bank-load: each bank first gets a load region on its own core that
// copies Wg_j and Wu_j (input 1 of the gate and of the up). Wd_j has another
// type, so the down keeps reading it from DRAM.
//   bank 0: load (7,3), gate (6,3), up (7,2), down (7,4), glu (5,3)
//   bank 1: load (0,0), gate (0,1), up (1,0), down (0,2), glu (1,1)
// #8: total distance to (7,4) and (0,2) is at least 9 (columns 2-4), first
// free at (0,3).
// LOAD: remark: input 1 of generic #3 stays a direct read of DRAM bank 0: its type differs from the bank's load copies
// LOAD: remark: input 1 of generic #7 stays a direct read of DRAM bank 1: its type differs from the bank's load copies
// LOAD: remark: policy=bank-load gives groups=0,0,6,3,6,3,0,1,7,2,7,2,0,2,5,3,5,3,0,3,7,4,7,4,0,4,0,1,0,1,0,5,1,0,1,0,0,6,1,1,1,1,0,7,0,2,0,2,0,8,0,3,0,3 loads=0,1,7,3,7,3,1,1,7,3,7,3,4,1,0,0,0,0,5,1,0,0,0,0
// LOAD: remark: load region on #ttcore.core_range<(7,3), (7,3)> for 2 inputs
// LOAD: remark: load region on #ttcore.core_range<(0,0), (0,0)> for 2 inputs
// LOAD-LABEL: IR Dump After D2MMaterializeSpatialGroups
// LOAD: d2m.spatial {edges = [{{.*}}], grid_ranges = [#ttcore.core_range<(7,3), (7,3)>, #ttcore.core_range<(6,3), (6,3)>, #ttcore.core_range<(7,2), (7,2)>, #ttcore.core_range<(5,3), (5,3)>, #ttcore.core_range<(7,4), (7,4)>, #ttcore.core_range<(0,0), (0,0)>, #ttcore.core_range<(0,1), (0,1)>, #ttcore.core_range<(1,0), (1,0)>, #ttcore.core_range<(1,1), (1,1)>, #ttcore.core_range<(0,2), (0,2)>, #ttcore.core_range<(0,3), (0,3)>]}

// The eleven regions run in one program: the first (the load on (7,3)) and
// the last (the sum on (0,3)) are in the same enqueue_program.
// METAL: "ttmetal.enqueue_program"{{.*}}#ttmetal.core_range<7x3, 1x1>{{.*}}#ttmetal.core_range<0x3, 1x1>

// ERR-GROUPS: error: policy=bank cannot be combined with groups or loads
// ERR-POLICY: error: policy must be none, greedy, bank or bank-load, got nearest
func.func @glu2(%x: tensor<32x256xbf16>,
                %wg0: tensor<256x128xbf16> {d2m.dram_bank = 0 : i64},
                %wu0: tensor<256x128xbf16> {d2m.dram_bank = 0 : i64},
                %wd0: tensor<128x256xbf16> {d2m.dram_bank = 0 : i64},
                %wg1: tensor<256x128xbf16> {d2m.dram_bank = 1 : i64},
                %wu1: tensor<256x128xbf16> {d2m.dram_bank = 1 : i64},
                %wd1: tensor<128x256xbf16> {d2m.dram_bank = 1 : i64}) -> tensor<32x256xbf16> {
  %g0 = "ttir.matmul"(%x, %wg0) : (tensor<32x256xbf16>, tensor<256x128xbf16>) -> tensor<32x128xbf16>
  %u0 = "ttir.matmul"(%x, %wu0) : (tensor<32x256xbf16>, tensor<256x128xbf16>) -> tensor<32x128xbf16>
  %s0 = "ttir.silu"(%g0) : (tensor<32x128xbf16>) -> tensor<32x128xbf16>
  %h0 = "ttir.multiply"(%s0, %u0) : (tensor<32x128xbf16>, tensor<32x128xbf16>) -> tensor<32x128xbf16>
  %d0 = "ttir.matmul"(%h0, %wd0) : (tensor<32x128xbf16>, tensor<128x256xbf16>) -> tensor<32x256xbf16>
  %g1 = "ttir.matmul"(%x, %wg1) : (tensor<32x256xbf16>, tensor<256x128xbf16>) -> tensor<32x128xbf16>
  %u1 = "ttir.matmul"(%x, %wu1) : (tensor<32x256xbf16>, tensor<256x128xbf16>) -> tensor<32x128xbf16>
  %s1 = "ttir.silu"(%g1) : (tensor<32x128xbf16>) -> tensor<32x128xbf16>
  %h1 = "ttir.multiply"(%s1, %u1) : (tensor<32x128xbf16>, tensor<32x128xbf16>) -> tensor<32x128xbf16>
  %d1 = "ttir.matmul"(%h1, %wd1) : (tensor<32x128xbf16>, tensor<128x256xbf16>) -> tensor<32x256xbf16>
  %y = "ttir.add"(%d0, %d1) : (tensor<32x256xbf16>, tensor<32x256xbf16>) -> tensor<32x256xbf16>
  return %y : tensor<32x256xbf16>
}
