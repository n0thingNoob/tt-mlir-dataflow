// RUN: ttmlir-opt --ttcore-register-device --d2m-split-by-dram-bank="dram-args-and-results=true remarks=true" --verify-diagnostics --split-input-file %s | FileCheck %s
// RUN: ttmlir-opt --ttcore-register-device --d2m-split-by-dram-bank="remarks=true" %s --split-input-file 2>&1 >/dev/null | FileCheck %s --check-prefix=NO-DRAM

// d2m-split-by-dram-bank: a gated MLP (matmul -> elementwise -> matmul with
// argument weights) split over the 12 DRAM banks of the mock system desc.

// NO-DRAM-COUNT-6: remark: bank split: nothing split, the arguments are not in DRAM (dram-args-and-results is not set)

// F = 12 tiles: one tile per bank. Gate and up are split along N (rows of
// their [F, H] weights), down along K (columns of its [H, F] weight); the 12
// partial results are summed by a chain of adds.
// CHECK-LABEL: func.func @even
// CHECK: %[[G0:.*]] = "ttir.slice_static"(%arg1) <{{.}}begins = [0 : i32, 0 : i32], ends = [32 : i32, 64 : i32], step = [1 : i32, 1 : i32]{{.}}> {{.}}d2m.dram_bank = 0 : i64{{.}} : (tensor<384x64xbf16>) -> tensor<32x64xbf16>
// CHECK: "ttir.matmul"(%arg0, %[[G0]]) <{{.}}transpose_a = false, transpose_b = true{{.}}> : (tensor<32x64xbf16>, tensor<32x64xbf16>) -> tensor<32x32xbf16>
// CHECK: "ttir.silu"
// CHECK: %[[U0:.*]] = "ttir.slice_static"(%arg2) <{{.}}begins = [0 : i32, 0 : i32], ends = [32 : i32, 64 : i32]{{.*}}{{.}}d2m.dram_bank = 0 : i64{{.}}
// CHECK: "ttir.matmul"(%arg0, %[[U0]])
// CHECK: "ttir.multiply"
// CHECK: %[[D0:.*]] = "ttir.slice_static"(%arg3) <{{.}}begins = [0 : i32, 0 : i32], ends = [64 : i32, 32 : i32], step = [1 : i32, 1 : i32]{{.}}> {{.}}d2m.dram_bank = 0 : i64{{.}} : (tensor<64x384xbf16>) -> tensor<64x32xbf16>
// CHECK: %[[Y0:.*]] = "ttir.matmul"(%{{.*}}, %[[D0]]) <{{.}}transpose_a = false, transpose_b = true{{.}}> : (tensor<32x32xbf16>, tensor<64x32xbf16>) -> tensor<32x64xbf16>
// CHECK: "ttir.slice_static"(%arg1) <{{.}}begins = [32 : i32, 0 : i32], ends = [64 : i32, 64 : i32]{{.*}}{{.}}d2m.dram_bank = 1 : i64{{.}}
// CHECK: "ttir.slice_static"(%arg3) <{{.}}begins = [0 : i32, 352 : i32], ends = [64 : i32, 384 : i32]{{.*}}{{.}}d2m.dram_bank = 11 : i64{{.}}
// CHECK: %[[Y11:.*]] = "ttir.matmul"
// CHECK: %[[A1:.*]] = "ttir.add"(%[[Y0]], %{{.*}})
// CHECK-COUNT-9: "ttir.add"
// CHECK: %[[A11:.*]] = "ttir.add"(%{{.*}}, %[[Y11]])
// CHECK-NEXT: return %[[A11]]

func.func @even(%x: tensor<32x64xbf16>, %wg: tensor<384x64xbf16>, %wu: tensor<384x64xbf16>, %wd: tensor<64x384xbf16>) -> tensor<32x64xbf16> {
  // expected-remark @+1 {{bank split: 4 ops and the matmul that ends them split over 12 DRAM banks, F = 12 tiles: 1 per bank}}
  %g = "ttir.matmul"(%x, %wg) <{transpose_a = false, transpose_b = true}> : (tensor<32x64xbf16>, tensor<384x64xbf16>) -> tensor<32x384xbf16>
  %s = "ttir.silu"(%g) : (tensor<32x384xbf16>) -> tensor<32x384xbf16>
  %u = "ttir.matmul"(%x, %wu) <{transpose_a = false, transpose_b = true}> : (tensor<32x64xbf16>, tensor<384x64xbf16>) -> tensor<32x384xbf16>
  %h = "ttir.multiply"(%s, %u) : (tensor<32x384xbf16>, tensor<32x384xbf16>) -> tensor<32x384xbf16>
  %y = "ttir.matmul"(%h, %wd) <{transpose_a = false, transpose_b = true}> : (tensor<32x384xbf16>, tensor<64x384xbf16>) -> tensor<32x64xbf16>
  return %y : tensor<32x64xbf16>
}
// -----

// F = 13 tiles: bank 0 gets 2 tiles, banks 1-11 one each (no padding).
// CHECK-LABEL: func.func @uneven
// CHECK: "ttir.slice_static"(%arg1) <{{.}}begins = [0 : i32, 0 : i32], ends = [64 : i32, 64 : i32]{{.*}}{{.}}d2m.dram_bank = 0 : i64{{.}} : (tensor<416x64xbf16>) -> tensor<64x64xbf16>
// CHECK: "ttir.slice_static"(%arg3) <{{.}}begins = [0 : i32, 0 : i32], ends = [64 : i32, 64 : i32]{{.*}}{{.}}d2m.dram_bank = 0 : i64{{.}} : (tensor<64x416xbf16>) -> tensor<64x64xbf16>
// CHECK: "ttir.slice_static"(%arg1) <{{.}}begins = [64 : i32, 0 : i32], ends = [96 : i32, 64 : i32]{{.*}}{{.}}d2m.dram_bank = 1 : i64{{.}} : (tensor<416x64xbf16>) -> tensor<32x64xbf16>
// CHECK: "ttir.slice_static"(%arg3) <{{.}}begins = [0 : i32, 384 : i32], ends = [64 : i32, 416 : i32]{{.*}}{{.}}d2m.dram_bank = 11 : i64{{.}} : (tensor<64x416xbf16>) -> tensor<64x32xbf16>

func.func @uneven(%x: tensor<32x64xbf16>, %wg: tensor<416x64xbf16>, %wu: tensor<416x64xbf16>, %wd: tensor<64x416xbf16>) -> tensor<32x64xbf16> {
  // expected-remark @+1 {{bank split: 4 ops and the matmul that ends them split over 12 DRAM banks, F = 13 tiles: 2 on bank 0, 1 on banks 1-11}}
  %g = "ttir.matmul"(%x, %wg) <{transpose_a = false, transpose_b = true}> : (tensor<32x64xbf16>, tensor<416x64xbf16>) -> tensor<32x416xbf16>
  %s = "ttir.silu"(%g) : (tensor<32x416xbf16>) -> tensor<32x416xbf16>
  %u = "ttir.matmul"(%x, %wu) <{transpose_a = false, transpose_b = true}> : (tensor<32x64xbf16>, tensor<416x64xbf16>) -> tensor<32x416xbf16>
  %h = "ttir.multiply"(%s, %u) : (tensor<32x416xbf16>, tensor<32x416xbf16>) -> tensor<32x416xbf16>
  %y = "ttir.matmul"(%h, %wd) <{transpose_a = false, transpose_b = true}> : (tensor<32x416xbf16>, tensor<64x416xbf16>) -> tensor<32x64xbf16>
  return %y : tensor<32x64xbf16>
}

// -----

// An [M, F] value of the chain is returned: splitting would need a concat.
// CHECK-LABEL: func.func @leak
// CHECK-NOT: ttir.slice_static
// CHECK: return
func.func @leak(%x: tensor<32x64xbf16>, %wg: tensor<384x64xbf16>, %wu: tensor<384x64xbf16>, %wd: tensor<64x384xbf16>) -> tensor<32x384xbf16> {
  // expected-remark @+1 {{bank split: matmul not split: its result reaches func.return, which is not an elementwise op of the chain or a matmul that contracts it over F with an argument}}
  %g = "ttir.matmul"(%x, %wg) <{transpose_a = false, transpose_b = true}> : (tensor<32x64xbf16>, tensor<384x64xbf16>) -> tensor<32x384xbf16>
  %s = "ttir.silu"(%g) : (tensor<32x384xbf16>) -> tensor<32x384xbf16>
  %u = "ttir.matmul"(%x, %wu) <{transpose_a = false, transpose_b = true}> : (tensor<32x64xbf16>, tensor<384x64xbf16>) -> tensor<32x384xbf16>
  %h = "ttir.multiply"(%s, %u) : (tensor<32x384xbf16>, tensor<32x384xbf16>) -> tensor<32x384xbf16>
  %y_unused = "ttir.matmul"(%h, %wd) <{transpose_a = false, transpose_b = true}> : (tensor<32x384xbf16>, tensor<64x384xbf16>) -> tensor<32x64xbf16>
  return %h : tensor<32x384xbf16>
}

// -----

// F = 11 tiles: fewer tiles than banks.
// CHECK-LABEL: func.func @small
// CHECK-NOT: ttir.slice_static
// CHECK: return
func.func @small(%x: tensor<32x64xbf16>, %wg: tensor<352x64xbf16>, %wu: tensor<352x64xbf16>, %wd: tensor<64x352xbf16>) -> tensor<32x64xbf16> {
  // expected-remark @+1 {{bank split: matmul not split: F = 352 is 11 tiles, fewer than the 12 banks}}
  %g = "ttir.matmul"(%x, %wg) <{transpose_a = false, transpose_b = true}> : (tensor<32x64xbf16>, tensor<352x64xbf16>) -> tensor<32x352xbf16>
  %s = "ttir.silu"(%g) : (tensor<32x352xbf16>) -> tensor<32x352xbf16>
  %u = "ttir.matmul"(%x, %wu) <{transpose_a = false, transpose_b = true}> : (tensor<32x64xbf16>, tensor<352x64xbf16>) -> tensor<32x352xbf16>
  %h = "ttir.multiply"(%s, %u) : (tensor<32x352xbf16>, tensor<32x352xbf16>) -> tensor<32x352xbf16>
  %y = "ttir.matmul"(%h, %wd) <{transpose_a = false, transpose_b = true}> : (tensor<32x352xbf16>, tensor<64x352xbf16>) -> tensor<32x64xbf16>
  return %y : tensor<32x64xbf16>
}

// -----

// Arguments that already carry d2m.dram_bank: split by hand, left alone.
// CHECK-LABEL: func.func @banked
// CHECK-NOT: ttir.slice_static
// CHECK: return
// expected-remark @+1 {{bank split: nothing split, arguments already carry d2m.dram_bank}}
func.func @banked(%x: tensor<32x64xbf16>, %wg: tensor<384x64xbf16> {d2m.dram_bank = 3 : i64}, %wu: tensor<384x64xbf16>, %wd: tensor<64x384xbf16>) -> tensor<32x64xbf16> {
  %g = "ttir.matmul"(%x, %wg) <{transpose_a = false, transpose_b = true}> : (tensor<32x64xbf16>, tensor<384x64xbf16>) -> tensor<32x384xbf16>
  %s = "ttir.silu"(%g) : (tensor<32x384xbf16>) -> tensor<32x384xbf16>
  %u = "ttir.matmul"(%x, %wu) <{transpose_a = false, transpose_b = true}> : (tensor<32x64xbf16>, tensor<384x64xbf16>) -> tensor<32x384xbf16>
  %h = "ttir.multiply"(%s, %u) : (tensor<32x384xbf16>, tensor<32x384xbf16>) -> tensor<32x384xbf16>
  %y = "ttir.matmul"(%h, %wd) <{transpose_a = false, transpose_b = true}> : (tensor<32x384xbf16>, tensor<64x384xbf16>) -> tensor<32x64xbf16>
  return %y : tensor<32x64xbf16>
}

// -----

// Weights stored [K, N] (transpose_b = false) and no elementwise op between
// the matmuls: the first weight is split along its columns, the second along
// its rows.
// CHECK-LABEL: func.func @no_transpose
// CHECK: "ttir.slice_static"(%arg1) <{{.}}begins = [0 : i32, 0 : i32], ends = [64 : i32, 32 : i32]{{.*}}d2m.dram_bank = 0 : i64{{.}} : (tensor<64x384xbf16>) -> tensor<64x32xbf16>
// CHECK: "ttir.slice_static"(%arg2) <{{.}}begins = [0 : i32, 0 : i32], ends = [32 : i32, 64 : i32]{{.*}}d2m.dram_bank = 0 : i64{{.}} : (tensor<384x64xbf16>) -> tensor<32x64xbf16>
// CHECK: "ttir.slice_static"(%arg2) <{{.}}begins = [352 : i32, 0 : i32], ends = [384 : i32, 64 : i32]{{.*}}d2m.dram_bank = 11 : i64{{.}}
// CHECK-COUNT-11: "ttir.add"
// CHECK-NEXT: return
func.func @no_transpose(%x: tensor<32x64xbf16>, %w1: tensor<64x384xbf16>, %w2: tensor<384x64xbf16>) -> tensor<32x64xbf16> {
  // expected-remark @+1 {{bank split: 1 op and the matmul that ends them split over 12 DRAM banks, F = 12 tiles: 1 per bank}}
  %h = "ttir.matmul"(%x, %w1) <{transpose_a = false, transpose_b = false}> : (tensor<32x64xbf16>, tensor<64x384xbf16>) -> tensor<32x384xbf16>
  %y = "ttir.matmul"(%h, %w2) <{transpose_a = false, transpose_b = false}> : (tensor<32x384xbf16>, tensor<384x64xbf16>) -> tensor<32x64xbf16>
  return %y : tensor<32x64xbf16>
}
