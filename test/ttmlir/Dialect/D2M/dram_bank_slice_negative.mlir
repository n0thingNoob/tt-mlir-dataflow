// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m="dram-args-and-results=true" --split-input-file --verify-diagnostics %s
// RUN: not ttmlir-opt --ttcore-register-device --ttir-to-d2m %s --split-input-file 2>&1 | FileCheck %s --check-prefix=NO-DRAM

// NO-DRAM: error: 'ttir.slice_static' op d2m.dram_bank needs dram-args-and-results and a tile-aligned unit-step 2-D slice of a function argument

func.func @bank_out_of_range(%w: tensor<128x64xbf16>) -> tensor<64x64xbf16> {
  // expected-error @+1 {{'ttir.slice_static' op d2m.dram_bank must be a DRAM bank index below 12}}
  %0 = "ttir.slice_static"(%w) <{begins = [0 : i32, 0 : i32], ends = [64 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> {d2m.dram_bank = 12 : i64} : (tensor<128x64xbf16>) -> tensor<64x64xbf16>
  return %0 : tensor<64x64xbf16>
}

// -----

func.func @not_tile_aligned(%w: tensor<128x64xbf16>) -> tensor<48x64xbf16> {
  // expected-error @+1 {{'ttir.slice_static' op d2m.dram_bank needs dram-args-and-results and a tile-aligned unit-step 2-D slice of a function argument}}
  %0 = "ttir.slice_static"(%w) <{begins = [16 : i32, 0 : i32], ends = [64 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> {d2m.dram_bank = 0 : i64} : (tensor<128x64xbf16>) -> tensor<48x64xbf16>
  return %0 : tensor<48x64xbf16>
}

// -----

func.func @not_an_argument(%a: tensor<128x64xbf16>, %b: tensor<128x64xbf16>) -> tensor<64x64xbf16> {
  %0 = "ttir.add"(%a, %b) : (tensor<128x64xbf16>, tensor<128x64xbf16>) -> tensor<128x64xbf16>
  // expected-error @+1 {{'ttir.slice_static' op d2m.dram_bank needs dram-args-and-results and a tile-aligned unit-step 2-D slice of a function argument}}
  %1 = "ttir.slice_static"(%0) <{begins = [0 : i32, 0 : i32], ends = [64 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> {d2m.dram_bank = 0 : i64} : (tensor<128x64xbf16>) -> tensor<64x64xbf16>
  return %1 : tensor<64x64xbf16>
}
