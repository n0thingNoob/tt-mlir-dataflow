// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-fuse-elementwise=true spatial-policy=bank resident-parameters=true" --mlir-print-ir-after=d2m-hoist-resident-parameters --mlir-print-local-scope %s -o %t.mlir 2>&1 | FileCheck %s --check-prefix=HOIST
// RUN: FileCheck %s --check-prefix=METAL --input-file=%t.mlir

// Resident parameters with d2m.dram_bank slices (the form d2m-split-by-dram-bank
// emits): @glu_load writes each weight once into its staging buffer and tilizes
// each slice into its bank; @glu takes the four tiled slices as arguments that
// carry their bank, so the kernels address them in that bank. Per call only x
// is written, once for both matmuls that read it.

// HOIST-LABEL: IR Dump After D2MHoistResidentParameters
// HOIST: func.func @glu_load(
// HOIST-SAME: d2m.resident_load, d2m.resident_params = [1, 2]
// HOIST-COUNT-2: d2m.to_layout %arg
// HOIST-COUNT-4: d2m.view_layout
// HOIST: return
// HOIST: func.func @glu(%arg0: tensor<32x64xbf16>
// HOIST-SAME: {d2m.dram_bank = 0 : i64, d2m.resident_of = 1 : i64}
// HOIST-SAME: {d2m.dram_bank = 0 : i64, d2m.resident_of = 2 : i64}
// HOIST-SAME: {d2m.dram_bank = 1 : i64, d2m.resident_of = 1 : i64}
// HOIST-SAME: {d2m.dram_bank = 1 : i64, d2m.resident_of = 2 : i64}
// HOIST-SAME: d2m.resident_run, d2m.run_inputs = [0]
// HOIST-NOT: d2m.view_layout {{.*}}d2m.dram_bank
// HOIST-COUNT-1: d2m.to_layout %arg0,
// HOIST-NOT: d2m.to_layout %arg0,

// The load program: two staging writes and one tilize per slice, nothing
// copied on return; the tiled slices are created in banks 0 and 1.
// METAL-LABEL: func.func @glu_load
// METAL-COUNT-2: ttmetal.enqueue_write_buffer
// METAL-NOT: ttmetal.enqueue_write_buffer
// METAL-COUNT-4: ttmetal.enqueue_program
// METAL-NOT: ttmetal.enqueue_program
// METAL: return
// METAL-LABEL: func.func @glu(
// METAL-COUNT-1: ttmetal.enqueue_write_buffer
// METAL-NOT: ttmetal.enqueue_write_buffer
#param = #ttcore.argument_type<parameter>
func.func @glu(%x: tensor<32x64xbf16>, %wg: tensor<128x64xbf16> {ttcore.argument_type = #param}, %wd: tensor<64x128xbf16> {ttcore.argument_type = #param}) -> tensor<32x64xbf16> {
  %sg0 = "ttir.slice_static"(%wg) <{begins = [0 : i32, 0 : i32], ends = [64 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> {d2m.dram_bank = 0 : i64} : (tensor<128x64xbf16>) -> tensor<64x64xbf16>
  %sg1 = "ttir.slice_static"(%wg) <{begins = [64 : i32, 0 : i32], ends = [128 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> {d2m.dram_bank = 1 : i64} : (tensor<128x64xbf16>) -> tensor<64x64xbf16>
  %sd0 = "ttir.slice_static"(%wd) <{begins = [0 : i32, 0 : i32], ends = [64 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> {d2m.dram_bank = 0 : i64} : (tensor<64x128xbf16>) -> tensor<64x64xbf16>
  %sd1 = "ttir.slice_static"(%wd) <{begins = [0 : i32, 64 : i32], ends = [64 : i32, 128 : i32], step = [1 : i32, 1 : i32]}> {d2m.dram_bank = 1 : i64} : (tensor<64x128xbf16>) -> tensor<64x64xbf16>
  %g0 = "ttir.matmul"(%x, %sg0) <{transpose_a = false, transpose_b = true}> : (tensor<32x64xbf16>, tensor<64x64xbf16>) -> tensor<32x64xbf16>
  %y0 = "ttir.matmul"(%g0, %sd0) <{transpose_a = false, transpose_b = true}> : (tensor<32x64xbf16>, tensor<64x64xbf16>) -> tensor<32x64xbf16>
  %g1 = "ttir.matmul"(%x, %sg1) <{transpose_a = false, transpose_b = true}> : (tensor<32x64xbf16>, tensor<64x64xbf16>) -> tensor<32x64xbf16>
  %y1 = "ttir.matmul"(%g1, %sd1) <{transpose_a = false, transpose_b = true}> : (tensor<32x64xbf16>, tensor<64x64xbf16>) -> tensor<32x64xbf16>
  %y = "ttir.add"(%y0, %y1) : (tensor<32x64xbf16>, tensor<32x64xbf16>) -> tensor<32x64xbf16>
  return %y : tensor<32x64xbf16>
}
