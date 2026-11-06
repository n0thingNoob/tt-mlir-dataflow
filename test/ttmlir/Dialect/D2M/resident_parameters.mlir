// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="dram-args-and-results=true resident-parameters=true" --mlir-print-ir-after=d2m-hoist-resident-parameters --mlir-print-local-scope %s -o %t.mlir 2>&1 | FileCheck %s --check-prefix=HOIST
// RUN: FileCheck %s --check-prefix=METAL --input-file=%t.mlir
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="dram-args-and-results=true" --mlir-print-ir-before=d2m-lower-to-layout %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=OFF
// RUN: sed 's/argument_type<parameter>/argument_type<input>/' %s > %t.input.mlir
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="dram-args-and-results=true resident-parameters=true" --mlir-print-ir-after=d2m-hoist-resident-parameters %t.input.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=INPUTS
// RUN: not ttmlir-opt --ttir-to-ttmetal-pipeline="resident-parameters=true" %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

// resident-parameters: the weights (ttcore.argument_type = parameter) are
// written and tilized once by @glu_load, which returns the tiled DRAM buffers;
// @glu takes them as trailing arguments and only writes x per call. %wg is
// read by two matmuls and x by two: each gets one to_layout.

// HOIST-LABEL: IR Dump After D2MHoistResidentParameters
// HOIST: func.func @glu_load(%arg0: tensor<128x64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg1: tensor<64x128xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> (tensor<4x2x1x1x!ttcore.tile<32x32, bf16>, {{.*}}dram, sharded>>, tensor<2x4x1x1x!ttcore.tile<32x32, bf16>, {{.*}}dram, sharded>>)
// HOIST-SAME: d2m.resident_load, d2m.resident_params = [1, 2]
// HOIST-COUNT-2: d2m.to_layout %arg
// HOIST-NOT: d2m.to_layout
// HOIST: return
// HOIST: func.func @glu(%arg0: tensor<32x64xbf16>, %arg1: tensor<4x2x1x1x!ttcore.tile<32x32, bf16>, {{.*}}> {d2m.resident_of = 1 : i64}, %arg2: tensor<2x4x1x1x!ttcore.tile<32x32, bf16>, {{.*}}> {d2m.resident_of = 2 : i64})
// HOIST-SAME: d2m.resident_run, d2m.run_inputs = [0]
// HOIST: d2m.to_layout %arg0,
// HOIST-NOT: d2m.to_layout %arg

// One write per weight in the load program, one write (x) per call in @glu.
// METAL-LABEL: func.func @glu_load
// METAL-SAME: d2m.resident_end
// METAL-COUNT-2: ttmetal.enqueue_write_buffer
// METAL-NOT: ttmetal.enqueue_write_buffer
// METAL: return
// METAL-LABEL: func.func @glu(
// METAL-COUNT-1: ttmetal.enqueue_write_buffer
// METAL-NOT: ttmetal.enqueue_write_buffer
// METAL: return

// Without the option nothing changes: x and %wg are still laid out per use.
// OFF-LABEL: IR Dump Before D2MLowerToLayout
// OFF-NOT: glu_load
// OFF-COUNT-2: d2m.to_layout %arg0,

// No parameters: no load function, the to_layouts of x and of %wg are merged.
// INPUTS-LABEL: IR Dump After D2MHoistResidentParameters
// INPUTS-NOT: glu_load
// INPUTS-COUNT-1: d2m.to_layout %arg0,
// INPUTS-NOT: d2m.to_layout %arg0,

// ERR: error: 'func.func' op resident parameters must be laid out in DRAM (dram-args-and-results)
#param = #ttcore.argument_type<parameter>
func.func @glu(%x: tensor<32x64xbf16>, %wg: tensor<128x64xbf16> {ttcore.argument_type = #param}, %wd: tensor<64x128xbf16> {ttcore.argument_type = #param}) -> tensor<32x64xbf16> {
  %g = "ttir.matmul"(%x, %wg) <{transpose_a = false, transpose_b = true}> : (tensor<32x64xbf16>, tensor<128x64xbf16>) -> tensor<32x128xbf16>
  %u = "ttir.matmul"(%x, %wg) <{transpose_a = false, transpose_b = true}> : (tensor<32x64xbf16>, tensor<128x64xbf16>) -> tensor<32x128xbf16>
  %h = "ttir.multiply"(%g, %u) : (tensor<32x128xbf16>, tensor<32x128xbf16>) -> tensor<32x128xbf16>
  %y = "ttir.matmul"(%h, %wd) <{transpose_a = false, transpose_b = true}> : (tensor<32x128xbf16>, tensor<64x128xbf16>) -> tensor<32x64xbf16>
  return %y : tensor<32x64xbf16>
}
