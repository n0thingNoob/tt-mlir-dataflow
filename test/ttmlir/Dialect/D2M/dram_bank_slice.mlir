// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m="dram-args-and-results=true" --mlir-print-local-scope %s | FileCheck %s --check-prefix=D2M
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-fuse-elementwise=true spatial-policy=bank" --mlir-print-ir-before=d2m-lower-to-layout --mlir-print-local-scope %s -o %t.mlir 2>&1 | FileCheck %s --check-prefix=LTL
// RUN: FileCheck %s --check-prefix=METAL --input-file=%t.mlir
// RUN: FileCheck %s --check-prefix=BUF --input-file=%t.mlir
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="execution-strategy=spatial dram-args-and-results=true spatial-policy=bank" %s | FileCheck %s --check-prefix=METAL

// d2m.dram_bank on ttir.slice_static: weight slices placed in DRAM banks
// without splitting the weight arguments. Each argument gets one row-major
// DRAM staging copy; each slice is a view of it that carries the bank, and
// the consumer tilizes the view into a buffer in that bank.

// D2M-LABEL: func.func @glu
// D2M-DAG: %[[SD:.*]] = d2m.to_layout %arg2, %{{.*}} : tensor<64x128xbf16> into tensor<1x1x64x128xbf16, {{.*}}dram, sharded>>
// D2M-DAG: %[[SG:.*]] = d2m.to_layout %arg1, %{{.*}} : tensor<128x64xbf16> into tensor<1x1x128x64xbf16, {{.*}}dram, sharded>>
// D2M-DAG: d2m.view_layout %[[SG]] remapping = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)> {d2m.dram_bank = 0 : i64}
// D2M-DAG: d2m.view_layout %[[SG]] remapping = affine_map<(d0, d1, d2, d3) -> (0, 0, d2 + 64, d3)> {d2m.dram_bank = 1 : i64}
// D2M-DAG: d2m.view_layout %[[SD]] remapping = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)> {d2m.dram_bank = 0 : i64}
// D2M-DAG: d2m.view_layout %[[SD]] remapping = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3 + 64)> {d2m.dram_bank = 1 : i64}
// D2M-NOT: d2m.to_layout %arg1,
// D2M-NOT: d2m.to_layout %arg2,
// D2M: d2m.empty() {d2m.dram_bank = 0 : i64} : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, {{.*}}dram, sharded>>
// D2M: d2m.empty() {d2m.dram_bank = 0 : i64} : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, {{.*}}dram, sharded>>
// D2M: d2m.empty() {d2m.dram_bank = 1 : i64} : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, {{.*}}dram, sharded>>
// D2M: d2m.empty() {d2m.dram_bank = 1 : i64} : tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, {{.*}}dram, sharded>>

// The view -> host -> device round trips fold away (MaterializeViewReturns
// leaves them alone): each slice is one DRAM view -> tiled DRAM to_layout.
// LTL-LABEL: IR Dump Before D2MLowerToLayout
// LTL-COUNT-4: d2m.to_layout %view{{(_[0-9]+)?}}, %{{.*}} : tensor<1x1x64x64xbf16, {{.*}}dram, sharded>> into tensor<1x1x2x2x!ttcore.tile<32x32, bf16>, {{.*}}dram, sharded>>

// One host write per weight argument (and one per x copy), not one per slice.
// METAL-LABEL: func.func @glu
// METAL-DAG: "ttmetal.enqueue_write_buffer"(%arg1, %{{.*}}) : (memref<128x64xbf16>
// METAL-DAG: "ttmetal.enqueue_write_buffer"(%arg2, %{{.*}}) : (memref<64x128xbf16>
// METAL-DAG: "ttmetal.enqueue_write_buffer"(%arg0, %{{.*}}) : (memref<32x64xbf16>
// METAL-DAG: "ttmetal.enqueue_write_buffer"(%arg0, %{{.*}}) : (memref<32x64xbf16>
// METAL-NOT: ttmetal.enqueue_write_buffer

// The tiled slices are DRAM buffers in banks 0 and 1.
// BUF-DAG: "ttmetal.create_buffer"() <{address = {{[0-9]+}} : i64}> {d2m.dram_bank = 0 : i64} : () -> memref<1x1x2x2x!ttcore.tile<32x32, bf16>, {{.*}}dram>>
// BUF-DAG: "ttmetal.create_buffer"() <{address = {{[0-9]+}} : i64}> {d2m.dram_bank = 1 : i64} : () -> memref<1x1x2x2x!ttcore.tile<32x32, bf16>, {{.*}}dram>>
func.func @glu(%x: tensor<32x64xbf16>, %wg: tensor<128x64xbf16>, %wd: tensor<64x128xbf16>) -> tensor<32x64xbf16> {
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
