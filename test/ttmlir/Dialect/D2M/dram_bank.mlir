// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m="dram-args-and-results=true" %s | FileCheck %s --check-prefix=D2M
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="dram-args-and-results=true" --mlir-print-ir-after=d2m-lower-dma-to-fully-indexed-form %s -o %t.mlir 2>&1 | FileCheck %s --check-prefix=DMA
// RUN: FileCheck %s --check-prefix=METAL --input-file=%t.mlir
// RUN: sed 's/dram_bank = 5/dram_bank = 12/' %s > %t.bad.mlir
// RUN: not ttmlir-opt --ttcore-register-device --ttir-to-d2m="dram-args-and-results=true" %t.bad.mlir 2>&1 | FileCheck %s --check-prefix=ERR

// d2m.dram_bank on a function argument: the argument's DRAM copy places its
// shard i in bank (5 + i) mod 12 instead of i mod 12.

// The attribute moves to the DRAM buffer the argument is copied into.
// D2M-LABEL: func.func @mm
// D2M: %[[W:.*]] = d2m.empty() {d2m.dram_bank = 5 : i64} : tensor<1x1x16x4x!ttcore.tile<32x32, bf16>
// D2M: d2m.to_layout %arg1, %[[W]]

// Kernels address the weight's shards with the bank rotated by 5 (the tilize
// that writes the weight and the matmul that reads it); the address inside
// the bank does not change.
// DMA: IR Dump After D2MLowerDMAToFullyIndexedForm
// DMA: affine.apply affine_map<{{.*}}mod 12 + 5) floordiv 12) * 12 + 5)>
// DMA: affine.apply affine_map<{{.*}}mod 12 + 5) floordiv 12) * 12 + 5)>

// The host side reads the attribute from the buffer (flatbuffer distribution
// cores start at bank 5).
// METAL: "ttmetal.create_buffer"() <{address = {{[0-9]+}} : i64}> {d2m.dram_bank = 5 : i64} : () -> memref<{{.*}}#dram>

// ERR: error: 'func.func' op argument 1: d2m.dram_bank must be a DRAM bank index below 12
func.func @mm(%x: tensor<32x512xbf16>, %w: tensor<512x128xbf16> {d2m.dram_bank = 5 : i64}) -> tensor<32x128xbf16> {
  %0 = "ttir.matmul"(%x, %w) : (tensor<32x512xbf16>, tensor<512x128xbf16>) -> tensor<32x128xbf16>
  return %0 : tensor<32x128xbf16>
}
