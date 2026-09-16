// RUN: ttmlir-opt %s --ttcore-register-device | sed 's/dram_unreserved_end = [0-9]*/dram_unreserved_end = 4096/' > %t.mlir
// RUN: not ttmlir-opt %t.mlir --d2m-allocate='emit-resource-report=true' --mlir-print-ir-after-failure 2>&1 | FileCheck %s

// The registered fixture reserves the first 1024 bytes. A physical channel
// larger than the usable interval must not make its reserved tail allocatable.
// CHECK: required DRAM memory usage 65536 exceeds memory capacity 3072
// CHECK: dram_capacity_bytes = 3072 : i64
// CHECK-SAME: status = "dram_capacity_exceeded"
func.func @reserved_tail() -> memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.memory_space<dram>> {
  %a = memref.alloc() : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.memory_space<dram>>
  return %a : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.memory_space<dram>>
}
