// RUN: ttmlir-opt --ttcore-register-device --d2m-allocate %s | FileCheck %s
// RUN: ttmlir-opt --ttcore-register-device --d2m-allocate="resident-dram-base=1048576" %s | FileCheck %s --check-prefix=BASE

// Resident parameters: @load writes the parameters and returns them as DRAM
// buffers, @run takes them as arguments. Allocate handles @load first, even
// though it comes second, and starts @run's DRAM buffers at the end of the
// buffers @load returns (its freed staging buffer does not count), so the run
// function never overwrites a resident parameter. A function without the
// attributes is allocated as before, from the unreserved DRAM base.

!dram_t = memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #ttcore.memory_space<dram>>

// CHECK-LABEL: func.func @run
// CHECK-SAME: d2m.dram_end = 787456 : i64
// CHECK: memref.alloc() {address = 525312 : i64
// BASE-LABEL: func.func @run
// BASE: memref.alloc() {address = 1572864 : i64
func.func @run(%w: !dram_t) -> !dram_t attributes {d2m.resident_run} {
  %out = memref.alloc() : !dram_t
  return %out : !dram_t
}

// CHECK-LABEL: func.func @load
// CHECK-SAME: d2m.resident_end = 525312 : i64
// CHECK: memref.alloc() {address = 1024 : i64
// CHECK: memref.alloc() {address = 263168 : i64
// BASE-LABEL: func.func @load
// BASE-SAME: d2m.resident_end = 1572864 : i64
// BASE: memref.alloc() {address = 1048576 : i64
func.func @load() -> !dram_t attributes {d2m.resident_load} {
  %staging = memref.alloc() : !dram_t
  %w = memref.alloc() : !dram_t
  memref.dealloc %staging : !dram_t
  return %w : !dram_t
}

// CHECK-LABEL: func.func @plain
// CHECK-NOT: d2m.dram_end
// CHECK: memref.alloc() {address = 1024 : i64
// BASE-LABEL: func.func @plain
// BASE: memref.alloc() {address = 1024 : i64
func.func @plain() -> !dram_t {
  %out = memref.alloc() : !dram_t
  return %out : !dram_t
}
