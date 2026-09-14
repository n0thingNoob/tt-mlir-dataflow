// RUN: ttmlir-opt %s --ttcore-register-device --d2m-allocate='emit-resource-report=true available-l1-addr-range=1024,132096' | FileCheck %s
// RUN: ttmlir-opt %s --ttcore-register-device --d2m-allocate | FileCheck %s --check-prefix=DEFAULT
// RUN: not ttmlir-opt %s --ttcore-register-device --d2m-allocate='emit-resource-report=true available-l1-addr-range=1024,1056' --mlir-print-ir-after-failure 2>&1 | FileCheck %s --check-prefix=OOM
// RUN: sed 's/func.func @empty() {/func.func @empty() attributes {d2m.allocation_report = {status = "stale"}} {/' %s | ttmlir-opt --ttcore-register-device --d2m-allocate='emit-resource-report=true available-l1-addr-range=1024,132096' | FileCheck %s
// RUN: ttmlir-opt %S/allocate_intermediate_outputs.mlir --ttcore-register-device --d2m-reblock-generics='test-buffer-size-policy=max' --d2m-allocate='allow-l1-output-spilling=true test-assume-l1-capacity=12288 emit-resource-report=true' | FileCheck %s --check-prefix=SPILL
// RUN: sed 's/memory_space<l1>/memory_space<dram>/' %S/allocate_oom.mlir | not ttmlir-opt --ttcore-register-device --d2m-allocate='emit-resource-report=true' --mlir-print-ir-after-failure 2>&1 | FileCheck %s --check-prefix=DRAM-OOM
// RUN: ttmlir-opt %S/../Transforms/dataflow_planning.mlir --d2m-fe-pipeline='enable-dataflow-planning=true use-explicit-block-factors=true emit-resource-report=true' | FileCheck %s --check-prefix=PIPELINE
// RUN: sed 's/ {d2m.blocking_loop = [012]}//' %S/reblock_explicit_plan.mlir | not ttmlir-opt --ttcore-register-device --d2m-allocate='emit-resource-report=true' --mlir-print-ir-after-failure 2>&1 | FileCheck %s --check-prefix=INVALID-FORM
// RUN: sed 's/func.func @empty() {/func.func @empty() attributes {d2m.allocation_report = {status = "stale"}} {/' %s | not ttmlir-opt --ttcore-register-device --d2m-allocate='emit-resource-report=true available-l1-addr-range=1024,1056' --mlir-print-ir-after-failure 2>&1 | FileCheck %s --check-prefix=OOM --implicit-check-not=stale

!l1 = memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.memory_space<l1>>
!dram = memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.memory_space<dram>>

// SPILL: l1_to_dram_count = 1 : i64
// SPILL-SAME: memory_space = "dram", offset_bytes = {{[0-9]+}} : i64, original_memory_space = "l1"
// SPILL-SAME: status = "success"
// DRAM-OOM: required DRAM memory usage
// DRAM-OOM: status = "dram_capacity_exceeded"
// PIPELINE: d2m.allocation_report = {{.*}}status = "success"
// INVALID-FORM: d2m.allocation_report = {dram_capacity_bytes = {{[0-9]+}} : i64, l1_capacity_bytes = {{[0-9]+}} : i64, status = "failed", version = 1 : i64}

// DEFAULT-NOT: d2m.allocation_report
// CHECK-LABEL: func.func @l1()
// CHECK-SAME: dram_usage_bytes = 0 : i64
// CHECK-SAME: l1_capacity_bytes = 131072 : i64
// CHECK-SAME: l1_to_dram_count = 0 : i64
// CHECK-SAME: l1_usage_bytes = 65536 : i64
// CHECK-SAME: placements = [{has_request = true, id = 0 : i64, memory_space = "l1", offset_bytes = 0 : i64, original_memory_space = "l1", size_bytes = 65536 : i64}]
// CHECK-SAME: status = "success", version = 1 : i64
// CHECK: address = 1024 : i64
// OOM: required L1 memory usage 65536 exceeds memory capacity 32
// OOM: l1_usage_bytes = 65536 : i64, status = "l1_capacity_exceeded", version = 1 : i64
func.func @l1() -> !l1 {
  %a = memref.alloc() : !l1
  return %a : !l1
}

// CHECK-LABEL: func.func @dram()
// CHECK-SAME: dram_usage_bytes = {{[1-9][0-9]*}} : i64
// CHECK-SAME: l1_to_dram_count = 0 : i64
// CHECK-SAME: l1_usage_bytes = 0 : i64
// CHECK-SAME: memory_space = "dram"
// CHECK-SAME: original_memory_space = "dram"
// CHECK-SAME: status = "success"
func.func @dram() -> !dram {
  %a = memref.alloc() : !dram
  return %a : !dram
}

// CHECK-LABEL: func.func @empty()
// CHECK-SAME: dram_usage_bytes = 0 : i64
// CHECK-SAME: l1_usage_bytes = 0 : i64, placements = [], status = "success"
func.func @empty() {
  return
}
