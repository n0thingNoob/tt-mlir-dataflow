// RUN: ttmlir-opt %s --d2m-dataflow-allocation-feedback='dump-plan=true' -o %t 2>&1 | FileCheck %s --check-prefix=TRACE
// RUN: FileCheck %s --input-file=%t --check-prefix=CONSUMED
// RUN: ttmlir-opt %s --d2m-dataflow-allocation-feedback='keep-report=true' | FileCheck %s --check-prefix=KEPT
// RUN: not ttmlir-opt %t --d2m-dataflow-allocation-feedback 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: sed 's/status = "success"/status = "failed"/' %s | not ttmlir-opt --d2m-dataflow-allocation-feedback 2>&1 | FileCheck %s --check-prefix=FAILED
// RUN: sed 's/version = 1 : i64/version = 2 : i64/' %s | not ttmlir-opt --d2m-dataflow-allocation-feedback 2>&1 | FileCheck %s --check-prefix=VERSION
// RUN: ttmlir-opt %S/dataflow_blocking_passthrough.mlir --d2m-fe-pipeline='enable-dataflow-planning=true dump-dataflow-plan=true' -o /dev/null 2>&1 | FileCheck %s --check-prefix=PIPELINE
// RUN: ttmlir-opt %S/dataflow_blocking_passthrough.mlir --d2m-fe-pipeline='enable-dataflow-planning=true emit-resource-report=true' | FileCheck %s --check-prefix=KEPT

// TRACE: d2m-dataflow-allocation function=@main status=success l1-usage=512 dram-usage=0 l1-to-dram=0
// CONSUMED-NOT: d2m.allocation_report
// CONSUMED: func.func @main
// CONSUMED-NOT: d2m.allocation_report
// KEPT: d2m.allocation_report
// MISSING: missing d2m.allocation_report
// FAILED: allocation did not succeed: failed
// VERSION: expected allocation report version 1
// PIPELINE: d2m-dataflow-plan function=@main
// PIPELINE: d2m-dataflow-blocking function=@main
// PIPELINE: d2m-dataflow-allocation function=@main status=success
func.func private @external()
func.func @main() attributes {d2m.allocation_report = {
    version = 1 : i64, status = "success",
    l1_capacity_bytes = 1024 : i64, dram_capacity_bytes = 8192 : i64,
    l1_usage_bytes = 512 : i64, dram_usage_bytes = 0 : i64,
    l1_to_dram_count = 0 : i64}} {
  return
}
