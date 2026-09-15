// RUN: ttmlir-opt %s --d2m-fe-pipeline -o %t.default
// RUN: ttmlir-opt %s --d2m-fe-pipeline='enable-dataflow-planning=true dump-dataflow-plan=true' -o %t.planned 2>&1 | FileCheck %s --check-prefix=TRACE
// RUN: diff %t.default %t.planned
// RUN: ttmlir-opt %s --d2m-fe-pipeline='dump-dataflow-plan=true' -o /dev/null 2>&1 | FileCheck %s --check-prefix=DISABLED --allow-empty
// RUN: ttmlir-opt %S/../allocate/allocate_reblock_auto.mlir --ttcore-register-device --d2m-reblock-generics -o %t.reblock
// RUN: ttmlir-opt %S/../allocate/allocate_reblock_auto.mlir --ttcore-register-device --d2m-dataflow-blocking-planning --d2m-reblock-generics -o %t.passthrough
// RUN: diff %t.reblock %t.passthrough
// RUN: ttmlir-opt %S/../allocate/reblock_explicit_plan.mlir --ttcore-register-device --d2m-reblock-generics='use-explicit-block-factors=true' -o %t.explicit
// RUN: ttmlir-opt %S/../allocate/reblock_explicit_plan.mlir --ttcore-register-device --d2m-dataflow-blocking-planning --d2m-reblock-generics='use-explicit-block-factors=true' -o %t.explicit-passthrough
// RUN: diff %t.explicit %t.explicit-passthrough

// TRACE: d2m-dataflow-plan function=@main
// TRACE: d2m-dataflow-blocking function=@main candidate=0 policy=passthrough current-factors=[
// DISABLED-NOT: d2m-dataflow-
func.func @main(%a: tensor<64x64xbf16>, %b: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  %0 = "ttir.matmul"(%a, %b) : (tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
  %1 = "ttir.relu"(%0) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
  return %1 : tensor<64x64xbf16>
}
