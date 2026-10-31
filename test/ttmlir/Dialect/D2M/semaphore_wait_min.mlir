// RUN: ttmlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// A min wait round-trips; a min wait with a reset is rejected.

// CHECK-LABEL: func.func private @wait_min
// CHECK: d2m.semaphore_wait %{{.*}}, %{{.*}} min : !d2m.local_semaphore
func.func private @wait_min() attributes {d2m.thread = #d2m.thread<datamovement>} {
  %sem0 = d2m.get_arg(0) : !d2m.local_semaphore
  %c3 = arith.constant 3 : index
  d2m.semaphore_wait %sem0, %c3 min : !d2m.local_semaphore
  return
}

// -----

func.func private @wait_min_reset() attributes {d2m.thread = #d2m.thread<datamovement>} {
  %sem0 = d2m.get_arg(0) : !d2m.local_semaphore
  %c3 = arith.constant 3 : index
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{a min wait cannot reset the semaphore}}
  d2m.semaphore_wait %sem0, %c3 reset %c0 min : !d2m.local_semaphore
  return
}
