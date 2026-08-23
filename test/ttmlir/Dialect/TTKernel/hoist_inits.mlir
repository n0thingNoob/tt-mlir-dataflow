// RUN: ttmlir-opt --ttkernel-hoist-inits %s | FileCheck %s

!cb0 = !ttkernel.cb<8, !ttcore.tile<32x32, f32>>
!cb1 = !ttkernel.cb<8, !ttcore.tile<32x32, f32>>
!cb2 = !ttkernel.cb<8, !ttcore.tile<32x32, f32>>

module {
  // Hardware startup resets the unpack/pack/SFPU configuration. It must be
  // ordered before a specialized init even when conversion emitted the
  // specialized init first.
  // CHECK-LABEL: func.func @startup_precedes_specialized_init
  func.func @startup_precedes_specialized_init() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
    %in = "ttkernel.get_compile_time_arg_val"() <{arg_index = 0 : i32}> : () -> !cb0
    %scale = "ttkernel.get_compile_time_arg_val"() <{arg_index = 1 : i32}> : () -> !cb1
    %out = "ttkernel.get_compile_time_arg_val"() <{arg_index = 2 : i32}> : () -> !cb2
    // CHECK: %[[OUT:.*]] = ttkernel.get_compile_time_arg_val(2)
    // CHECK-NEXT: ttkernel.compute_kernel_hw_startup(%{{.*}}, %{{.*}}, %[[OUT]])
    // CHECK-NEXT: ttkernel.init_sfpu(%{{.*}}, %[[OUT]])
    "ttkernel.init_sfpu"(%in, %out) : (!cb0, !cb2) -> ()
    "ttkernel.compute_kernel_hw_startup"(%in, %scale, %out) : (!cb0, !cb1, !cb2) -> ()
    return
  }
}
