// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%S/Inputs/wormhole_n300_num_cbs_64.ttsys" %s | FileCheck %s

// A Wormhole n300 system descriptor written by a runtime that recorded the
// host-side NUM_CIRCULAR_BUFFERS (64). Wormhole device firmware has 32
// circular buffers, so the loaded chip descriptor is clamped to 32.
// CHECK: arch = <wormhole_b0>
// CHECK-SAME: num_cbs = 32

func.func @f(%a: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  return %a : tensor<32x32xbf16>
}
