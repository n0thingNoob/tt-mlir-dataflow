// RUN: rm -rf %t.run
// RUN: %dataflow_python %S/../Transforms/Inputs/run_dataflow_passthrough_hardware.py --compiler %ttmlir_tools/ttmlir-opt --translate %ttmlir_tools/ttmlir-translate --metal-home %dataflow_metal --device %dataflow_device --output %t.run --runs 3 | FileCheck %s
// CHECK: PASS: planner off/on, 3 runs each
