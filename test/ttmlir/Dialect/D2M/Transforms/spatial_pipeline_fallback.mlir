// RUN: ttmlir-opt %s --ttcore-register-device --ttir-to-d2m -o %t.base
// RUN: sed 's/workerGrid = #ttcore.grid<8x8/workerGrid = #ttcore.grid<1x2/' %t.base | ttmlir-opt --d2m-spatial-planning="prepare-pipelines=true dump-regions=true" -o /dev/null 2>&1 | FileCheck %s --check-prefix=CORES
// RUN: sed 's/l1_size = 1499136/l1_size = 65536/' %t.base | ttmlir-opt --d2m-spatial-planning="prepare-pipelines=true dump-regions=true" -o /dev/null 2>&1 | FileCheck %s --check-prefix=L1
// RUN: sed 's/num_cbs = 32/num_cbs = 2/' %t.base | ttmlir-opt --d2m-spatial-planning="prepare-pipelines=true dump-regions=true" -o /dev/null 2>&1 | FileCheck %s --check-prefix=CBS
// RUN: sed 's/d2m.tile_relu/d2m.tile_exp/' %t.base | ttmlir-opt --d2m-spatial-planning="prepare-pipelines=true dump-regions=true" -o /dev/null 2>&1 | FileCheck %s --check-prefix=COMPUTE
// RUN: ttmlir-opt %t.base --d2m-spatial-planning="prepare-pipelines=true" -o %t.prepared
// RUN: ttmlir-opt %t.prepared --d2m-spatial-planning="prepare-pipelines=true" -o %t.again
// RUN: diff %t.prepared %t.again
// RUN: not ttmlir-opt %s --d2m-fe-pipeline="execution-strategy=spatial test-assume-l1-capacity=8192 allow-l1-output-spilling=true" -o /dev/null 2>&1 | FileCheck %s --check-prefix=NOSPILL

// CORES: pipeline fallback R0: insufficient worker cores
// CORES: selected S1 members=[G1, G2] reason=parallel:
// L1: pipeline fallback R0: full tensors and conservative CB reservation exceed L1
// CBS: pipeline fallback R0: insufficient circular buffer ports for one program
// COMPUTE: pipeline fallback R0: unsupported compute or destination alias
// NOSPILL: required L1 memory usage
// NOSPILL-SAME: exceeds memory capacity 8192

func.func @main(%a: tensor<32x32xbf16>, %b: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  %0 = "ttir.add"(%a, %b) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %1 = "ttir.relu"(%0) : (tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %2 = "ttir.neg"(%0) : (tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %3 = "ttir.add"(%1, %2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %3 : tensor<32x32xbf16>
}
