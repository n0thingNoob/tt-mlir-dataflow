// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m="dram-args-and-results=true" --d2m-materialize-view-returns="dram-returns=true" --mlir-print-local-scope %s | FileCheck %s
// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m="dram-args-and-results=true" --d2m-materialize-view-returns --mlir-print-local-scope %s | FileCheck %s --check-prefix=VIEWSPACE
// RUN: ttmlir-opt --d2m-fe-pipeline="dram-args-and-results=true" --mlir-print-local-scope %s | FileCheck %s --check-prefix=PIPELINE

// The function returns a reshape of an intermediate (the exp result, in L1).
// The reshape is a view, which keeps the memory space of the tensor it views;
// d2m-materialize-view-returns copies it into a buffer of its own before the
// transfer to host. With dram-returns that buffer is in DRAM, like every other
// returned result under dram-args-and-results; without it the buffer takes the
// view's memory space (L1).

// CHECK-LABEL: func.func @reshape_result
// CHECK: %[[VIEW:.*]] = d2m.view_layout {{.*}} -> tensor<1x1x128x64xbf16, {{.*}}logical_shape = 128x64,{{.*}}, l1, sharded>>
// CHECK: %[[BUF:.*]] = d2m.empty() : tensor<1x1x128x64xbf16, {{.*}}logical_shape = 128x64,{{.*}}, dram, sharded>>
// CHECK: %[[COPY:.*]] = d2m.generic
// CHECK-NEXT: ins(%[[VIEW]] :
// CHECK-NEXT: outs(%[[BUF]] :
// CHECK: d2m.to_layout %[[COPY]], %{{.*}} : tensor<1x1x128x64xbf16, {{.*}}, dram, sharded>> into tensor<128x64xbf16>

// VIEWSPACE-LABEL: func.func @reshape_result
// VIEWSPACE: d2m.view_layout {{.*}} -> tensor<1x1x128x64xbf16, {{.*}}, l1, sharded>>
// VIEWSPACE: d2m.empty() : tensor<1x1x128x64xbf16, {{.*}}logical_shape = 128x64,{{.*}}, l1, sharded>>
// VIEWSPACE: d2m.generic
// VIEWSPACE: d2m.to_layout %{{.*}}, %{{.*}} : tensor<1x1x128x64xbf16, {{.*}}, l1, sharded>> into tensor<128x64xbf16>

// The pipeline passes dram-args-and-results on: the host reads the returned
// value from DRAM.
// PIPELINE-LABEL: func.func @reshape_result
// PIPELINE: d2m.to_host %{{.*}}, %{{.*}} layout = <logical_shape = 128x64, {{.*}}, dram, sharded> : memref<{{.*}}, #ttcore.memory_space<dram>> into memref<128x64xbf16>
func.func @reshape_result(%x: tensor<64x128xbf16>) -> tensor<128x64xbf16> {
  %0 = "ttir.exp"(%x) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
  %1 = "ttir.reshape"(%0) <{shape = [128 : i32, 64 : i32]}> : (tensor<64x128xbf16>) -> tensor<128x64xbf16>
  return %1 : tensor<128x64xbf16>
}

// TTIRToD2M moves every op result to host, so the reshape below also reaches a
// device-to-host to_layout as a view and is materialized; it is not returned,
// so its copy stays in the view's memory space (L1) even with dram-returns.
// The exp that is returned writes DRAM as before.
// CHECK-LABEL: func.func @reshape_mid
// CHECK: %[[VIEW:.*]] = d2m.view_layout {{.*}} -> tensor<1x1x128x64xbf16, {{.*}}, l1, sharded>>
// CHECK: d2m.generic
// CHECK-NEXT: ins(%[[VIEW]] :
// CHECK-NEXT: outs(%{{.*}} : tensor<1x1x128x64xbf16, {{.*}}, l1, sharded>>)
// CHECK: d2m.generic
// CHECK-NEXT: ins(
// CHECK-NEXT: outs(%{{.*}} : tensor<1x1x4x2x!ttcore.tile<32x32, bf16>, {{.*}}, dram, sharded>>)
// CHECK: d2m.tile_exp
// CHECK-NOT: d2m.generic
// CHECK: return
func.func @reshape_mid(%x: tensor<64x128xbf16>) -> tensor<128x64xbf16> {
  %0 = "ttir.exp"(%x) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
  %1 = "ttir.reshape"(%0) <{shape = [128 : i32, 64 : i32]}> : (tensor<64x128xbf16>) -> tensor<128x64xbf16>
  %2 = "ttir.exp"(%1) : (tensor<128x64xbf16>) -> tensor<128x64xbf16>
  return %2 : tensor<128x64xbf16>
}
