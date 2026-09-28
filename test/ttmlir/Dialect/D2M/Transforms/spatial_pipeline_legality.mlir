// An existing downstream option changes traversal AFTER selection. The entire
// prepared group must fall back, without leaving any ready synchronization.
// RUN: ttmlir-opt %S/../../../Silicon/TTMetal/n150/spatial/auto_gemm_chain.mlir --ttir-to-ttmetal-pipeline="execution-strategy=spatial spatial-pipeline-max-shards=1 dump-spatial-planning=true matmul-interchange=1,0,2" --mlir-print-op-generic -o %t.swap 2> %t.swap.report
// RUN: FileCheck %s --check-prefix=FALLBACK --input-file=%t.swap.report
// RUN: FileCheck %s --check-prefix=NOSYNC --input-file=%t.swap --implicit-check-not=semaphore --implicit-check-not=d2m.pipeline_
// RUN: ttmlir-translate %t.swap --ttmetal-to-flatbuffer -o %t.swap.ttm
// RUN: ttmlir-opt %S/../../../Silicon/TTMetal/n150/spatial/auto_gemm_chain.mlir --ttir-to-ttmetal-pipeline="execution-strategy=spatial spatial-pipeline-max-shards=1 dump-spatial-planning=true matmul-interchange=2,0,1" --mlir-print-op-generic -o %t.kouter 2> %t.kouter.report
// RUN: FileCheck %s --check-prefix=FALLBACK --input-file=%t.kouter.report
// RUN: FileCheck %s --check-prefix=NOSYNC --input-file=%t.kouter --implicit-check-not=semaphore --implicit-check-not=d2m.pipeline_
// RUN: ttmlir-translate %t.kouter --ttmetal-to-flatbuffer -o %t.kouter.ttm
// FALLBACK: pipeline traversal fallback: stage 0:
// FALLBACK-COUNT-3: emitted temporal fallback
// FALLBACK-NOT: emitted L1 tile pipeline
// NOSYNC: ttmetal.enqueue_program

// Reuse the canonical prepared IR to inject extra stores. Repeating the same
// write is legal sequential memory behavior, but would publish too many ready
// notifications if admitted into the pipeline.
// RUN: ttmlir-opt %S/../../../Silicon/TTMetal/n150/spatial/auto_gemm_chain.mlir --d2m-fe-pipeline="execution-strategy=spatial spatial-pipeline-max-shards=1" --mlir-print-ir-before=d2m-materialize-spatial-pipelines --mlir-disable-threading --mlir-print-op-generic -o /dev/null 2> %t.prepared
// RUN: sed '/"d2m.remote_store"/p' %t.prepared | ttmlir-opt --d2m-materialize-spatial-pipelines="dump-regions=true" --mlir-print-op-generic -o %t.duplicate 2> %t.duplicate.report
// RUN: FileCheck %s --check-prefix=FALLBACK --input-file=%t.duplicate.report
// RUN: FileCheck %s --check-prefix=SINGLETON --input-file=%t.duplicate --implicit-check-not=d2m.pipeline_ --implicit-check-not=semaphore
// RUN: ttmlir-opt %t.duplicate --d2m-materialize-spatial-pipelines --mlir-print-op-generic -o %t.again
// RUN: diff %t.duplicate %t.again
// RUN: ttmlir-opt %t.duplicate --ttcore-mark-functions-as-forward --d2m-mark-synchronized-buffers --d2m-allocate --d2m-lower-multicast-loads --d2m-generic-lower-to-explicit-form --canonicalize --d2m-be-pipeline --d2m-to-ttkernel-pre-emitc-pipeline --d2m-to-ttmetal-pipeline --d2m-emitc-pipeline --ttcore-wrap-device-module --mlir-print-op-generic -o %t.lowered
// RUN: ttmlir-translate %t.lowered --ttmetal-to-flatbuffer -o %t.duplicate.ttm
// SINGLETON-COUNT-3: "d2m.spatial"
