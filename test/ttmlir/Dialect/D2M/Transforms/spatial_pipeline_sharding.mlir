// Region-wide selection, actual per-core traversal and shared full-K RHS.
// RUN: ttmlir-opt %S/../../../Silicon/TTMetal/n150/spatial/auto_gemm_chain.mlir --d2m-fe-pipeline="execution-strategy=spatial dump-spatial-planning=true" --mlir-print-op-generic --mlir-print-ir-after=d2m-materialize-spatial-pipelines --mlir-disable-threading -o %t.two 2> %t.two.report
// RUN: FileCheck %s --check-prefix=TWO --input-file=%t.two.report
// RUN: FileCheck %s --check-prefix=LOCAL --input-file=%t.two.report
// RUN: ttmlir-opt %S/../../../Silicon/TTMetal/n150/spatial/auto_gemm_chain.mlir --ttir-to-ttmetal-pipeline="execution-strategy=spatial" --mlir-print-op-generic -o %t.metal
// RUN: FileCheck %s --check-prefix=METAL --input-file=%t.metal
// RUN: ttmlir-translate %t.metal --ttmetal-to-flatbuffer -o %t.ttm
// TWO: pipeline grid G0 P=2 cores=6
// TWO-SAME: #ttcore.core_range<(0,0), (0,1)>
// TWO-SAME: #ttcore.core_range<(0,2), (0,3)>
// TWO-SAME: #ttcore.core_range<(0,4), (0,5)>
// TWO-NOT: fallback
// LOCAL: "d2m.spatial"
// LOCAL: block_factors = [1, 4, 1]
// LOCAL-SAME: grid = #ttcore.grid<2x1
// LOCAL: memref<1x3x!ttcore.tile<32x32, bf16>>
// LOCAL: memref<3x1x!ttcore.tile<32x32, bf16>>
// LOCAL: d2m.pipeline_shards = 2
// LOCAL: "arith.subi"
// LOCAL: "d2m.semaphore_wait"
// LOCAL: d2m.pipeline_stage = 1
// LOCAL: block_factors = [1, 3, 1]
// LOCAL: "arith.subi"
// LOCAL: "d2m.semaphore_wait"
// LOCAL: d2m.pipeline_stage = 2
// METAL: #ttmetal.compute_config<@{{[^,]+}}, #ttmetal.core_range<0x0, 1x2>{{.*}}#ttmetal.compute_config<@{{[^,]+}}, #ttmetal.core_range<0x2, 1x2>{{.*}}#ttmetal.compute_config<@{{[^,]+}}, #ttmetal.core_range<0x4, 1x2>
// METAL: callee = "noc_semaphore_inc"
// METAL: callee = "experimental::semaphore_wait_min"

// Odd tile-row counts can select P=3; no special-cased powers of two.
// RUN: sed 's/64x/96x/g' %S/../../../Silicon/TTMetal/n150/spatial/auto_gemm_chain.mlir > %t.three.input
// RUN: ttmlir-opt %t.three.input --d2m-fe-pipeline="execution-strategy=spatial dump-spatial-planning=true" -o %t.three 2> %t.three.report
// RUN: FileCheck %s --check-prefix=THREE --input-file=%t.three.report
// THREE: pipeline grid G0 P=3 cores=9
// THREE-NOT: fallback

// A genuine 2D physical rectangle still represents M-only logical sharding.
// RUN: sed 's/64x/128x/g' %S/../../../Silicon/TTMetal/n150/spatial/auto_gemm_chain.mlir > %t.four.input
// RUN: ttmlir-opt %t.four.input --ttir-to-ttmetal-pipeline="execution-strategy=spatial override-device-shape=6,2 dump-spatial-planning=true" --mlir-print-op-generic -o %t.four 2> %t.four.report
// RUN: FileCheck %s --check-prefix=FOUR --input-file=%t.four.report
// RUN: ttmlir-translate %t.four --ttmetal-to-flatbuffer -o %t.four.ttm
// FOUR: pipeline grid G0 P=4 cores=12
// FOUR-SAME: #ttcore.core_range<(0,0), (1,1)>
// FOUR-SAME: #ttcore.core_range<(2,0), (3,1)>
// FOUR-SAME: #ttcore.core_range<(4,0), (5,1)>
// FOUR-NOT: fallback

// Ten cores fit nine by volume, but not three identical three-core rectangles.
// RUN: ttmlir-opt %t.three.input --d2m-fe-pipeline="execution-strategy=spatial override-device-shape=2,5 dump-spatial-planning=true" -o %t.geometry 2> %t.geometry.report
// RUN: FileCheck %s --check-prefix=GEOMETRY --input-file=%t.geometry.report
// GEOMETRY: reject P=3: stage rectangles do not fit worker grid
// GEOMETRY: pipeline grid G0 P=1 cores=3

// Upper bounds limit total region use and still require divisibility.
// RUN: ttmlir-opt %t.four.input --d2m-fe-pipeline="execution-strategy=spatial override-device-shape=1,6 dump-spatial-planning=true" -o %t.budget 2> %t.budget.report
// RUN: FileCheck %s --check-prefix=TWO --input-file=%t.budget.report
// RUN: ttmlir-opt %t.four.input --d2m-fe-pipeline="execution-strategy=spatial spatial-pipeline-max-shards=3 dump-spatial-planning=true" -o %t.divisor 2> %t.divisor.report
// RUN: FileCheck %s --check-prefix=TWO --input-file=%t.divisor.report

// Preserve the single-core execution protocol on demand.
// RUN: ttmlir-opt %t.four.input --d2m-fe-pipeline="execution-strategy=spatial spatial-pipeline-max-shards=1 dump-spatial-planning=true" -o %t.one 2> %t.one.report
// RUN: FileCheck %s --check-prefix=ONE --input-file=%t.one.report
// ONE: pipeline grid G0 P=1 cores=3

// GridSelection must reject all candidates before inserting readiness state.
// Check just through grid selection so the ordinary temporal allocator remains
// free to reject an independently impossible overall program.
// RUN: ttmlir-opt %t.three.input --d2m-fe-pipeline="execution-strategy=spatial" --mlir-print-ir-before=d2m-grid-selection --mlir-disable-threading -o /dev/null 2> %t.pre-grid
// RUN: ttmlir-opt %t.pre-grid --d2m-grid-selection="test-assume-l1-capacity=8192 dump-spatial-planning=true" --mlir-print-op-generic -o %t.l1 2> %t.l1.report
// RUN: FileCheck %s --check-prefix=TEMPORAL --input-file=%t.l1 --implicit-check-not=d2m.pipeline_group --implicit-check-not=semaphore
// RUN: FileCheck %s --check-prefix=CAPACITY --input-file=%t.l1.report
// CAPACITY: reject P=3: pipeline tensors, full RHS and CB reservation exceed L1
// CAPACITY: reject P=1: pipeline tensors, full RHS and CB reservation exceed L1
// CAPACITY: pipeline grid G0 temporal fallback:
// TEMPORAL: "d2m.generic"

// A shared program needs enough CB ports, even though each stage is smaller.
// RUN: sed 's/num_cbs = 32/num_cbs = 4/' %t.pre-grid | ttmlir-opt --d2m-grid-selection="dump-spatial-planning=true" --mlir-print-op-generic -o %t.cb 2> %t.cb.report
// RUN: FileCheck %s --check-prefix=CB --input-file=%t.cb.report
// RUN: FileCheck %s --check-prefix=TEMPORAL --input-file=%t.cb --implicit-check-not=d2m.pipeline_group --implicit-check-not=semaphore
// CB: reject P=3: insufficient circular buffer ports
// CB: reject P=1: insufficient circular buffer ports
// CB: temporal fallback:

// Resource exhaustion at the region level clears the complete group.
// RUN: ttmlir-opt %t.pre-grid --d2m-grid-selection="override-device-shape=1,2 dump-spatial-planning=true" --mlir-print-op-generic -o %t.cores 2> %t.cores.report
// RUN: FileCheck %s --check-prefix=CORES --input-file=%t.cores.report
// RUN: FileCheck %s --check-prefix=TEMPORAL --input-file=%t.cores --implicit-check-not=d2m.pipeline_group --implicit-check-not=semaphore
// CORES: temporal fallback: insufficient worker cores

// Invalid options and corrupted preparation contracts are errors, not fallback.
// RUN: not ttmlir-opt %t.pre-grid --d2m-grid-selection="spatial-pipeline-max-shards=-1" -o /dev/null 2>&1 | FileCheck %s --check-prefix=NEGATIVE
// NEGATIVE: spatial-pipeline-max-shards must be nonnegative
// RUN: sed 's/d2m.pipeline_stage = 1/d2m.pipeline_stage = 7/' %t.pre-grid | not ttmlir-opt --d2m-grid-selection -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONTRACT
// CONTRACT: invalid prepared spatial pipeline contract

// More than one local tile row per shard exercises local, cumulative readiness.
// RUN: ttmlir-opt %t.four.input --d2m-fe-pipeline="execution-strategy=spatial spatial-pipeline-max-shards=2" --mlir-print-ir-after=d2m-materialize-spatial-pipelines --mlir-disable-threading --mlir-print-op-generic -o /dev/null 2> %t.local-rows
// RUN: FileCheck %s --check-prefix=ROWS --input-file=%t.local-rows
// ROWS: block_factors = [2, 4, 1]
// ROWS: d2m.pipeline_stage = 0
// ROWS: block_factors = [2, 4]
// ROWS: "arith.subi"
// ROWS: "d2m.semaphore_wait"
// ROWS: d2m.pipeline_stage = 1
// ROWS: block_factors = [2, 3, 1]
// ROWS: "arith.subi"
// ROWS: "d2m.semaphore_wait"

// Multi-core unsupported traversal falls back as a whole without counters.
// RUN: ttmlir-opt %t.four.input --ttir-to-ttmetal-pipeline="execution-strategy=spatial spatial-pipeline-max-shards=2 matmul-interchange=1,0,2 dump-spatial-planning=true" --mlir-print-op-generic -o %t.interchange 2> %t.interchange.report
// RUN: FileCheck %s --check-prefix=LATE --input-file=%t.interchange.report
// RUN: FileCheck %s --check-prefix=LATE-IR --input-file=%t.interchange --implicit-check-not=global_semaphore --implicit-check-not=experimental::semaphore_wait_min --implicit-check-not=d2m.pipeline_
// RUN: ttmlir-translate %t.interchange --ttmetal-to-flatbuffer -o %t.interchange.ttm
// LATE: pipeline traversal fallback:
// LATE-COUNT-3: emitted temporal fallback
// LATE-IR: ttmetal.enqueue_program

// An override cannot enlarge the pipeline's real descriptor resource budget.
// RUN: ttmlir-opt %t.pre-grid --d2m-grid-selection="override-device-shape=64,64 dump-spatial-planning=true" -o /dev/null 2> %t.override.report
// RUN: FileCheck %s --check-prefix=OVERRIDE --input-file=%t.override.report
// OVERRIDE: pipeline grid G0 P=3 cores=9
// OVERRIDE-SAME: #ttcore.core_range<(1,0), (1,2)>

// A fork/join has one independent distributed counter per edge. All four
// two-core stages share one spatial program; the two join inputs never share
// a readiness counter.
// RUN: ttmlir-opt %S/../../../Silicon/TTMetal/n150/spatial/auto_diamond.mlir --d2m-fe-pipeline="execution-strategy=spatial spatial-pipeline-max-shards=2" --mlir-print-op-generic -o %t.diamond
// RUN: FileCheck %s --check-prefix=DIAMOND --input-file=%t.diamond
// DIAMOND-COUNT-4: "d2m.create_global_semaphore"
// DIAMOND: "d2m.spatial"
// DIAMOND-SAME: #ttcore.core_range<(0,0), (0,1)>
// DIAMOND-SAME: #ttcore.core_range<(0,2), (0,3)>
// DIAMOND-SAME: #ttcore.core_range<(0,4), (0,5)>
// DIAMOND-SAME: #ttcore.core_range<(0,6), (0,7)>
// DIAMOND-COUNT-4: d2m.pipeline_shards = 2
