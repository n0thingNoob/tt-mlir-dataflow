// RUN: ttmlir-opt %s --ttcore-register-device --d2m-reblock-generics='use-explicit-block-factors=true test-buffer-size-policy=max' | FileCheck %s --check-prefix=PLAN
// RUN: ttmlir-opt %s --ttcore-register-device --d2m-reblock-generics | FileCheck %s --check-prefix=AUTO
// RUN: ttmlir-opt %s --ttcore-register-device --d2m-reblock-generics='use-explicit-block-factors=true' --d2m-reblock-generics='use-explicit-block-factors=true' | FileCheck %s --check-prefix=PLAN
// RUN: sed 's/array<i64: 1, 1, 4>/array<i64: 1, 1, 1>/' %s | ttmlir-opt --ttcore-register-device --d2m-reblock-generics='use-explicit-block-factors=true' | FileCheck %s --check-prefix=IDENTITY
// RUN: sed 's/array<i64: 1, 1, 4>/array<i64: 1, 1>/' %s | not ttmlir-opt --ttcore-register-device --d2m-reblock-generics='use-explicit-block-factors=true' 2>&1 | FileCheck %s --check-prefix=RANK
// RUN: sed 's/array<i64: 1, 1, 4>/array<i64: 1, 1, 0>/' %s | not ttmlir-opt --ttcore-register-device --d2m-reblock-generics='use-explicit-block-factors=true' 2>&1 | FileCheck %s --check-prefix=POSITIVE
// RUN: sed 's/array<i64: 1, 1, 4>/array<i64: 1, 1, -1>/' %s | not ttmlir-opt --ttcore-register-device --d2m-reblock-generics='use-explicit-block-factors=true' 2>&1 | FileCheck %s --check-prefix=POSITIVE
// RUN: sed 's/array<i64: 1, 1, 4>/array<i64: 1, 1, 3>/' %s | not ttmlir-opt --ttcore-register-device --d2m-reblock-generics='use-explicit-block-factors=true' 2>&1 | FileCheck %s --check-prefix=DIVISIBILITY
// RUN: sed 's/array<i64: 1, 1, 4>/array<i64: 1, 1, 9223372036854775807>/' %s | not ttmlir-opt --ttcore-register-device --d2m-reblock-generics='use-explicit-block-factors=true' 2>&1 | FileCheck %s --check-prefix=DIVISIBILITY
// RUN: sed 's/array<i64: 1, 1, 4>/"bad"/' %s | not ttmlir-opt --ttcore-register-device --d2m-reblock-generics='use-explicit-block-factors=true' 2>&1 | FileCheck %s --check-prefix=TYPE
// RUN: ttmlir-opt %s --ttcore-register-device --d2m-reblock-generics='use-explicit-block-factors=true' --d2m-allocate='test-assume-l1-capacity=8388608 emit-resource-report=true' | FileCheck %s --check-prefix=ALLOCATE
// RUN: sed -e 's/func.func @planned_matmul()/func.func @planned_matmul(%lhs: !tensor)/' -e '/^  %lhs = memref.alloc()/d' %s | ttmlir-opt --ttcore-register-device --d2m-reblock-generics='use-explicit-block-factors=true' --d2m-allocate='test-assume-l1-capacity=8388608 emit-resource-report=true' | FileCheck %s --check-prefix=EXTERNAL
// RUN: sed 's/array<i64: 1, 1, 4>/array<i64: 16, 16, 16>/' %s | not ttmlir-opt --ttcore-register-device --d2m-reblock-generics='use-explicit-block-factors=true' 2>&1 | FileCheck %s --check-prefix=DOMAIN
// RUN: not ttmlir-opt %s --ttcore-register-device --d2m-reblock-generics='use-explicit-block-factors=true num-stream-buffers=0' 2>&1 | FileCheck %s --check-prefix=DEPTH
// RUN: ttmlir-opt %S/allocate_reblock_auto.mlir --ttcore-register-device --d2m-reblock-generics -o %t.default
// RUN: ttmlir-opt %S/allocate_reblock_auto.mlir --ttcore-register-device --d2m-reblock-generics='use-explicit-block-factors=true' -o %t.unplanned
// RUN: diff %t.default %t.unplanned

// PLAN: d2m.generic {block_factors = [1, 1, 4], d2m.planned_block_factors = array<i64: 1, 1, 4>
// PLAN: memref.alloc() {{.*}} : memref<16x4x!ttcore.tile<32x32, f32>
// PLAN: memref.alloc() {{.*}} : memref<4x16x!ttcore.tile<32x32, f32>
// AUTO: d2m.generic {block_factors = [1, 1, 16]
// IDENTITY: d2m.generic {block_factors = [1, 1, 1]
// RANK: invalid d2m.planned_block_factors: expected one planned block factor per iteration dimension
// POSITIVE: invalid d2m.planned_block_factors: planned factors must be positive multiples of current factors
// DIVISIBILITY: invalid d2m.planned_block_factors: planned factors must evenly divide the remaining shard
// TYPE: d2m.planned_block_factors must be an array<i64>
// ALLOCATE: d2m.allocation_report = {{.*}}status = "success"
// ALLOCATE: d2m.generic {block_factors = [1, 1, 4]
// EXTERNAL: d2m.allocation_report = {{.*}}has_request = false
// DOMAIN: invalid d2m.planned_block_factors: unsupported reblocking
// DEPTH: invalid d2m.planned_block_factors: num-stream-buffers must be positive

#l1 = #ttcore.memory_space<l1>
!tile = !ttcore.tile<32x32, f32>
!tensor = memref<1x1x16x16x!tile, #ttcore.shard<65536x4096, 1>, #l1>
!buffer = memref<16x16x!tile, #l1>
#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (k, n)>
#out = affine_map<(m, n, k) -> (m, n)>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>

func.func @planned_matmul() -> !tensor {
  %lhs = memref.alloc() : !tensor
  %rhs = memref.alloc() : !tensor
  %out = memref.alloc() : !tensor
  d2m.generic {block_factors = [1, 1, 1], d2m.planned_block_factors = array<i64: 1, 1, 4>, grid = #ttcore.grid<1x1>, indexing_maps = [#lhs, #rhs, #out], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<unified>]}
      ins(%lhs, %rhs : !tensor, !tensor) outs(%out : !tensor) {
    %bf0 = d2m.get_block_factor(0) : index
    %bf1 = d2m.get_block_factor(1) : index
    %bf2 = d2m.get_block_factor(2) : index
    affine.for %m = 0 to %bf0 {
      affine.for %n = 0 to %bf1 {
        affine.for %k = 0 to %bf2 {
          %a = memref.alloc() {d2m.synchronized_buffer = 2} : !buffer
          %b = memref.alloc() {d2m.synchronized_buffer = 2} : !buffer
          %c = memref.alloc() {d2m.synchronized_buffer = 2} : !buffer
          d2m.remote_load %a %lhs[%m, %k] : !buffer, !tensor
          d2m.remote_load %b %rhs[%k, %n] : !buffer, !tensor
          linalg.generic {indexing_maps = [#lhs, #rhs, #out], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a, %b : !buffer, !buffer) outs(%c : !buffer) {
          ^bb0(%x: !tile, %y: !tile, %z: !tile):
            %v = "d2m.tile_matmul"(%x, %y, %z) : (!tile, !tile, !tile) -> !tile
            linalg.yield %v : !tile
          }
          d2m.remote_store %out[%m, %n] %c : !tensor, !buffer
        } {d2m.blocking_loop = 2}
      } {d2m.blocking_loop = 1}
    } {d2m.blocking_loop = 0}
  }
  return %out : !tensor
}
