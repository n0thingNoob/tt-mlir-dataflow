// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MDATAFLOWBLOCKINGPLANNING
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
class D2MDataflowBlockingPlanningPass final
    : public impl::D2MDataflowBlockingPlanningBase<
          D2MDataflowBlockingPlanningPass> {
public:
  using Base =
      impl::D2MDataflowBlockingPlanningBase<D2MDataflowBlockingPlanningPass>;
  using Base::Base;

  void runOnOperation() override {
    for (auto function : getOperation().getOps<func::FuncOp>()) {
      unsigned ordinal = 0;
      function.walk([&](GenericOp generic) {
        if (generic->getParentOfType<SpatialOp>() ||
            generic->getParentOfType<GenericOp>()) {
          return;
        }
        if (dumpPlan) {
          llvm::errs() << "d2m-dataflow-blocking function=@"
                       << function.getSymName() << " candidate=" << ordinal
                       << " policy=passthrough current-factors=[";
          llvm::interleaveComma(generic.getBlockFactorsValue(), llvm::errs());
          llvm::errs() << "]\n";
        }
        ++ordinal;
      });
    }
    // In particular, do not freeze the input factors with a planned-factors
    // attribute: doing so would disable the existing reblocking heuristic.
    markAllAnalysesPreserved();
  }
};
} // namespace
} // namespace mlir::tt::d2m
