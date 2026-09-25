// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "SpatialMapping.h"
#include "ttmlir/Dialect/D2M/Utils/SpatialPipeline.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MSPATIALPLANNING
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

class D2MSpatialPlanningPass
    : public impl::D2MSpatialPlanningBase<D2MSpatialPlanningPass> {
public:
  using impl::D2MSpatialPlanningBase<
      D2MSpatialPlanningPass>::D2MSpatialPlanningBase;

  void runOnOperation() final {
    SmallVector<SpatializationAnalysisResult, 0> analyses;
    SmallVector<SmallVector<SelectedSpatialGroup, 0>> plans;
    for (func::FuncOp func : getOperation().getOps<func::FuncOp>()) {
      for (Block &block : func.getBody()) {
        SpatializationAnalysisResult result = analyzeSpatialization(block);
        numGenericNodes += result.dag.nodes.size();
        for (const SpatialGenericNode &node : result.dag.nodes) {
          numDependencyEdges += node.predecessors.size();
        }
        if (dumpRegions) {
          printSpatializationAnalysis(llvm::errs(), block, result);
        }
        if (preparePipelines && !result.dag.nodes.empty() &&
            llvm::all_of(result.dag.nodes, [](const SpatialGenericNode &node) {
              return node.generic->hasAttr(spatial_pipeline::group);
            })) {
          continue;
        }
        if (materialize || preparePipelines) {
          auto plan =
              selectSpatialGroups(result, dumpRegions, preparePipelines);
          if (failed(plan)) {
            signalPassFailure();
            return;
          }
          plans.push_back(std::move(*plan));
          analyses.push_back(std::move(result));
        }
      }
    }
    if (!materialize && !preparePipelines) {
      markAllAnalysesPreserved();
      return;
    }
    // Selection above has preflighted every block before this mutation phase.
    for (auto [analysis, plan] : llvm::zip(analyses, plans)) {
      if (dumpRegions && !analysis.dag.nodes.empty()) {
        auto func =
            analysis.dag.nodes.front().generic->getParentOfType<func::FuncOp>();
        llvm::errs() << "spatial_selected_groups @" << func.getSymName()
                     << '\n';
      }
      materializeSpatialGroups(analysis.dag, plan, dumpRegions);
    }
  }
};

} // namespace
} // namespace mlir::tt::d2m
