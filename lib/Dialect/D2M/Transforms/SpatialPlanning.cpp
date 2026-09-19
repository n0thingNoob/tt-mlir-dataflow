// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MSPATIALPLANNING
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

struct GenericDAGNode {
  GenericOp generic;
  SmallVector<unsigned> predecessors;
};

// A projection of block-local tensor SSA dependencies onto generic operations.
// This is not a memory-dependence or cross-control-flow scheduling analysis.
using GenericDAG = SmallVector<GenericDAGNode>;

static GenericDAG buildGenericDAG(Block &block) {
  GenericDAG dag;
  llvm::DenseMap<Operation *, unsigned> nodeIndices;
  for (GenericOp generic : block.getOps<GenericOp>()) {
    if (!generic.hasTensorSemantics()) {
      continue;
    }
    nodeIndices[generic.getOperation()] = dag.size();
    dag.push_back({generic, {}});
  }

  for (GenericDAGNode &node : dag) {
    llvm::SetVector<Value> capturedValues;
    getUsedValuesDefinedAbove(node.generic->getRegions(), capturedValues);
    SmallVector<Value> worklist(node.generic->getOperands());
    llvm::append_range(worklist, capturedValues);
    llvm::SmallPtrSet<Operation *, 16> visited;
    llvm::SetVector<unsigned> predecessors;
    while (!worklist.empty()) {
      Operation *def = worklist.pop_back_val().getDefiningOp();
      if (!def || def->getBlock() != &block || !visited.insert(def).second) {
        continue;
      }
      auto producer = nodeIndices.find(def);
      if (producer != nodeIndices.end()) {
        predecessors.insert(producer->second);
        continue;
      }
      // Region-bearing operations are opaque scheduling boundaries. Ordinary
      // SSA operations, including layout views, preserve dependency paths.
      if (def->getNumRegions() == 0) {
        llvm::append_range(worklist, def->getOperands());
      }
    }
    node.predecessors.assign(predecessors.begin(), predecessors.end());
    llvm::sort(node.predecessors);
  }
  return dag;
}

struct SpatialGroup {
  SmallVector<unsigned> nodes;
  // One physical core range per generic, matching nodes in order.
  SmallVector<ttcore::CoreRangeAttr> coreRanges;
};

struct SpatialPlan {
  SmallVector<SpatialGroup> groups;
};

static SmallVector<SpatialGroup> selectGroups(const GenericDAG &) {
  // TODO (spatial-planning): Select legal inter-generic groups. Account for
  // dependency paths and side-effect/control-flow boundaries before moving ops.
  return {};
}

static std::optional<SpatialPlan> assignCoreRanges(const GenericDAG &,
                                                   ArrayRef<SpatialGroup>) {
  // TODO (spatial-planning): Assign disjoint, device-bounded physical core
  // ranges. Leave per-generic grids and block factors to the existing passes.
  return std::nullopt;
}

static std::optional<SpatialPlan> evaluateCandidate(const GenericDAG &,
                                                    const SpatialPlan &) {
  // TODO (spatial-planning): Evaluate candidates and select a plan. Search and
  // cost modeling belong here, including future automatic strategy selection.
  return std::nullopt;
}

class D2MSpatialPlanningPass
    : public impl::D2MSpatialPlanningBase<D2MSpatialPlanningPass> {
public:
  using impl::D2MSpatialPlanningBase<
      D2MSpatialPlanningPass>::D2MSpatialPlanningBase;

  void runOnOperation() final {
    for (func::FuncOp func : getOperation().getOps<func::FuncOp>()) {
      for (Block &block : func.getBody()) {
        GenericDAG dag = buildGenericDAG(block);
        numGenericNodes += dag.size();
        for (const GenericDAGNode &node : dag) {
          numDependencyEdges += node.predecessors.size();
        }
        auto groups = selectGroups(dag);
        if (groups.empty()) {
          continue;
        }
        auto candidate = assignCoreRanges(dag, groups);
        if (!candidate || !evaluateCandidate(dag, *candidate)) {
          continue;
        }
        // TODO (spatial-planning): Materialize a selected plan as d2m.spatial,
        // preserving SSA uses, DPS outputs and spatial yields. Until
        // implemented, fail explicitly if a future policy starts returning
        // selected plans.
        func.emitOpError("spatial plan materialization is not implemented");
        signalPassFailure();
        return;
      }
    }
    markAllAnalysesPreserved();
  }
};

} // namespace
} // namespace mlir::tt::d2m
