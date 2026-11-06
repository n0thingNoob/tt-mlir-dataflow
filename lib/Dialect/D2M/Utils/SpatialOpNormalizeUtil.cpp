// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/SpatialOpNormalizeUtil.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::d2m {
namespace detail {

static d2m::GenericOp getSingleGenericOpFromSpatialRegion(Region &region) {
  TT_assertv(!region.empty(), "each spatial region must not be empty.");
  Block &regionBlock = region.front();
  auto genericOps = llvm::to_vector(regionBlock.getOps<d2m::GenericOp>());
  TT_assertv(genericOps.size() == 1u,
             "each spatial region must contain exactly one d2m.generic op.");
  return genericOps.front();
}

// Hoist every op except d2m.generic out of each spatial region: prefix ops
// move before the spatial (forward moveBefore order) unless they read a region
// block argument (a spatial edge, e.g. a view reblocking it for the generic),
// suffix ops (after generic
// until an optional d2m.spatial_yield or block end) move after the spatial
// with generic-result uses rewired to spatial results. Any existing
// d2m.spatial_yield ops in the region are erased, then a fresh yield of the
// generic's results is created at the end of the region block.
static void hoistNonGenericOpsAroundSpatial(d2m::SpatialOp spatialOp) {
  mlir::OpBuilder builder(spatialOp.getContext());
  Operation *suffixInsertTail = spatialOp.getOperation();
  unsigned cumulativeGenericResults = 0;

  for (Region &spatialRegion : spatialOp->getRegions()) {
    d2m::GenericOp genericOp =
        getSingleGenericOpFromSpatialRegion(spatialRegion);
    Block &regionBlock = spatialRegion.front();

    llvm::SmallVector<Operation *> beforeOps;
    llvm::SmallVector<Operation *> afterOps;
    bool seenGeneric = false;
    for (Operation &regionBodyOp : regionBlock) {
      if (&regionBodyOp == genericOp.getOperation()) {
        seenGeneric = true;
        continue;
      }
      if (!seenGeneric) {
        beforeOps.push_back(&regionBodyOp);
      } else {
        if (mlir::isa<d2m::SpatialYieldOp>(&regionBodyOp)) {
          break;
        }
        afterOps.push_back(&regionBodyOp);
      }
    }

    for (d2m::SpatialYieldOp yieldOp :
         llvm::to_vector(regionBlock.getOps<d2m::SpatialYieldOp>())) {
      yieldOp.erase();
    }

    llvm::SmallPtrSet<Value, 4> fromEdges(regionBlock.args_begin(),
                                          regionBlock.args_end());
    for (Operation *prefixOp : beforeOps) {
      if (llvm::any_of(prefixOp->getOperands(),
                       [&](Value v) { return fromEdges.contains(v); })) {
        fromEdges.insert(prefixOp->result_begin(), prefixOp->result_end());
        continue;
      }
      prefixOp->moveBefore(spatialOp);
    }

    if (!afterOps.empty()) {
      const unsigned resultBase = cumulativeGenericResults;
      const unsigned numGenResults = genericOp->getNumResults();
      const unsigned numSpatialResults = spatialOp->getNumResults();
      TT_assertv(((numGenResults == 0u) ||
                  (resultBase + numGenResults <= numSpatialResults)),
                 "spatial op result count must cover generic results for "
                 "suffix hoist remapping");

      for (unsigned resultIndex = 0; resultIndex < numGenResults;
           ++resultIndex) {
        Value genericResult = genericOp.getResult(resultIndex);
        Value spatialResult = spatialOp.getResult(resultBase + resultIndex);
        genericResult.replaceUsesWithIf(spatialResult, [&](OpOperand &operand) {
          return llvm::is_contained(afterOps, operand.getOwner());
        });
      }

      for (Operation *suffixOp : afterOps) {
        for (Value operand : suffixOp->getOperands()) {
          TT_assertv(!llvm::is_contained(genericOp.getResults(), operand),
                     "suffix op operand should use spatial op results after "
                     "remapping");
        }
      }

      for (Operation *suffixOp : afterOps) {
        suffixOp->moveAfter(suffixInsertTail);
        suffixInsertTail = suffixOp;
      }
    }

    cumulativeGenericResults += genericOp->getNumResults();

    builder.setInsertionPointToEnd(&regionBlock);
    builder.create<d2m::SpatialYieldOp>(genericOp.getLoc(),
                                        genericOp.getResults());
  }
}

// Reassign d2m.spatial ins/outs from generics' operands; when result count
// matches outs, update each spatial result's type from the corresponding out
// value only when that out is a ranked tensor (spatial results are tensors;
// outs may be memref). Generic inputs that are the spatial's own region block
// arguments (edge sinks), or values defined in the region (views of them), are
// not visible outside the spatial and are skipped. Edges index generic outputs
// per region, so they need no remapping here.
static void rebuildSpatialOpInsOutsAndResultTypes(d2m::SpatialOp spatialOp) {
  llvm::SetVector<Value> inputs;
  llvm::SmallVector<Value> outputs;
  for (Region &region : spatialOp->getRegions()) {
    d2m::GenericOp genericOp = getSingleGenericOpFromSpatialRegion(region);
    for (Value input : genericOp.getInputs()) {
      auto blockArg = mlir::dyn_cast<BlockArgument>(input);
      if (blockArg && blockArg.getOwner()->getParentOp() == spatialOp) {
        continue;
      }
      if (Operation *def = input.getDefiningOp();
          def && spatialOp->isProperAncestor(def)) {
        continue;
      }
      inputs.insert(input);
    }
    for (Value output : genericOp.getOutputs()) {
      outputs.push_back(output);
    }
  }
  spatialOp.getInputsMutable().assign(inputs.getArrayRef());
  spatialOp.getOutputsMutable().assign(outputs);
  if (spatialOp->getNumResults() == outputs.size()) {
    for (auto [result, outVal] : llvm::zip(spatialOp->getResults(), outputs)) {
      if (mlir::isa<RankedTensorType>(outVal.getType())) {
        result.setType(outVal.getType());
      }
    }
  }
}

void normalizeSingleSpatialOp(d2m::SpatialOp spatialOp) {
  hoistNonGenericOpsAroundSpatial(spatialOp);
  rebuildSpatialOpInsOutsAndResultTypes(spatialOp);
}

} // namespace detail

void normalizeSpatialOpsInModule(ModuleOp module) {
  SmallVector<d2m::SpatialOp> spatials;
  module.walk([&](d2m::SpatialOp spatialOp) { spatials.push_back(spatialOp); });

  for (d2m::SpatialOp spatialOp : spatials) {
    detail::normalizeSingleSpatialOp(spatialOp);
  }
}

void normalizeSpatialOpContainingGeneric(GenericOp genericOp) {
  for (Region *r = genericOp->getParentRegion(); r;) {
    Operation *parent = r->getParentOp();
    if (!parent) {
      break;
    }
    if (auto spatial = mlir::dyn_cast<d2m::SpatialOp>(parent)) {
      detail::normalizeSingleSpatialOp(spatial);
      return;
    }
    r = parent->getParentRegion();
  }
}

} // namespace mlir::tt::d2m
