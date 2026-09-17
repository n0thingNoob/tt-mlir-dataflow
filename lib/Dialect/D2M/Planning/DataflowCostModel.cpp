// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Planning/DataflowCostModel.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <tuple>
#include <utility>

namespace mlir::tt::d2m {

static std::optional<uint64_t>
getProgramInitiationInterval(const DataflowProgramVariant &variant) {
  const ProgramCostEstimate &cost = variant.getCost();
  if (cost.initiationIntervalCycles) {
    return cost.initiationIntervalCycles;
  }
  if (!cost.computeCycles || !cost.dataMovementCycles) {
    return std::nullopt;
  }
  return std::max(*cost.computeCycles, *cost.dataMovementCycles);
}

static uint32_t getCoreCount(llvm::ArrayRef<int64_t> gridShape) {
  constexpr uint64_t maxCoreCount = std::numeric_limits<uint32_t>::max();
  uint64_t count = 1;
  for (int64_t dim : gridShape) {
    if (dim <= 0) {
      return 0;
    }
    uint64_t unsignedDim = static_cast<uint64_t>(dim);
    if (count > maxCoreCount / unsignedDim) {
      return static_cast<uint32_t>(maxCoreCount);
    }
    count *= unsignedDim;
  }
  return static_cast<uint32_t>(count);
}

static uint32_t saturatingAdd(uint32_t lhs, uint32_t rhs) {
  constexpr uint32_t max = std::numeric_limits<uint32_t>::max();
  return rhs > max - lhs ? max : lhs + rhs;
}

DataflowPlanCost
AnalyticalDataflowCostModel::evaluate(const DataflowGraph &,
                                      const DataflowMappingPlan &plan) const {
  DataflowPlanCost result;
  result.programCount = static_cast<uint32_t>(plan.getPrograms().size());
  result.confidence = 1.0F;

  bool allCyclesKnown = true;
  uint64_t cycleSum = 0;
  uint64_t maxInitiationInterval = 0;
  for (const DataflowPlannedProgram &program : plan.getPrograms()) {
    const DataflowProgramVariant &variant = program.variant;
    const ProgramResourceEstimate &resources = variant.getResources();
    result.dramBytes += resources.dramBytes;
    result.nocBytes += resources.nocBytes;
    result.peakL1BytesPerCore =
        std::max(result.peakL1BytesPerCore, resources.l1BytesPerCore);

    uint32_t coreCount = getCoreCount(variant.getGridShape());
    if (plan.getStrategy() == DataflowPlanStrategy::Spatial) {
      result.occupiedCores = saturatingAdd(result.occupiedCores, coreCount);
    } else {
      result.occupiedCores = std::max(result.occupiedCores, coreCount);
    }

    std::optional<uint64_t> initiationInterval =
        getProgramInitiationInterval(variant);
    if (!initiationInterval) {
      allCyclesKnown = false;
      result.confidence = 0.0F;
      continue;
    }
    cycleSum += *initiationInterval;
    maxInitiationInterval =
        std::max(maxInitiationInterval, *initiationInterval);
    result.confidence =
        std::min(result.confidence, variant.getCost().confidence);
  }

  if (allCyclesKnown) {
    if (plan.getStrategy() == DataflowPlanStrategy::Spatial) {
      result.initiationIntervalCycles = maxInitiationInterval;
    } else {
      result.latencyCycles = cycleSum;
    }
  }

  return result;
}

} // namespace mlir::tt::d2m
