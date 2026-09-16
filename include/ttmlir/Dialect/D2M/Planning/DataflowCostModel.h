// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_PLANNING_DATAFLOWCOSTMODEL_H
#define TTMLIR_DIALECT_D2M_PLANNING_DATAFLOWCOSTMODEL_H

#include "ttmlir/Dialect/D2M/Planning/DataflowPlan.h"
#include <cstdint>
#include <optional>

namespace mlir::tt::d2m {

struct DataflowPlanCost {
  std::optional<uint64_t> latencyCycles;
  std::optional<uint64_t> initiationIntervalCycles;
  std::optional<uint64_t> dramBytes;
  std::optional<uint64_t> nocBytes;
  std::optional<uint64_t> peakL1BytesPerCore;
  uint32_t occupiedCores = 0;
  std::optional<uint32_t> spillCount;
  uint32_t programCount = 0;
  float confidence = 0.0F;
};

class DataflowCostModel {
public:
  virtual ~DataflowCostModel() = default;

  virtual DataflowPlanCost evaluate(const DataflowGraph &graph,
                                    const DataflowMappingPlan &plan) const = 0;
};

/// Aggregates optional per-program estimates without inventing unavailable
/// cycle data. Temporal latency and spatial initiation interval are reported
/// only when every participating program has a cycle estimate.
class AnalyticalDataflowCostModel final : public DataflowCostModel {
public:
  DataflowPlanCost evaluate(const DataflowGraph &graph,
                            const DataflowMappingPlan &plan) const override;
};

} // namespace mlir::tt::d2m

#endif
