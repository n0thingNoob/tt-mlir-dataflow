// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_PIPELINES_DATAFLOWEXECUTION_H
#define TTMLIR_DIALECT_D2M_PIPELINES_DATAFLOWEXECUTION_H

#include "ttmlir/Dialect/D2M/Analysis/DataflowAllocationFeedback.h"
#include "ttmlir/Dialect/D2M/Pipelines/D2MPipelines.h"
#include "ttmlir/Dialect/D2M/Planning/DataflowContracts.h"
#include "ttmlir/Dialect/D2M/Planning/DataflowCostModel.h"

namespace mlir::tt::d2m {

enum class DataflowAttemptStatus { Accepted, Rejected, Unsupported, Error };

struct DataflowFunctionFeedback {
  std::string function;
  DataflowAllocationFeedback allocation;
};

struct DataflowStageReport {
  DataflowStage stage;
  bool completed = false;
  std::string failedPass;
  std::string diagnostic;
  SmallVector<DataflowFunctionFeedback> functions;
};

/// Evaluates observations only. This optional policy cannot override failure
/// or decide correctness. Allocation usage is not a transfer-traffic estimate.
class DataflowRealizationCostModel {
public:
  virtual ~DataflowRealizationCostModel() = default;
  virtual DataflowPlanCost
  evaluate(ArrayRef<DataflowStageReport> reports) const = 0;
};

struct DataflowAttemptResult {
  DataflowAttemptStatus status = DataflowAttemptStatus::Error;
  std::string reason;
  SmallVector<DataflowStageReport> reports;
  std::optional<DataflowPlanCost> cost;
  // Present only for Accepted. This is the verified artifact, not a plan to
  // replay against the caller's original module.
  OwningOpRef<ModuleOp> artifact;
};

/// Run through D2M backend on a clone of a device-body module (no host
/// wrapper). Input is TTIR; Blocking is the preparation pipeline output;
/// Allocation is the allocation pipeline output including explicit-form
/// lowering. Accepted means D2M backend success, not serialization, device
/// correctness or speedup.
DataflowAttemptResult
runDataflowAttempt(ModuleOp input, DataflowStage inputStage,
                   const ttmetal::D2MPipelineOptions &options,
                   const DataflowRequirements &requirements = {},
                   const DataflowRealizationCostModel *costModel = nullptr);

} // namespace mlir::tt::d2m
#endif
