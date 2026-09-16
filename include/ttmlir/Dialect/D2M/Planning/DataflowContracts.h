// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_PLANNING_DATAFLOWCONTRACTS_H
#define TTMLIR_DIALECT_D2M_PLANNING_DATAFLOWCONTRACTS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include <memory>
#include <optional>
#include <string>

namespace mlir::tt::d2m {

enum class DataflowStage { Input, Blocking, Allocation, Backend };

/// A selector is interpreted only at the declared blocking boundary. It is
/// not an identity transported from TTIR through fusion and layout lowering.
struct DataflowBlockingRequest {
  std::string function;
  unsigned genericOrdinal = 0;
  SmallVector<int64_t> factors;
};

/// Frontends, including a future DSL, supply requirements without IR pointers.
/// Unspecified decisions remain owned by the existing pipeline.
struct DataflowRequirements {
  SmallVector<DataflowBlockingRequest> blocking;
  std::optional<bool> allowIntermediateOutputSpilling;
};

class DataflowStagePlan;

class DataflowBinding {
  friend class DataflowStagePlan;
  std::shared_ptr<const int> owner;
  std::string function;
  unsigned ordinal = 0;
};

/// Owns a read-only snapshot. Bindings can be resolved against an equivalent
/// clone, never against rewritten IR or another stage/plan. This conservative
/// v1 contract intentionally trades comparison cost for explicit invalidation.
class DataflowStagePlan {
public:
  DataflowStagePlan(ModuleOp module, DataflowStage stage);
  FailureOr<DataflowBinding> bind(StringRef function, unsigned ordinal,
                                  std::string &reason) const;
  FailureOr<GenericOp> resolve(const DataflowBinding &binding, ModuleOp module,
                               DataflowStage stage, std::string &reason) const;

private:
  OwningOpRef<ModuleOp> snapshot;
  DataflowStage stage;
  std::shared_ptr<const int> identity = std::make_shared<const int>(0);
};

} // namespace mlir::tt::d2m
#endif
