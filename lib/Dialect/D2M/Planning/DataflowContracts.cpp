// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Planning/DataflowContracts.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OperationSupport.h"

namespace mlir::tt::d2m {

static FailureOr<GenericOp> lookup(ModuleOp module, StringRef name,
                                   unsigned ordinal, std::string &reason) {
  auto function = module.lookupSymbol<func::FuncOp>(name);
  if (!function || function.isDeclaration() ||
      !llvm::hasSingleElement(function.getBody())) {
    reason = "expected a defined single-block function";
    return failure();
  }
  for (GenericOp generic : function.getBody().front().getOps<GenericOp>()) {
    if (ordinal-- == 0) {
      return generic;
    }
  }
  reason = "unknown top-level generic ordinal";
  return failure();
}

DataflowStagePlan::DataflowStagePlan(ModuleOp module, DataflowStage stage)
    : snapshot(cast<ModuleOp>(module->clone())), stage(stage) {}

FailureOr<DataflowBinding> DataflowStagePlan::bind(StringRef function,
                                                   unsigned ordinal,
                                                   std::string &reason) const {
  reason.clear();
  if (failed(lookup(*snapshot, function, ordinal, reason))) {
    return failure();
  }
  DataflowBinding binding;
  binding.owner = identity;
  binding.function = function.str();
  binding.ordinal = ordinal;
  return binding;
}

FailureOr<GenericOp> DataflowStagePlan::resolve(const DataflowBinding &binding,
                                                ModuleOp module,
                                                DataflowStage atStage,
                                                std::string &reason) const {
  reason.clear();
  if (binding.owner != identity || atStage != stage) {
    reason = "binding belongs to another plan or stage";
    return failure();
  }
  if (!OperationEquivalence::isEquivalentTo(
          snapshot.get().getOperation(), module,
          OperationEquivalence::IgnoreLocations)) {
    reason = "stage IR changed; rebuild the plan and bindings";
    return failure();
  }
  return lookup(module, binding.function, binding.ordinal, reason);
}

} // namespace mlir::tt::d2m
