// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Pipelines/DataflowExecution.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "ttmlir/Dialect/D2M/Analysis/BlockFactorAnalysis.h"
#include "llvm/Support/raw_ostream.h"
#include <mutex>

namespace mlir::tt::d2m {
namespace {
constexpr llvm::StringLiteral requirementIdAttr = "d2m.execution_requirement";

class CheckBlockingRequirements
    : public PassWrapper<CheckBlockingRequirements, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CheckBlockingRequirements)
  explicit CheckBlockingRequirements(const DataflowRequirements &requirements)
      : requirements(requirements) {}
  StringRef getArgument() const final {
    return "d2m-check-blocking-requirements";
  }
  void runOnOperation() override {
    SmallVector<bool> seen(requirements.blocking.size(), false);
    bool valid = true;
    getOperation().walk([&](GenericOp generic) {
      auto attr = generic->getAttrOfType<IntegerAttr>(requirementIdAttr);
      if (!attr) {
        return;
      }
      auto id = attr.getInt();
      if (id < 0 || static_cast<size_t>(id) >= seen.size() || seen[id] ||
          generic.getBlockFactorsValue() != requirements.blocking[id].factors) {
        generic.emitOpError("blocking requirement was not preserved");
        valid = false;
        return;
      }
      seen[id] = true;
      generic->removeAttr(requirementIdAttr);
    });
    if (!valid || llvm::is_contained(seen, false)) {
      getOperation().emitError(
          "missing or changed blocking requirement after reblocking");
      signalPassFailure();
    }
  }

private:
  DataflowRequirements requirements;
};

LogicalResult
applyBlockingRequirements(ModuleOp module,
                          const DataflowRequirements &requirements,
                          uint32_t numBuffers, std::string &reason) {
  bool reserved = false;
  module.walk(
      [&](Operation *op) { reserved |= op->hasAttr(requirementIdAttr); });
  if (reserved) {
    reason = "input contains reserved d2m.execution_requirement metadata";
    return failure();
  }
  DataflowStagePlan plan(module, DataflowStage::Blocking);
  SmallVector<GenericOp> targets;
  // Resolve every binding before mutation: attaching the first attribute
  // invalidates the original snapshot for subsequent resolutions.
  for (const auto &request : requirements.blocking) {
    auto binding = plan.bind(request.function, request.genericOrdinal, reason);
    if (failed(binding)) {
      return failure();
    }
    auto generic =
        plan.resolve(*binding, module, DataflowStage::Blocking, reason);
    if (failed(generic)) {
      return failure();
    }
    if (llvm::is_contained(targets, *generic)) {
      reason = "duplicate blocking request for the same generic";
      return failure();
    }
    auto expected =
        DenseI64ArrayAttr::get(module.getContext(), request.factors);
    if (auto previous =
            (*generic)->getAttr(BlockFactorAnalysis::explicitFactorsAttrName);
        previous && previous != expected) {
      reason = "blocking request conflicts with existing explicit factors";
      return failure();
    }
    if (failed(BlockFactorAnalysis::validateExplicitFactors(
            *generic, request.factors, numBuffers, reason))) {
      return failure();
    }
    targets.push_back(*generic);
  }
  Builder builder(module.getContext());
  for (auto [id, target] : llvm::enumerate(targets)) {
    target->setAttr(requirementIdAttr, builder.getI64IntegerAttr(id));
    target->setAttr(
        BlockFactorAnalysis::explicitFactorsAttrName,
        builder.getDenseI64ArrayAttr(requirements.blocking[id].factors));
  }
  return success();
}

class FailureRecorder : public PassInstrumentation {
public:
  FailureRecorder(DataflowStageReport &report, std::mutex &mutex)
      : report(report), mutex(mutex) {}
  void runAfterPassFailed(Pass *pass, Operation *) override {
    std::lock_guard<std::mutex> lock(mutex);
    if (report.failedPass.empty()) {
      report.failedPass = pass->getArgument().str();
    }
  }

private:
  DataflowStageReport &report;
  std::mutex &mutex;
};
} // namespace

DataflowAttemptResult
runDataflowAttempt(ModuleOp input, DataflowStage inputStage,
                   const ttmetal::D2MPipelineOptions &options,
                   const DataflowRequirements &requirements,
                   const DataflowRealizationCostModel *costModel) {
  DataflowAttemptResult result;
  if (options.ttnnMode ||
      (inputStage != DataflowStage::Input &&
       inputStage != DataflowStage::Blocking &&
       inputStage != DataflowStage::Allocation) ||
      (inputStage == DataflowStage::Allocation &&
       (!requirements.blocking.empty() ||
        requirements.allowIntermediateOutputSpilling.has_value()))) {
    result.status = DataflowAttemptStatus::Unsupported;
    result.reason = "unsupported target, starting stage or pass requirements";
    return result;
  }
  if (inputStage != DataflowStage::Input &&
      !input->getAttrOfType<ttcore::SystemDescAttr>(
          ttcore::SystemDescAttr::name)) {
    result.status = DataflowAttemptStatus::Unsupported;
    result.reason = "prepared input must carry its system descriptor";
    return result;
  }
  OwningOpRef<ModuleOp> working = cast<ModuleOp>(input->clone());
  ttmetal::D2MPipelineOptions localOptions;
  localOptions.copyOptionValuesFrom(options);
  localOptions.emitResourceReport = true;
  if (requirements.allowIntermediateOutputSpilling) {
    localOptions.allowL1OutputSpilling =
        *requirements.allowIntermediateOutputSpilling;
  }
  if (!requirements.blocking.empty()) {
    localOptions.useExplicitBlockFactors = true;
  }

  auto runStage = [&](DataflowStage stage, auto populate) -> LogicalResult {
    result.reports.push_back({stage, false, {}, {}, {}});
    auto &report = result.reports.back();
    std::mutex diagnosticMutex;
    ScopedDiagnosticHandler handler(input.getContext(), [&](Diagnostic &diag) {
      std::lock_guard<std::mutex> lock(diagnosticMutex);
      llvm::raw_string_ostream stream(report.diagnostic);
      diag.print(stream);
      stream << '\n';
      return success();
    });
    PassManager pm(input.getContext());
    pm.addInstrumentation(
        std::make_unique<FailureRecorder>(report, diagnosticMutex));
    populate(pm, localOptions);
    bool passed = succeeded(pm.run(*working));
    if (stage == DataflowStage::Allocation) {
      working->walk([&](func::FuncOp function) {
        auto attr =
            function->getAttrOfType<DictionaryAttr>("d2m.allocation_report");
        if (!attr) {
          if (passed && !function.isDeclaration()) {
            passed = false;
            report.diagnostic += "missing allocation feedback for " +
                                 function.getSymName().str();
          }
          return;
        }
        std::string reason;
        auto feedback = readDataflowAllocationFeedback(attr, reason);
        if (failed(feedback)) {
          passed = false;
          report.diagnostic += reason;
          return;
        }
        report.functions.push_back({function.getSymName().str(), *feedback});
        if (feedback->status == "success" &&
            requirements.allowIntermediateOutputSpilling == false &&
            (!feedback->intermediateOutputSpillCount ||
             *feedback->intermediateOutputSpillCount != 0)) {
          passed = false;
          result.status = DataflowAttemptStatus::Rejected;
          report.diagnostic +=
              "intermediate output spill prohibition could not be verified";
        }
        if (!options.emitResourceReport) {
          function->removeAttr("d2m.allocation_report");
        }
      });
    }
    report.completed = passed;
    if (!passed) {
      result.reason =
          report.diagnostic.empty()
              ? "pipeline failed without a diagnostic: " + report.failedPass
              : report.diagnostic;
      for (const auto &function : report.functions) {
        if (function.allocation.status == "l1_capacity_exceeded" ||
            function.allocation.status == "dram_capacity_exceeded") {
          result.status = DataflowAttemptStatus::Rejected;
        }
      }
      return failure();
    }
    return success();
  };

  if (inputStage == DataflowStage::Input &&
      failed(runStage(DataflowStage::Blocking,
                      ttmetal::createD2MFrontendPreparationPipeline))) {
    return result;
  }
  if (inputStage != DataflowStage::Allocation) {
    if (failed(applyBlockingRequirements(*working, requirements,
                                         localOptions.numStreamBuffers,
                                         result.reason))) {
      result.status = DataflowAttemptStatus::Unsupported;
      return result;
    }
    auto allocation = [&](OpPassManager &pm,
                          const ttmetal::D2MPipelineOptions &opts) {
      ttmetal::createD2MReblockingPipeline(pm, opts);
      if (!requirements.blocking.empty()) {
        pm.addPass(std::make_unique<CheckBlockingRequirements>(requirements));
      }
      ttmetal::createD2MMemoryAllocationPipeline(pm, opts);
    };
    if (failed(runStage(DataflowStage::Allocation, allocation))) {
      return result;
    }
  }
  if (failed(runStage(DataflowStage::Backend,
                      ttmetal::createD2MBackendPipeline))) {
    return result;
  }
  result.status = DataflowAttemptStatus::Accepted;
  if (costModel) {
    result.cost = costModel->evaluate(result.reports);
  }
  result.artifact = std::move(working);
  return result;
}
} // namespace mlir::tt::d2m
