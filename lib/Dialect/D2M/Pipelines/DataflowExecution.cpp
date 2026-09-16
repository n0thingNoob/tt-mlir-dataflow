// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Pipelines/DataflowExecution.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"
#include <mutex>

namespace mlir::tt::d2m {
namespace {
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
      !requirements.blocking.empty() ||
      requirements.allowIntermediateOutputSpilling.has_value()) {
    result.status = DataflowAttemptStatus::Unsupported;
    result.reason = "unsupported target, starting stage or pass requirements";
    return result;
  }
  OwningOpRef<ModuleOp> working = cast<ModuleOp>(input->clone());
  ttmetal::D2MPipelineOptions localOptions;
  localOptions.copyOptionValuesFrom(options);
  localOptions.emitResourceReport = true;

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
  if (inputStage != DataflowStage::Allocation &&
      failed(runStage(DataflowStage::Allocation,
                      ttmetal::createD2MAllocationPipeline))) {
    return result;
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
