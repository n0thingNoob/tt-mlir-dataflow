// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/Analysis/DataflowAllocationFeedback.h"

#include "llvm/Support/raw_ostream.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MDATAFLOWALLOCATIONFEEDBACK
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
class D2MDataflowAllocationFeedbackPass final
    : public impl::D2MDataflowAllocationFeedbackBase<
          D2MDataflowAllocationFeedbackPass> {
public:
  using Base = impl::D2MDataflowAllocationFeedbackBase<
      D2MDataflowAllocationFeedbackPass>;
  using Base::Base;

  void runOnOperation() override {
    SmallVector<func::FuncOp> functions;
    WalkResult result = getOperation().walk([&](func::FuncOp function) {
      if (function.isDeclaration()) {
        return WalkResult::advance();
      }
      std::string reason;
      auto feedback = readDataflowAllocationFeedback(
          function->getAttrOfType<DictionaryAttr>("d2m.allocation_report"),
          reason);
      if (failed(feedback)) {
        function.emitOpError() << reason;
        return WalkResult::interrupt();
      }
      if (feedback->status != "success") {
        function.emitOpError()
            << "allocation did not succeed: " << feedback->status;
        return WalkResult::interrupt();
      }
      if (dumpPlan) {
        llvm::errs() << "d2m-dataflow-allocation function=@"
                     << function.getSymName() << " status=" << feedback->status
                     << " l1-usage=" << *feedback->l1UsageBytes
                     << " dram-usage=" << *feedback->dramUsageBytes
                     << " l1-to-dram=" << *feedback->l1ToDramCount << "\n";
      }
      functions.push_back(function);
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }
    if (!keepReport) {
      for (auto function : functions) {
        function->removeAttr("d2m.allocation_report");
      }
    }
  }
};
} // namespace
} // namespace mlir::tt::d2m
