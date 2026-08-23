// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTKernel/Transforms/Passes.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::tt::ttkernel {
#define GEN_PASS_DEF_TTKERNELHOISTINITS
#include "ttmlir/Dialect/TTKernel/Transforms/Passes.h.inc"

namespace {

class TTKernelFunctionRewriter : public OpRewritePattern<func::FuncOp> {
public:
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const final {
    DenseMap<scf::ForOp, SmallVector<OperationName>> initOps;

    op.walk([&initOps](Operation *op) {
      if (op->hasTrait<ttkernel::TTKernelInitOpTrait>()) {
        auto forParent = op->getParentOfType<scf::ForOp>();
        while (forParent) {
          auto forInitOps = initOps.lookup(forParent);
          initOps[forParent].push_back(op->getName());
          forParent = forParent->getParentOfType<scf::ForOp>();
        }
      }
    });

    op.walk([&](Operation *op) {
      if (op->hasTrait<ttkernel::TTKernelInitOpTrait>()) {
        Operation *highestLiftableLoop = nullptr;
        scf::ForOp curr = op->getParentOfType<scf::ForOp>();
        while (curr) {
          assert(initOps.contains(curr) &&
                 "Init op's parent loop should be in the initOps map.");
          auto currLoopInitOps = initOps.lookup(curr);
          assert(std::find(currLoopInitOps.begin(), currLoopInitOps.end(),
                           op->getName()) != currLoopInitOps.end() &&
                 "Init op should be inside the parent loop's initOps map.");

          // This condition should be smarter, in the sense that we should have
          // a lookup table of conflicting inits and detect whether we can keep
          // going. For now, assume all inits conflict.
          if (currLoopInitOps.size() == 1) {
            highestLiftableLoop = curr;
          } else {
            break;
          }
          curr = curr->getParentOfType<scf::ForOp>();
        }
        if (highestLiftableLoop) {
          rewriter.moveOpBefore(op, highestLiftableLoop);
        }
      }
    });

    return success();
  }
};

static void orderHardwareStartupBeforeSpecializedInits(func::FuncOp func) {
  // compute_kernel_hw_startup establishes the baseline unpack/pack/SFPU
  // configuration and must precede every specialized init. Moving several
  // init ops before the same outer loop otherwise leaves their order dependent
  // on walk/move order, which can place startup after init_sfpu and silently
  // reset it in fused kernels.
  Block &entry = func.getBody().front();
  ComputeKernelHWStartupOp startup;
  for (Operation &entryOp : entry) {
    if (auto candidate = dyn_cast<ComputeKernelHWStartupOp>(entryOp)) {
      startup = candidate;
    }
  }
  if (!startup) {
    return;
  }

  Operation *latestOperandDef = nullptr;
  for (Value operand : startup->getOperands()) {
    Operation *def = operand.getDefiningOp();
    if (!def || def->getBlock() != &entry) {
      continue;
    }
    if (!latestOperandDef || latestOperandDef->isBeforeInBlock(def)) {
      latestOperandDef = def;
    }
  }
  if (latestOperandDef && startup->getPrevNode() != latestOperandDef) {
    startup->moveAfter(latestOperandDef);
  }

  // Preserve the original order of direct entry-block init operations while
  // moving any specialized init that preceded startup to immediately after
  // it. All of their operands already dominate startup at this point.
  SmallVector<Operation *> precedingInits;
  for (Operation &entryOp : entry) {
    if (&entryOp == startup.getOperation()) {
      break;
    }
    if (entryOp.hasTrait<ttkernel::TTKernelInitOpTrait>()) {
      precedingInits.push_back(&entryOp);
    }
  }
  Operation *anchor = startup.getOperation();
  for (Operation *init : precedingInits) {
    init->moveAfter(anchor);
    anchor = init;
  }
}

} // namespace

namespace {
class TTKernelHoistInits
    : public impl::TTKernelHoistInitsBase<TTKernelHoistInits> {
public:
  using impl::TTKernelHoistInitsBase<
      TTKernelHoistInits>::TTKernelHoistInitsBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTKernelFunctionRewriter>(&getContext());
    walkAndApplyPatterns(getOperation(), std::move(patterns));
    orderHardwareStartupBeforeSpecializedInits(getOperation());
  }
};
} // namespace

} // namespace mlir::tt::ttkernel
