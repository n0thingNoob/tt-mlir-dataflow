// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/Analysis/BlockFactorAnalysis.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/StringSwitch.h"

namespace mlir::tt::d2m {

#define GEN_PASS_DEF_D2MREBLOCKGENERICS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

class D2MReblockGenerics final
    : public impl::D2MReblockGenericsBase<D2MReblockGenerics> {
  using Base = impl::D2MReblockGenericsBase<D2MReblockGenerics>;
  using Base::Base;
  using BufferSizePolicy = BlockFactorAnalysis::BufferSizePolicy;

  static std::optional<BufferSizePolicy>
  parseBufferSizePolicy(StringRef policy) {
    return llvm::StringSwitch<std::optional<BufferSizePolicy>>(policy)
        .Case("auto", BufferSizePolicy::Auto)
        .Case("bounded", BufferSizePolicy::Bounded)
        .Case("auto-mn", BufferSizePolicy::AutoMN)
        .Case("min", BufferSizePolicy::Min)
        .Case("max", BufferSizePolicy::Max)
        .Default(std::nullopt);
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    std::optional<BufferSizePolicy> parsedBufferSizePolicy =
        parseBufferSizePolicy(testBufferSizePolicy);
    if (!parsedBufferSizePolicy) {
      moduleOp.emitOpError()
          << "invalid test-buffer-size-policy '" << testBufferSizePolicy
          << "' (expected one of: auto, bounded, auto-mn, min, max)";
      return signalPassFailure();
    }

    BlockFactorAnalysis::Options bfOpts;
    bfOpts.policy = *parsedBufferSizePolicy;
    bfOpts.numBuffers = numStreamBuffers;
    bfOpts.useExplicitBlockFactors = useExplicitBlockFactors;

    if (moduleOp
            ->walk([&](func::FuncOp funcOp) -> WalkResult {
              if (funcOp.isDeclaration()) {
                return WalkResult::advance();
              }
              if (failed(reblockGenerics(funcOp, bfOpts))) {
                return WalkResult::interrupt();
              }
              return WalkResult::advance();
            })
            .wasInterrupted()) {
      signalPassFailure();
    }
  }

  LogicalResult reblockGenerics(func::FuncOp funcOp,
                                const BlockFactorAnalysis::Options &bfOpts) {
    IRRewriter rewriter(funcOp->getContext());

    SmallVector<GenericOp> genericOps;
    auto collect = [&](GenericOp genericOp) {
      genericOps.push_back(genericOp);
    };
    if (useExplicitBlockFactors) {
      funcOp.walk(collect);
    } else {
      funcOp.getBody().front().walk(collect);
    }

    // Validate all requests before rebuilding any generic in this function.
    llvm::DenseMap<Operation *, SmallVector<int64_t>> plannedFactors;
    if (useExplicitBlockFactors) {
      for (GenericOp genericOp : genericOps) {
        Attribute attr =
            genericOp->getAttr(BlockFactorAnalysis::explicitFactorsAttrName);
        if (!attr) {
          continue;
        }
        auto factors = dyn_cast<DenseI64ArrayAttr>(attr);
        if (!factors) {
          return genericOp.emitOpError()
                 << "d2m.planned_block_factors must be an array<i64>";
        }
        std::string reason;
        if (failed(BlockFactorAnalysis::validateExplicitFactors(
                genericOp, factors.asArrayRef(), bfOpts.numBuffers, reason))) {
          return genericOp.emitOpError()
                 << "invalid d2m.planned_block_factors: " << reason;
        }
        plannedFactors[genericOp] = llvm::to_vector(factors.asArrayRef());
      }
    }

    BlockFactorAnalysis blockFactorAnalysis(funcOp, bfOpts);
    for (GenericOp oldGenericOp : genericOps) {
      const BlockFactorAnalysis::Result *bfResult =
          blockFactorAnalysis.lookup(oldGenericOp);
      auto planned = plannedFactors.find(oldGenericOp);
      if (!bfResult && planned == plannedFactors.end()) {
        continue;
      }
      ArrayRef<int64_t> factors = planned != plannedFactors.end()
                                      ? planned->second
                                      : bfResult->reblockedFactors;

      SmallVector<int64_t> oldBlockFactors =
          oldGenericOp.getBlockFactorsValue();
      if (ArrayRef<int64_t>(oldBlockFactors) == factors) {
        continue;
      }

      rewriter.setInsertionPoint(oldGenericOp);
      FailureOr<ParallelizedGeneric> reblocked =
          oldGenericOp.withParallelization(rewriter, std::nullopt, factors,
                                           /*generateReturnView=*/true);
      if (failed(reblocked)) {
        oldGenericOp.emitOpError()
            << "failed to rebuild generic op with updated block factors";
        return failure();
      }

      TT_assertv(oldGenericOp.getOutputs().size() == 1u,
                 "reblocking expects a single output operand");
      Operation *sequenceAnchor = reblocked->returnView.getOperation();
      Value newOutput = reblocked->returnView.getResult();

      if (oldGenericOp.getNumResults() > 0) {
        TT_assert(oldGenericOp.getNumResults() == 1u);
        oldGenericOp.getResult(0).replaceAllUsesWith(newOutput);
      } else {
        auto getContainingOpInBlock = [&](Operation *op) -> Operation * {
          Operation *current = op;
          while (current && current->getBlock() != sequenceAnchor->getBlock()) {
            current = current->getParentOp();
          }
          return current;
        };
        oldGenericOp.getOutputs().front().replaceUsesWithIf(
            newOutput, [&](OpOperand &use) {
              Operation *ownerInBlock = getContainingOpInBlock(use.getOwner());
              return ownerInBlock &&
                     ownerInBlock != oldGenericOp.getOperation() &&
                     sequenceAnchor->isBeforeInBlock(ownerInBlock);
            });
      }

      rewriter.eraseOp(oldGenericOp);
    }

    return success();
  }
};

} // namespace

} // namespace mlir::tt::d2m
