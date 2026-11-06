// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/Analysis/BlockFactorAnalysis.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/StringSwitch.h"

namespace mlir::tt::d2m {

#define GEN_PASS_DEF_D2MREBLOCKGENERICS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// A streaming spatial edge with `block = RxC` asks for blocks of R x C tiles
// per core at both ends: for a generic in a spatial region, the loop dims
// that index such an edge operand get factor (shard extent / block extent).
// Ends that cannot take the block keep `factors`;
// d2m-materialize-spatial-streams then decides on the blocks it finds.
static void applyStreamBlockHints(GenericOp generic,
                                  SmallVector<int64_t> &factors) {
  auto spatial = mlir::dyn_cast_if_present<SpatialOp>(generic->getParentOp());
  if (!spatial || !spatial.getEdgesAttr()) {
    return;
  }
  unsigned region = generic->getParentRegion()->getRegionNumber();
  for (auto edge : spatial.getEdgesAttr().getAsRange<SpatialEdgeAttr>()) {
    if (edge.getStreamBlock().empty()) {
      continue;
    }
    Value buffer = spatial.getEdgeBuffer(edge);
    for (OpOperand &operand : generic->getOpOperands()) {
      bool isEnd =
          (edge.getProducerRegion() == region && generic.isDpsInit(&operand) &&
           generic.getDpsInitsMutable()[edge.getProducerOutput()]
                   .getOperandNumber() == operand.getOperandNumber()) ||
          (edge.getConsumerRegion() == region && generic.isDpsInput(&operand) &&
           SpatialOp::stripViews(operand.get()) == buffer);
      if (!isEnd) {
        continue;
      }
      auto type = mlir::dyn_cast<MemRefType>(operand.get().getType());
      AffineMap map = generic.getIndexingMap(operand.getOperandNumber());
      ArrayRef<int64_t> block = edge.getStreamBlock();
      if (!type || type.getRank() != 4 || map.getNumResults() != 2) {
        continue;
      }
      ArrayRef<int64_t> shard = type.getShape().drop_front(2);
      SmallVector<int64_t> hinted = factors;
      bool fits = true;
      for (unsigned r = 0; r < 2; ++r) {
        auto dim = mlir::dyn_cast<AffineDimExpr>(map.getResult(r));
        if (!dim || shard[r] % block[r] != 0) {
          fits = false;
          break;
        }
        hinted[dim.getPosition()] = shard[r] / block[r];
      }
      if (fits) {
        factors = hinted;
      }
    }
  }
}

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
    BlockFactorAnalysis blockFactorAnalysis(funcOp, bfOpts);

    SmallVector<GenericOp> genericOps;
    funcOp.getBody().front().walk(
        [&](GenericOp genericOp) { genericOps.push_back(genericOp); });

    // The factors of every generic first, so that a load can follow its
    // consumers' blocks (alignLoadsToStreams).
    llvm::DenseMap<Operation *, SmallVector<int64_t>> factorsOf;
    for (GenericOp generic : genericOps) {
      const BlockFactorAnalysis::Result *bfResult =
          blockFactorAnalysis.lookup(generic);
      SmallVector<int64_t> factors = bfResult ? bfResult->reblockedFactors
                                              : generic.getBlockFactorsValue();
      applyStreamBlockHints(generic, factors);
      factorsOf[generic] = factors;
    }
    alignLoadsToStreams(genericOps, factorsOf);

    for (GenericOp oldGenericOp : genericOps) {
      SmallVector<int64_t> oldBlockFactors =
          oldGenericOp.getBlockFactorsValue();
      SmallVector<int64_t> factors = factorsOf.lookup(oldGenericOp);
      if (factors.empty() || oldBlockFactors == factors) {
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

      if (oldGenericOp.getNumResults() > 0) {
        TT_assertv(oldGenericOp.getOutputs().size() == 1u,
                   "reblocking a generic with results expects a single "
                   "output operand");
        oldGenericOp.getResult(0).replaceAllUsesWith(
            reblocked->returnView.getResult());
        rewriter.eraseOp(oldGenericOp);
        continue;
      }
      // Buffer form: every output gets a view of the reblocked output back to
      // its old type, and the uses after the generic read that view. (Only a
      // data-movement load has more than one output.)
      Operation *sequenceAnchor = reblocked->returnView.getOperation();
      auto getContainingOpInBlock = [&](Operation *op) -> Operation * {
        Operation *current = op;
        while (current && current->getBlock() != sequenceAnchor->getBlock()) {
          current = current->getParentOp();
        }
        return current;
      };
      for (auto [index, oldOutput] :
           llvm::enumerate(oldGenericOp.getOutputs())) {
        Value newOutput = reblocked->returnView.getResult();
        if (index > 0) {
          rewriter.setInsertionPointAfter(sequenceAnchor);
          auto view = rewriter.create<ViewLayoutOp>(
              oldGenericOp.getLoc(), oldOutput.getType(),
              reblocked->genericOp.getOutputs()[index]);
          sequenceAnchor = view;
          newOutput = view.getResult();
        }
        oldOutput.replaceUsesWithIf(newOutput, [&](OpOperand &use) {
          Operation *ownerInBlock = getContainingOpInBlock(use.getOwner());
          return ownerInBlock && ownerInBlock != oldGenericOp.getOperation() &&
                 sequenceAnchor->isBeforeInBlock(ownerInBlock);
        });
      }

      rewriter.eraseOp(oldGenericOp);
    }

    return success();
  }

  // A data-movement generic (only remote loads and stores, such as a
  // spatial-loads load region) that produces a streaming edge without a
  // `block` hint takes the blocks its consumer reads: along each copy dim,
  // (its shard extent / the consumer's block extent).
  void alignLoadsToStreams(
      ArrayRef<GenericOp> generics,
      llvm::DenseMap<Operation *, SmallVector<int64_t>> &factorsOf) {
    for (GenericOp producer : generics) {
      auto spatial =
          mlir::dyn_cast_if_present<SpatialOp>(producer->getParentOp());
      if (!spatial || !spatial.getEdgesAttr() ||
          !isDataMovementOnly(producer)) {
        continue;
      }
      unsigned region = producer->getParentRegion()->getRegionNumber();
      SmallVector<int64_t> &factors = factorsOf[producer];
      for (auto edge : spatial.getEdgesAttr().getAsRange<SpatialEdgeAttr>()) {
        if (edge.getProducerRegion() != region || !edge.isStream() ||
            !edge.getStreamBlock().empty()) {
          continue;
        }
        GenericOp consumer = *spatial.getRegion(edge.getConsumerRegion())
                                  .getOps<GenericOp>()
                                  .begin();
        Value buffer = spatial.getEdgeBuffer(edge);
        OpOperand *read = nullptr;
        for (OpOperand &operand : consumer->getOpOperands()) {
          if (consumer.isDpsInput(&operand) &&
              SpatialOp::stripViews(operand.get()) == buffer) {
            read = &operand;
          }
        }
        OpOperand &written =
            producer.getDpsInitsMutable()[edge.getProducerOutput()];
        if (!read || !factorsOf.contains(consumer)) {
          continue;
        }
        auto readType = mlir::dyn_cast<MemRefType>(read->get().getType());
        auto writtenType = mlir::dyn_cast<MemRefType>(written.get().getType());
        AffineMap readMap = consumer.getIndexingMap(read->getOperandNumber());
        AffineMap writtenMap =
            producer.getIndexingMap(written.getOperandNumber());
        if (!readType || !writtenType || readType.getRank() != 4 ||
            writtenType.getRank() != 4 || readMap.getNumResults() != 2 ||
            writtenMap.getNumResults() != 2) {
          continue;
        }
        ArrayRef<int64_t> consumerFactors = factorsOf[consumer];
        SmallVector<int64_t> aligned = factors;
        bool fits = true;
        for (unsigned r = 0; r < 2; ++r) {
          auto readDim = mlir::dyn_cast<AffineDimExpr>(readMap.getResult(r));
          auto writtenDim =
              mlir::dyn_cast<AffineDimExpr>(writtenMap.getResult(r));
          if (!readDim || !writtenDim) {
            fits = false;
            break;
          }
          int64_t readShard = readType.getShape()[2 + r];
          int64_t writtenShard = writtenType.getShape()[2 + r];
          int64_t blocks = consumerFactors[readDim.getPosition()];
          if (blocks <= 0 || readShard % blocks != 0 ||
              writtenShard % (readShard / blocks) != 0) {
            fits = false;
            break;
          }
          aligned[writtenDim.getPosition()] =
              writtenShard / (readShard / blocks);
        }
        if (fits) {
          factors = aligned;
        }
      }
    }
  }

  static bool isDataMovementOnly(GenericOp generic) {
    bool computes = false;
    generic->walk([&](Operation *op) {
      if (op != generic.getOperation() &&
          (mlir::isa<linalg::GenericOp>(op) ||
           op->getName().getStringRef().starts_with("d2m.tile_"))) {
        computes = true;
      }
    });
    return !computes;
  }
};

} // namespace

} // namespace mlir::tt::d2m
