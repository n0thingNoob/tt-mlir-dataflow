// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/SpatialPipeline.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MMATERIALIZESPATIALPIPELINES
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"
namespace {

// Buffer-form views and allocations can be hoisted without executing a stage.
static bool collectDefinitions(Value value, Operation *anchor,
                               llvm::SetVector<Operation *> &preparations) {
  Operation *def = value.getDefiningOp();
  if (!def || def->getBlock() != anchor->getBlock() ||
      def->isBeforeInBlock(anchor)) {
    return true;
  }
  if (def->getNumRegions() ||
      (!isa<memref::AllocOp>(def) && !isMemoryEffectFree(def))) {
    return false;
  }
  for (Value operand : def->getOperands()) {
    if (!collectDefinitions(operand, anchor, preparations)) {
      return false;
    }
  }
  preparations.insert(def);
  return true;
}

struct ReadyEdge {
  unsigned producer;
  unsigned consumer;
  RemoteLoadOp load;
};

static Value storageRoot(Value value) {
  while (auto view = value.getDefiningOp<ViewLayoutOp>()) {
    value = view.getInput();
  }
  return value;
}

// Validate the deferred contract before hoisting buffers or creating
// semaphores.
static LogicalResult collectReadyEdges(ArrayRef<GenericOp> stages,
                                       SmallVectorImpl<ReadyEdge> &edges) {
  for (unsigned id = 0; id < stages.size(); ++id) {
    GenericOp generic = stages[id];
    auto stageId = generic->getAttrOfType<IntegerAttr>(spatial_pipeline::stage);
    auto core =
        generic->getAttrOfType<ttcore::CoreRangeAttr>(spatial_pipeline::core);
    auto producers =
        generic->getAttrOfType<DenseI64ArrayAttr>(spatial_pipeline::inputs);
    if (!stageId || stageId.getInt() != id || !core || !producers ||
        static_cast<size_t>(producers.size()) != generic.getInputs().size() ||
        generic.getOutputs().size() != 1 ||
        generic.getBlockFactorsValue().size() != 2) {
      return generic.emitOpError("invalid prepared spatial pipeline contract");
    }
    for (auto [inputId, producerId] : llvm::enumerate(producers.asArrayRef())) {
      if (producerId == -1) {
        continue;
      }
      if (producerId < 0 || producerId >= id) {
        return generic.emitOpError(
            "pipeline producer must precede its consumer");
      }
      Value input = generic.getInputs()[inputId];
      GenericOp producer = stages[producerId];
      if (storageRoot(input) != storageRoot(producer.getOutputs()[0]) ||
          ttcore::getMemorySpace(input) != ttcore::MemorySpace::DeviceL1) {
        return generic.emitOpError(
            "pipeline edge must share the producer's L1 storage");
      }
      SmallVector<RemoteLoadOp> loads;
      generic.walk([&](RemoteLoadOp load) {
        if (load.getMemref() == input) {
          loads.push_back(load);
        }
      });
      if (loads.size() != 1 || loads.front().getIndices().size() != 2 ||
          cast<MemRefType>(loads.front().getLocalBuffer().getType())
                  .getNumElements() != 1) {
        return generic.emitOpError(
            "pipeline requires one single-tile remote load per internal input");
      }
      edges.push_back({static_cast<unsigned>(producerId), id, loads.front()});
    }
  }
  return success();
}

static void insertReadyWait(OpBuilder &builder, GenericOp consumer,
                            RemoteLoadOp load, Value semaphore) {
  builder.setInsertionPoint(load);
  auto indices = load.getIndices();
  auto factors = consumer.getBlockFactorsValue();
  Value columns =
      builder.create<arith::ConstantIndexOp>(load.getLoc(), factors[1]);
  Value row = builder.create<arith::MulIOp>(load.getLoc(), indices[0], columns);
  Value ordinal = builder.create<arith::AddIOp>(load.getLoc(), row, indices[1]);
  Value one = builder.create<arith::ConstantIndexOp>(load.getLoc(), 1);
  Value count = builder.create<arith::AddIOp>(load.getLoc(), ordinal, one);
  auto wait = builder.create<SemaphoreWaitOp>(load.getLoc(), semaphore, count);
  wait->setAttr(spatial_pipeline::wait, builder.getUnitAttr());
}

class D2MMaterializeSpatialPipelines
    : public impl::D2MMaterializeSpatialPipelinesBase<
          D2MMaterializeSpatialPipelines> {
public:
  using Base =
      impl::D2MMaterializeSpatialPipelinesBase<D2MMaterializeSpatialPipelines>;
  using Base::Base;

  void runOnOperation() final {
    for (auto func : getOperation().getOps<func::FuncOp>()) {
      if (auto expected = func->getAttrOfType<IntegerAttr>(
              spatial_pipeline::expectedStages)) {
        int64_t actual = 0;
        for (Block &block : func.getBody()) {
          for (auto generic : block.getOps<GenericOp>()) {
            actual += generic->hasAttr(spatial_pipeline::group);
          }
        }
        if (actual != expected.getInt()) {
          func.emitOpError(
              "prepared pipeline stages were lost before materialization");
          return signalPassFailure();
        }
      }
      for (Block &block : func.getBody()) {
        llvm::MapVector<int64_t, SmallVector<GenericOp>> groups;
        for (auto generic : block.getOps<GenericOp>()) {
          if (auto group = generic->getAttrOfType<IntegerAttr>(
                  spatial_pipeline::group)) {
            groups[group.getInt()].push_back(generic);
          }
        }
        for (auto &[id, stages] : groups) {
          if (failed(materialize(stages))) {
            return signalPassFailure();
          }
        }
      }
      func->removeAttr(spatial_pipeline::expectedStages);
    }
  }

  LogicalResult materialize(ArrayRef<GenericOp> stages) {
    GenericOp first = stages.front();
    OpBuilder builder(first);
    llvm::SetVector<Operation *> preparations;
    llvm::SetVector<Value> inputs;
    SmallVector<Value> outputs;
    SmallVector<Attribute> ranges;
    for (GenericOp stage : stages) {
      if (stage.getNumResults() || stage.getNumRegions() != 1) {
        return stage.emitOpError(
            "prepared spatial stage must be bufferized with one region");
      }
      for (Value operand : stage->getOperands()) {
        if (!collectDefinitions(operand, first, preparations)) {
          return stage.emitOpError(
              "pipeline buffer preparation cannot dominate all stages");
        }
      }
      ranges.push_back(stage->getAttr(spatial_pipeline::core));
      inputs.insert(stage.getInputs().begin(), stage.getInputs().end());
      outputs.append(stage.getOutputs().begin(), stage.getOutputs().end());
    }
    SmallVector<ReadyEdge> edges;
    if (failed(collectReadyEdges(stages, edges))) {
      return failure();
    }
    for (Operation *prep : preparations) {
      prep->moveBefore(first);
    }
    builder.setInsertionPoint(first);
    auto device = ttcore::lookupDeviceOp(first);
    auto grid = device.getDeviceAttr().getWorkerGrid().getShape();
    Type uint32 = IntegerType::get(&getContext(), 32, IntegerType::Unsigned);
    auto memory = ttcore::MemorySpaceAttr::get(&getContext(),
                                               ttcore::MemorySpace::DeviceL1);
    auto layout = ttcore::ShardLayoutAttr::get({1, 1}, uint32, 1);
    auto type =
        MemRefType::get({grid[0], grid[1], 1, 1}, uint32, layout, memory);
    auto zero = builder.getIntegerAttr(uint32, 0);
    SmallVector<SmallVector<Attribute>> signals(stages.size());
    for (const ReadyEdge &edge : edges) {
      unsigned producerId = edge.producer;
      unsigned consumerId = edge.consumer;
      GenericOp consumer = stages[consumerId];
      GenericOp producer = stages[producerId];
      builder.setInsertionPoint(first);
      auto backing = builder.create<memref::AllocOp>(first.getLoc(), type);
      auto sem = builder.create<CreateGlobalSemaphoreOp>(
          first.getLoc(), GlobalSemaphoreType::get(&getContext()), backing,
          zero);
      builder.create<ResetGlobalSemaphoreOp>(first.getLoc(), sem, zero);
      unsigned index = producer.getAdditionalArgs().size();
      producer.getAdditionalArgsMutable().append(sem.getResult());
      consumer.getAdditionalArgsMutable().append(sem.getResult());
      auto core =
          cast<ttcore::CoreRangeAttr>(ranges[consumerId]).getStartCoord();
      // Semaphore destinations use the sending generic's virtual grid.
      // Kernel outlining applies its placement offset exactly once.
      auto origin =
          cast<ttcore::CoreRangeAttr>(ranges[producerId]).getStartCoord();
      signals[producerId].push_back(builder.getDenseI64ArrayAttr(
          {index, core.getY() - origin.getY(), core.getX() - origin.getX()}));
      insertReadyWait(builder, consumer, edge.load, sem);
    }
    for (auto [i, stage] : llvm::enumerate(stages)) {
      stage->setAttr(spatial_pipeline::signals,
                     builder.getArrayAttr(signals[i]));
    }
    builder.setInsertionPoint(first);
    auto spatial = builder.create<SpatialOp>(
        first.getLoc(), TypeRange{}, inputs.getArrayRef(), outputs,
        builder.getArrayAttr(ranges), stages.size());
    for (auto [i, stage] : llvm::enumerate(stages)) {
      Block *body = builder.createBlock(&spatial.getRegions()[i]);
      stage->moveBefore(body, body->end());
    }
    if (dumpRegions) {
      llvm::errs() << "emitted L1 tile pipeline ";
      spatial.print(llvm::errs());
      llvm::errs() << '\n';
    }
    return success();
  }
};
} // namespace
} // namespace mlir::tt::d2m
