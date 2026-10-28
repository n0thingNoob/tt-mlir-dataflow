// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"

#include <limits>

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MLOWERSPATIALEDGES
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Local semaphores available per core per program (tt-metal NUM_SEMAPHORES).
constexpr unsigned kMaxLocalSemaphoresPerCore = 16;

// Datamovement thread regions of `generic` that contain an op matching `pred`.
template <typename Pred>
static SmallVector<Region *> getDatamovementRegions(GenericOp generic,
                                                    Pred pred) {
  SmallVector<Region *> regions;
  for (auto [index, region] : llvm::enumerate(generic.getRegions())) {
    if (generic.getRegionThreadType(index) != ThreadType::Datamovement) {
      continue;
    }
    bool found = false;
    region.walk([&](Operation *op) {
      if (pred(op)) {
        found = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (found) {
      regions.push_back(&region);
    }
  }
  return regions;
}

// Logical worker coordinates (y, x) of every core of `generic`'s grid.
static SmallVector<std::pair<int64_t, int64_t>>
getLogicalCores(GenericOp generic) {
  ArrayRef<int64_t> shape = generic.getGrid().getShape();
  AffineMap virtToPhys = generic.getGrid().getVirtToPhysicalMap();
  SmallVector<std::pair<int64_t, int64_t>> cores;
  for (int64_t y = 0; y < shape[0]; ++y) {
    for (int64_t x = 0; x < shape[1]; ++x) {
      if (virtToPhys.isEmpty()) {
        cores.push_back({y, x});
        continue;
      }
      // (vy, vx) -> (device, y, x), or (y, x) without the device result.
      SmallVector<int64_t> phys = virtToPhys.compose({y, x});
      cores.push_back({phys[phys.size() - 2], phys[phys.size() - 1]});
    }
  }
  return cores;
}

static unsigned countLocalSemaphoreArgs(GenericOp generic) {
  return llvm::count_if(generic.getAdditionalArgs(), [](Value arg) {
    return mlir::isa<LocalSemaphoreType>(arg.getType());
  });
}

static LogicalResult lowerEdge(SpatialOp spatial, SpatialEdgeAttr edge,
                               OpBuilder &builder) {
  auto getGeneric = [&](unsigned region) {
    return *spatial.getRegion(region).getOps<GenericOp>().begin();
  };
  GenericOp producer = getGeneric(edge.getProducerRegion());
  GenericOp consumer = getGeneric(edge.getConsumerRegion());
  Value edgeBuffer = spatial.getEdgeSource(edge);

  SmallVector<Region *> writers =
      getDatamovementRegions(producer, [&](Operation *op) {
        auto write = mlir::dyn_cast<DMAWriteOp>(op);
        return write && write.getDst() == edgeBuffer;
      });
  SmallVector<Region *> readers =
      getDatamovementRegions(consumer, [&](Operation *op) {
        auto read = mlir::dyn_cast<DMAReadOp>(op);
        return read && read.getSrc() == edgeBuffer;
      });
  if (writers.empty() || readers.empty()) {
    return spatial.emitOpError("edge ")
           << edge
           << ": expected DMA writes of the edge buffer in the producer and "
              "DMA reads of it in the consumer (is the edge buffer aliased?)";
  }
  // The increment must follow every write of the edge buffer: each write's
  // transaction has to be waited on inside its thread.
  for (Region *region : writers) {
    WalkResult result = region->walk([&](DMAWriteOp write) {
      if (write.getDst() == edgeBuffer &&
          llvm::none_of(write->getUsers(),
                        [](Operation *user) { return isa<DMAWaitOp>(user); })) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      return spatial.emitOpError("edge ")
             << edge << ": producer DMA write of the edge buffer has no wait";
    }
  }

  for (GenericOp generic : {producer, consumer}) {
    if (countLocalSemaphoreArgs(generic) + 1 > kMaxLocalSemaphoresPerCore) {
      return spatial.emitOpError("edge ")
             << edge << ": more than " << kMaxLocalSemaphoresPerCore
             << " local semaphores on one core range";
    }
  }

  // When the consumer cores fill their bounding box, one multicast increment
  // reaches all of them. The NoC atomic has no loopback variant, so the
  // producer cores must lie outside that box; regions must not overlap anyway.
  SmallVector<std::pair<int64_t, int64_t>> consumerCores =
      getLogicalCores(consumer);
  int64_t minY = std::numeric_limits<int64_t>::max(), maxY = 0;
  int64_t minX = std::numeric_limits<int64_t>::max(), maxX = 0;
  for (auto [y, x] : consumerCores) {
    minY = std::min(minY, y);
    maxY = std::max(maxY, y);
    minX = std::min(minX, x);
    maxX = std::max(maxX, x);
  }
  for (auto [y, x] : getLogicalCores(producer)) {
    if (y >= minY && y <= maxY && x >= minX && x <= maxX) {
      return spatial.emitOpError("edge ")
             << edge << ": producer core (" << y << ", " << x
             << ") lies inside the consumer's core range";
    }
  }
  const int64_t height = maxY - minY + 1, width = maxX - minX + 1;
  const bool multicast =
      consumerCores.size() > 1 &&
      static_cast<int64_t>(consumerCores.size()) == height * width;

  Location loc = spatial.getLoc();
  builder.setInsertionPoint(spatial);
  Value semaphore = builder.create<CreateLocalSemaphoreOp>(
      loc, builder.getType<LocalSemaphoreType>(), 0u);
  producer.getAdditionalArgsMutable().append(semaphore);
  consumer.getAdditionalArgsMutable().append(semaphore);

  // Producer: after its last op, each writing thread bumps the semaphore of
  // every consumer core once. semaphore_inc core indices are virtual
  // coordinates of the enclosing (producer) generic, mapped to physical cores
  // by d2m-generic-regions-to-funcs (a multicast keeps its shape), so express
  // each consumer core in the producer grid's virtual coordinates.
  AffineMap producerPhysToVirt = producer.getGrid().getPhysicalToVirtMap();
  auto toProducerVirt = [&](int64_t y, int64_t x) {
    if (producerPhysToVirt.isEmpty()) {
      return std::make_pair(y, x);
    }
    SmallVector<int64_t> virt = producerPhysToVirt.compose({y, x});
    return std::make_pair(virt[virt.size() - 2], virt[virt.size() - 1]);
  };
  for (Region *region : writers) {
    Block &block = region->front();
    builder.setInsertionPointToEnd(&block);
    Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
    if (multicast) {
      auto [y, x] = toProducerVirt(minY, minX);
      Value coreY = builder.create<arith::ConstantIndexOp>(loc, y);
      Value coreX = builder.create<arith::ConstantIndexOp>(loc, x);
      Value mcastY = builder.create<arith::ConstantIndexOp>(loc, height);
      Value mcastX = builder.create<arith::ConstantIndexOp>(loc, width);
      builder.create<SemaphoreIncOp>(loc, semaphore, one,
                                     ValueRange{coreY, coreX},
                                     ValueRange{mcastY, mcastX},
                                     /*startDevice=*/ValueRange(),
                                     /*deviceMcastShape=*/ValueRange());
      continue;
    }
    for (auto [consumerY, consumerX] : consumerCores) {
      auto [y, x] = toProducerVirt(consumerY, consumerX);
      Value coreY = builder.create<arith::ConstantIndexOp>(loc, y);
      Value coreX = builder.create<arith::ConstantIndexOp>(loc, x);
      builder.create<SemaphoreIncOp>(loc, semaphore, one,
                                     ValueRange{coreY, coreX});
    }
  }

  // Consumer: every reading thread waits for all producer signals before it
  // does anything else.
  int64_t expected =
      static_cast<int64_t>(getLogicalCores(producer).size() * writers.size());
  for (Region *region : readers) {
    builder.setInsertionPointToStart(&region->front());
    Value count = builder.create<arith::ConstantIndexOp>(loc, expected);
    builder.create<SemaphoreWaitOp>(loc, semaphore, count,
                                    /*resetValue=*/nullptr);
  }
  return success();
}

class D2MLowerSpatialEdges
    : public impl::D2MLowerSpatialEdgesBase<D2MLowerSpatialEdges> {
public:
  using impl::D2MLowerSpatialEdgesBase<
      D2MLowerSpatialEdges>::D2MLowerSpatialEdgesBase;

  void runOnOperation() final {
    if (ttnnMode) {
      return;
    }
    OpBuilder builder(&getContext());
    WalkResult result = getOperation()->walk([&](SpatialOp spatial) {
      if (!spatial.hasEdges()) {
        return WalkResult::advance();
      }
      for (auto edge : spatial.getEdgesAttr().getAsRange<SpatialEdgeAttr>()) {
        if (failed(lowerEdge(spatial, edge, builder))) {
          return WalkResult::interrupt();
        }
      }
      spatial.removeEdgesAttr();
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::tt::d2m
