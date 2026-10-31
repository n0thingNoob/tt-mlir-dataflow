// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/MapVector.h"

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

// One consumer region of an edge group: its reading threads, its cores and,
// when the cores fill their bounding box, that box for a multicast increment.
struct EdgeConsumer {
  SmallVector<Region *> readers;
  SmallVector<std::pair<int64_t, int64_t>> cores;
  int64_t minY, minX, height, width;
  bool multicast;
};

// True when the transaction `tx` of a DMA write is waited on in its thread.
// Besides a direct d2m.dma_wait this accepts the software-pipelined form a
// blocked matmul uses: the transaction is yielded out of an scf.for, waited
// at the start of the next iteration (on the iteration argument) and once
// more after the loop (on the loop result).
static bool isWaited(Value tx) {
  bool direct = llvm::any_of(
      tx.getUsers(), [](Operation *user) { return isa<DMAWaitOp>(user); });
  if (direct) {
    return true;
  }
  for (OpOperand &use : tx.getUses()) {
    auto yield = mlir::dyn_cast<scf::YieldOp>(use.getOwner());
    auto loop =
        yield ? mlir::dyn_cast<scf::ForOp>(yield->getParentOp()) : nullptr;
    if (!loop) {
      continue;
    }
    unsigned i = use.getOperandNumber();
    if (isWaited(loop.getRegionIterArgs()[i]) && isWaited(loop.getResult(i))) {
      return true;
    }
  }
  return false;
}

// Lower the edges that leave one producer output (one per consumer region)
// with a single semaphore: the producer bumps every consumer core once per
// writing thread, and every consumer waits for all of those signals.
static LogicalResult lowerEdgeGroup(SpatialOp spatial,
                                    ArrayRef<SpatialEdgeAttr> edges,
                                    OpBuilder &builder) {
  auto getGeneric = [&](unsigned region) {
    return *spatial.getRegion(region).getOps<GenericOp>().begin();
  };
  GenericOp producer = getGeneric(edges.front().getProducerRegion());
  Value edgeBuffer = spatial.getEdgeSource(edges.front());
  auto missingAccess = [&](SpatialEdgeAttr edge) {
    return spatial.emitOpError("edge ")
           << edge
           << ": expected DMA writes of the edge buffer in the producer and "
              "DMA reads of it in the consumer (is the edge buffer aliased?)";
  };

  SmallVector<Region *> writers =
      getDatamovementRegions(producer, [&](Operation *op) {
        auto write = mlir::dyn_cast<DMAWriteOp>(op);
        return write && write.getDst() == edgeBuffer;
      });
  if (writers.empty()) {
    return missingAccess(edges.front());
  }
  // The increment must follow every write of the edge buffer: each write's
  // transaction has to be waited on inside its thread.
  for (Region *region : writers) {
    WalkResult result = region->walk([&](DMAWriteOp write) {
      if (write.getDst() == edgeBuffer && !isWaited(write->getResult(0))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      return spatial.emitOpError("edge ")
             << edges.front()
             << ": producer DMA write of the edge buffer has no wait";
    }
  }

  SmallVector<GenericOp> generics = {producer};
  SmallVector<EdgeConsumer> consumers;
  SmallVector<std::pair<int64_t, int64_t>> producerCores =
      getLogicalCores(producer);
  for (SpatialEdgeAttr edge : edges) {
    GenericOp generic = getGeneric(edge.getConsumerRegion());
    EdgeConsumer &consumer = consumers.emplace_back();
    consumer.readers = getDatamovementRegions(generic, [&](Operation *op) {
      auto read = mlir::dyn_cast<DMAReadOp>(op);
      return read && read.getSrc() == edgeBuffer;
    });
    if (consumer.readers.empty()) {
      return missingAccess(edge);
    }
    generics.push_back(generic);

    // When the consumer cores fill their bounding box, one multicast
    // increment reaches all of them. The NoC atomic has no loopback variant,
    // so the producer cores must lie outside that box; regions must not
    // overlap anyway.
    consumer.cores = getLogicalCores(generic);
    int64_t minY = std::numeric_limits<int64_t>::max(), maxY = 0;
    int64_t minX = std::numeric_limits<int64_t>::max(), maxX = 0;
    for (auto [y, x] : consumer.cores) {
      minY = std::min(minY, y);
      maxY = std::max(maxY, y);
      minX = std::min(minX, x);
      maxX = std::max(maxX, x);
    }
    for (auto [y, x] : producerCores) {
      if (y >= minY && y <= maxY && x >= minX && x <= maxX) {
        return spatial.emitOpError("edge ")
               << edge << ": producer core (" << y << ", " << x
               << ") lies inside the consumer's core range";
      }
    }
    consumer.minY = minY;
    consumer.minX = minX;
    consumer.height = maxY - minY + 1;
    consumer.width = maxX - minX + 1;
    consumer.multicast = consumer.cores.size() > 1 &&
                         static_cast<int64_t>(consumer.cores.size()) ==
                             consumer.height * consumer.width;
  }

  for (GenericOp generic : generics) {
    if (countLocalSemaphoreArgs(generic) + 1 > kMaxLocalSemaphoresPerCore) {
      return spatial.emitOpError("edge ")
             << edges.front() << ": more than " << kMaxLocalSemaphoresPerCore
             << " local semaphores on one core range";
    }
  }

  Location loc = spatial.getLoc();
  builder.setInsertionPoint(spatial);
  Value semaphore = builder.create<CreateLocalSemaphoreOp>(
      loc, builder.getType<LocalSemaphoreType>(), 0u);
  for (GenericOp generic : generics) {
    generic.getAdditionalArgsMutable().append(semaphore);
  }

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
    for (const EdgeConsumer &consumer : consumers) {
      if (consumer.multicast) {
        auto [y, x] = toProducerVirt(consumer.minY, consumer.minX);
        Value coreY = builder.create<arith::ConstantIndexOp>(loc, y);
        Value coreX = builder.create<arith::ConstantIndexOp>(loc, x);
        Value mcastY =
            builder.create<arith::ConstantIndexOp>(loc, consumer.height);
        Value mcastX =
            builder.create<arith::ConstantIndexOp>(loc, consumer.width);
        builder.create<SemaphoreIncOp>(loc, semaphore, one,
                                       ValueRange{coreY, coreX},
                                       ValueRange{mcastY, mcastX},
                                       /*startDevice=*/ValueRange(),
                                       /*deviceMcastShape=*/ValueRange());
        continue;
      }
      for (auto [consumerY, consumerX] : consumer.cores) {
        auto [y, x] = toProducerVirt(consumerY, consumerX);
        Value coreY = builder.create<arith::ConstantIndexOp>(loc, y);
        Value coreX = builder.create<arith::ConstantIndexOp>(loc, x);
        builder.create<SemaphoreIncOp>(loc, semaphore, one,
                                       ValueRange{coreY, coreX});
      }
    }
  }

  // Consumers: every reading thread waits for all producer signals before it
  // does anything else.
  int64_t expected =
      static_cast<int64_t>(producerCores.size() * writers.size());
  for (const EdgeConsumer &consumer : consumers) {
    for (Region *region : consumer.readers) {
      builder.setInsertionPointToStart(&region->front());
      Value count = builder.create<arith::ConstantIndexOp>(loc, expected);
      builder.create<SemaphoreWaitOp>(loc, semaphore, count,
                                      /*resetValue=*/nullptr);
    }
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
      // Edges that leave the same producer output share one semaphore.
      llvm::MapVector<std::pair<unsigned, unsigned>,
                      SmallVector<SpatialEdgeAttr>>
          groups;
      for (auto edge : spatial.getEdgesAttr().getAsRange<SpatialEdgeAttr>()) {
        groups[{edge.getProducerRegion(), edge.getProducerOutput()}].push_back(
            edge);
      }
      for (auto &[source, edges] : groups) {
        if (failed(lowerEdgeGroup(spatial, edges, builder))) {
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
