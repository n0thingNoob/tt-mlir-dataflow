// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
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

// True when a write barrier follows `write` in its thread: a d2m.dma_wait on
// a write transaction in the thread's top-level block, after the op that
// holds the write. A write dma_wait lowers to noc_async_write_barrier, which
// waits for every write the thread issued before it. This covers transactions
// carried out of nested loops and scf.if (a reblocked generic) that isWaited
// does not follow.
static bool isFollowedByWriteBarrier(DMAWriteOp write, Region &thread) {
  Operation *top = thread.front().findAncestorOpInBlock(*write);
  for (Operation *op = top ? top->getNextNode() : nullptr; op;
       op = op->getNextNode()) {
    auto wait = mlir::dyn_cast<DMAWaitOp>(op);
    if (wait && mlir::cast<MemTxType>(wait.getMemTx().getType()).getDmaType() ==
                    DMAType::Write) {
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
  // Either end may access the edge buffer through a view of it (a reblocked
  // generic), so DMA operands are compared with their views stripped.
  Value edgeBuffer = spatial.getEdgeBuffer(edges.front());
  auto isEdgeBuffer = [&](Value value) {
    return SpatialOp::stripViews(value) == edgeBuffer;
  };
  auto missingAccess = [&](SpatialEdgeAttr edge) {
    return spatial.emitOpError("edge ")
           << edge
           << ": expected DMA writes of the edge buffer in the producer and "
              "DMA reads of it in the consumer (is the edge buffer aliased?)";
  };

  SmallVector<Region *> writers =
      getDatamovementRegions(producer, [&](Operation *op) {
        auto write = mlir::dyn_cast<DMAWriteOp>(op);
        return write && isEdgeBuffer(write.getDst());
      });
  if (writers.empty()) {
    return missingAccess(edges.front());
  }
  // The increment must follow every write of the edge buffer: each write's
  // transaction has to be waited on inside its thread.
  for (Region *region : writers) {
    WalkResult result = region->walk([&](DMAWriteOp write) {
      if (isEdgeBuffer(write.getDst()) && !isWaited(write->getResult(0)) &&
          !isFollowedByWriteBarrier(write, *region)) {
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
      return read && isEdgeBuffer(read.getSrc());
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

// The scf.for blocking loops (d2m.blocking_loop) around `op` in its thread,
// outer first.
static SmallVector<scf::ForOp> blockingLoops(Operation *op) {
  SmallVector<scf::ForOp> loops;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto loop = mlir::dyn_cast<scf::ForOp>(parent);
        loop && loop->hasAttr("d2m.blocking_loop")) {
      loops.push_back(loop);
    }
    if (mlir::isa<GenericOp>(parent)) {
      break;
    }
  }
  std::reverse(loops.begin(), loops.end());
  return loops;
}

// The ops of one streaming edge in one thread: the DMA ops that move its
// blocks, all in the body of the same innermost blocking loop, and the block
// count and ordinal of the current iteration.
struct BlockSite {
  SmallVector<scf::ForOp> loops;          // all blocking loops, outer first
  SmallVector<scf::ForOp> blockLoops;     // the non-reduction ones
  SmallVector<scf::ForOp> reductionLoops; // a block is final in their last
                                          // iteration
  Operation *first = nullptr; // first op of the loop body holding a DMA op
  Operation *last = nullptr;  // last such op
  int64_t numBlocks = 1;
};

// For writes, reduction loops do not count blocks: the producer writes a block
// once per reduction, finally in the last iteration. For reads every
// iteration reads a block.
static FailureOr<BlockSite> findBlockSite(ArrayRef<Operation *> dmaOps,
                                          SpatialOp spatial,
                                          SpatialEdgeAttr edge, bool isWrite) {
  BlockSite site;
  site.loops = blockingLoops(dmaOps.front());
  for (Operation *op : dmaOps) {
    if (blockingLoops(op) != site.loops) {
      return spatial.emitOpError("edge ")
             << edge << ": its DMA ops in one thread are in different loops";
    }
  }
  if (site.loops.empty()) {
    return spatial.emitOpError("edge ")
           << edge << ": a streaming edge needs its DMA ops in blocking loops";
  }
  for (scf::ForOp loop : site.loops) {
    std::optional<int64_t> lb = getConstantIntValue(loop.getLowerBound());
    std::optional<int64_t> ub = getConstantIntValue(loop.getUpperBound());
    std::optional<int64_t> step = getConstantIntValue(loop.getStep());
    if (!lb || !ub || !step || *lb != 0 || *step != 1) {
      return spatial.emitOpError("edge ")
             << edge << ": a blocking loop around it is not 0..N step 1";
    }
    if (isWrite && loop->hasAttr("d2m.reduction_loop")) {
      site.reductionLoops.push_back(loop);
      continue;
    }
    site.blockLoops.push_back(loop);
    site.numBlocks *= *ub;
  }
  Block *body = site.loops.back().getBody();
  for (Operation *op : dmaOps) {
    Operation *top = body->findAncestorOpInBlock(*op);
    if (!site.first || top->isBeforeInBlock(site.first)) {
      site.first = top;
    }
    if (!site.last || site.last->isBeforeInBlock(top)) {
      site.last = top;
    }
  }
  return site;
}

// The block ordinal of the current iteration: the blocking loops' induction
// variables, linearized row-major.
static Value blockOrdinal(OpBuilder &builder, Location loc,
                          ArrayRef<scf::ForOp> loops) {
  Value ordinal = builder.create<arith::ConstantIndexOp>(loc, 0);
  for (scf::ForOp loop : loops) {
    Value extent = loop.getUpperBound();
    ordinal = builder.create<arith::MulIOp>(loc, ordinal, extent);
    ordinal =
        builder.create<arith::AddIOp>(loc, ordinal, loop.getInductionVar());
  }
  return ordinal;
}

// Lowers one streaming edge whose buffer is a ring in the consumer's L1
// (d2m-materialize-spatial-streams). One local semaphore: the consumer core's
// copy counts the blocks written into its ring (HEAD), the producer core's
// copy counts the blocks the consumer has read (CREDIT). For block n:
//   producer: wait CREDIT >= n + 1 - depth; write the block; barrier;
//             increment HEAD on the consumer core.
//   consumer: wait HEAD >= n + 1; read the block; barrier; increment CREDIT on
//             the producer core.
// The producer finally waits for all credits, so no increment is in flight
// when the program ends.
static LogicalResult lowerStreamEdge(SpatialOp spatial, SpatialEdgeAttr edge,
                                     OpBuilder &builder) {
  auto getGeneric = [&](unsigned region) {
    return *spatial.getRegion(region).getOps<GenericOp>().begin();
  };
  GenericOp producer = getGeneric(edge.getProducerRegion());
  GenericOp consumer = getGeneric(edge.getConsumerRegion());
  Value ring = spatial.getEdgeBuffer(edge);
  auto isRing = [&](Value value) {
    return SpatialOp::stripViews(value) == ring;
  };

  // The DMA ops that move the ring's blocks, and the threads that hold them.
  auto collect = [&](GenericOp generic, bool isWrite,
                     SmallVector<Operation *> &ops,
                     SmallVector<Region *> &threads) {
    for (auto [index, region] : llvm::enumerate(generic.getRegions())) {
      if (generic.getRegionThreadType(index) != ThreadType::Datamovement) {
        continue;
      }
      size_t before = ops.size();
      region.walk([&](Operation *op) {
        auto write = mlir::dyn_cast<DMAWriteOp>(op);
        auto read = mlir::dyn_cast<DMAReadOp>(op);
        if ((isWrite && write && isRing(write.getDst())) ||
            (!isWrite && read && isRing(read.getSrc()))) {
          ops.push_back(op);
        }
      });
      if (ops.size() != before) {
        threads.push_back(&region);
      }
    }
  };
  SmallVector<Operation *> writes, reads;
  SmallVector<Region *> writers, readers;
  collect(producer, /*isWrite=*/true, writes, writers);
  collect(consumer, /*isWrite=*/false, reads, readers);
  if (writers.size() != 1 || readers.size() != 1) {
    return spatial.emitOpError("edge ")
           << edge << ": a streaming edge needs one writing thread in the "
           << "producer and one reading thread in the consumer, got "
           << writers.size() << " and " << readers.size();
  }
  FailureOr<BlockSite> writeSite =
      findBlockSite(writes, spatial, edge, /*isWrite=*/true);
  FailureOr<BlockSite> readSite =
      findBlockSite(reads, spatial, edge, /*isWrite=*/false);
  if (failed(writeSite) || failed(readSite)) {
    return failure();
  }
  if (writeSite->numBlocks != readSite->numBlocks) {
    return spatial.emitOpError("edge ")
           << edge << ": the producer writes " << writeSite->numBlocks
           << " blocks but the consumer reads " << readSite->numBlocks;
  }
  for (GenericOp generic : {producer, consumer}) {
    if (countLocalSemaphoreArgs(generic) + 1 > kMaxLocalSemaphoresPerCore) {
      return spatial.emitOpError("edge ")
             << edge << ": more than " << kMaxLocalSemaphoresPerCore
             << " local semaphores on one core range";
    }
  }

  Location loc = spatial.getLoc();
  builder.setInsertionPoint(spatial);
  Value semaphore = builder.create<CreateLocalSemaphoreOp>(
      loc, builder.getType<LocalSemaphoreType>(), 0u);
  producer.getAdditionalArgsMutable().append(semaphore);
  consumer.getAdditionalArgsMutable().append(semaphore);

  // semaphore_inc core indices are virtual coordinates of the issuing
  // generic. Core k of one region pairs with core k of the other, and both
  // virtual grids are shifted copies of their cores
  // (d2m-materialize-spatial-streams checks both), so the partner of virtual
  // core k is k + (the target's core 0 in the issuer's virtual coordinates).
  auto coreIn = [&](GenericOp issuer, GenericOp target) {
    auto [y, x] = getLogicalCores(target).front();
    AffineMap physToVirt = issuer.getGrid().getPhysicalToVirtMap();
    if (!physToVirt.isEmpty()) {
      SmallVector<int64_t> virt = physToVirt.compose({y, x});
      y = virt[virt.size() - 2];
      x = virt[virt.size() - 1];
    }
    return std::make_pair(y, x);
  };
  auto increment = [&](GenericOp issuer, GenericOp target) {
    auto [y, x] = coreIn(issuer, target);
    Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value coreY = builder.create<arith::ConstantIndexOp>(loc, y);
    Value coreX = builder.create<arith::ConstantIndexOp>(loc, x);
    if (issuer.getGrid().getGridVolume() > 1) {
      coreY = builder.create<arith::AddIOp>(
          loc, builder.create<CoreIndexOp>(loc, 0), coreY);
      coreX = builder.create<arith::AddIOp>(
          loc, builder.create<CoreIndexOp>(loc, 1), coreX);
    }
    builder.create<SemaphoreIncOp>(loc, semaphore, one,
                                   ValueRange{coreY, coreX});
  };
  auto waitAtLeast = [&](Value value) {
    builder.create<SemaphoreWaitOp>(loc, semaphore, value,
                                    /*resetValue=*/nullptr,
                                    builder.getUnitAttr());
  };
  auto barrier = [&](DMAType type) {
    Value tx = builder.create<NullTxOp>(loc, type);
    builder.create<DMAWaitOp>(loc, tx);
  };

  // Producer: wait for a free slot, then signal the written block.
  int64_t depth = edge.getStreamDepth();
  builder.setInsertionPoint(writeSite->first);
  Value n = blockOrdinal(builder, loc, writeSite->blockLoops);
  auto waitForSlot = [&]() {
    Value credits = builder.create<arith::SubIOp>(
        loc, n, builder.create<arith::ConstantIndexOp>(loc, depth - 1));
    credits = builder.create<arith::MaxSIOp>(
        loc, credits, builder.create<arith::ConstantIndexOp>(loc, 0));
    waitAtLeast(credits);
  };
  if (writeSite->reductionLoops.empty()) {
    waitForSlot();
    builder.setInsertionPointAfter(writeSite->last);
    barrier(DMAType::Write);
    increment(producer, consumer);
  } else {
    // A reduction producer rewrites its output block in every iteration of
    // its reduction loops; only the last one is final. The others would
    // overwrite a slot the consumer may still read, so only the final
    // iteration writes the ring (the compute thread keeps its partial sums in
    // its own output CB) and signals it; the others yield a null transaction
    // to whatever waited on the write.
    auto write = mlir::dyn_cast<DMAWriteOp>(writeSite->first);
    if (writes.size() != 1 || writeSite->first != writeSite->last || !write) {
      return spatial.emitOpError("edge ")
             << edge << ": a reduction producer must write each block with "
             << "one DMA op directly in its innermost blocking loop";
    }
    Value isFinal;
    for (scf::ForOp loop : writeSite->reductionLoops) {
      Value last = builder.create<arith::SubIOp>(
          loc, loop.getUpperBound(),
          builder.create<arith::ConstantIndexOp>(loc, 1));
      Value cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                 loop.getInductionVar(), last);
      isFinal =
          isFinal ? builder.create<arith::AndIOp>(loc, isFinal, cond) : cond;
    }
    Type txType = write.getResult().getType();
    auto ifOp = builder.create<scf::IfOp>(loc, TypeRange{txType}, isFinal,
                                          /*withElseRegion=*/true);
    write.getResult().replaceAllUsesWith(ifOp.getResult(0));
    builder.setInsertionPointToStart(ifOp.thenBlock());
    waitForSlot();
    write->moveBefore(ifOp.thenBlock(), ifOp.thenBlock()->end());
    builder.setInsertionPointToEnd(ifOp.thenBlock());
    barrier(DMAType::Write);
    increment(producer, consumer);
    builder.create<scf::YieldOp>(loc, write.getResult());
    builder.setInsertionPointToStart(ifOp.elseBlock());
    builder.create<scf::YieldOp>(
        loc, builder.create<NullTxOp>(loc, DMAType::Write).getResult());
  }
  builder.setInsertionPointToEnd(&writers.front()->front());
  waitAtLeast(
      builder.create<arith::ConstantIndexOp>(loc, writeSite->numBlocks));

  // Consumer: wait for the block, then return its slot.
  builder.setInsertionPoint(readSite->first);
  Value m = blockOrdinal(builder, loc, readSite->blockLoops);
  waitAtLeast(builder.create<arith::AddIOp>(
      loc, m, builder.create<arith::ConstantIndexOp>(loc, 1)));
  builder.setInsertionPointAfter(readSite->last);
  barrier(DMAType::Read);
  increment(consumer, producer);
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
        // A streaming edge whose ring d2m-materialize-spatial-streams built
        // streams. Any other edge, including a streaming edge without a ring
        // (that pass did not run), gets the whole-tensor handoff, which is
        // correct for any edge.
        Operation *bufferDef =
            spatial.getEdgeBuffer(edges.front()).getDefiningOp();
        if (edges.size() == 1 && edges.front().isStream() && bufferDef &&
            bufferDef->hasAttr(utils::kStreamRingAttr)) {
          if (failed(lowerStreamEdge(spatial, edges.front(), builder))) {
            return WalkResult::interrupt();
          }
          continue;
        }
        for (SpatialEdgeAttr edge : edges) {
          if (edge.isStream()) {
            spatial.emitRemark("edge ")
                << edge << " asks for a streaming handoff without a ring; "
                << "it uses the whole-tensor handoff";
          }
        }
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
