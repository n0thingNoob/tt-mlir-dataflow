// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"

#include <string>

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MMATERIALIZESPATIALSTREAMS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Upper bound on index evaluations per edge; an edge above it is not
// analyzed and keeps the whole-tensor handoff.
constexpr int64_t kMaxEvaluations = 1 << 22;

// A tile set of the edge buffer: linear tile ids in the buffer's logical tile
// grid, sorted, plus the shard (grid coordinates, linearized) that holds them.
struct Block {
  SmallVector<int64_t> tiles;
  int64_t owner = -1;
  bool operator==(const Block &other) const { return tiles == other.tiles; }
};

// One blocking loop around an access: its iteration-space dim, extent and
// whether it is a reduction.
struct Loop {
  affine::AffineForOp op;
  int64_t extent;
  bool reduction;
};

// Evaluates `map` at constant operands (dims, then symbols). AffineMap::compose
// takes no symbols, and loop bounds use a symbol for the block factor.
static std::optional<SmallVector<int64_t>> evalMap(AffineMap map,
                                                   ArrayRef<int64_t> operands) {
  if (operands.size() != map.getNumInputs()) {
    return std::nullopt;
  }
  MLIRContext *ctx = map.getContext();
  SmallVector<AffineExpr> dims, symbols;
  for (unsigned i = 0; i < map.getNumDims(); ++i) {
    dims.push_back(getAffineConstantExpr(operands[i], ctx));
  }
  for (unsigned i = 0; i < map.getNumSymbols(); ++i) {
    symbols.push_back(
        getAffineConstantExpr(operands[map.getNumDims() + i], ctx));
  }
  SmallVector<int64_t> results;
  for (AffineExpr expr : map.getResults()) {
    auto constant = mlir::dyn_cast<AffineConstantExpr>(
        expr.replaceDimsAndSymbols(dims, symbols));
    if (!constant) {
      return std::nullopt;
    }
    results.push_back(constant.getValue());
  }
  return results;
}

// Evaluates the index arithmetic of an access in affine blocked form for one
// core and one iteration of the blocking loops.
class IndexEvaluator {
public:
  IndexEvaluator(GenericOp generic) : generic(generic) {
    blockFactors = generic.getBlockFactorsValue();
    // Same mapping as LowerToExplicitForm: d2m.block_offset(dim) is
    // block_factor(dim) * core_index(g) when loop dim `dim` is grid result `g`
    // of the output indexing map, and 0 otherwise.
    loopDimToGridDim.assign(blockFactors.size(), -1);
    AffineMap output =
        generic.getIndexingMap(generic.getOutputs().getBeginOperandIndex());
    AffineMap gridMapping = generic.getGrid().getPhysicalToVirtMap();
    unsigned numGridDims =
        gridMapping.isEmpty() ? 2 : gridMapping.getNumResults() - 1;
    for (size_t dim = 0; dim < blockFactors.size(); ++dim) {
      std::optional<unsigned> pos = output.getResultPosition(
          getAffineDimExpr(static_cast<unsigned>(dim), generic.getContext()));
      if (pos && *pos < numGridDims) {
        loopDimToGridDim[dim] = *pos;
      }
    }
  }

  void setCore(ArrayRef<int64_t> core) {
    this->core.assign(core.begin(), core.end());
  }
  void setIV(Value iv, int64_t value) { ivs[iv] = value; }

  std::optional<int64_t> eval(Value value) const {
    if (auto it = ivs.find(value); it != ivs.end()) {
      return it->second;
    }
    Operation *def = value.getDefiningOp();
    if (!def) {
      return std::nullopt;
    }
    if (auto constant = mlir::dyn_cast<arith::ConstantOp>(def)) {
      if (auto attr = mlir::dyn_cast<IntegerAttr>(constant.getValue())) {
        return attr.getInt();
      }
      return std::nullopt;
    }
    auto binary = [&](auto fn) -> std::optional<int64_t> {
      std::optional<int64_t> lhs = eval(def->getOperand(0));
      std::optional<int64_t> rhs = eval(def->getOperand(1));
      if (!lhs || !rhs) {
        return std::nullopt;
      }
      return fn(*lhs, *rhs);
    };
    if (mlir::isa<arith::AddIOp>(def)) {
      return binary([](int64_t a, int64_t b) { return a + b; });
    }
    if (mlir::isa<arith::SubIOp>(def)) {
      return binary([](int64_t a, int64_t b) { return a - b; });
    }
    if (mlir::isa<arith::MulIOp>(def)) {
      return binary([](int64_t a, int64_t b) { return a * b; });
    }
    if (auto apply = mlir::dyn_cast<affine::AffineApplyOp>(def)) {
      SmallVector<int64_t> operands;
      for (Value operand : apply.getMapOperands()) {
        std::optional<int64_t> v = eval(operand);
        if (!v) {
          return std::nullopt;
        }
        operands.push_back(*v);
      }
      std::optional<SmallVector<int64_t>> results =
          evalMap(apply.getAffineMap(), operands);
      if (!results) {
        return std::nullopt;
      }
      return results->front();
    }
    if (auto factor = mlir::dyn_cast<GetBlockFactorOp>(def)) {
      return blockFactor(factor.getDim());
    }
    if (auto offset = mlir::dyn_cast<BlockOffsetOp>(def)) {
      std::optional<int64_t> factor = blockFactor(offset.getDim());
      if (!factor) {
        return std::nullopt;
      }
      int64_t gridDim = loopDimToGridDim[offset.getDim()];
      return gridDim < 0 ? 0 : *factor * core[gridDim];
    }
    if (auto coreIndex = mlir::dyn_cast<CoreIndexOp>(def)) {
      if (static_cast<size_t>(coreIndex.getDim()) < core.size()) {
        return core[coreIndex.getDim()];
      }
    }
    return std::nullopt;
  }

private:
  std::optional<int64_t> blockFactor(int64_t dim) const {
    if (dim < 0 || static_cast<size_t>(dim) >= blockFactors.size()) {
      return std::nullopt;
    }
    return blockFactors[dim];
  }

  GenericOp generic;
  SmallVector<int64_t> blockFactors;
  SmallVector<int64_t> loopDimToGridDim;
  SmallVector<int64_t> core;
  llvm::DenseMap<Value, int64_t> ivs;
};

// The blocks one access touches, in program order, per core of its generic
// (row-major over the generic's grid). `finalOnly` keeps only the last
// iteration of every reduction loop around the access (a producer's output
// block is complete only then).
struct AccessTrace {
  // core (linearized) -> blocks in program order
  SmallVector<SmallVector<Block>> perCore;
  SmallVector<SmallVector<int64_t>> cores;
};

static std::string coreString(ArrayRef<int64_t> core) {
  std::string s = "(";
  llvm::raw_string_ostream os(s);
  llvm::interleave(core, os, ", ");
  os << ")";
  return s;
}

static std::string blockShape(const Block &block, int64_t tileCols) {
  int64_t minR = INT64_MAX, maxR = INT64_MIN, minC = INT64_MAX,
          maxC = INT64_MIN;
  for (int64_t t : block.tiles) {
    minR = std::min(minR, t / tileCols);
    maxR = std::max(maxR, t / tileCols);
    minC = std::min(minC, t % tileCols);
    maxC = std::max(maxC, t % tileCols);
  }
  int64_t rows = maxR - minR + 1, cols = maxC - minC + 1;
  std::string s = std::to_string(rows) + "x" + std::to_string(cols);
  if (rows * cols != static_cast<int64_t>(block.tiles.size())) {
    s += " (not a rectangle)";
  }
  return s;
}

class EdgeAnalysis {
public:
  EdgeAnalysis(SpatialOp spatial, SpatialEdgeAttr edge)
      : spatial(spatial), edge(edge) {}

  // On success fills `summary`; on failure fills `reason`.
  bool run() {
    Value buffer = spatial.getEdgeBuffer(edge);
    auto bufferType = mlir::dyn_cast<MemRefType>(buffer.getType());
    if (!bufferType || bufferType.getRank() % 2 != 0) {
      return fail("the edge buffer is not a sharded memref");
    }
    if (!mlir::isa<ttcore::TileType>(bufferType.getElementType())) {
      return fail("the edge is not tiled");
    }
    int64_t rank = bufferType.getRank() / 2;
    ArrayRef<int64_t> shape = bufferType.getShape();
    gridShape.assign(shape.begin(), shape.begin() + rank);
    shardShape.assign(shape.begin() + rank, shape.end());
    for (int64_t d = 0; d < rank; ++d) {
      tileShape.push_back(gridShape[d] * shardShape[d]);
    }
    if (rank != 2) {
      return fail("the edge buffer is not 2-D");
    }

    GenericOp producer = generic(edge.getProducerRegion());
    GenericOp consumer = generic(edge.getConsumerRegion());
    SmallVector<Operation *> stores, loads;
    producer.walk([&](RemoteStoreOp op) {
      if (SpatialOp::stripViews(op.getMemref()) == buffer) {
        stores.push_back(op);
      }
    });
    consumer.walk([&](RemoteLoadOp op) {
      if (SpatialOp::stripViews(op.getMemref()) == buffer) {
        loads.push_back(op);
      }
    });
    if (stores.size() != 1) {
      return fail("the producer has " + std::to_string(stores.size()) +
                  " remote_store ops of it (expected one)");
    }
    if (loads.size() != 1) {
      return fail("the consumer has " + std::to_string(loads.size()) +
                  " remote_load ops of it (expected one)");
    }
    auto load = mlir::cast<RemoteLoadOp>(loads.front());
    if (load.isMcast() && consumer.getGrid().getGridVolume() > 1) {
      return fail("the consumer loads it with a multicast inside its region");
    }

    std::optional<AccessTrace> writes =
        trace(producer, stores.front(),
              mlir::cast<RemoteStoreOp>(stores.front()).getMemref(),
              mlir::cast<RemoteStoreOp>(stores.front()).getIndices(),
              /*finalOnly=*/true);
    if (!writes) {
      return false;
    }
    std::optional<AccessTrace> reads =
        trace(consumer, load, load.getMemref(), load.getIndices(),
              /*finalOnly=*/false);
    if (!reads) {
      return false;
    }

    // The producer's final writes per shard (owner), from the one core that
    // writes it.
    llvm::DenseMap<int64_t, SmallVector<Block>> writesOf;
    llvm::DenseMap<int64_t, size_t> writerOf;
    for (auto [c, blocks] : llvm::enumerate(writes->perCore)) {
      for (const Block &block : blocks) {
        auto [it, inserted] = writerOf.try_emplace(block.owner, c);
        if (!inserted && it->second != c) {
          return fail("producer cores " +
                      coreString(writes->cores[it->second]) + " and " +
                      coreString(writes->cores[c]) + " write the same shard");
        }
        writesOf[block.owner].push_back(block);
      }
    }
    for (auto &[owner, blocks] : writesOf) {
      llvm::DenseSet<int64_t> seen;
      for (const Block &block : blocks) {
        for (int64_t t : block.tiles) {
          if (!seen.insert(t).second) {
            return fail("the producer's final writes cover a tile twice");
          }
        }
      }
    }

    // Pair every consumer core with the one producer shard it reads.
    llvm::DenseMap<int64_t, size_t> readerOf;
    int64_t blocksPerPair = -1;
    std::string blockText;
    SmallVector<Block> noWrites;
    for (auto [c, blocks] : llvm::enumerate(reads->perCore)) {
      if (blocks.empty()) {
        continue;
      }
      int64_t owner = blocks.front().owner;
      for (const Block &block : blocks) {
        if (block.owner != owner) {
          return fail("consumer core " + coreString(reads->cores[c]) +
                      " reads it from more than one producer core");
        }
      }
      auto [it, inserted] = readerOf.try_emplace(owner, c);
      if (!inserted) {
        return fail("consumer cores " + coreString(reads->cores[it->second]) +
                    " and " + coreString(reads->cores[c]) +
                    " read the same producer core");
      }
      auto found = writesOf.find(owner);
      const SmallVector<Block> &w =
          found == writesOf.end() ? noWrites : found->second;
      if (w.empty()) {
        return fail("consumer core " + coreString(reads->cores[c]) +
                    " reads a shard the producer never writes");
      }
      if (blocks.size() != w.size() ||
          !llvm::all_of(llvm::zip(blocks, w), [](auto pair) {
            return std::get<0>(pair) == std::get<1>(pair);
          })) {
        std::string why = "consumer core " + coreString(reads->cores[c]) +
                          " reads " + std::to_string(blocks.size()) +
                          " blocks but the producer writes " +
                          std::to_string(w.size());
        if (blocks.size() == w.size()) {
          why = "consumer core " + coreString(reads->cores[c]) +
                " reads the blocks in another order or shape than the "
                "producer writes them";
        }
        if (std::optional<int64_t> d = minDepth(w, blocks)) {
          why += "; a ring would need " + std::to_string(*d) + " of " +
                 std::to_string(w.size()) + " blocks";
        }
        return fail(why);
      }
      if (blocksPerPair >= 0 &&
          blocksPerPair != static_cast<int64_t>(w.size())) {
        return fail("core pairs see different block counts");
      }
      blocksPerPair = w.size();
      blockText = blockShape(w.front(), tileShape[1]);
    }
    if (blocksPerPair < 0) {
      return fail("the consumer does not read it");
    }
    if (blocksPerPair == 1) {
      return fail("one block per core pair, nothing to stream");
    }
    summary = std::to_string(readerOf.size()) + " core pair" +
              (readerOf.size() == 1 ? "" : "s") + ", " +
              std::to_string(blocksPerPair) + " blocks of " + blockText +
              " tiles each, in the producer's order";
    return true;
  }

  std::string reason;
  std::string summary;

private:
  bool fail(const std::string &why) {
    reason = why;
    return false;
  }

  GenericOp generic(unsigned region) {
    return *spatial.getRegion(region).getOps<GenericOp>().begin();
  }

  // Smallest ring depth (in producer blocks) that lets the consumer read `r`
  // while the producer writes `w` in order: the most blocks live at a read.
  static std::optional<int64_t> minDepth(ArrayRef<Block> w, ArrayRef<Block> r) {
    if (static_cast<int64_t>(w.size() * r.size()) > kMaxEvaluations) {
      return std::nullopt;
    }
    auto meets = [](const Block &a, const Block &b) {
      auto i = a.tiles.begin(), j = b.tiles.begin();
      while (i != a.tiles.end() && j != b.tiles.end()) {
        if (*i == *j) {
          return true;
        }
        *i < *j ? ++i : ++j;
      }
      return false;
    };
    SmallVector<int64_t> lastUse(w.size(), -1);
    SmallVector<int64_t> need(r.size(), 0);
    for (auto [j, read] : llvm::enumerate(r)) {
      for (auto [i, write] : llvm::enumerate(w)) {
        if (meets(write, read)) {
          need[j] = i + 1;
          lastUse[i] = j;
        }
      }
    }
    int64_t depth = 1;
    for (size_t j = 0; j < r.size(); ++j) {
      for (size_t i = 0; i < w.size(); ++i) {
        if (lastUse[i] >= static_cast<int64_t>(j)) {
          depth = std::max<int64_t>(depth, need[j] - static_cast<int64_t>(i));
          break;
        }
      }
    }
    return depth;
  }

  // Enumerates the blocks `op` accesses on every core of `generic`.
  std::optional<AccessTrace> trace(GenericOp generic, Operation *op,
                                   Value memref, ValueRange indices,
                                   bool finalOnly) {
    // The blocking loops around the access, outer first.
    SmallVector<Loop> loops;
    for (Operation *parent = op->getParentOp(); parent != generic;
         parent = parent->getParentOp()) {
      auto loop = mlir::dyn_cast<affine::AffineForOp>(parent);
      if (!loop || !loop->hasAttr("d2m.blocking_loop")) {
        fail("an access of it is not directly inside blocking loops");
        return std::nullopt;
      }
      loops.push_back({loop, 0, loop->hasAttr("d2m.reduction_loop")});
    }
    std::reverse(loops.begin(), loops.end());

    // The view chain from the accessed memref to the edge buffer.
    auto memrefType = mlir::cast<MemRefType>(memref.getType());
    AffineMap toBuffer = AffineMap::getMultiDimIdentityMap(
        memrefType.getRank(), memrefType.getContext());
    if (Operation *def = memref.getDefiningOp()) {
      if (auto view = mlir::dyn_cast<ViewOpInterface>(def)) {
        if (view.isComposite()) {
          fail("an access of it goes through a composite view");
          return std::nullopt;
        }
        toBuffer = applyViews(def).second;
      }
    }
    int64_t rank = memrefType.getRank() / 2;
    if (toBuffer.getNumDims() != memrefType.getRank() ||
        toBuffer.getNumResults() != 2 * gridShape.size()) {
      fail("an access of it has a view of another rank");
      return std::nullopt;
    }
    if (static_cast<int64_t>(indices.size()) != rank) {
      fail("an access of it does not index its grid");
      return std::nullopt;
    }
    ArrayRef<int64_t> accessShard = memrefType.getShape().drop_front(rank);

    IndexEvaluator evaluator(generic);
    AccessTrace result;
    ArrayRef<int64_t> grid = generic.getGrid().getShape();
    int64_t shardVolume = 1;
    for (int64_t d : accessShard) {
      shardVolume *= d;
    }
    SmallVector<int64_t> core(grid.size(), 0);
    int64_t budget = kMaxEvaluations;
    for (bool coresLeft = true; coresLeft;) {
      evaluator.setCore(core);
      result.cores.push_back(llvm::to_vector(core));
      SmallVector<Block> &blocks = result.perCore.emplace_back();
      // Loop extents may depend only on block factors, never on the core.
      for (Loop &loop : loops) {
        if (!loop.op.hasConstantLowerBound() ||
            loop.op.getConstantLowerBound() != 0 ||
            loop.op.getStepAsInt() != 1) {
          fail("a blocking loop around it does not start at 0 with step 1");
          return std::nullopt;
        }
        AffineMap ub = loop.op.getUpperBoundMap();
        SmallVector<int64_t> operands;
        for (Value v : loop.op.getUpperBoundOperands()) {
          std::optional<int64_t> e = evaluator.eval(v);
          if (!e) {
            fail("a blocking loop bound cannot be evaluated");
            return std::nullopt;
          }
          operands.push_back(*e);
        }
        std::optional<SmallVector<int64_t>> bound = evalMap(ub, operands);
        if (ub.getNumResults() != 1 || !bound) {
          fail("a blocking loop bound cannot be evaluated");
          return std::nullopt;
        }
        loop.extent = bound->front();
      }
      SmallVector<int64_t> iv(loops.size(), 0);
      bool itersLeft =
          llvm::all_of(loops, [](Loop &l) { return l.extent > 0; });
      while (itersLeft) {
        bool isFinal = true;
        for (auto [l, loop] : llvm::enumerate(loops)) {
          evaluator.setIV(loop.op.getInductionVar(), iv[l]);
          if (loop.reduction && iv[l] != loop.extent - 1) {
            isFinal = false;
          }
        }
        if (!finalOnly || isFinal) {
          budget -= shardVolume;
          if (budget < 0) {
            fail("too many blocks to analyze");
            return std::nullopt;
          }
          SmallVector<int64_t> coord;
          for (Value index : indices) {
            std::optional<int64_t> e = evaluator.eval(index);
            if (!e) {
              fail("an index of an access of it cannot be evaluated");
              return std::nullopt;
            }
            coord.push_back(*e);
          }
          Block &block = blocks.emplace_back();
          SmallVector<int64_t> shardPos(rank, 0);
          for (int64_t n = 0; n < shardVolume; ++n) {
            SmallVector<int64_t> full(coord);
            full.append(shardPos);
            std::optional<SmallVector<int64_t>> mapped =
                evalMap(toBuffer, full);
            if (!mapped) {
              fail("a view of it cannot be evaluated");
              return std::nullopt;
            }
            SmallVector<int64_t> &b = *mapped;
            int64_t row = b[0] * shardShape[0] + b[2];
            int64_t col = b[1] * shardShape[1] + b[3];
            block.tiles.push_back(row * tileShape[1] + col);
            int64_t owner = b[0] * gridShape[1] + b[1];
            if (block.owner >= 0 && block.owner != owner) {
              fail("one access of it spans two shards");
              return std::nullopt;
            }
            block.owner = owner;
            for (int64_t d = rank - 1; d >= 0; --d) {
              if (++shardPos[d] < accessShard[d]) {
                break;
              }
              shardPos[d] = 0;
            }
          }
          llvm::sort(block.tiles);
        }
        itersLeft = false;
        for (int64_t l = static_cast<int64_t>(loops.size()) - 1; l >= 0; --l) {
          if (++iv[l] < loops[l].extent) {
            itersLeft = true;
            break;
          }
          iv[l] = 0;
        }
      }
      coresLeft = false;
      for (int64_t d = static_cast<int64_t>(grid.size()) - 1; d >= 0; --d) {
        if (++core[d] < grid[d]) {
          coresLeft = true;
          break;
        }
        core[d] = 0;
      }
    }
    return result;
  }

  SpatialOp spatial;
  SpatialEdgeAttr edge;
  SmallVector<int64_t> gridShape, shardShape, tileShape;
};

static std::string edgeName(SpatialEdgeAttr edge) {
  return std::to_string(edge.getProducerRegion()) + ":" +
         std::to_string(edge.getProducerOutput()) + " -> " +
         std::to_string(edge.getConsumerRegion()) + ":" +
         std::to_string(edge.getConsumerArg());
}

class D2MMaterializeSpatialStreams
    : public impl::D2MMaterializeSpatialStreamsBase<
          D2MMaterializeSpatialStreams> {
public:
  using impl::D2MMaterializeSpatialStreamsBase<
      D2MMaterializeSpatialStreams>::D2MMaterializeSpatialStreamsBase;

  void runOnOperation() final {
    MLIRContext *ctx = &getContext();
    getOperation()->walk([&](SpatialOp spatial) {
      if (!spatial.hasEdges()) {
        return;
      }
      SmallVector<SpatialEdgeAttr> edges =
          llvm::to_vector(spatial.getEdgesAttr().getAsRange<SpatialEdgeAttr>());
      // Edges that leave one output share one protocol (R11): they stream
      // together or not at all.
      llvm::MapVector<std::pair<unsigned, unsigned>, SmallVector<unsigned>>
          groups;
      for (auto [index, edge] : llvm::enumerate(edges)) {
        if (edge.isStream()) {
          groups[{edge.getProducerRegion(), edge.getProducerOutput()}]
              .push_back(index);
        }
      }
      bool changed = false;
      for (auto &[source, members] : groups) {
        SmallVector<std::string> summaries;
        std::string failure;
        for (unsigned index : members) {
          EdgeAnalysis analysis(spatial, edges[index]);
          if (!analysis.run()) {
            failure = "edge " + edgeName(edges[index]) +
                      " stays a whole-tensor handoff: " + analysis.reason;
            break;
          }
          summaries.push_back("edge " + edgeName(edges[index]) +
                              " can stream: " + analysis.summary);
        }
        if (failure.empty()) {
          for (const std::string &s : summaries) {
            spatial.emitRemark(s);
          }
          continue;
        }
        if (members.size() > 1) {
          failure += " (so do the other edges that leave output " +
                     std::to_string(source.second) + " of region " +
                     std::to_string(source.first) + ")";
        }
        spatial.emitRemark(failure);
        for (unsigned index : members) {
          SpatialEdgeAttr edge = edges[index];
          edges[index] = SpatialEdgeAttr::get(
              ctx, edge.getProducerRegion(), edge.getProducerOutput(),
              edge.getConsumerRegion(), edge.getConsumerArg());
        }
        changed = true;
      }
      if (changed) {
        spatial.setEdgesAttr(
            ArrayAttr::get(ctx, llvm::to_vector_of<Attribute>(edges)));
      }
    });
  }
};

} // namespace
} // namespace mlir::tt::d2m
