// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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
  int64_t columns;
  bool wholeRow;
  bool sharded;
};

// Only exact reblocking views preserve the logical row-major tile sequence.
// Sharing an allocation alone does not prove that two views name the same tile.
static Value reblockRoot(Value value) {
  while (auto view = value.getDefiningOp<ViewLayoutOp>()) {
    if (view.getReinterpretLayout() || !view.isReblockOnly()) {
      return {};
    }
    value = view.getInput();
  }
  return value;
}

static SmallVector<int64_t> tiledShape(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type || !type.hasStaticShape() || type.getRank() != 4) {
    return {};
  }
  auto shape = type.getShape();
  return {shape[0] * shape[2], shape[1] * shape[3]};
}

// GenerateOuterLoops produces iv + block_offset(dim). A single-core stage
// has zero logical block offset even when placed at a nonzero physical core.
static bool isLoopIndex(Value value, affine::AffineForOp loop, unsigned dim,
                        bool requireOffset = false) {
  if (value == loop.getInductionVar()) {
    return !requireOffset;
  }
  auto add = value.getDefiningOp<arith::AddIOp>();
  if (!add) {
    return false;
  }
  for (unsigned i = 0; i != 2; ++i) {
    auto offset = add->getOperand(i).getDefiningOp<BlockOffsetOp>();
    if (offset && offset.getDim() == dim &&
        add->getOperand(1 - i) == loop.getInductionVar()) {
      return true;
    }
  }
  return false;
}

// Prove the standard loop nest, rather than inferring traversal from shape.
// K has exactly one outer iteration: the linalg body computes the full panel
// reduction before the one output store and therefore before any notification.
static std::string
checkStageTraversal(GenericOp generic,
                    SmallVectorImpl<RemoteLoadOp> &inputLoads) {
  auto factors = generic.getBlockFactorsValue();
  bool matmul = generic.hasReduction();
  auto shardAttr =
      generic->getAttrOfType<IntegerAttr>(spatial_pipeline::shards);
  int64_t shards = shardAttr ? shardAttr.getInt() : 1;
  if (factors.size() != (matmul ? 3u : 2u) ||
      generic.getGrid().getShape() != ArrayRef<int64_t>({shards, 1}) ||
      generic.getOutputs().size() != 1 || (matmul && factors[2] != 1)) {
    return "requires an M-only grid and a complete, unsplit K reduction";
  }
  auto outputShape = tiledShape(generic.getOutputs()[0]);
  if (outputShape.size() != 2 || outputShape[0] != factors[0] * shards ||
      outputShape[1] != factors[1] || factors[0] <= 0 || factors[1] <= 0 ||
      factors[0] > INT32_MAX / factors[1]) {
    return "output blocking must cover the row-major tile grid exactly once";
  }
  SmallVector<affine::AffineForOp> loops;
  Block *body = &generic.getRegion(0).front();
  for (unsigned dim = 0; dim < factors.size(); ++dim) {
    SmallVector<affine::AffineForOp> nested(
        body->getOps<affine::AffineForOp>());
    if (nested.size() != 1) {
      return "requires one canonical loop per blocking dimension";
    }
    auto loop = nested.front();
    auto bound = loop.getUpperBoundOperands();
    auto factor = bound.size() == 1
                      ? bound.front().getDefiningOp<GetBlockFactorOp>()
                      : GetBlockFactorOp();
    bool correctUpper =
        loop.hasConstantUpperBound()
            ? loop.getConstantUpperBound() == factors[dim]
            : factor && factor.getDim() == dim &&
                  loop.getUpperBoundMap() ==
                      AffineMap::get(
                          0, 1, getAffineSymbolExpr(0, generic.getContext()));
    if (!loop.hasConstantLowerBound() || loop.getConstantLowerBound() != 0 ||
        loop.getStep() != 1 || !correctUpper) {
      return "incompatible blocking traversal: expected row, column, then full "
             "K";
    }
    loops.push_back(loop);
    body = loop.getBody();
  }
  SmallVector<RemoteLoadOp> loads;
  SmallVector<RemoteStoreOp> stores;
  SmallVector<linalg::GenericOp> computes;
  generic.walk([&](RemoteLoadOp op) { loads.push_back(op); });
  generic.walk([&](RemoteStoreOp op) { stores.push_back(op); });
  generic.walk([&](linalg::GenericOp op) { computes.push_back(op); });
  if (loads.size() != generic.getInputs().size() || stores.size() != 1 ||
      computes.size() != 1 || stores.front()->getBlock() != body ||
      computes.front()->getBlock() != body ||
      !computes.front()->isBeforeInBlock(stores.front())) {
    return "requires unconditional loads, one compute and one completed store "
           "per tile";
  }
  auto compute = computes.front();
  auto store = stores.front();
  if (compute.getIndexingMapsArray() != generic.getIndexingMapsValue() ||
      compute.getNumDpsInits() != 1 ||
      static_cast<size_t>(compute.getNumDpsInputs()) !=
          generic.getInputs().size() ||
      store.getLocalBuffer() != compute.getDpsInits()[0] ||
      cast<MemRefType>(store.getLocalBuffer().getType()).getShape() !=
          ArrayRef<int64_t>({1, 1}) ||
      store.getMemref() != generic.getOutputs()[0] ||
      store.getIndices().size() != 2 ||
      !isLoopIndex(store.getIndices()[0], loops[0], 0, shards > 1) ||
      !isLoopIndex(store.getIndices()[1], loops[1], 1)) {
    return "output store does not publish one complete row-major tile";
  }
  for (auto [id, input] : llvm::enumerate(generic.getInputs())) {
    auto matchesInput = [operand = input](RemoteLoadOp load) {
      return load.getMemref() == operand;
    };
    if (llvm::count_if(loads, matchesInput) != 1) {
      return "requires exactly one load per input and iteration";
    }
    RemoteLoadOp load = *llvm::find_if(loads, matchesInput);
    unsigned rowDim = matmul && id == 1 ? 2 : 0;
    unsigned colDim = matmul && id == 0 ? 2 : 1;
    auto shape = tiledShape(input);
    if (shape.size() != 2 || load->getBlock() != body ||
        !load->isBeforeInBlock(compute) || load.getIndices().size() != 2 ||
        !isLoopIndex(load.getIndices()[0], loops[rowDim], rowDim,
                     shards > 1 && rowDim == 0) ||
        !isLoopIndex(load.getIndices()[1], loops[colDim], colDim)) {
      return "input traversal does not match its tile/panel dependency";
    }
    SmallVector<int64_t> panel = {matmul && id == 1 ? shape[0] : 1,
                                  matmul && id == 0 ? shape[1] : 1};
    if (cast<MemRefType>(load.getLocalBuffer().getType()).getShape() !=
            ArrayRef<int64_t>(panel) ||
        load.getLocalBuffer() != compute.getDpsInputs()[id]) {
      return "input blocking does not provide the complete tile or K panel";
    }
    inputLoads.push_back(load);
  }
  return {};
}

static std::string collectReadyEdges(ArrayRef<GenericOp> stages,
                                     SmallVectorImpl<ReadyEdge> &edges) {
  for (auto [id, stageRef] : llvm::enumerate(stages)) {
    GenericOp generic = stageRef;
    // Reuse the validated loads in operand order when constructing edges.
    SmallVector<RemoteLoadOp> inputLoads;
    if (auto reason = checkStageTraversal(generic, inputLoads);
        !reason.empty()) {
      return "stage " + std::to_string(id) + ": " + reason;
    }
    auto producers =
        generic->getAttrOfType<DenseI64ArrayAttr>(spatial_pipeline::inputs);
    for (auto [inputId, producerId] : llvm::enumerate(producers.asArrayRef())) {
      if (producerId == -1) {
        continue;
      }
      Value input = generic.getInputs()[inputId];
      GenericOp producer = stages[producerId];
      Value output = producer.getOutputs()[0];
      auto shape = tiledShape(output);
      bool wholeRow = generic.hasReduction();
      Value root = reblockRoot(input);
      if (!root || root != reblockRoot(output) || tiledShape(input) != shape ||
          ttcore::getMemorySpace(input) != ttcore::MemorySpace::DeviceL1 ||
          (wholeRow && inputId != 0)) {
        return "edge does not preserve producer L1 tile coordinates";
      }
      edges.push_back({static_cast<unsigned>(producerId),
                       static_cast<unsigned>(id), inputLoads[inputId], shape[1],
                       wholeRow, generic.getGrid().getShape()[0] > 1});
    }
  }
  return {};
}

static void insertReadyWait(OpBuilder &builder, const ReadyEdge &edge,
                            Value semaphore) {
  RemoteLoadOp load = edge.load;
  builder.setInsertionPoint(load);
  auto indices = load.getIndices();
  Value columns =
      builder.create<arith::ConstantIndexOp>(load.getLoc(), edge.columns);
  Value one = builder.create<arith::ConstantIndexOp>(load.getLoc(), 1);
  Value rowIndex = indices[0];
  if (edge.sharded) {
    Value offset = builder.create<BlockOffsetOp>(load.getLoc(), int64_t{0});
    rowIndex = builder.create<arith::SubIOp>(load.getLoc(), rowIndex, offset);
  }
  Value count;
  if (edge.wholeRow) {
    Value nextRow = builder.create<arith::AddIOp>(load.getLoc(), rowIndex, one);
    count = builder.create<arith::MulIOp>(load.getLoc(), nextRow, columns);
  } else {
    Value row = builder.create<arith::MulIOp>(load.getLoc(), rowIndex, columns);
    Value ordinal =
        builder.create<arith::AddIOp>(load.getLoc(), row, indices[1]);
    count = builder.create<arith::AddIOp>(load.getLoc(), ordinal, one);
  }
  auto wait = builder.create<SemaphoreWaitOp>(load.getLoc(), semaphore, count);
  wait->setAttr(spatial_pipeline::wait, builder.getUnitAttr());
}

// Late fallback retains the already established layouts and placement. There
// is no ready protocol to undo: the entire group is checked before mutation.
static void materializeTemporalStages(ArrayRef<GenericOp> stages, bool dump) {
  for (GenericOp stage : stages) {
    OpBuilder builder(stage);
    auto range = stage->getAttr(spatial_pipeline::core);
    auto spatial = builder.create<SpatialOp>(
        stage.getLoc(), TypeRange{}, stage.getInputs(), stage.getOutputs(),
        builder.getArrayAttr({range}), 1);
    for (StringRef attr :
         {spatial_pipeline::group, spatial_pipeline::stage,
          spatial_pipeline::core, spatial_pipeline::inputs,
          spatial_pipeline::signals, spatial_pipeline::shards}) {
      stage->removeAttr(attr);
    }
    stage->setAttr(spatial_pipeline::noSpill, builder.getUnitAttr());
    Block *body = builder.createBlock(&spatial.getRegions().front());
    stage->moveBefore(body, body->end());
    if (dump) {
      llvm::errs() << "emitted temporal fallback ";
      spatial.print(llvm::errs());
      llvm::errs() << '\n';
    }
  }
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
    auto device = ttcore::lookupDeviceOp(first);
    if (!device) {
      return first.emitOpError(
          "prepared pipeline requires a registered device");
    }
    auto grid = device.getDeviceAttr().getWorkerGrid().getShape();
    llvm::SmallDenseSet<std::pair<int64_t, int64_t>> occupied;
    int64_t commonShards = 0;
    SmallVector<int64_t> commonRangeShape;
    for (auto [id, stageRef] : llvm::enumerate(stages)) {
      GenericOp stage = stageRef;
      auto stageId = stage->getAttrOfType<IntegerAttr>(spatial_pipeline::stage);
      auto core =
          stage->getAttrOfType<ttcore::CoreRangeAttr>(spatial_pipeline::core);
      auto producers =
          stage->getAttrOfType<DenseI64ArrayAttr>(spatial_pipeline::inputs);
      if (!stageId || stageId.getInt() != static_cast<int64_t>(id) || !core ||
          !producers ||
          static_cast<size_t>(producers.size()) != stage.getInputs().size() ||
          llvm::any_of(
              producers.asArrayRef(),
              [stageIndex = static_cast<int64_t>(id)](int64_t producer) {
                return producer < -1 || producer >= stageIndex;
              })) {
        return stage.emitOpError("invalid prepared spatial pipeline contract");
      }
      auto start = core.getStartCoord();
      auto end = core.getEndCoord();
      auto shardAttr =
          stage->getAttrOfType<IntegerAttr>(spatial_pipeline::shards);
      int64_t shards = shardAttr ? shardAttr.getInt() : 1;
      SmallVector<int64_t> rangeShape = {end.getY() - start.getY() + 1,
                                         end.getX() - start.getX() + 1};
      if (id == 0) {
        commonShards = shards;
        commonRangeShape = rangeShape;
      }
      if (grid.size() != 2 || shards <= 0 || shards != commonShards ||
          rangeShape != commonRangeShape || rangeShape[0] <= 0 ||
          rangeShape[1] <= 0 || rangeShape[0] * rangeShape[1] != shards ||
          start.getY() < 0 || start.getX() < 0 || end.getY() >= grid[0] ||
          end.getX() >= grid[1]) {
        return stage.emitOpError(
            "pipeline requires compatible legal M-shard ranges");
      }
      auto forward = stage.getGrid().getVirtToPhysicalMap();
      auto inverse = stage.getGrid().getPhysicalToVirtMap();
      for (int64_t shard = 0; shard < shards; ++shard) {
        int64_t y = start.getY() + shard / rangeShape[1];
        int64_t x = start.getX() + shard % rangeShape[1];
        SmallVector<int64_t> physical = {0, shard, 0};
        SmallVector<int64_t> logical = {0, y, x};
        if (forward && !forward.isEmpty()) {
          if (forward.getNumDims() != 2 || forward.getNumSymbols() ||
              forward.getNumResults() != 3) {
            return stage.emitOpError("invalid pipeline forward grid mapping");
          }
          physical = forward.compose({shard, 0});
        }
        if (inverse && !inverse.isEmpty()) {
          if (inverse.getNumDims() != 2 || inverse.getNumSymbols() ||
              inverse.getNumResults() != 3) {
            return stage.emitOpError("invalid pipeline inverse grid mapping");
          }
          logical = inverse.compose({y, x});
        }
        if (physical != SmallVector<int64_t>({0, y, x}) ||
            logical != SmallVector<int64_t>({0, shard, 0})) {
          return stage.emitOpError(
              "pipeline grid mapping does not cover its range exactly once");
        }
      }
      for (int64_t y = start.getY(); y <= end.getY(); ++y) {
        for (int64_t x = start.getX(); x <= end.getX(); ++x) {
          if (!occupied.insert({y, x}).second) {
            return stage.emitOpError("pipeline core ranges overlap");
          }
        }
      }
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
    if (auto reason = collectReadyEdges(stages, edges); !reason.empty()) {
      if (dumpRegions) {
        llvm::errs() << "pipeline traversal fallback: " << reason << '\n';
      }
      materializeTemporalStages(stages, dumpRegions);
      return success();
    }
    for (Operation *prep : preparations) {
      prep->moveBefore(first);
    }
    builder.setInsertionPoint(first);
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
      if (commonShards == 1) {
        signals[producerId].push_back(builder.getDenseI64ArrayAttr(
            {index, core.getY() - origin.getY(), core.getX() - origin.getX()}));
      } else {
        signals[producerId].push_back(builder.getDenseI64ArrayAttr(
            {index, core.getY(), core.getX(), commonRangeShape[1]}));
      }
      insertReadyWait(builder, edge, sem);
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
