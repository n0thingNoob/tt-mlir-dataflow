// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "SpatialMapping.h"
#include "ttmlir/Dialect/D2M/Utils/SpatialPipeline.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::tt::d2m {
namespace {

static bool isPreparation(Operation *op) {
  if (!op->getRegions().empty()) {
    return false;
  }
  if (isa<EmptyOp, tensor::EmptyOp>(op)) {
    return true;
  }
  if (isa<ToLayoutOp>(op)) {
    return llvm::all_of(op->getOperandTypes(),
                        [](Type type) { return isa<RankedTensorType>(type); });
  }
  return isMemoryEffectFree(op) &&
         (op->hasTrait<OpTrait::ConstantLike>() ||
          isa<ViewLayoutOp, tensor::CastOp, tensor::CollapseShapeOp,
              tensor::ExpandShapeOp, tensor::ExtractSliceOp>(op));
}

// Collect a topological closure of the small, explicitly supported set of
// tensor preparation operations. Never move a generic or speculate effects.
static bool collectPreparations(Value value, Operation *insertion,
                                DominanceInfo &dominance,
                                llvm::SetVector<Operation *> &preparations) {
  if (dominance.dominates(value, insertion)) {
    return true;
  }
  Operation *def = value.getDefiningOp();
  if (!def || def->getBlock() != insertion->getBlock() || !isPreparation(def)) {
    return false;
  }
  for (Value operand : def->getOperands()) {
    if (!collectPreparations(operand, insertion, dominance, preparations)) {
      return false;
    }
  }
  preparations.insert(def);
  return true;
}

// Fresh destinations with no escaping aliases are sufficient for this
// prototype. Uses inside the owning generic are expected (remote_store).
static bool freshDestination(Value value, Operation *owner,
                             llvm::SmallPtrSetImpl<Operation *> &visited) {
  Operation *def = value.getDefiningOp();
  if (!def || !visited.insert(def).second) {
    return false;
  }
  for (Operation *user : value.getUsers()) {
    if (user != owner && !owner->isProperAncestor(user)) {
      return false;
    }
  }
  if (isa<EmptyOp, tensor::EmptyOp>(def)) {
    return true;
  }
  if (auto layout = dyn_cast<ToLayoutOp>(def)) {
    // A tensor layout conversion writes its DPS destination, not its input.
    return freshDestination(layout.getOutput(), def, visited);
  }
  return false;
}

static bool safeDestinations(GenericOp first, GenericOp second) {
  llvm::SmallPtrSet<Operation *, 8> roots;
  for (GenericOp generic : {first, second}) {
    for (Value output : generic.getOutputs()) {
      if (!freshDestination(output, generic, roots)) {
        return false;
      }
    }
  }
  return true;
}

static Attribute range(MLIRContext *ctx, int64_t y, int64_t x, int64_t h,
                       int64_t w) {
  return ttcore::CoreRangeAttr::get(
      ctx, ttcore::CoreCoordAttr::get(ctx, y, x),
      ttcore::CoreCoordAttr::get(ctx, y + h - 1, x + w - 1));
}
// A legality check only: no scores or alternative mapping search. The memory
// bound intentionally overestimates concurrent storage for this first policy.
static std::string checkPipelineCandidate(
    const SpatialGenericDAG &dag, const RegionCandidate &candidate,
    ArrayRef<int64_t> shape, ttcore::ChipDescAttr chip,
    DominanceInfo &dominance, llvm::SetVector<Operation *> &preparations) {
  std::string rejection;
  RankedTensorType common;
  uint64_t bytes = 0;
  unsigned cbCount = 0;
  llvm::SmallPtrSet<Operation *, 8> destinations;
  GenericOp anchor = dag.nodes[candidate.members.front()].generic;
  for (unsigned id : candidate.members) {
    GenericOp g = dag.nodes[id].generic;
    if (g.getNumResults() != 1 || !g.isAllParallel() || g.getNumDims() != 2 ||
        g.getGrid().getShape() != ArrayRef<int64_t>({1, 1}) ||
        !llvm::all_of(
            dag.nodes[id].captures,
            [&](Value v) { return llvm::is_contained(g->getOperands(), v); }) ||
        !llvm::all_of(g.getIndexingMapsValue(),
                      [](AffineMap map) { return map.isIdentity(); })) {
      rejection = "requires single-output 2D elementwise stages";
      break;
    }
    llvm::SetVector<Value> distinctInputs;
    distinctInputs.insert(g.getInputs().begin(), g.getInputs().end());
    if (distinctInputs.size() != g.getInputs().size()) {
      rejection = "repeated inputs require temporal fallback";
      break;
    }
    auto type = cast<RankedTensorType>(g.getResult(0).getType());
    auto tile = dyn_cast<ttcore::TileType>(type.getElementType());
    auto layout = cast<ttcore::MetalLayoutAttr>(type.getEncoding());
    auto logical = layout.getLogicalShape();
    if (!tile || !tile.getElementType().isBF16() || tile.getHeight() != 32 ||
        tile.getWidth() != 32 || type.getRank() != 4 ||
        !llvm::all_of(g.getInputs(),
                      [&](Value v) { return v.getType() == type; }) ||
        logical.size() != 2 || logical[0] <= 0 || logical[1] <= 0 ||
        logical[0] % 32 || logical[1] % 32 || (common && common != type)) {
      rejection = "requires matching tile-aligned static 2D BF16 layouts";
      break;
    }
    common = type;
    unsigned computeCount = 0;
    bool unsupported = false;
    g.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name.starts_with("d2m.tile_")) {
        ++computeCount;
        unsupported |= name != "d2m.tile_add" && name != "d2m.tile_relu" &&
                       name != "d2m.tile_negative";
      }
    });
    if (unsupported || computeCount != 1 ||
        !freshDestination(g.getOutputs()[0], g, destinations)) {
      rejection = "unsupported compute or destination alias";
      break;
    }
    // Conservatively reserve all full tensors plus CBs and
    // semaphore/alignment overhead on every core. Allocation performs the
    // final address check.
    cbCount += g.getInputs().size() + g.getOutputs().size();
    bytes += type.getNumElements() * 2048 * (g.getInputs().size() + 1) + 65536;
    for (Value operand : g->getOperands()) {
      const auto *edge =
          llvm::find_if(dag.dependencies, [&](const SpatialDependency &d) {
            return d.consumer == id && d.target == operand && d.producer &&
                   llvm::is_contained(candidate.members, *d.producer);
          });
      if (edge != dag.dependencies.end()) {
        Value original = operand;
        // Only cancel an exact tiled -> plain -> same tiled round trip.
        if (original != edge->source) {
          auto outer = original.getDefiningOp<ToLayoutOp>();
          auto inner = outer ? outer.getInput().getDefiningOp<ToLayoutOp>()
                             : ToLayoutOp();
          auto plain =
              inner ? dyn_cast<RankedTensorType>(inner.getResult(0).getType())
                    : RankedTensorType();
          if (!inner || inner.getInput() != edge->source || !plain ||
              plain.getEncoding() || !plain.getElementType().isBF16() ||
              plain.getShape() != logical ||
              original.getType() != edge->source.getType()) {
            rejection = "internal layout transformation is not an identity "
                        "round trip";
            break;
          }
        }
      } else if (!collectPreparations(operand, anchor, dominance,
                                      preparations)) {
        rejection = "external input does not dominate the pipeline";
        break;
      }
    }
    if (!rejection.empty()) {
      break;
    }
  }
  if (candidate.members.size() > static_cast<uint64_t>(shape[0] * shape[1])) {
    rejection = "insufficient worker cores";
  }
  if (bytes > chip.getL1Size() - chip.getL1UnreservedBase()) {
    rejection = "full tensors and conservative CB reservation exceed L1";
  }
  if (cbCount > chip.getNumCBs()) {
    rejection = "insufficient circular buffer ports for one program";
  }
  return rejection;
}

// Record the deferred pipeline contract; buffer-form materialization runs
// after layout and reblocking have established the final tile traversal.
static void prepareSpatialPipeline(const SpatialGenericDAG &dag,
                                   const SelectedSpatialGroup &group,
                                   bool dump) {
  GenericOp first = dag.nodes[group.members.front()].generic;
  OpBuilder builder(first);
  auto func = first->getParentOfType<func::FuncOp>();
  auto previous =
      func->getAttrOfType<IntegerAttr>(spatial_pipeline::expectedStages);
  func->setAttr(spatial_pipeline::expectedStages,
                builder.getI64IntegerAttr((previous ? previous.getInt() : 0) +
                                          group.members.size()));
  for (auto [stage, id] : llvm::enumerate(group.members)) {
    GenericOp generic = dag.nodes[id].generic;
    generic->setAttr(spatial_pipeline::group,
                     builder.getI64IntegerAttr(group.members.front()));
    generic->setAttr(spatial_pipeline::stage, builder.getI64IntegerAttr(stage));
    generic->setAttr(spatial_pipeline::core, group.ranges[stage]);
    SmallVector<int64_t> producers(generic.getInputs().size(), -1);
    for (const auto &edge : dag.dependencies) {
      if (edge.consumer != id || !edge.producer ||
          !llvm::is_contained(group.members, *edge.producer)) {
        continue;
      }
      auto input = llvm::find(generic.getInputs(), edge.target);
      if (input == generic.getInputs().end()) {
        continue;
      }
      unsigned index = std::distance(generic.getInputs().begin(), input);
      producers[index] = std::distance(
          group.members.begin(), llvm::find(group.members, *edge.producer));
      Value target = edge.target;
      target.replaceUsesWithIf(edge.source, [&](OpOperand &use) {
        return use.getOwner() == generic ||
               generic->isProperAncestor(use.getOwner());
      });
    }
    generic->setAttr(spatial_pipeline::inputs,
                     builder.getDenseI64ArrayAttr(producers));
  }
  if (dump) {
    llvm::errs() << "selected pipeline members=[";
    llvm::interleaveComma(group.members, llvm::errs(),
                          [](unsigned id) { llvm::errs() << "G" << id; });
    llvm::errs() << "] internal=L1->NoC->L1 ranges="
                 << builder.getArrayAttr(group.ranges) << '\n';
  }
}

} // namespace

FailureOr<SmallVector<SelectedSpatialGroup, 0>>
selectSpatialGroups(const SpatializationAnalysisResult &analysis, bool dump,
                    bool pipelines) {
  const auto &dag = analysis.dag;
  SmallVector<SelectedSpatialGroup, 0> groups;
  if (dag.nodes.empty()) {
    return groups;
  }
  GenericOp first = dag.nodes.front().generic;
  auto device = ttcore::lookupDeviceOp(first);
  if (!device) {
    first.emitOpError("spatial materialization requires a registered device");
    return failure();
  }
  if (device.getDeviceAttr().getChipIds().size() != 1) {
    first.emitOpError("spatial materialization supports only a single device");
    return failure();
  }
  auto shape = device.getDeviceAttr().getWorkerGrid().getShape();
  if (shape.size() != 2 || shape[0] <= 0 || shape[1] <= 0) {
    first.emitOpError(
        "spatial materialization requires a positive 2D worker grid");
    return failure();
  }
  for (const auto &node : dag.nodes) {
    GenericOp generic = node.generic;
    auto grid = generic.getGrid();
    auto size = grid.getShape();
    if (generic.getOutputs().empty() ||
        !llvm::all_of(generic.getOutputs(), [](Value value) {
          auto tensor = dyn_cast<RankedTensorType>(value.getType());
          return tensor &&
                 isa_and_nonnull<ttcore::MetalLayoutAttr>(tensor.getEncoding());
        })) {
      generic.emitOpError(
          "spatial materialization requires TTMetal tensor outputs");
      return failure();
    }
    if (size.size() != 2 || !grid.getPhysicalToVirtMap().isEmpty() ||
        !grid.getVirtToPhysicalMap().isEmpty() ||
        generic.isExplicitDatamovementForm() ||
        generic->hasAttr("d2m.skip_grid_selection") || size[0] <= 0 ||
        size[1] <= 0 || size[0] > shape[0] || size[1] > shape[1] ||
        !llvm::all_of(generic->getOperandTypes(), [](Type type) {
          auto tensor = dyn_cast<RankedTensorType>(type);
          return tensor && tensor.hasStaticShape();
        })) {
      generic.emitOpError("spatial materialization supports only static tensor "
                          "generics with ordinary 2D grids fitting the device");
      return failure();
    }
  }
  llvm::BitVector usedCandidates(analysis.candidates.size());
  llvm::BitVector covered(dag.nodes.size());
  DominanceInfo dominance;
  if (pipelines) {
    auto system = ttcore::getCurrentScopeSystemDesc(first);
    auto chip = system.getChipDescs().front();
    for (auto [candidateId, candidate] : llvm::enumerate(analysis.candidates)) {
      if (!llvm::is_contained(candidate.kinds,
                              RegionCandidateKind::ProducerConsumerChain) &&
          !llvm::is_contained(candidate.kinds, RegionCandidateKind::ForkJoin)) {
        continue;
      }
      if (llvm::any_of(candidate.members,
                       [&](unsigned id) { return covered[id]; })) {
        continue;
      }
      llvm::SetVector<Operation *> preparations;
      std::string rejection = checkPipelineCandidate(
          dag, candidate, shape, chip, dominance, preparations);
      if (!rejection.empty()) {
        if (dump) {
          llvm::errs() << "  pipeline fallback R" << candidateId << ": "
                       << rejection << '\n';
        }
        continue;
      }
      SelectedSpatialGroup group;
      group.pipeline = true;
      group.members = candidate.members;
      group.preparations.assign(preparations.begin(), preparations.end());
      for (auto [stage, id] : llvm::enumerate(group.members)) {
        group.ranges.push_back(range(first.getContext(), stage / shape[1],
                                     stage % shape[1], 1, 1));
        covered.set(id);
      }
      usedCandidates.set(candidateId);
      groups.push_back(std::move(group));
    }
  }
  for (unsigned a = 0; a < dag.nodes.size(); ++a) {
    if (covered[a]) {
      continue;
    }
    GenericOp ga = dag.nodes[a].generic;
    SelectedSpatialGroup group;
    group.members.push_back(a);
    group.reason = "temporal fallback: no legal unused parallel partner";
    for (auto [candidateId, candidate] : llvm::enumerate(analysis.candidates)) {
      if (!llvm::is_contained(candidate.kinds,
                              RegionCandidateKind::ParallelBranches) ||
          !llvm::is_contained(candidate.members, a)) {
        continue;
      }
      for (unsigned b : candidate.members) {
        if (b <= a || covered[b]) {
          continue;
        }
        GenericOp gb = dag.nodes[b].generic;
        std::string rejection;
        llvm::SetVector<Operation *> preparations;
        auto as = ga.getGrid().getShape();
        auto bs = gb.getGrid().getShape();
        bool horizontal =
            std::max(as[0], bs[0]) <= shape[0] && as[1] + bs[1] <= shape[1];
        bool vertical =
            as[0] + bs[0] <= shape[0] && std::max(as[1], bs[1]) <= shape[1];
        if (dag.nodes[a].interval != dag.nodes[b].interval ||
            !dag.independent(a, b)) {
          rejection = "dependency or analysis boundary";
        } else if (!safeDestinations(ga, gb)) {
          rejection = "shared or unknown destination storage";
        } else if (!horizontal && !vertical) {
          rejection = "two grids do not fit side by side";
        } else {
          for (Value input : gb->getOperands()) {
            if (!collectPreparations(input, ga, dominance, preparations)) {
              rejection = "input preparation cannot be safely hoisted";
              break;
            }
          }
          for (Value capture : dag.nodes[b].captures) {
            if (!collectPreparations(capture, ga, dominance, preparations)) {
              rejection = "captured value cannot be safely hoisted";
              break;
            }
          }
        }
        if (!rejection.empty()) {
          if (dump) {
            llvm::errs() << "  skip R" << candidateId << " pair G" << a << ",G"
                         << b << ": " << rejection << '\n';
          }
          continue;
        }
        usedCandidates.set(candidateId);
        group.members.push_back(b);
        group.preparations.assign(preparations.begin(), preparations.end());
        group.ranges = {range(ga.getContext(), 0, 0, as[0], as[1]),
                        range(ga.getContext(), horizontal ? 0 : as[0],
                              horizontal ? as[1] : 0, bs[0], bs[1])};
        group.reason =
            "parallel: independent pair from R" + std::to_string(candidateId);
        break;
      }
      if (group.members.size() == 2) {
        break;
      }
    }
    if (group.members.size() == 1) {
      group.ranges.push_back(range(ga.getContext(), 0, 0, shape[0], shape[1]));
    }
    for (unsigned id : group.members) {
      assert(!covered[id] && "generic selected twice");
      covered.set(id);
    }
    groups.push_back(std::move(group));
  }
  if (dump) {
    for (auto [id, candidate] : llvm::enumerate(analysis.candidates)) {
      if (usedCandidates[id]) {
        continue;
      }
      llvm::errs()
          << "  not selected R" << id << ": "
          << (llvm::is_contained(candidate.kinds,
                                 RegionCandidateKind::ParallelBranches)
                  ? "members already covered or no legal pair"
                  : "structural candidate; use pairs or temporal singletons")
          << '\n';
    }
  }
  assert(covered.all() && "unmapped generic");
  return groups;
}

void materializeSpatialGroups(const SpatialGenericDAG &dag,
                              ArrayRef<SelectedSpatialGroup> groups,
                              bool dump) {
  for (auto [groupId, group] : llvm::enumerate(groups)) {
    GenericOp first = dag.nodes[group.members.front()].generic;
    for (Operation *prep : group.preparations) {
      // A previous group may already have hoisted a shared preparation.
      if (!prep->isBeforeInBlock(first)) {
        prep->moveBefore(first);
      }
    }
    if (group.pipeline) {
      prepareSpatialPipeline(dag, group, dump);
      continue;
    }
    llvm::SetVector<Value> inputs;
    SmallVector<Value> outputs;
    SmallVector<Type> types;
    for (unsigned id : group.members) {
      GenericOp generic = dag.nodes[id].generic;
      generic->setAttr(spatial_pipeline::noSpill,
                       UnitAttr::get(generic.getContext()));
      inputs.insert(generic.getInputs().begin(), generic.getInputs().end());
      outputs.append(generic.getOutputs().begin(), generic.getOutputs().end());
      types.append(generic.getResultTypes().begin(),
                   generic.getResultTypes().end());
    }
    OpBuilder builder(first);
    auto spatial = builder.create<SpatialOp>(
        first.getLoc(), types, inputs.getArrayRef(), outputs,
        builder.getArrayAttr(group.ranges), group.members.size());
    unsigned resultIndex = 0;
    for (auto [regionId, id] : llvm::enumerate(group.members)) {
      GenericOp generic = dag.nodes[id].generic;
      Block *body = builder.createBlock(&spatial.getRegions()[regionId]);
      generic->moveBefore(body, body->end());
      builder.setInsertionPointToEnd(body);
      builder.create<SpatialYieldOp>(generic.getLoc(), generic.getResults());
      for (Value result : generic.getResults()) {
        Value replacement = spatial.getResult(resultIndex++);
        result.replaceUsesWithIf(replacement, [&](OpOperand &use) {
          return !spatial->isProperAncestor(use.getOwner());
        });
      }
    }
    if (dump) {
      llvm::errs() << "selected S" << groupId << " members=[";
      llvm::interleaveComma(group.members, llvm::errs(),
                            [](unsigned id) { llvm::errs() << "G" << id; });
      llvm::errs() << "] reason=" << group.reason
                   << " ranges=" << spatial.getGridRanges() << "\nemitted ";
      spatial.print(llvm::errs());
      llvm::errs() << '\n';
    }
  }
}
} // namespace mlir::tt::d2m
