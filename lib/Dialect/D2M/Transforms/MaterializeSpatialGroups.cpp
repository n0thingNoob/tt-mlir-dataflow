// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Utils/SpatialGroupUtils.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MMATERIALIZESPATIALGROUPS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

static bool containsCore(ttcore::CoreRangeAttr r, int64_t y, int64_t x) {
  return r.getStartCoord().getY() <= y && y <= r.getEndCoord().getY() &&
         r.getStartCoord().getX() <= x && x <= r.getEndCoord().getX();
}

static bool overlaps(ttcore::CoreRangeAttr a, ttcore::CoreRangeAttr b) {
  return a.getStartCoord().getY() <= b.getEndCoord().getY() &&
         b.getStartCoord().getY() <= a.getEndCoord().getY() &&
         a.getStartCoord().getX() <= b.getEndCoord().getX() &&
         b.getStartCoord().getX() <= a.getEndCoord().getX();
}

struct Group {
  int64_t id;
  // Members in block order (this is also the region order).
  SmallVector<GenericOp> members;
};

class D2MMaterializeSpatialGroupsPass
    : public impl::D2MMaterializeSpatialGroupsBase<
          D2MMaterializeSpatialGroupsPass> {
public:
  using impl::D2MMaterializeSpatialGroupsBase<
      D2MMaterializeSpatialGroupsPass>::D2MMaterializeSpatialGroupsBase;

  void runOnOperation() final {
    WalkResult result = getOperation().walk([&](func::FuncOp func) {
      for (Block &block : func.getBody()) {
        llvm::MapVector<int64_t, Group> groups;
        for (GenericOp generic : block.getOps<GenericOp>()) {
          auto id = generic->getAttrOfType<IntegerAttr>(kSpatialGroupAttr);
          if (!id) {
            continue;
          }
          Group &group = groups[id.getInt()];
          group.id = id.getInt();
          group.members.push_back(generic);
        }
        for (auto &[id, group] : groups) {
          if (failed(materialize(group, block))) {
            return WalkResult::interrupt();
          }
        }
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }

private:
  LogicalResult checkRanges(Group &group,
                            SmallVectorImpl<ttcore::CoreRangeAttr> &ranges);
  LogicalResult materialize(Group &group, Block &block);
};

LogicalResult D2MMaterializeSpatialGroupsPass::checkRanges(
    Group &group, SmallVectorImpl<ttcore::CoreRangeAttr> &ranges) {
  ttcore::DeviceAttr device = ttcore::lookupDevice(group.members.front());
  ArrayRef<int64_t> workerGrid = device.getWorkerGrid().getShape();
  for (GenericOp generic : group.members) {
    if (generic.getNumResults() == 0 ||
        generic.getNumResults() != generic.getOutputs().size()) {
      return generic.emitOpError("spatial group member must be in tensor "
                                 "form and return each of its outputs");
    }
    auto range =
        generic->getAttrOfType<ttcore::CoreRangeAttr>(kSpatialRangeAttr);
    if (!range) {
      return generic.emitOpError("spatial group member needs a '")
             << kSpatialRangeAttr << "' core range";
    }
    if (range.getStartCoord().getY() < 0 || range.getStartCoord().getX() < 0 ||
        range.getEndCoord().getY() >= workerGrid[0] ||
        range.getEndCoord().getX() >= workerGrid[1] ||
        range.getStartCoord().getY() > range.getEndCoord().getY() ||
        range.getStartCoord().getX() > range.getEndCoord().getX()) {
      return generic.emitOpError("core range ")
             << range << " is empty or outside the worker grid";
    }
    for (size_t i = 0; i + 1 < excludeCores.size(); i += 2) {
      if (containsCore(range, excludeCores[i], excludeCores[i + 1])) {
        return generic.emitOpError("core range ")
               << range << " contains excluded core (" << excludeCores[i]
               << ", " << excludeCores[i + 1] << ")";
      }
    }
    for (auto [i, other] : llvm::enumerate(ranges)) {
      if (overlaps(range, other)) {
        return generic.emitOpError("core range ")
               << range << " overlaps " << other << " of group member " << i;
      }
    }
    ranges.push_back(range);
  }
  return success();
}

LogicalResult D2MMaterializeSpatialGroupsPass::materialize(Group &group,
                                                           Block &block) {
  // An earlier group may have moved some members after its spatial.
  llvm::sort(group.members,
             [](GenericOp a, GenericOp b) { return a->isBeforeInBlock(b); });
  SmallVector<ttcore::CoreRangeAttr> ranges;
  if (failed(checkRanges(group, ranges))) {
    return failure();
  }

  llvm::DenseMap<Operation *, unsigned> memberIndex;
  for (auto [i, generic] : llvm::enumerate(group.members)) {
    memberIndex[generic] = i;
  }

  // Convexity: no generic outside the group may lie on a path between two
  // members. Members are in block order, so walk the generics between the
  // first and the last member and propagate "reachable from a member".
  {
    llvm::DenseSet<Operation *> reachable(group.members.begin(),
                                          group.members.end());
    for (Operation &op :
         llvm::make_range(group.members.front()->getIterator(),
                          std::next(group.members.back()->getIterator()))) {
      auto generic = mlir::dyn_cast<GenericOp>(&op);
      if (!generic) {
        continue;
      }
      SmallVector<Value> operands(generic->getOperands());
      bool fromMember = false;
      bool fromOutsidePath = false;
      for (Value operand : operands) {
        for (Operation *p : producerGenerics(operand, block)) {
          if (!reachable.contains(p)) {
            continue;
          }
          fromMember = true;
          fromOutsidePath |= !memberIndex.contains(p);
        }
      }
      if (memberIndex.contains(generic) && fromOutsidePath) {
        return generic.emitOpError("spatial group ")
               << group.id << " is not convex: a generic outside the group "
               << "lies on a path into this member";
      }
      if (fromMember) {
        reachable.insert(generic);
      }
    }
  }

  // Classify every member input: an edge from another member (through a
  // foldable to_layout / no-op mask chain) or a value from outside.
  // A composite_view operand (a concatenation) is classified per input: each
  // input read from a member is its own edge (`compositeInput` is its index
  // in the view; -1 for a plain operand), and the view is rebuilt inside the
  // consumer region over the edges' block arguments.
  struct EdgeUse {
    unsigned operandIndex;
    unsigned producer;
    unsigned producerResult;
    int compositeInput = -1;
  };
  SmallVector<SmallVector<EdgeUse>> edgeUses(group.members.size());
  for (auto [c, generic] : llvm::enumerate(group.members)) {
    for (OpOperand &operand : generic.getInputsMutable()) {
      bool readsMember =
          llvm::any_of(producerGenerics(operand.get(), block),
                       [&](Operation *p) { return memberIndex.contains(p); });
      if (!readsMember) {
        continue;
      }
      SmallVector<std::pair<Value, int>> sources = {{operand.get(), -1}};
      if (auto composite = operand.get().getDefiningOp<CompositeViewOp>()) {
        sources.clear();
        for (auto [i, input] : llvm::enumerate(composite.getInputs())) {
          if (llvm::any_of(producerGenerics(input, block), [&](Operation *p) {
                return memberIndex.contains(p);
              })) {
            sources.push_back({input, static_cast<int>(i)});
          }
        }
      }
      for (auto [source, compositeInput] : sources) {
        OpResult src = traceEdgeSource(source);
        if (!src || !memberIndex.contains(src.getOwner())) {
          return generic.emitOpError("input ")
                 << operand.getOperandNumber()
                 << " reads another member of spatial group " << group.id
                 << " through ops that cannot be an edge (only to_layout, "
                    "no-op mask and a composite_view of such chains)";
        }
        Type dst = source.getType();
        if (!isL1Layout(src.getType()) ||
            (!isL1Layout(dst) && !isDRAMLayout(dst))) {
          return generic.emitOpError("input ")
                 << operand.getOperandNumber() << " reads another member of "
                 << "spatial group " << group.id
                 << " outside L1; an edge buffer must stay in L1";
        }
        if (!sameLayoutExceptMemorySpace(src.getType(), dst)) {
          return generic.emitOpError("input ")
                 << operand.getOperandNumber()
                 << " needs a layout change from its producer in spatial group "
                 << group.id << "; resharding an edge is not supported";
        }
        edgeUses[c].push_back({operand.getOperandNumber(),
                               memberIndex[src.getOwner()],
                               src.getResultNumber(), compositeInput});
      }
    }
  }

  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);
  // Values whose defining chain may become dead once the members stop using
  // them (folded edge chains and replaced output inits).
  llvm::SetVector<Operation *> maybeDead;
  auto noteMaybeDead = [&](Value v) {
    if (Operation *def = v.getDefiningOp()) {
      maybeDead.insert(def);
    }
  };
  GenericOp last = group.members.back();
  builder.setInsertionPoint(last);

  // Fresh L1 outputs. An edge buffer must be a d2m.empty on the producer's
  // cores; the other outputs get the same form that canonicalization would
  // give them.
  for (GenericOp generic : group.members) {
    for (OpOperand &init : generic.getOutputsMutable()) {
      Value fresh = builder.create<EmptyOp>(
          generic.getLoc(), init.get().getType(), nullptr, nullptr);
      Value old = init.get();
      old.replaceUsesWithIf(fresh, [&](OpOperand &use) {
        return generic->isAncestor(use.getOwner());
      });
      noteMaybeDead(old);
    }
  }

  // A to_layout that GridSelection may resize can feed only one generic
  // (recordGenericConsumer): give each member its own copy.
  for (GenericOp generic : group.members) {
    for (OpOperand &operand : generic.getInputsMutable()) {
      auto toLayout = operand.get().getDefiningOp<ToLayoutOp>();
      if (!toLayout || toLayout->getBlock() != &block) {
        continue;
      }
      bool shared = llvm::any_of(toLayout->getUsers(), [&](Operation *user) {
        return !generic->isAncestor(user);
      });
      if (!shared ||
          llvm::any_of(edgeUses[memberIndex[generic]], [&](EdgeUse &e) {
            return e.operandIndex == operand.getOperandNumber();
          })) {
        continue;
      }
      auto init = builder.create<EmptyOp>(
          toLayout.getLoc(), toLayout.getOutput().getType(), nullptr, nullptr);
      if (Operation *oldInit = toLayout.getOutput().getDefiningOp()) {
        utils::copyDramBank(oldInit, init);
      }
      auto copy = builder.create<ToLayoutOp>(toLayout.getLoc(),
                                             toLayout.getInput(), init);
      Value old = operand.get();
      old.replaceUsesWithIf(copy.getResult(0), [&](OpOperand &use) {
        return generic->isAncestor(use.getOwner());
      });
    }
  }

  // Build the spatial after the last member.
  SmallVector<Type> resultTypes;
  SmallVector<Value> outputs;
  llvm::SetVector<Value> inputs;
  for (auto [c, generic] : llvm::enumerate(group.members)) {
    llvm::append_range(resultTypes, generic->getResultTypes());
    llvm::append_range(outputs, generic.getOutputs());
    for (OpOperand &operand : generic.getInputsMutable()) {
      if (llvm::none_of(edgeUses[c], [&](EdgeUse &e) {
            return e.operandIndex == operand.getOperandNumber();
          })) {
        inputs.insert(operand.get());
      } else if (auto composite =
                     operand.get().getDefiningOp<CompositeViewOp>()) {
        for (auto [i, input] : llvm::enumerate(composite.getInputs())) {
          int inputIndex = static_cast<int>(i);
          if (llvm::none_of(edgeUses[c], [&](EdgeUse &e) {
                return e.operandIndex == operand.getOperandNumber() &&
                       e.compositeInput == inputIndex;
              })) {
            inputs.insert(input);
          }
        }
      }
    }
  }
  builder.setInsertionPointAfter(last);
  SmallVector<Attribute> gridRanges(ranges.begin(), ranges.end());
  SmallVector<Attribute> edges;
  auto spatial = builder.create<SpatialOp>(
      last.getLoc(), resultTypes, inputs.getArrayRef(), outputs,
      builder.getArrayAttr(gridRanges), /*edges=*/nullptr,
      group.members.size());

  // Spatial result index of each member's first result.
  SmallVector<unsigned> resultBase;
  unsigned base = 0;
  for (GenericOp generic : group.members) {
    resultBase.push_back(base);
    base += generic.getNumResults();
  }

  for (unsigned c = 0; c < group.members.size(); ++c) {
    GenericOp generic = group.members[c];
    Block *body = builder.createBlock(&spatial->getRegion(c));
    // One block argument per (producer, result) read by this region.
    llvm::MapVector<std::pair<unsigned, unsigned>, BlockArgument> args;
    // Composite operands: their inputs, with the edge inputs replaced by
    // block arguments as they are created.
    llvm::MapVector<unsigned, SmallVector<Value>> compositeInputs;
    for (EdgeUse &use : edgeUses[c]) {
      auto key = std::make_pair(use.producer, use.producerResult);
      OpOperand &operand = generic->getOpOperand(use.operandIndex);
      Type edgeType =
          group.members[use.producer]->getResult(use.producerResult).getType();
      auto [it, inserted] = args.try_emplace(key, nullptr);
      if (inserted) {
        it->second = body->addArgument(edgeType, generic.getLoc());
        edges.push_back(SpatialEdgeAttr::get(ctx, use.producer,
                                             use.producerResult, c,
                                             it->second.getArgNumber()));
      }
      if (use.compositeInput >= 0) {
        auto composite = operand.get().getDefiningOp<CompositeViewOp>();
        auto [entry, first] = compositeInputs.try_emplace(
            use.operandIndex, llvm::to_vector(composite.getInputs()));
        entry->second[use.compositeInput] = it->second;
        continue;
      }
      // Retype a collapsed_intervals-only or memory-space-only difference to
      // the producer's type (R5 needs equal types on both ends): a DRAM
      // operand then reads the producer's L1 buffer directly.
      Value old = operand.get();
      old.replaceUsesWithIf(it->second, [&](OpOperand &u) {
        return generic->isAncestor(u.getOwner());
      });
      noteMaybeDead(old);
    }
    builder.setInsertionPointToEnd(body);
    for (auto &[operandIndex, viewInputs] : compositeInputs) {
      Value old = generic->getOperand(operandIndex);
      auto composite = old.getDefiningOp<CompositeViewOp>();
      auto view = builder.create<CompositeViewOp>(
          composite.getLoc(), composite.getType(), viewInputs,
          composite.getDimAttr(), composite.getLogicalSizesAttr());
      old.replaceUsesWithIf(view.getResult(), [&](OpOperand &u) {
        return generic->isAncestor(u.getOwner());
      });
      noteMaybeDead(old);
    }
    generic->moveBefore(body, body->end());
    builder.setInsertionPointToEnd(body);
    builder.create<SpatialYieldOp>(generic.getLoc(), generic->getResults());
  }
  if (!edges.empty()) {
    spatial.setEdgesAttr(builder.getArrayAttr(edges));
  }

  // Uses outside the spatial now read its results.
  for (auto [c, generic] : llvm::enumerate(group.members)) {
    for (OpResult r : generic->getResults()) {
      r.replaceUsesWithIf(
          spatial->getResult(resultBase[c] + r.getResultNumber()),
          [&](OpOperand &u) { return !spatial->isAncestor(u.getOwner()); });
    }
    generic->removeAttr(kSpatialGroupAttr);
    generic->removeAttr(kSpatialRangeAttr);
  }

  // Erase what the fold left dead, then move every op before the spatial
  // that depends on a spatial result to after it (convexity guarantees that
  // no member needs them).
  while (!maybeDead.empty()) {
    Operation *def = maybeDead.pop_back_val();
    if (!def->use_empty() ||
        !mlir::isa<ToLayoutOp, MaskOp, EmptyOp, CompositeViewOp>(def)) {
      continue;
    }
    for (Value operand : def->getOperands()) {
      noteMaybeDead(operand);
    }
    def->erase();
  }
  llvm::DenseSet<Operation *> moved{spatial};
  Operation *tail = spatial;
  SmallVector<Operation *> toMove;
  for (Operation &op : block) {
    if (&op == spatial.getOperation()) {
      break;
    }
    bool dependsOnSpatial = llvm::any_of(op.getOperands(), [&](Value v) {
      return v.getDefiningOp() && moved.contains(v.getDefiningOp());
    });
    if (dependsOnSpatial) {
      moved.insert(&op);
      toMove.push_back(&op);
    }
  }
  for (Operation *op : toMove) {
    op->moveAfter(tail);
    tail = op;
  }
  return success();
}

} // namespace
} // namespace mlir::tt::d2m
