// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MHOISTRESIDENTPARAMETERS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// The d2m.empty `op` writes into, if it is used by `op` only.
EmptyOp ownEmpty(Operation *op, Value output) {
  auto empty = output.getDefiningOp<EmptyOp>();
  return empty && empty->hasOneUse() ? empty : nullptr;
}

// Keep one d2m.to_layout per argument and target: the other to_layouts of the
// same argument with the same result type into an identically attributed
// empty are replaced by the first, so each argument is written and laid out
// once per call however many ops read it.
void mergeArgumentToLayouts(func::FuncOp func, IRRewriter &rewriter) {
  Block &body = func.getBody().front();
  for (BlockArgument arg : func.getArguments()) {
    SmallVector<ToLayoutOp> toLayouts;
    for (Operation *user : arg.getUsers()) {
      auto toLayout = mlir::dyn_cast<ToLayoutOp>(user);
      if (toLayout && toLayout->getBlock() == &body &&
          toLayout.getInput() == arg && toLayout->getNumResults() == 1 &&
          ownEmpty(toLayout, toLayout.getOutput())) {
        toLayouts.push_back(toLayout);
      }
    }
    llvm::sort(toLayouts, [](ToLayoutOp a, ToLayoutOp b) {
      return a->isBeforeInBlock(b);
    });
    SmallVector<ToLayoutOp> kept;
    for (ToLayoutOp toLayout : toLayouts) {
      auto empty = toLayout.getOutput().getDefiningOp<EmptyOp>();
      ToLayoutOp *same = llvm::find_if(kept, [&](ToLayoutOp k) {
        auto keptEmpty = k.getOutput().getDefiningOp<EmptyOp>();
        return k.getResult(0).getType() == toLayout.getResult(0).getType() &&
               keptEmpty->getAttrDictionary() == empty->getAttrDictionary();
      });
      if (same == kept.end()) {
        kept.push_back(toLayout);
        continue;
      }
      rewriter.replaceAllUsesWith(toLayout.getResult(0), same->getResult(0));
      rewriter.eraseOp(toLayout);
      rewriter.eraseOp(empty);
    }
  }
}

// The layout ops that only depend on parameter arguments: to_layouts and masks
// (with the empties they write into) and views, starting from the parameters.
// A view that ops outside the set also read stays in the run function, and
// so does anything whose input is not a parameter or a value of the set.
llvm::SetVector<Operation *> collectParameterLayouts(func::FuncOp func) {
  Block &body = func.getBody().front();
  llvm::SetVector<Operation *> set;
  SmallVector<Value> worklist;
  for (BlockArgument arg : func.getArguments()) {
    if (ttcore::isConstOrParamArg(arg, func)) {
      worklist.push_back(arg);
    }
  }
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    for (Operation *user : value.getUsers()) {
      if (user->getBlock() != &body || set.contains(user) ||
          user->getOperand(0) != value) {
        continue;
      }
      if (mlir::isa<ToLayoutOp, MaskOp>(user)) {
        if (user->getNumResults() != 1) {
          continue;
        }
        EmptyOp empty = ownEmpty(user, user->getOperand(1));
        if (!empty) {
          continue;
        }
        set.insert(empty);
      } else if (!mlir::isa<ViewLayoutOp>(user)) {
        continue;
      }
      set.insert(user);
      worklist.push_back(user->getResult(0));
    }
  }

  auto inSet = [&](Value value) {
    if (auto arg = mlir::dyn_cast<BlockArgument>(value)) {
      return ttcore::isConstOrParamArg(arg, func);
    }
    return set.contains(value.getDefiningOp());
  };
  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation *op : llvm::to_vector(set)) {
      if (mlir::isa<EmptyOp>(op)) {
        continue;
      }
      bool drop = !inSet(op->getOperand(0));
      if (mlir::isa<ViewLayoutOp>(op)) {
        drop = drop || llvm::any_of(op->getUsers(), [&](Operation *user) {
                 return !set.contains(user);
               });
      }
      if (drop) {
        set.remove(op);
        if (!mlir::isa<ViewLayoutOp>(op)) {
          set.remove(op->getOperand(1).getDefiningOp());
        }
        changed = true;
      }
    }
  }
  return set;
}

// The parameter argument a hoisted value comes from.
unsigned sourceParameter(Value value) {
  while (auto *def = value.getDefiningOp()) {
    value = def->getOperand(0);
  }
  return mlir::cast<BlockArgument>(value).getArgNumber();
}

LogicalResult hoist(func::FuncOp func, IRRewriter &rewriter,
                    SymbolTable &symbols) {
  llvm::SetVector<Operation *> set = collectParameterLayouts(func);
  SmallVector<Operation *> ops = llvm::to_vector(set);
  llvm::sort(ops,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });

  // Values the run function still reads: the load function's results.
  SmallVector<Value> results;
  for (Operation *op : ops) {
    for (Value result : op->getResults()) {
      if (llvm::any_of(result.getUsers(),
                       [&](Operation *user) { return !set.contains(user); })) {
        results.push_back(result);
      }
    }
  }
  if (results.empty()) {
    return success();
  }
  for (Value result : results) {
    auto type = mlir::cast<RankedTensorType>(result.getType());
    auto layout =
        mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(type.getEncoding());
    if (!layout || layout.getMemorySpace() != ttcore::MemorySpace::DeviceDRAM) {
      return func.emitOpError()
             << "resident parameters must be laid out in DRAM "
                "(dram-args-and-results): "
             << type;
    }
  }

  // Load function: the parameters the hoisted ops read, in argument order.
  SmallVector<unsigned> params;
  for (BlockArgument arg : func.getArguments()) {
    if (ttcore::isConstOrParamArg(arg, func) &&
        llvm::any_of(arg.getUsers(),
                     [&](Operation *user) { return set.contains(user); })) {
      params.push_back(arg.getArgNumber());
    }
  }
  SmallVector<Type> paramTypes = llvm::map_to_vector(
      params, [&](unsigned i) { return func.getArgument(i).getType(); });
  SmallVector<Type> resultTypes =
      llvm::map_to_vector(results, [](Value v) { return v.getType(); });

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(func);
  auto load = rewriter.create<func::FuncOp>(
      func.getLoc(), (func.getSymName() + "_load").str(),
      rewriter.getFunctionType(paramTypes, resultTypes));
  symbols.insert(load);
  if (Attribute kind = func->getAttr("tt.function_type")) {
    load->setAttr("tt.function_type", kind);
  }
  load->setAttr(utils::kResidentLoadAttr, rewriter.getUnitAttr());
  load->setAttr("d2m.resident_params",
                rewriter.getI64ArrayAttr(llvm::map_to_vector(
                    params, [](unsigned i) { return int64_t(i); })));
  for (auto [i, param] : llvm::enumerate(params)) {
    load.setArgAttrs(i, func.getArgAttrDict(param));
  }
  Block *loadBody = load.addEntryBlock();
  rewriter.setInsertionPointToStart(loadBody);
  IRMapping mapping;
  for (auto [i, param] : llvm::enumerate(params)) {
    mapping.map(func.getArgument(param), loadBody->getArgument(i));
  }
  for (Operation *op : ops) {
    rewriter.clone(*op, mapping);
  }
  rewriter.create<func::ReturnOp>(
      func.getLoc(),
      llvm::map_to_vector(results, [&](Value v) { return mapping.lookup(v); }));

  // Run function: the loaded values become trailing arguments.
  for (Value result : results) {
    SmallVector<NamedAttribute> attrs = {rewriter.getNamedAttr(
        "d2m.resident_of",
        rewriter.getI64IntegerAttr(sourceParameter(result)))};
    if (std::optional<int64_t> bank = utils::getDramBank(result)) {
      attrs.push_back(rewriter.getNamedAttr(utils::kDramBankAttr,
                                            rewriter.getI64IntegerAttr(*bank)));
    }
    unsigned index = func.getNumArguments();
    if (failed(func.insertArgument(index, result.getType(),
                                   rewriter.getDictionaryAttr(attrs),
                                   result.getLoc()))) {
      return failure();
    }
    rewriter.replaceUsesWithIf(
        result, func.getArgument(index),
        [&](OpOperand &use) { return !set.contains(use.getOwner()); });
  }
  for (Operation *op : llvm::reverse(ops)) {
    rewriter.eraseOp(op);
  }

  unsigned original = func.getNumArguments() - results.size();
  llvm::BitVector unused(func.getNumArguments());
  SmallVector<int64_t> runInputs;
  for (unsigned i = 0; i < original; ++i) {
    if (ttcore::isConstOrParamArg(func.getArgument(i), func) &&
        func.getArgument(i).use_empty()) {
      unused.set(i);
    } else {
      runInputs.push_back(i);
    }
  }
  if (failed(func.eraseArguments(unused))) {
    return failure();
  }
  func->setAttr(utils::kResidentRunAttr, rewriter.getUnitAttr());
  func->setAttr("d2m.run_inputs", rewriter.getI64ArrayAttr(runInputs));
  return success();
}

class D2MHoistResidentParameters
    : public impl::D2MHoistResidentParametersBase<D2MHoistResidentParameters> {
public:
  using impl::D2MHoistResidentParametersBase<
      D2MHoistResidentParameters>::D2MHoistResidentParametersBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    SymbolTable symbols(module);
    IRRewriter rewriter(&getContext());
    SmallVector<func::FuncOp> funcs;
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (!func.isDeclaration() && func.isPublic() &&
          !func->hasAttr(utils::kResidentLoadAttr)) {
        funcs.push_back(func);
      }
    }
    for (func::FuncOp func : funcs) {
      mergeArgumentToLayouts(func, rewriter);
      if (failed(hoist(func, rewriter, symbols))) {
        return signalPassFailure();
      }
    }
  }
};

} // namespace
} // namespace mlir::tt::d2m
