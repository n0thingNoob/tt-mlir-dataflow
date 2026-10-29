// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/SpatialGroupUtils.h"

#include "llvm/ADT/DenseSet.h"

namespace mlir::tt::d2m {

bool samePhysicalLayout(Type a, Type b) {
  if (a == b) {
    return true;
  }
  auto ta = mlir::dyn_cast<RankedTensorType>(a);
  auto tb = mlir::dyn_cast<RankedTensorType>(b);
  if (!ta || !tb || ta.getShape() != tb.getShape() ||
      ta.getElementType() != tb.getElementType()) {
    return false;
  }
  auto la = mlir::dyn_cast_or_null<ttcore::MetalLayoutAttr>(ta.getEncoding());
  auto lb = mlir::dyn_cast_or_null<ttcore::MetalLayoutAttr>(tb.getEncoding());
  return la && lb && la.getLogicalShape().size() <= 2 &&
         la.getLogicalShape() == lb.getLogicalShape() &&
         la.getDimAlignments() == lb.getDimAlignments() &&
         la.getMemorySpace() == lb.getMemorySpace() &&
         la.getMemoryLayout() == lb.getMemoryLayout();
}

// A mask whose fold returns its input (no maskable padding) is a no-op.
static bool isNoOpMask(MaskOp mask) {
  SmallVector<OpFoldResult> folded;
  if (failed(mask->fold(folded)) || folded.size() != 1) {
    return false;
  }
  return mlir::dyn_cast_if_present<Value>(folded.front()) == mask.getInput();
}

OpResult traceEdgeSource(Value v) {
  while (true) {
    Operation *def = v.getDefiningOp();
    if (!def) {
      return nullptr;
    }
    if (mlir::isa<GenericOp>(def)) {
      return mlir::cast<OpResult>(v);
    }
    if (auto toLayout = mlir::dyn_cast<ToLayoutOp>(def)) {
      v = toLayout.getInput();
      continue;
    }
    if (auto mask = mlir::dyn_cast<MaskOp>(def); mask && isNoOpMask(mask)) {
      v = mask.getInput();
      continue;
    }
    return nullptr;
  }
}

llvm::SetVector<Operation *> producerGenerics(Value v, Block &block) {
  llvm::SetVector<Operation *> producers;
  llvm::DenseSet<Operation *> visited;
  SmallVector<Value> worklist{v};
  while (!worklist.empty()) {
    Operation *def = worklist.pop_back_val().getDefiningOp();
    if (!def || def->getBlock() != &block || !visited.insert(def).second) {
      continue;
    }
    if (mlir::isa<GenericOp>(def)) {
      producers.insert(def);
      continue;
    }
    if (def->getNumRegions() == 0) {
      llvm::append_range(worklist, def->getOperands());
    }
  }
  return producers;
}

} // namespace mlir::tt::d2m
