// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_SPATIALGROUPUTILS_H
#define TTMLIR_DIALECT_D2M_UTILS_SPATIALGROUPUTILS_H

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "llvm/ADT/SetVector.h"

namespace mlir::tt::d2m {

// Attributes that put a top-level generic into a spatial group: the group id
// (i64) and the core range of its region (#ttcore.core_range). Set by
// d2m-spatial-planning or by hand, consumed by d2m-materialize-spatial-groups.
inline constexpr llvm::StringLiteral kSpatialGroupAttr = "d2m.spatial_group";
inline constexpr llvm::StringLiteral kSpatialRangeAttr = "d2m.spatial_range";

// True when `a` and `b` describe the same physical L1 tensor: equal device
// shape and element type, and layouts that differ at most in
// collapsed_intervals with logical rank <= 2 (where any collapse is the
// identity, so a to_layout between them only reinterprets metadata).
bool samePhysicalLayout(Type a, Type b);

// Follows `v` back through d2m.to_layout ops and d2m.mask ops that fold to
// their input, to the generic result it carries. Returns null if the chain
// contains anything else or does not start at a generic. Only such chains can
// become a d2m.spatial edge.
OpResult traceEdgeSource(Value v);

// Generics in `block` that `v` depends on through region-free ops.
llvm::SetVector<Operation *> producerGenerics(Value v, Block &block);

} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_UTILS_SPATIALGROUPUTILS_H
