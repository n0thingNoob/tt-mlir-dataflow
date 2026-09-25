// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_D2M_TRANSFORMS_SPATIALMAPPING_H
#define TTMLIR_D2M_TRANSFORMS_SPATIALMAPPING_H

#include "ttmlir/Dialect/D2M/Analysis/SpatializationAnalysis.h"

namespace mlir::tt::d2m {
struct SelectedSpatialGroup {
  SmallVector<unsigned> members;
  SmallVector<Attribute> ranges;
  SmallVector<Operation *> preparations;
  std::string reason;
  bool pipeline = false;
};

// Selection validates the entire block without changing IR.
FailureOr<SmallVector<SelectedSpatialGroup, 0>>
selectSpatialGroups(const SpatializationAnalysisResult &analysis, bool dump,
                    bool pipelines = false);
void materializeSpatialGroups(const SpatialGenericDAG &dag,
                              ArrayRef<SelectedSpatialGroup> groups, bool dump);
} // namespace mlir::tt::d2m
#endif
