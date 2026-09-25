// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_SPATIALPIPELINE_H
#define TTMLIR_DIALECT_D2M_UTILS_SPATIALPIPELINE_H

#include "llvm/ADT/StringRef.h"

namespace mlir::tt::d2m::spatial_pipeline {
// Internal contract between planning, layout, allocation and DMA lowering.
// Generic rebuilds must preserve discardable attributes. Group and stage IDs
// are stable within a function block; each stage owns one physical worker core.
inline constexpr llvm::StringLiteral group = "d2m.pipeline_group";
inline constexpr llvm::StringLiteral stage = "d2m.pipeline_stage";
inline constexpr llvm::StringLiteral core = "d2m.pipeline_core";
// One producer stage ID per input (-1 for an external input).
inline constexpr llvm::StringLiteral inputs = "d2m.pipeline_inputs";
// Triples of (additional argument index, destination y, destination x).
// Coordinates are relative to the sending stage's virtual single-core grid.
inline constexpr llvm::StringLiteral signals = "d2m.pipeline_signals";
// A cumulative, non-resetting wait that belongs only on data-movement threads.
inline constexpr llvm::StringLiteral wait = "d2m.pipeline_wait";
inline constexpr llvm::StringLiteral expectedStages =
    "d2m.pipeline_expected_stages";
inline constexpr llvm::StringLiteral noSpill = "d2m.spatial_no_spill";
} // namespace mlir::tt::d2m::spatial_pipeline

#endif
