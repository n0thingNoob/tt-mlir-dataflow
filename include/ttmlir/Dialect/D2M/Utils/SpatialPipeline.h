// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_SPATIALPIPELINE_H
#define TTMLIR_DIALECT_D2M_UTILS_SPATIALPIPELINE_H

#include "llvm/ADT/StringRef.h"

namespace mlir::tt::d2m::spatial_pipeline {
// Internal contract between planning, layout, allocation and DMA lowering.
// Generic rebuilds must preserve discardable attributes. Group and stage IDs
// are stable within a function block; each stage owns a disjoint physical
// worker range.
inline constexpr llvm::StringLiteral group = "d2m.pipeline_group";
inline constexpr llvm::StringLiteral stage = "d2m.pipeline_stage";
inline constexpr llvm::StringLiteral shards = "d2m.pipeline_shards";
// Semaphore destinations already expressed in device logical worker
// coordinates.
inline constexpr llvm::StringLiteral absoluteDestination =
    "d2m.pipeline_absolute_destination";
inline constexpr llvm::StringLiteral core = "d2m.pipeline_core";
// One producer stage ID per input (-1 for an external input).
inline constexpr llvm::StringLiteral inputs = "d2m.pipeline_inputs";
// Triples for the P=1 baseline: (argument index, relative destination y, x).
// For P>1: (argument index, destination y, x, rectangle width), indexed
// by this core's M shard ID. Destinations are absolute logical workers.
inline constexpr llvm::StringLiteral signals = "d2m.pipeline_signals";
// A cumulative, non-resetting wait that belongs only on data-movement threads.
inline constexpr llvm::StringLiteral wait = "d2m.pipeline_wait";
inline constexpr llvm::StringLiteral expectedStages =
    "d2m.pipeline_expected_stages";
inline constexpr llvm::StringLiteral noSpill = "d2m.spatial_no_spill";
} // namespace mlir::tt::d2m::spatial_pipeline

#endif
