// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_ANALYSIS_SPATIALIZATIONANALYSIS_H
#define TTMLIR_DIALECT_D2M_ANALYSIS_SPATIALIZATIONANALYSIS_H

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <string>
#include <utility>

namespace mlir::tt::d2m {

// All operation/value references belong to the input IR. Results are valid only
// until that IR changes; neither the analysis nor its printer modifies the IR.
struct SpatialGenericNode {
  GenericOp generic;
  unsigned interval = 0;
  SmallVector<unsigned> predecessors;
  SmallVector<unsigned> successors;
  SmallVector<Value> inputs;
  SmallVector<Value> outputs;
  SmallVector<Value> results;
  SmallVector<Value> captures;
  SmallVector<std::string> computeOps;
  std::string opType;
  SmallVector<std::string> resourceHints;
};

struct SpatialDependency {
  // An absent producer denotes an input defined outside the analyzed DAG.
  std::optional<unsigned> producer;
  unsigned consumer;
  Value source;
  Value target;
  unsigned targetIndex;
  bool captured;
  // One representative backward path per source/result and target binding,
  // ordered from consumer to producer. This is not an enumeration of paths.
  SmallVector<Operation *> via;
};

struct SpatialExternalUse {
  unsigned producer;
  Value value;
  Operation *user;
  unsigned operandIndex;
};

struct SpatialGenericDAG {
  SmallVector<SpatialGenericNode, 0> nodes;
  SmallVector<SpatialDependency> dependencies;
  SmallVector<SpatialExternalUse> externalUses;
  SmallVector<Operation *> barriers;
  // reachable[a][b] means that b transitively depends on a, across intervals
  // as well as within them. Intervals constrain grouping, not data dependence.
  SmallVector<llvm::BitVector> reachable;

  bool independent(unsigned a, unsigned b) const;
};

enum class RegionCandidateKind {
  ProducerConsumerChain,
  ParallelBranches,
  ForkJoin,
  Singleton
};

struct RegionCandidateContext {
  std::optional<unsigned> producer;
  std::optional<unsigned> consumer;
  Value externalInput;
};

// A planning candidate, not an MLIR Region or an executable spatial mapping.
// Candidates may overlap; no candidate is selected or assigned resources here.
struct RegionCandidate {
  SmallVector<unsigned> members;
  SmallVector<RegionCandidateKind> kinds;
  SmallVector<std::string> reasons;
  SmallVector<RegionCandidateContext> contexts;
  // Indices into SpatialGenericDAG::dependencies / externalUses.
  SmallVector<unsigned> internalDependencies;
  SmallVector<unsigned> incomingDependencies;
  SmallVector<unsigned> outgoingDependencies;
  SmallVector<unsigned> externalUses;
  SmallVector<std::pair<unsigned, unsigned>> independentPairs;
};

struct SpatializationAnalysisResult {
  SpatialGenericDAG dag;
  SmallVector<RegionCandidate, 0> candidates;
};

SpatialGenericDAG buildSpatialGenericDAG(Block &block);
SmallVector<RegionCandidate, 0>
formSpatialRegionCandidates(const SpatialGenericDAG &dag);
SpatializationAnalysisResult analyzeSpatialization(Block &block);
void printSpatializationAnalysis(llvm::raw_ostream &os, Block &block,
                                 const SpatializationAnalysisResult &result);

} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_ANALYSIS_SPATIALIZATIONANALYSIS_H
