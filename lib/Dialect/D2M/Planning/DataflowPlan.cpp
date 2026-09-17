// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Planning/DataflowPlan.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <tuple>
#include <utility>

#include "ttmlir/Dialect/D2M/Planning/DataflowCostModel.h"

namespace mlir::tt::d2m {

DataflowProgramVariant::DataflowProgramVariant(
    unsigned variantId, llvm::SmallVector<GenericOp> members,
    llvm::SmallVector<int64_t> gridShape,
    llvm::SmallVector<int64_t> blockFactors, ProgramResourceEstimate resources,
    ProgramCostEstimate cost)
    : variantId(variantId), members(std::move(members)),
      gridShape(std::move(gridShape)), blockFactors(std::move(blockFactors)),
      resources(resources), cost(cost) {}

llvm::StringRef stringifyDataflowConnectionKind(DataflowConnectionKind kind) {
  switch (kind) {
  case DataflowConnectionKind::Materialized:
    return "materialized";
  case DataflowConnectionKind::L1Stream:
    return "l1-stream";
  case DataflowConnectionKind::NocStream:
    return "noc-stream";
  case DataflowConnectionKind::Dram:
    return "dram";
  }
  llvm_unreachable("unknown dataflow connection kind");
}

llvm::StringRef stringifyDataflowPlanStrategy(DataflowPlanStrategy strategy) {
  switch (strategy) {
  case DataflowPlanStrategy::Temporal:
    return "temporal-fallback";
  case DataflowPlanStrategy::Fused:
    return "fused";
  case DataflowPlanStrategy::Spatial:
    return "spatial";
  }
  llvm_unreachable("unknown dataflow plan strategy");
}

DataflowMappingPlan::DataflowMappingPlan(
    func::FuncOp function, Block *block, unsigned scopeOrdinal,
    DataflowPlanStrategy strategy,
    llvm::SmallVector<DataflowPlannedProgram, 0> programs,
    llvm::SmallVector<DataflowPlannedConnection> connections)
    : function(function), block(block), scopeOrdinal(scopeOrdinal),
      strategy(strategy), programs(std::move(programs)),
      connections(std::move(connections)) {}

llvm::StringRef stringifyDataflowRejectionKind(DataflowRejectionKind kind) {
  switch (kind) {
  case DataflowRejectionKind::None:
    return "none";
  case DataflowRejectionKind::InvalidGraph:
    return "invalid-graph";
  case DataflowRejectionKind::InvalidVariant:
    return "invalid-variant";
  case DataflowRejectionKind::ResourceLimit:
    return "resource-limit";
  case DataflowRejectionKind::Unsupported:
    return "unsupported";
  }
  llvm_unreachable("unknown dataflow rejection kind");
}

DataflowFeasibilityResult DataflowFeasibilityResult::success() {
  return {true, DataflowRejectionKind::None, {}};
}

DataflowFeasibilityResult
DataflowFeasibilityResult::reject(DataflowRejectionKind kind,
                                  llvm::StringRef message) {
  return {false, kind, message.str()};
}

DataflowFeasibilityResult StructuralDataflowFeasibilityModel::evaluate(
    const DataflowGraph &graph, const DataflowMappingPlan &plan) const {
  if (plan.getFunction() != graph.getFunction() ||
      plan.getBlock() != graph.getBlock() ||
      plan.getScopeOrdinal() != graph.getScopeOrdinal()) {
    return DataflowFeasibilityResult::reject(
        DataflowRejectionKind::InvalidGraph,
        "plan identity does not match its node graph");
  }

  llvm::DenseMap<Operation *, unsigned> nodeToProgram;
  llvm::DenseSet<Operation *> graphNodes;
  for (GenericOp node : graph.getOperations()) {
    if (!node || !graphNodes.insert(node).second) {
      return DataflowFeasibilityResult::reject(
          DataflowRejectionKind::InvalidGraph,
          "graph has null or duplicate nodes");
    }
  }
  for (auto [programIndex, program] : llvm::enumerate(plan.getPrograms())) {
    const DataflowProgramVariant &variant = program.variant;
    if (variant.getMembers().empty() || variant.getGridShape().empty() ||
        llvm::any_of(variant.getGridShape(),
                     [](int64_t dim) { return dim <= 0; })) {
      return DataflowFeasibilityResult::reject(
          DataflowRejectionKind::InvalidVariant,
          "program variant must have members and a positive grid shape");
    }
    for (GenericOp member : variant.getMembers()) {
      if (!member || !graphNodes.contains(member)) {
        return DataflowFeasibilityResult::reject(
            DataflowRejectionKind::InvalidVariant,
            "program variant references an unknown node");
      }
      if (!nodeToProgram.try_emplace(member, programIndex).second) {
        return DataflowFeasibilityResult::reject(
            DataflowRejectionKind::InvalidVariant,
            "node is covered by more than one program variant");
      }
    }
  }
  if (nodeToProgram.size() != graphNodes.size()) {
    return DataflowFeasibilityResult::reject(
        DataflowRejectionKind::InvalidGraph,
        "mapping plan does not cover every graph node");
  }

  llvm::DenseSet<std::pair<unsigned, unsigned>> connections;
  for (const DataflowPlannedConnection &connection : plan.getConnections()) {
    if (connection.producerProgram >= plan.getPrograms().size() ||
        connection.consumerProgram >= plan.getPrograms().size() ||
        connection.producerProgram == connection.consumerProgram ||
        connection.bufferDepth == 0) {
      return DataflowFeasibilityResult::reject(
          DataflowRejectionKind::InvalidGraph,
          "plan contains an invalid program connection");
    }
    if (!connections
             .insert({connection.producerProgram, connection.consumerProgram})
             .second) {
      return DataflowFeasibilityResult::reject(
          DataflowRejectionKind::InvalidGraph,
          "plan contains a duplicate program connection");
    }
  }

  for (DataflowEdge edge : graph.getEdges()) {
    if (!edge.producer || !edge.consumer ||
        !nodeToProgram.contains(edge.producer) ||
        !nodeToProgram.contains(edge.consumer) || !edge.producerValue ||
        edge.producerValue.getDefiningOp() != edge.producer ||
        edge.consumerOperand >= edge.consumer.getOperands().size() ||
        edge.consumer.getOperands()[edge.consumerOperand] !=
            edge.consumerValue) {
      return DataflowFeasibilityResult::reject(
          DataflowRejectionKind::InvalidGraph,
          "node graph contains an invalid dependency");
    }
    unsigned producerProgram = nodeToProgram.lookup(edge.producer);
    unsigned consumerProgram = nodeToProgram.lookup(edge.consumer);
    if (producerProgram == consumerProgram) {
      continue;
    }
    if (!connections.contains({producerProgram, consumerProgram})) {
      return DataflowFeasibilityResult::reject(
          DataflowRejectionKind::InvalidGraph,
          "plan does not preserve a producer-consumer dependency");
    }
    if (plan.getStrategy() == DataflowPlanStrategy::Temporal &&
        producerProgram >= consumerProgram) {
      return DataflowFeasibilityResult::reject(
          DataflowRejectionKind::InvalidGraph,
          "temporal plan does not preserve dependency order");
    }
  }

  return DataflowFeasibilityResult::success();
}

DataflowMappingPlan buildTemporalFallbackPlan(const DataflowGraph &graph) {
  llvm::SmallVector<DataflowPlannedProgram, 0> programs;
  programs.reserve(graph.getOperations().size());
  for (auto item : llvm::enumerate(graph.getOperations())) {
    GenericOp operation = item.value();
    programs.push_back({DataflowProgramVariant(
                            /*variantId=*/0, {operation},
                            llvm::to_vector(operation.getGrid().getShape()),
                            operation.getBlockFactorsValue()),
                        /*coreOffset=*/{}});
  }

  llvm::SmallVector<DataflowPlannedConnection> connections;
  connections.reserve(graph.getEdges().size());
  llvm::DenseMap<Operation *, unsigned> programIndices;
  for (auto [index, operation] : llvm::enumerate(graph.getOperations())) {
    programIndices.try_emplace(operation, index);
  }
  llvm::DenseSet<std::pair<unsigned, unsigned>> connected;
  for (DataflowEdge edge : graph.getEdges()) {
    auto endpoints = std::make_pair(programIndices.lookup(edge.producer),
                                    programIndices.lookup(edge.consumer));
    if (!connected.insert(endpoints).second) {
      continue;
    }
    connections.push_back({endpoints.first, endpoints.second,
                           DataflowConnectionKind::Materialized,
                           /*bufferDepth=*/1});
  }

  return DataflowMappingPlan(graph.getFunction(), graph.getBlock(),
                             graph.getScopeOrdinal(),
                             DataflowPlanStrategy::Temporal,
                             std::move(programs), std::move(connections));
}

static void printOptionalCycles(llvm::raw_ostream &os,
                                std::optional<uint64_t> cycles) {
  if (cycles) {
    os << *cycles;
  } else {
    os << "unknown";
  }
}

void printDataflowPlan(llvm::raw_ostream &os, const DataflowGraph &graph,
                       const DataflowMappingPlan &plan,
                       const DataflowPlanCost &cost) {
  os << "d2m-dataflow-plan function=@" << plan.getFunction().getSymName()
     << " scope=" << plan.getScopeOrdinal()
     << " strategy=" << stringifyDataflowPlanStrategy(plan.getStrategy())
     << " nodes=" << graph.getOperations().size()
     << " dependencies=" << graph.getEdges().size()
     << " programs=" << plan.getPrograms().size()
     << " connections=" << plan.getConnections().size() << " latency=";
  printOptionalCycles(os, cost.latencyCycles);
  os << " ii=";
  printOptionalCycles(os, cost.initiationIntervalCycles);
  os << "\n";
}

LogicalResult TemporalDataflowPlanMaterializer::materialize(
    const DataflowGraph &, const DataflowMappingPlan &plan) const {
  return success(plan.getStrategy() == DataflowPlanStrategy::Temporal);
}

} // namespace mlir::tt::d2m
