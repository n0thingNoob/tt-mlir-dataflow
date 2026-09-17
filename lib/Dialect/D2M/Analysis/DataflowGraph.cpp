// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/DataflowGraph.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <tuple>
#include <utility>

namespace mlir::tt::d2m {

DataflowGraph::DataflowGraph(func::FuncOp function, Block *block,
                             unsigned scopeOrdinal,
                             llvm::SmallVector<GenericOp> operations,
                             llvm::SmallVector<DataflowEdge> edges)
    : function(function), block(block), scopeOrdinal(scopeOrdinal),
      operations(std::move(operations)), edges(std::move(edges)) {}

static void collectDefiningResults(
    Value value, const llvm::DenseMap<Operation *, unsigned> &nodes,
    llvm::DenseSet<Value> &visited, llvm::SmallVector<OpResult> &results) {
  if (!visited.insert(value).second) {
    return;
  }
  auto result = dyn_cast<OpResult>(value);
  if (!result) {
    return;
  }
  Operation *definingOp = result.getOwner();
  if (nodes.contains(definingOp)) {
    results.push_back(result);
    return;
  }
  if (definingOp->getParentOfType<GenericOp>()) {
    return;
  }
  for (Value operand : definingOp->getOperands()) {
    collectDefiningResults(operand, nodes, visited, results);
  }
}

static llvm::SmallVector<DataflowEdge>
buildNodeEdges(llvm::ArrayRef<GenericOp> operations) {
  // Ordinals determine output ordering only; nodes and edges retain IR handles.
  llvm::DenseMap<Operation *, unsigned> order;
  for (auto [index, operation] : llvm::enumerate(operations)) {
    order.try_emplace(operation, index);
  }
  llvm::SmallVector<DataflowEdge> edges;
  for (GenericOp consumer : operations) {
    for (auto [inputIndex, input] : llvm::enumerate(consumer.getOperands())) {
      llvm::DenseSet<Value> visited;
      llvm::SmallVector<OpResult> producers;
      collectDefiningResults(input, order, visited, producers);
      llvm::sort(producers, [&](OpResult lhs, OpResult rhs) {
        return std::make_pair(order.lookup(lhs.getOwner()),
                              lhs.getResultNumber()) <
               std::make_pair(order.lookup(rhs.getOwner()),
                              rhs.getResultNumber());
      });
      for (OpResult result : producers) {
        auto producer = cast<GenericOp>(result.getOwner());
        if (producer != consumer) {
          edges.push_back({producer, consumer, result, input,
                           static_cast<unsigned>(inputIndex)});
        }
      }
    }
  }
  return edges;
}

llvm::SmallVector<DataflowGraph> buildDataflowGraphs(ModuleOp module) {
  llvm::SmallVector<DataflowGraph> graphs;

  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    llvm::DenseMap<Block *, unsigned> blockToScope;
    llvm::SmallVector<Block *> blocks;
    llvm::SmallVector<llvm::SmallVector<GenericOp>> operationsByBlock;

    function.walk([&](GenericOp genericOp) {
      if (genericOp->getParentOfType<SpatialOp>() ||
          genericOp->getParentOfType<GenericOp>()) {
        return;
      }

      Block *block = genericOp->getBlock();
      auto [it, inserted] = blockToScope.try_emplace(
          block, static_cast<unsigned>(operationsByBlock.size()));
      if (inserted) {
        blocks.push_back(block);
        operationsByBlock.emplace_back();
      }
      operationsByBlock[it->second].push_back(genericOp);
    });

    for (size_t scope = 0; scope < operationsByBlock.size(); ++scope) {
      llvm::SmallVector<GenericOp> operations =
          std::move(operationsByBlock[scope]);
      llvm::SmallVector<DataflowEdge> edges = buildNodeEdges(operations);
      graphs.emplace_back(function, blocks[scope], static_cast<unsigned>(scope),
                          std::move(operations), std::move(edges));
    }
  }

  return graphs;
}

} // namespace mlir::tt::d2m
