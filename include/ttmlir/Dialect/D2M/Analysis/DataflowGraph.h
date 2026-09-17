// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_ANALYSIS_DATAFLOWGRAPH_H
#define TTMLIR_DIALECT_D2M_ANALYSIS_DATAFLOWGRAPH_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::d2m {

struct DataflowEdge {
  GenericOp producer;
  GenericOp consumer;
  // Producer result and exact consumer operand, including intervening view ops.
  Value producerValue;
  Value consumerValue;
  unsigned consumerOperand = 0;
};

/// A read-only graph of program-level d2m.generic operations in one block.
/// Op/value handles are valid only while that IR remains unchanged; rebuild
/// after cloning or rewriting. Ordinals are used only for deterministic dumps.
class DataflowGraph {
public:
  DataflowGraph(func::FuncOp function, Block *block, unsigned scopeOrdinal,
                llvm::SmallVector<GenericOp> operations,
                llvm::SmallVector<DataflowEdge> edges);

  func::FuncOp getFunction() const { return function; }
  Block *getBlock() const { return block; }
  unsigned getScopeOrdinal() const { return scopeOrdinal; }
  llvm::ArrayRef<GenericOp> getOperations() const { return operations; }
  llvm::ArrayRef<DataflowEdge> getEdges() const { return edges; }

private:
  func::FuncOp function;
  Block *block;
  unsigned scopeOrdinal;
  llvm::SmallVector<GenericOp> operations;
  llvm::SmallVector<DataflowEdge> edges;
};

/// Discover one graph per block, excluding generics nested in spatial/generic
/// regions. Edges describe SSA operands before bufferization, not memory alias
/// dependences or control-flow dependences across blocks.
llvm::SmallVector<DataflowGraph> buildDataflowGraphs(ModuleOp module);

} // namespace mlir::tt::d2m

#endif
