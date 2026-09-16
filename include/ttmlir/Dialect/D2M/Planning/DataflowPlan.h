// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_PLANNING_DATAFLOWPLAN_H
#define TTMLIR_DIALECT_D2M_PLANNING_DATAFLOWPLAN_H

#include "mlir/Support/LogicalResult.h"
#include "ttmlir/Dialect/D2M/Analysis/DataflowGraph.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <optional>
#include <string>

namespace llvm {
class raw_ostream;
}

namespace mlir::tt::d2m {

/// Aggregate resource estimate for an entire program, not one device kernel.
struct ProgramResourceEstimate {
  std::optional<uint64_t> l1BytesPerCore;
  std::optional<uint64_t> dramBytes;
  std::optional<uint64_t> nocBytes;
  std::optional<uint32_t> cbCount;
  std::optional<uint32_t> dstTiles;
};

/// Program-level estimate encompassing its compute and data-movement kernels.
struct ProgramCostEstimate {
  std::optional<uint64_t> computeCycles;
  std::optional<uint64_t> dataMovementCycles;
  std::optional<uint64_t> initiationIntervalCycles;
  float confidence = 0.0F;
};

/// One possible implementation of one or more node computations. A
/// multi-member variant represents fusion. Future staged variants may appear
/// more than once in a plan after the feasibility model learns their contract.
class DataflowProgramVariant {
public:
  DataflowProgramVariant(unsigned variantId,
                         llvm::SmallVector<GenericOp> members,
                         llvm::SmallVector<int64_t> gridShape,
                         llvm::SmallVector<int64_t> blockFactors,
                         ProgramResourceEstimate resources = {},
                         ProgramCostEstimate cost = {});

  unsigned getVariantId() const { return variantId; }
  llvm::ArrayRef<GenericOp> getMembers() const { return members; }
  llvm::ArrayRef<int64_t> getGridShape() const { return gridShape; }
  llvm::ArrayRef<int64_t> getBlockFactors() const { return blockFactors; }
  const ProgramResourceEstimate &getResources() const { return resources; }
  const ProgramCostEstimate &getCost() const { return cost; }

private:
  unsigned variantId;
  llvm::SmallVector<GenericOp> members;
  llvm::SmallVector<int64_t> gridShape;
  llvm::SmallVector<int64_t> blockFactors;
  ProgramResourceEstimate resources;
  ProgramCostEstimate cost;
};

struct DataflowPlannedProgram {
  DataflowProgramVariant variant;
  llvm::SmallVector<int64_t> coreOffset;
};

enum class DataflowConnectionKind {
  Materialized,
  L1Stream,
  NocStream,
  Dram,
};

llvm::StringRef stringifyDataflowConnectionKind(DataflowConnectionKind kind);

/// Indices address program slots in this immutable plan, never source IR nodes.
struct DataflowPlannedConnection {
  unsigned producerProgram;
  unsigned consumerProgram;
  DataflowConnectionKind kind = DataflowConnectionKind::Materialized;
  uint32_t bufferDepth = 1;
};

enum class DataflowPlanStrategy {
  Temporal,
  Fused,
  Spatial,
};

llvm::StringRef stringifyDataflowPlanStrategy(DataflowPlanStrategy strategy);

/// An immutable external-dataflow decision. Search implementations construct
/// plans out of band and materialize only the selected plan.
class DataflowMappingPlan {
public:
  DataflowMappingPlan(func::FuncOp function, Block *block,
                      unsigned scopeOrdinal, DataflowPlanStrategy strategy,
                      llvm::SmallVector<DataflowPlannedProgram, 0> programs,
                      llvm::SmallVector<DataflowPlannedConnection> connections);

  func::FuncOp getFunction() const { return function; }
  Block *getBlock() const { return block; }
  unsigned getScopeOrdinal() const { return scopeOrdinal; }
  DataflowPlanStrategy getStrategy() const { return strategy; }
  llvm::ArrayRef<DataflowPlannedProgram> getPrograms() const {
    return programs;
  }
  llvm::ArrayRef<DataflowPlannedConnection> getConnections() const {
    return connections;
  }

private:
  func::FuncOp function;
  Block *block;
  unsigned scopeOrdinal;
  DataflowPlanStrategy strategy;
  llvm::SmallVector<DataflowPlannedProgram, 0> programs;
  llvm::SmallVector<DataflowPlannedConnection> connections;
};

enum class DataflowRejectionKind {
  None,
  InvalidGraph,
  InvalidVariant,
  ResourceLimit,
  Unsupported,
};

llvm::StringRef stringifyDataflowRejectionKind(DataflowRejectionKind kind);

struct DataflowFeasibilityResult {
  bool feasible;
  DataflowRejectionKind rejectionKind;
  std::string message;

  static DataflowFeasibilityResult success();
  static DataflowFeasibilityResult reject(DataflowRejectionKind kind,
                                          llvm::StringRef message);
};

class DataflowFeasibilityModel {
public:
  virtual ~DataflowFeasibilityModel() = default;

  virtual DataflowFeasibilityResult
  evaluate(const DataflowGraph &graph,
           const DataflowMappingPlan &plan) const = 0;
};

/// Enforces graph coverage, variant shape, and connection integrity. Hardware
/// resource models can compose stricter checks on top of this baseline.
class StructuralDataflowFeasibilityModel final
    : public DataflowFeasibilityModel {
public:
  DataflowFeasibilityResult
  evaluate(const DataflowGraph &graph,
           const DataflowMappingPlan &plan) const override;
};

struct DataflowPlanCost;

DataflowMappingPlan buildTemporalFallbackPlan(const DataflowGraph &graph);

void printDataflowPlan(llvm::raw_ostream &os, const DataflowGraph &graph,
                       const DataflowMappingPlan &plan,
                       const DataflowPlanCost &cost);

class DataflowPlanMaterializer {
public:
  virtual ~DataflowPlanMaterializer() = default;

  virtual LogicalResult materialize(const DataflowGraph &graph,
                                    const DataflowMappingPlan &plan) const = 0;
};

/// The temporal fallback already matches the input IR, so its materializer is
/// intentionally a no-op. Fused and spatial strategies provide their own
/// implementations without changing the planning pass boundary.
class TemporalDataflowPlanMaterializer final : public DataflowPlanMaterializer {
public:
  LogicalResult materialize(const DataflowGraph &graph,
                            const DataflowMappingPlan &plan) const override;
};

} // namespace mlir::tt::d2m

#endif
