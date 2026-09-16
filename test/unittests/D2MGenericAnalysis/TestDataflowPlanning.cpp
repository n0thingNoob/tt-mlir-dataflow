// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/DataflowAllocationFeedback.h"
#include "ttmlir/Dialect/D2M/Planning/DataflowCostModel.h"

#include "mlir/IR/Builders.h"
#include <gtest/gtest.h>

#include <limits>

namespace mlir::tt::d2m {
namespace {

TEST(DataflowPlanningTest, ReadsAllocationFeedback) {
  MLIRContext context;
  Builder builder(&context);
  NamedAttrList report;
  report.set("version", builder.getI64IntegerAttr(1));
  report.set("status", builder.getStringAttr("success"));
  report.set("l1_capacity_bytes", builder.getI64IntegerAttr(1024));
  report.set("dram_capacity_bytes", builder.getI64IntegerAttr(8192));
  report.set("l1_usage_bytes", builder.getI64IntegerAttr(512));
  report.set("dram_usage_bytes", builder.getI64IntegerAttr(2048));
  report.set("l1_to_dram_count", builder.getI64IntegerAttr(1));
  std::string reason;
  auto read = [&] {
    return readDataflowAllocationFeedback(report.getDictionary(&context),
                                          reason);
  };
  auto feedback = read();
  ASSERT_TRUE(succeeded(feedback));
  EXPECT_EQ(feedback->l1UsageBytes, 512u);
  EXPECT_EQ(feedback->dramUsageBytes, 2048u);
  EXPECT_EQ(feedback->l1ToDramCount, 1u);
  EXPECT_TRUE(reason.empty());

  report.erase("dram_usage_bytes");
  EXPECT_TRUE(failed(read()));
  report.set("status", builder.getStringAttr("l1_capacity_exceeded"));
  feedback = read();
  ASSERT_TRUE(succeeded(feedback));
  EXPECT_FALSE(feedback->dramUsageBytes.has_value());

  report.set("l1_usage_bytes", builder.getI64IntegerAttr(-1));
  EXPECT_TRUE(failed(read()));
  report.set("l1_usage_bytes", builder.getI64IntegerAttr(2048));
  report.set("dram_usage_bytes", builder.getI64IntegerAttr(0));
  report.set("status", builder.getStringAttr("success"));
  EXPECT_TRUE(failed(read()));
  EXPECT_EQ(reason, "successful allocation report exceeds capacity");
  report.set("version", builder.getI64IntegerAttr(2));
  EXPECT_TRUE(failed(read()));
  EXPECT_EQ(reason, "expected allocation report version 1");
  EXPECT_TRUE(failed(readDataflowAllocationFeedback({}, reason)));
  EXPECT_EQ(reason, "missing d2m.allocation_report");
}

DataflowProgramVariant
makeVariant(GenericOp member, llvm::SmallVector<int64_t> grid,
            ProgramResourceEstimate resources, uint64_t computeCycles,
            uint64_t dataMovementCycles, float confidence) {
  ProgramCostEstimate cost;
  cost.computeCycles = computeCycles;
  cost.dataMovementCycles = dataMovementCycles;
  cost.confidence = confidence;
  return DataflowProgramVariant(/*variantId=*/0, {member}, std::move(grid), {},
                                resources, cost);
}

TEST(DataflowPlanningTest, AggregatesTemporalPlanCost) {
  DataflowGraph graph({}, nullptr, /*scopeOrdinal=*/0, {}, {});
  llvm::SmallVector<DataflowPlannedProgram, 0> programs;
  programs.push_back(
      {makeVariant(/*member=*/{}, {1, 2},
                   {/*l1BytesPerCore=*/100, /*dramBytes=*/10,
                    /*nocBytes=*/20, /*cbCount=*/2, /*dstTiles=*/1},
                   /*computeCycles=*/100, /*dataMovementCycles=*/50,
                   /*confidence=*/0.8F),
       {}});
  programs.push_back(
      {makeVariant(/*member=*/{}, {2, 2},
                   {/*l1BytesPerCore=*/200, /*dramBytes=*/30,
                    /*nocBytes=*/40, /*cbCount=*/3, /*dstTiles=*/2},
                   /*computeCycles=*/60, /*dataMovementCycles=*/80,
                   /*confidence=*/0.6F),
       {}});
  DataflowMappingPlan plan({}, nullptr, /*scopeOrdinal=*/0,
                           DataflowPlanStrategy::Temporal, std::move(programs),
                           {});

  DataflowPlanCost cost = AnalyticalDataflowCostModel().evaluate(graph, plan);

  EXPECT_EQ(cost.latencyCycles, 180u);
  EXPECT_FALSE(cost.initiationIntervalCycles.has_value());
  EXPECT_EQ(cost.dramBytes, 40u);
  EXPECT_EQ(cost.nocBytes, 60u);
  EXPECT_EQ(cost.peakL1BytesPerCore, 200u);
  EXPECT_EQ(cost.occupiedCores, 4u);
  EXPECT_EQ(cost.programCount, 2u);
  EXPECT_FLOAT_EQ(cost.confidence, 0.6F);
}

TEST(DataflowPlanningTest, AggregatesSpatialPlanInitiationInterval) {
  DataflowGraph graph({}, nullptr, /*scopeOrdinal=*/0, {}, {});
  llvm::SmallVector<DataflowPlannedProgram, 0> programs;
  programs.push_back(
      {makeVariant(/*member=*/{}, {1, 2}, {}, /*computeCycles=*/100,
                   /*dataMovementCycles=*/50, /*confidence=*/0.8F),
       {0, 0}});
  programs.push_back(
      {makeVariant(/*member=*/{}, {2, 2}, {}, /*computeCycles=*/60,
                   /*dataMovementCycles=*/80, /*confidence=*/0.6F),
       {1, 0}});
  DataflowMappingPlan plan({}, nullptr, /*scopeOrdinal=*/0,
                           DataflowPlanStrategy::Spatial, std::move(programs),
                           {});

  DataflowPlanCost cost = AnalyticalDataflowCostModel().evaluate(graph, plan);

  EXPECT_FALSE(cost.latencyCycles.has_value());
  EXPECT_EQ(cost.initiationIntervalCycles, 100u);
  EXPECT_EQ(cost.occupiedCores, 6u);
  EXPECT_FLOAT_EQ(cost.confidence, 0.6F);
}

TEST(DataflowPlanningTest, LeavesCyclesUnknownWithoutProgramEstimate) {
  DataflowGraph graph({}, nullptr, /*scopeOrdinal=*/0, {}, {});
  llvm::SmallVector<DataflowPlannedProgram, 0> programs;
  programs.push_back(
      {DataflowProgramVariant(/*variantId=*/0, {GenericOp{}}, {1, 1}, {}), {}});
  DataflowMappingPlan plan({}, nullptr, /*scopeOrdinal=*/0,
                           DataflowPlanStrategy::Temporal, std::move(programs),
                           {});

  DataflowPlanCost cost = AnalyticalDataflowCostModel().evaluate(graph, plan);

  EXPECT_FALSE(cost.latencyCycles.has_value());
  EXPECT_FALSE(cost.initiationIntervalCycles.has_value());
  EXPECT_FLOAT_EQ(cost.confidence, 0.0F);
}

TEST(DataflowPlanningTest, UnknownResourcesDoNotBecomeZero) {
  DataflowGraph graph({}, nullptr, 0, {}, {});
  llvm::SmallVector<DataflowPlannedProgram, 0> programs;
  programs.push_back(
      {DataflowProgramVariant(0, {GenericOp{}}, {1, 1}, {}), {}});
  DataflowMappingPlan plan({}, nullptr, 0, DataflowPlanStrategy::Temporal,
                           std::move(programs), {});
  auto cost = AnalyticalDataflowCostModel().evaluate(graph, plan);
  EXPECT_FALSE(cost.dramBytes.has_value());
  EXPECT_FALSE(cost.nocBytes.has_value());
  EXPECT_FALSE(cost.peakL1BytesPerCore.has_value());
  EXPECT_FALSE(cost.spillCount.has_value());
}

TEST(DataflowPlanningTest, SaturatesSpatialCoreCount) {
  DataflowGraph graph({}, nullptr, /*scopeOrdinal=*/0, {}, {});
  llvm::SmallVector<DataflowPlannedProgram, 0> programs;
  programs.push_back(
      {makeVariant(/*member=*/{},
                   {std::numeric_limits<int64_t>::max(),
                    std::numeric_limits<int64_t>::max()},
                   {}, /*computeCycles=*/1, /*dataMovementCycles=*/1,
                   /*confidence=*/1.0F),
       {}});
  programs.push_back(
      {makeVariant(/*member=*/{}, {2, 2}, {}, /*computeCycles=*/1,
                   /*dataMovementCycles=*/1, /*confidence=*/1.0F),
       {}});
  DataflowMappingPlan plan({}, nullptr, /*scopeOrdinal=*/0,
                           DataflowPlanStrategy::Spatial, std::move(programs),
                           {});

  DataflowPlanCost cost = AnalyticalDataflowCostModel().evaluate(graph, plan);

  EXPECT_EQ(cost.occupiedCores, std::numeric_limits<uint32_t>::max());
}

TEST(DataflowPlanningTest, TreatsInvalidGridAsUsingNoCores) {
  DataflowGraph graph({}, nullptr, /*scopeOrdinal=*/0, {}, {});
  llvm::SmallVector<DataflowPlannedProgram, 0> programs;
  programs.push_back(
      {makeVariant(/*member=*/{}, {-1, 2}, {}, /*computeCycles=*/1,
                   /*dataMovementCycles=*/1, /*confidence=*/1.0F),
       {}});
  DataflowMappingPlan plan({}, nullptr, /*scopeOrdinal=*/0,
                           DataflowPlanStrategy::Temporal, std::move(programs),
                           {});

  DataflowPlanCost cost = AnalyticalDataflowCostModel().evaluate(graph, plan);

  EXPECT_EQ(cost.occupiedCores, 0u);
}

TEST(DataflowPlanningTest, RejectsUnknownNodeReference) {
  DataflowGraph graph({}, nullptr, /*scopeOrdinal=*/0, {}, {});
  llvm::SmallVector<DataflowPlannedProgram, 0> programs;
  programs.push_back(
      {DataflowProgramVariant(/*variantId=*/0, {GenericOp{}}, {1, 1}, {}), {}});
  DataflowMappingPlan plan({}, nullptr, /*scopeOrdinal=*/0,
                           DataflowPlanStrategy::Temporal, std::move(programs),
                           {});

  DataflowFeasibilityResult result =
      StructuralDataflowFeasibilityModel().evaluate(graph, plan);

  EXPECT_FALSE(result.feasible);
  EXPECT_EQ(result.rejectionKind, DataflowRejectionKind::InvalidVariant);
}

TEST(DataflowPlanningTest, RejectsInvalidNodeDependency) {
  DataflowGraph graph({}, nullptr, /*scopeOrdinal=*/0, {}, {{}});
  DataflowMappingPlan plan({}, nullptr, /*scopeOrdinal=*/0,
                           DataflowPlanStrategy::Temporal, {}, {});

  DataflowFeasibilityResult result =
      StructuralDataflowFeasibilityModel().evaluate(graph, plan);

  EXPECT_FALSE(result.feasible);
  EXPECT_EQ(result.rejectionKind, DataflowRejectionKind::InvalidGraph);
}

// Minimal registered ops exercise graph identity without unrelated lowering.
struct GraphFixture {
  MLIRContext context;
  OwningOpRef<ModuleOp> module;
  func::FuncOp function;
  GraphFixture() {
    context.loadDialect<D2MDialect, func::FuncDialect>();
    module = ModuleOp::create(UnknownLoc::get(&context));
    function = func::FuncOp::create(module->getLoc(), "graph",
                                    FunctionType::get(&context, {}, {}));
    module->push_back(function);
    function.addEntryBlock();
  }
  GenericOp generic(ValueRange inputs = {}, ValueRange outputs = {},
                    ValueRange additional = {}) {
    Builder builder(&context);
    OperationState state(module->getLoc(), GenericOp::getOperationName());
    state.addOperands(inputs);
    state.addOperands(outputs);
    state.addOperands(additional);
    state.addTypes({builder.getI32Type(), builder.getI32Type()});
    state.addAttribute("operandSegmentSizes",
                       builder.getDenseI32ArrayAttr(
                           {static_cast<int32_t>(inputs.size()),
                            static_cast<int32_t>(outputs.size()),
                            static_cast<int32_t>(additional.size())}));
    auto op = cast<GenericOp>(Operation::create(state));
    function.getBody().front().push_back(op);
    return op;
  }
};

TEST(DataflowPlanningTest, KeepsExactResultsAndRepeatedConsumerInputs) {
  GraphFixture fixture;
  auto producer = fixture.generic();
  auto consumer = fixture.generic(
      {producer.getResult(1), producer.getResult(0), producer.getResult(1)});
  auto graphs = buildDataflowGraphs(*fixture.module);
  ASSERT_EQ(graphs.size(), 1u);
  auto edges = graphs[0].getEdges();
  ASSERT_EQ(edges.size(), 3u);
  for (unsigned i = 0; i < edges.size(); ++i) {
    EXPECT_EQ(edges[i].producer, producer);
    EXPECT_EQ(edges[i].consumer, consumer);
    EXPECT_EQ(edges[i].consumerOperand, i);
    EXPECT_EQ(edges[i].consumerValue, consumer.getOperands()[i]);
    EXPECT_EQ(edges[i].producerValue, consumer.getOperands()[i]);
  }
}

TEST(DataflowPlanningTest, TracksOutputInitializersAndAdditionalOperands) {
  GraphFixture fixture;
  auto producer = fixture.generic();
  auto consumer =
      fixture.generic({}, producer.getResult(0), producer.getResult(1));
  auto graphs = buildDataflowGraphs(*fixture.module);
  ASSERT_EQ(graphs.size(), 1u);
  ASSERT_EQ(graphs[0].getEdges().size(), 2u);
  for (unsigned i = 0; i < 2; ++i) {
    const auto &edge = graphs[0].getEdges()[i];
    EXPECT_EQ(edge.producer, producer);
    EXPECT_EQ(edge.consumer, consumer);
    EXPECT_EQ(edge.consumerOperand, i);
    EXPECT_EQ(edge.producerValue, producer.getResult(i));
  }
}

TEST(DataflowPlanningTest, TracksDependenciesThroughInterveningOperations) {
  GraphFixture fixture;
  auto producer = fixture.generic();
  OperationState state(fixture.module->getLoc(),
                       "builtin.unrealized_conversion_cast");
  state.addOperands(producer.getResult(1));
  state.addTypes(producer.getResult(1).getType());
  Operation *view = Operation::create(state);
  fixture.function.getBody().front().push_back(view);
  auto consumer = fixture.generic(view->getResult(0));
  auto graphs = buildDataflowGraphs(*fixture.module);
  ASSERT_EQ(graphs.size(), 1u);
  ASSERT_EQ(graphs[0].getEdges().size(), 1u);
  const auto &edge = graphs[0].getEdges()[0];
  EXPECT_EQ(edge.producerValue, producer.getResult(1));
  EXPECT_EQ(edge.consumerValue, view->getResult(0));
  EXPECT_EQ(edge.consumer, consumer);
}

TEST(DataflowPlanningTest, RejectsReorderedTemporalDependency) {
  GraphFixture fixture;
  auto producer = fixture.generic();
  auto consumer = fixture.generic(producer.getResult(0));
  auto graphs = buildDataflowGraphs(*fixture.module);
  ASSERT_EQ(graphs.size(), 1u);
  const auto &graph = graphs[0];
  llvm::SmallVector<DataflowPlannedProgram, 0> programs;
  programs.push_back({DataflowProgramVariant(0, {consumer}, {1, 1}, {}), {}});
  programs.push_back({DataflowProgramVariant(0, {producer}, {1, 1}, {}), {}});
  DataflowMappingPlan plan(graph.getFunction(), graph.getBlock(), 0,
                           DataflowPlanStrategy::Temporal, std::move(programs),
                           {{1, 0, DataflowConnectionKind::Materialized, 1}});
  auto result = StructuralDataflowFeasibilityModel().evaluate(graph, plan);
  EXPECT_FALSE(result.feasible);
  EXPECT_EQ(result.rejectionKind, DataflowRejectionKind::InvalidGraph);
}

} // namespace
} // namespace mlir::tt::d2m
