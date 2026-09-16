// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/IR/OperationSupport.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "ttmlir/Dialect/D2M/Pipelines/DataflowExecution.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/Transforms/Passes.h"
#include "ttmlir/RegisterAll.h"
#include <gtest/gtest.h>

namespace mlir::tt::d2m {
class DataflowExecutionTest : public ::testing::Test {
protected:
  MLIRContext context;
  DataflowExecutionTest() {
    DialectRegistry registry;
    registerAllDialects(registry);
    registerAllExtensions(registry);
    context.appendDialectRegistry(registry);
  }
  OwningOpRef<ModuleOp> fixture(StringRef name) {
    return parseSourceFile<ModuleOp>(
        std::string(TTMLIR_D2M_TEST_DIR) + "/" + name.str(), &context);
  }
  bool equivalent(ModuleOp a, ModuleOp b) {
    return OperationEquivalence::isEquivalentTo(
        a, b, OperationEquivalence::IgnoreLocations);
  }
};

TEST_F(DataflowExecutionTest, MatchesNormalPipelineAndPreservesInput) {
  auto input = fixture("Transforms/dataflow_passthrough_e2e.mlir");
  ASSERT_TRUE(input);
  OwningOpRef<ModuleOp> original = cast<ModuleOp>(input->clone());
  OwningOpRef<ModuleOp> normal = cast<ModuleOp>(input->clone());
  ttmetal::D2MPipelineOptions options;
  PassManager pm(&context);
  ttmetal::createD2MFrontendPipeline(pm, options);
  ttmetal::createD2MBackendPipeline(pm, options);
  ASSERT_TRUE(succeeded(pm.run(*normal)));
  auto result = runDataflowAttempt(*input, DataflowStage::Input, options);
  ASSERT_EQ(result.status, DataflowAttemptStatus::Accepted) << result.reason;
  ASSERT_TRUE(result.artifact);
  EXPECT_TRUE(equivalent(*normal, *result.artifact));
  EXPECT_TRUE(equivalent(*input, *original));
  EXPECT_FALSE(result.cost.has_value());
  ASSERT_EQ(result.reports.size(), 3u);
  EXPECT_FALSE(result.reports[1].functions.empty());
}

TEST_F(DataflowExecutionTest,
       CapturesAllocationFailureAndDoesNotPoisonNextAttempt) {
  auto input = fixture("Transforms/dataflow_passthrough_e2e.mlir");
  ASSERT_TRUE(input);
  OwningOpRef<ModuleOp> original = cast<ModuleOp>(input->clone());
  ttmetal::D2MPipelineOptions small;
  small.testAssumel1Capacity = 16;
  auto rejected = runDataflowAttempt(*input, DataflowStage::Input, small);
  EXPECT_EQ(rejected.status, DataflowAttemptStatus::Rejected);
  EXPECT_FALSE(rejected.artifact);
  ASSERT_EQ(rejected.reports.size(), 2u);
  EXPECT_FALSE(rejected.reports.back().completed);
  EXPECT_EQ(rejected.reports.back().failedPass, "d2m-allocate");
  ASSERT_FALSE(rejected.reports.back().functions.empty());
  EXPECT_EQ(rejected.reports.back().functions[0].allocation.status,
            "l1_capacity_exceeded");
  EXPECT_TRUE(equivalent(*input, *original));
  ttmetal::D2MPipelineOptions normal;
  auto accepted = runDataflowAttempt(*input, DataflowStage::Input, normal);
  EXPECT_EQ(accepted.status, DataflowAttemptStatus::Accepted)
      << accepted.reason;
  EXPECT_TRUE(equivalent(*input, *original));
}

TEST_F(DataflowExecutionTest,
       CapturesBackendScratchFailureAfterAllocationSuccess) {
  auto input = fixture("allocate/lower_scratch_allocate_oom.mlir");
  ASSERT_TRUE(input);
  PassManager prepare(&context);
  prepare.addPass(ttcore::createTTCoreRegisterDevicePass());
  prepare.addPass(createD2MGenerateOuterLoops());
  ASSERT_TRUE(succeeded(prepare.run(*input)));
  OwningOpRef<ModuleOp> original = cast<ModuleOp>(input->clone());
  ttmetal::D2MPipelineOptions options;
  auto result = runDataflowAttempt(*input, DataflowStage::Blocking, options);
  EXPECT_EQ(result.status, DataflowAttemptStatus::Error);
  EXPECT_FALSE(result.artifact);
  ASSERT_EQ(result.reports.size(), 2u);
  EXPECT_TRUE(result.reports[0].completed);
  ASSERT_FALSE(result.reports[0].functions.empty());
  EXPECT_EQ(result.reports[0].functions[0].allocation.status, "success");
  EXPECT_FALSE(result.reports[1].completed);
  EXPECT_EQ(result.reports[1].failedPass, "d2m-lower-scratch-allocate");
  EXPECT_NE(result.reason.find("scratch buffer capacity"), std::string::npos);
  EXPECT_TRUE(equivalent(*input, *original));
}

TEST_F(DataflowExecutionTest, RejectsUnsupportedModeBeforeRunningPasses) {
  auto input = fixture("Transforms/dataflow_passthrough_e2e.mlir");
  ASSERT_TRUE(input);
  ttmetal::D2MPipelineOptions options;
  options.ttnnMode = true;
  auto result = runDataflowAttempt(*input, DataflowStage::Input, options);
  EXPECT_EQ(result.status, DataflowAttemptStatus::Unsupported);
  EXPECT_TRUE(result.reports.empty());
  EXPECT_FALSE(result.artifact);
}
TEST_F(DataflowExecutionTest, CostInterfacePreservesUnknownValues) {
  struct UnknownCost final : DataflowRealizationCostModel {
    DataflowPlanCost
    evaluate(ArrayRef<DataflowStageReport> reports) const override {
      EXPECT_EQ(reports.size(), 3u);
      return {};
    }
  } cost;
  auto input = fixture("Transforms/dataflow_passthrough_e2e.mlir");
  ASSERT_TRUE(input);
  ttmetal::D2MPipelineOptions options;
  auto result =
      runDataflowAttempt(*input, DataflowStage::Input, options, {}, &cost);
  ASSERT_EQ(result.status, DataflowAttemptStatus::Accepted) << result.reason;
  ASSERT_TRUE(result.cost);
  EXPECT_FALSE(result.cost->dramBytes);
  EXPECT_FALSE(result.cost->peakL1BytesPerCore);
}

TEST_F(DataflowExecutionTest, RejectsUnknownStartingStage) {
  auto input = fixture("Transforms/dataflow_passthrough_e2e.mlir");
  ASSERT_TRUE(input);
  ttmetal::D2MPipelineOptions options;
  auto result =
      runDataflowAttempt(*input, static_cast<DataflowStage>(99), options);
  EXPECT_EQ(result.status, DataflowAttemptStatus::Unsupported);
  EXPECT_TRUE(result.reports.empty());
}

} // namespace mlir::tt::d2m
