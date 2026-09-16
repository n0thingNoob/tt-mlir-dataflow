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

TEST_F(DataflowExecutionTest, AppliesExplicitBlockingAndPreservesBaseline) {
  auto input = fixture("allocate/reblock_explicit_plan.mlir");
  ASSERT_TRUE(input);
  PassManager prepare(&context);
  prepare.addPass(ttcore::createTTCoreRegisterDevicePass());
  ASSERT_TRUE(succeeded(prepare.run(*input)));
  OwningOpRef<ModuleOp> original = cast<ModuleOp>(input->clone());
  DataflowRequirements requirements;
  requirements.blocking.push_back({"planned_matmul", 0, {1, 1, 4}});
  ttmetal::D2MPipelineOptions options;
  options.testAssumel1Capacity = 8388608;
  auto result = runDataflowAttempt(*input, DataflowStage::Blocking, options,
                                   requirements);
  ASSERT_EQ(result.status, DataflowAttemptStatus::Accepted) << result.reason;
  ASSERT_TRUE(result.artifact);
  EXPECT_TRUE(equivalent(*input, *original));
  // The post-reblock check verified the realized factors; private IDs must not
  // leak into the accepted artifact.
  result.artifact->walk([&](Operation *op) {
    EXPECT_FALSE(op->hasAttr("d2m.execution_requirement"));
  });
  requirements.blocking[0].factors = {1, 1, 3};
  auto conflict = runDataflowAttempt(*input, DataflowStage::Blocking, options,
                                     requirements);
  EXPECT_EQ(conflict.status, DataflowAttemptStatus::Unsupported);
  EXPECT_NE(conflict.reason.find("conflicts"), std::string::npos);
  EXPECT_FALSE(conflict.artifact);
  EXPECT_TRUE(equivalent(*input, *original));
}

TEST_F(DataflowExecutionTest, RejectsUnsupportedBlockingWithoutMutatingInput) {
  auto input = fixture("allocate/reblock_explicit_plan.mlir");
  ASSERT_TRUE(input);
  PassManager prepare(&context);
  prepare.addPass(ttcore::createTTCoreRegisterDevicePass());
  ASSERT_TRUE(succeeded(prepare.run(*input)));
  input->walk([](GenericOp generic) {
    generic->removeAttr("d2m.planned_block_factors");
  });
  OwningOpRef<ModuleOp> original = cast<ModuleOp>(input->clone());
  DataflowRequirements requirements;
  requirements.blocking.push_back({"planned_matmul", 0, {1, 1, 3}});
  ttmetal::D2MPipelineOptions options;
  auto invalid = runDataflowAttempt(*input, DataflowStage::Blocking, options,
                                    requirements);
  EXPECT_EQ(invalid.status, DataflowAttemptStatus::Unsupported);
  EXPECT_NE(invalid.reason.find("divide"), std::string::npos);
  requirements.blocking[0].factors = {1, 1, 4};
  requirements.blocking.push_back(requirements.blocking[0]);
  auto duplicate = runDataflowAttempt(*input, DataflowStage::Blocking, options,
                                      requirements);
  EXPECT_EQ(duplicate.status, DataflowAttemptStatus::Unsupported);
  EXPECT_NE(duplicate.reason.find("duplicate"), std::string::npos);
  EXPECT_TRUE(equivalent(*input, *original));
}

TEST_F(DataflowExecutionTest,
       ReportsSuccessfulSpillAndHonorsOutputSpillPolicy) {
  auto input = fixture("allocate/allocate_intermediate_outputs.mlir");
  ASSERT_TRUE(input);
  PassManager prepare(&context);
  prepare.addPass(ttcore::createTTCoreRegisterDevicePass());
  ASSERT_TRUE(succeeded(prepare.run(*input)));
  OwningOpRef<ModuleOp> original = cast<ModuleOp>(input->clone());
  ttmetal::D2MPipelineOptions options;
  options.testBufferSizePolicy = "max";
  options.forceSpillToDramIfLegal = true;
  DataflowRequirements allow;
  allow.allowIntermediateOutputSpilling = true;
  auto spilled =
      runDataflowAttempt(*input, DataflowStage::Blocking, options, allow);
  ASSERT_EQ(spilled.status, DataflowAttemptStatus::Accepted) << spilled.reason;
  ASSERT_FALSE(spilled.reports[0].functions.empty());
  auto &feedback = spilled.reports[0].functions[0].allocation;
  ASSERT_TRUE(feedback.intermediateOutputSpillCount);
  EXPECT_GT(*feedback.intermediateOutputSpillCount, 0u);
  DataflowRequirements forbid;
  forbid.allowIntermediateOutputSpilling = false;
  forbid.blocking.push_back({"intermediate_chain", 0, {1, 1}});
  forbid.blocking.push_back({"intermediate_chain", 1, {1, 1}});
  auto kept =
      runDataflowAttempt(*input, DataflowStage::Blocking, options, forbid);
  ASSERT_EQ(kept.status, DataflowAttemptStatus::Accepted) << kept.reason;
  ASSERT_FALSE(kept.reports[0].functions.empty());
  EXPECT_EQ(
      kept.reports[0].functions[0].allocation.intermediateOutputSpillCount,
      std::optional<uint64_t>(0));
  EXPECT_TRUE(equivalent(*input, *original));
}

TEST_F(DataflowExecutionTest, RejectsPreparedInputWithoutDescriptor) {
  auto input = fixture("allocate/reblock_explicit_plan.mlir");
  ASSERT_TRUE(input);
  ttmetal::D2MPipelineOptions options;
  auto result = runDataflowAttempt(*input, DataflowStage::Blocking, options);
  EXPECT_EQ(result.status, DataflowAttemptStatus::Unsupported);
  EXPECT_NE(result.reason.find("system descriptor"), std::string::npos);
  EXPECT_TRUE(result.reports.empty());
}

} // namespace mlir::tt::d2m
