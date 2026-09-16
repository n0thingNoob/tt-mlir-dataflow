// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/Planning/DataflowContracts.h"
#include <gtest/gtest.h>

namespace mlir::tt::d2m {
TEST(DataflowContractsTest, RebindsCloneAndRejectsStaleStage) {
  MLIRContext context;
  context.loadDialect<D2MDialect, func::FuncDialect>();
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  auto function =
      func::FuncOp::create(loc, "main", builder.getFunctionType({}, {}));
  module->push_back(function);
  builder.setInsertionPointToStart(function.addEntryBlock());
  // Binding does not inspect kernel legality; that remains a pass contract.
  OperationState state(loc, GenericOp::getOperationName());
  auto generic = cast<GenericOp>(builder.create(state));
  builder.create<func::ReturnOp>(loc);
  DataflowStagePlan plan(*module, DataflowStage::Blocking);
  std::string reason;
  auto binding = plan.bind("main", 0, reason);
  ASSERT_TRUE(succeeded(binding));
  OwningOpRef<ModuleOp> clone = cast<ModuleOp>(module->clone());
  auto resolved =
      plan.resolve(*binding, *clone, DataflowStage::Blocking, reason);
  ASSERT_TRUE(succeeded(resolved));
  EXPECT_NE(resolved->getOperation(), generic.getOperation());
  EXPECT_TRUE(failed(
      plan.resolve(*binding, *clone, DataflowStage::Allocation, reason)));
  DataflowStagePlan other(*module, DataflowStage::Blocking);
  EXPECT_TRUE(
      failed(other.resolve(*binding, *clone, DataflowStage::Blocking, reason)));
  resolved->getOperation()->setAttr("changed", builder.getUnitAttr());
  EXPECT_TRUE(
      failed(plan.resolve(*binding, *clone, DataflowStage::Blocking, reason)));
  EXPECT_TRUE(succeeded(
      plan.resolve(*binding, *module, DataflowStage::Blocking, reason)));
  EXPECT_TRUE(failed(plan.bind("missing", 0, reason)));
  EXPECT_TRUE(failed(plan.bind("main", 1, reason)));
}
} // namespace mlir::tt::d2m
