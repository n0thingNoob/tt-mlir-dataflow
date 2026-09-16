// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/DataflowAllocationFeedback.h"
namespace mlir::tt::d2m {
FailureOr<DataflowAllocationFeedback>
readDataflowAllocationFeedback(DictionaryAttr report, std::string &reason) {
  reason.clear();
  if (!report) {
    reason = "missing d2m.allocation_report";
    return failure();
  }
  auto version = report.getAs<IntegerAttr>("version");
  if (!version || !version.getType().isInteger(64) || version.getInt() != 1) {
    reason = "expected allocation report version 1";
    return failure();
  }
  auto status = report.getAs<StringAttr>("status");
  if (!status ||
      (status.getValue() != "success" && status.getValue() != "failed" &&
       status.getValue() != "l1_capacity_exceeded" &&
       status.getValue() != "dram_capacity_exceeded")) {
    reason = "invalid allocation report status";
    return failure();
  }
  auto read = [&](StringRef name, bool required,
                  std::optional<uint64_t> &value) -> LogicalResult {
    Attribute attr = report.get(name);
    if (!attr && !required) {
      return success();
    }
    auto number = dyn_cast_or_null<IntegerAttr>(attr);
    if (!number || !number.getType().isInteger(64) || number.getInt() < 0) {
      reason = "expected nonnegative i64 field: " + name.str();
      return failure();
    }
    value = number.getInt();
    return success();
  };
  DataflowAllocationFeedback feedback;
  feedback.status = status.getValue().str();
  bool succeeded = feedback.status == "success";
  std::optional<uint64_t> l1Capacity, dramCapacity;
  if (failed(read("l1_capacity_bytes", true, l1Capacity)) ||
      failed(read("dram_capacity_bytes", true, dramCapacity)) ||
      failed(read("l1_usage_bytes", succeeded, feedback.l1UsageBytes)) ||
      failed(read("dram_usage_bytes", succeeded, feedback.dramUsageBytes)) ||
      failed(read("l1_to_dram_count", succeeded, feedback.l1ToDramCount)) ||
      failed(read("intermediate_output_spill_count", false,
                  feedback.intermediateOutputSpillCount))) {
    return failure();
  }
  feedback.l1CapacityBytes = *l1Capacity;
  feedback.dramCapacityBytes = *dramCapacity;
  if (succeeded && (*feedback.l1UsageBytes > feedback.l1CapacityBytes ||
                    *feedback.dramUsageBytes > feedback.dramCapacityBytes)) {
    reason = "successful allocation report exceeds capacity";
    return failure();
  }
  return feedback;
}

} // namespace mlir::tt::d2m
