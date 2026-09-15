// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_ANALYSIS_DATAFLOWALLOCATIONFEEDBACK_H
#define TTMLIR_DIALECT_D2M_ANALYSIS_DATAFLOWALLOCATIONFEEDBACK_H
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LogicalResult.h"
#include <cstdint>
#include <optional>
#include <string>
namespace mlir::tt::d2m {
struct DataflowAllocationFeedback {
  std::string status;
  uint64_t l1CapacityBytes = 0;
  uint64_t dramCapacityBytes = 0;
  std::optional<uint64_t> l1UsageBytes;
  std::optional<uint64_t> dramUsageBytes;
  std::optional<uint64_t> l1ToDramCount;
};

/// Read allocator feedback without treating missing estimates as zero.
FailureOr<DataflowAllocationFeedback>
readDataflowAllocationFeedback(DictionaryAttr report, std::string &reason);

} // namespace mlir::tt::d2m
#endif
