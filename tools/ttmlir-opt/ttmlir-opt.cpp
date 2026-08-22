// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "ttmlir/RegisterAll.h"

#ifdef TTMLIR_ENABLE_ASTRAIA
#include "astraia/Pipelines/Pipelines.h"
#include "astraia/Transforms/Passes.h"
#endif

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::tt::registerAllPasses();
#ifdef TTMLIR_ENABLE_ASTRAIA
  mlir::tt::astraia::registerPasses();
  mlir::tt::astraia::registerPipelines();
#endif

  mlir::DialectRegistry registry;
  mlir::tt::registerAllDialects(registry);
  mlir::tt::registerAllExtensions(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "ttmlir optimizer driver\n", registry));
}
