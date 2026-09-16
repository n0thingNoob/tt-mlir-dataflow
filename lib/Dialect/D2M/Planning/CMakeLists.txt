add_mlir_dialect_library(MLIRD2MPlanning
  DataflowPlan.cpp
  DataflowContracts.cpp
  DataflowCostModel.cpp

  LINK_LIBS PUBLIC
    MLIRD2MAnalysis
    MLIRD2MDialect
)
