set(TESTSUITE D2MGenericAnalysisTests)

add_mlir_unittest(${TESTSUITE}
    TestDataflowPlanning.cpp
    TestDataflowContracts.cpp
    TestDataflowExecution.cpp
    TestD2MGenericAnalysis.cpp
)

target_include_directories(${TESTSUITE}
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../
)

target_link_libraries(${TESTSUITE}
  PRIVATE
    TTMLIRCompilerStatic
    MLIRD2MPipelines
    TTMLIRTestUtils
    MLIRD2MPlanning
    MLIRD2MAnalysis
    MLIRD2MDialect
    MLIRD2MUtils
    MLIRLinalgSynchronizableOpInterfaceImpl
    MLIRTTTransforms
    MLIRTTCoreDialect
    MLIRTTIRDialect
    MLIRTTNNDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    coverage_config
)

target_compile_definitions(${TESTSUITE} PRIVATE
    TTMLIR_D2M_TEST_DIR="${PROJECT_SOURCE_DIR}/test/ttmlir/Dialect/D2M")
