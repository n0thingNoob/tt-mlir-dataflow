add_llvm_executable(ttmlir-opt ttmlir-opt.cpp)

llvm_update_compile_flags(ttmlir-opt)

if(TTMLIR_ENABLE_ASTRAIA)
  target_compile_definitions(ttmlir-opt PRIVATE TTMLIR_ENABLE_ASTRAIA)
  # Keep extension references ahead of the compiler's static archive so the
  # latter can satisfy the full transitive pipeline dependency graph.
  target_link_libraries(ttmlir-opt PRIVATE AstraiaPipelines AstraiaTransforms)
endif()

target_link_libraries(ttmlir-opt PRIVATE TTMLIRCompilerStatic MLIROptLib)
target_link_libraries(ttmlir-opt PUBLIC coverage_config)

mlir_check_all_link_libraries(ttmlir-opt)

install(TARGETS ttmlir-opt DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Test EXCLUDE_FROM_ALL)
