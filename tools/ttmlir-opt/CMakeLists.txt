add_llvm_executable(ttmlir-opt ttmlir-opt.cpp)

llvm_update_compile_flags(ttmlir-opt)
target_link_libraries(ttmlir-opt PRIVATE TTMLIRCompilerStatic MLIROptLib)
target_link_libraries(ttmlir-opt PUBLIC coverage_config)

if(TTMLIR_ENABLE_ASTRAIA)
  target_compile_definitions(ttmlir-opt PRIVATE TTMLIR_ENABLE_ASTRAIA)
  target_link_libraries(ttmlir-opt PRIVATE AstraiaTransforms)
endif()

mlir_check_all_link_libraries(ttmlir-opt)

install(TARGETS ttmlir-opt DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Test EXCLUDE_FROM_ALL)
