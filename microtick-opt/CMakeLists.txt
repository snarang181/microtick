get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
message(STATUS "dialect libs ${dialect_libs}")
message(STATUS "conversion libs ${conversion_libs}")
set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIROptLib     
  MLIRRegisterAllDialects       
)

add_executable(microtick-opt microtick-opt.cpp)

llvm_update_compile_flags(microtick-opt)
target_link_libraries(microtick-opt PRIVATE ${LIBS})