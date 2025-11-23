add_mlir_dialect_library(MLIRMicrotick
  MicrotickDialect.cpp
  MicrotickOps.cpp
  MicrotickPasses.cpp
  MicrotickLowerRuntime.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Microtick

  DEPENDS
    MLIRMicrotickOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
)
