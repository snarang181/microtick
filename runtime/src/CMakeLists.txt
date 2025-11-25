add_executable(microtick-engine
  engine.cpp
)

target_include_directories(microtick-engine PRIVATE
  ${PROJECT_SOURCE_DIR}/runtime/include
)
