//===- microtick-opt.cpp ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Microtick/MicrotickDialect.h"
#include "Microtick/MicrotickPasses.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<microtick::tick::TickDialect>();
  registerAllDialects(registry);

  // Register all passes.
  registerAllPasses();
  microtick::tick::registerMicrotickPasses();

  return asMainReturnCode(MlirOptMain(argc, argv, "MicroTick optimizer driver\n", registry));
}
